"""FastAPI server for hosting Liquid AI models with an OpenAI-compatible API."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from liquid_host.config import GenerationConfig, MODEL_REGISTRY, get_model_spec
from liquid_host.mcp_client import McpClientManager
from liquid_host.models.manager import ModelManager

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Liquid Host",
    description="Local inference server for Liquid AI LFM models",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Shared instances — initialized on startup
_manager: ModelManager | None = None
_mcp: McpClientManager | None = None


def get_manager() -> ModelManager:
    if _manager is None:
        raise HTTPException(503, "Server not initialized")
    return _manager


# ── Request / Response schemas (OpenAI-compatible) ─────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = 0.3
    max_tokens: int = Field(default=4096, alias="max_tokens")
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    min_p: float = 0.15
    stream: bool = False
    use_tools: bool = True


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "liquid-ai"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class ServerStatus(BaseModel):
    status: str
    loaded_model: str | None
    available_models: int
    mcp_connected: bool = False
    mcp_tools: list[str] = []


# ── Endpoints ──────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    mgr = get_manager()
    downloaded = mgr.list_downloaded()
    models = [
        ModelInfo(id=d["repo_id"]) for d in downloaded
    ]
    # Also include the currently loaded model
    if mgr.is_loaded:
        loaded_id = mgr._loaded_spec.repo_id
        if not any(m.id == loaded_id for m in models):
            models.insert(0, ModelInfo(id=loaded_id))
    return ModelListResponse(data=models)


@app.get("/status", response_model=ServerStatus)
async def status():
    mgr = get_manager()
    return ServerStatus(
        status="ready" if mgr.is_loaded else "no_model_loaded",
        loaded_model=mgr.loaded_model_name,
        available_models=len(MODEL_REGISTRY),
        mcp_connected=_mcp.is_connected if _mcp else False,
        mcp_tools=[t.name for t in _mcp.tools] if _mcp else [],
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    mgr = get_manager()
    if not mgr.is_loaded:
        raise HTTPException(503, "No model loaded. Load a model first via /load.")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Inject system prompt with current date/time and persona
    now = datetime.now().strftime("%A, %B %-d, %Y, %-I:%M %p")
    system_text = (
        f"Current date and time: {now}.\n\n"
        "You are a senior institutional financial research analyst. "
        "You provide rigorous, data-driven analysis grounded in primary sources such as "
        "earnings transcripts, SEC filings, company guidance, and macroeconomic data. "
        "Your responses should be precise, well-structured, and suitable for an institutional audience — "
        "portfolio managers, buy-side analysts, and investment committees. "
        "When analyzing companies, focus on key financial metrics, margin trends, revenue drivers, "
        "competitive positioning, and forward guidance. "
        "Cite specific figures, quarters, and sources where possible. "
        "Flag risks, assumptions, and areas of uncertainty clearly. "
        "Avoid speculation and retail-oriented language. "
        "If you have access to tools, use them proactively to retrieve relevant data before answering.\n\n"
        "IMPORTANT: Keep your <think> block extremely brief — 2-3 short sentences maximum. "
        "State only what you need to do, then do it immediately. No analysis, no weighing options, "
        "no restating the question, no self-dialogue in thinking. Just act.\n\n"
        "You may call multiple tools in a single response — there is no limit. "
        "Call as many as needed to fully answer the question.\n\n"
        "When calling a tool, write a short user-friendly status message on the line immediately "
        "before each tool call describing what you are doing, e.g.:\n"
        "Looking up recent NFLX earnings events...\n"
        "[find_events(bloomberg_ticker='NFLX', event_type='earnings')]\n"
        "This status line will be shown to the user while the tool runs."
    )
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = system_text + "\n\n" + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": system_text})

    logger.info("System prompt: %s", messages[0]["content"][:300])
    logger.info("Messages being sent to model (%d total):", len(messages))
    for i, m in enumerate(messages):
        logger.info("  msg[%d] role=%s content=%.200s", i, m["role"], m["content"])

    config = GenerationConfig(
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        min_p=request.min_p,
        repetition_penalty=request.repetition_penalty,
        top_p=request.top_p,
        do_sample=request.temperature > 0,
    )

    # Use tool-calling loop when MCP servers are connected
    use_tools = request.use_tools and _mcp and _mcp.is_connected

    logger.info(
        "POST /v1/chat/completions — %d messages, stream=%s, use_tools=%s, max_tokens=%d, temp=%.2f",
        len(messages), request.stream, use_tools, request.max_tokens, request.temperature,
    )

    _sse_headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    if use_tools:
        logger.info("Routing to tool-calling stream")
        return StreamingResponse(
            _stream_tool_response(mgr, messages, config),
            media_type="text/event-stream",
            headers=_sse_headers,
        )

    if request.stream:
        logger.info("Routing to standard stream")
        return StreamingResponse(
            _stream_response(mgr, messages, config),
            media_type="text/event-stream",
            headers=_sse_headers,
        )

    logger.info("Routing to non-streaming generate")
    response_text = mgr.generate(messages, config)
    logger.info("Response generated (%d chars)", len(response_text))

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=mgr.loaded_model_name or "unknown",
        choices=[
            ChatChoice(message=ChatMessage(role="assistant", content=response_text))
        ],
        usage=UsageInfo(),
    )


async def _stream_response(mgr: ModelManager, messages: list[dict], config: GenerationConfig):
    """Yield SSE chunks for streaming responses."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = mgr.loaded_model_name or "unknown"

    try:
        async for token, _ in mgr._async_stream_raw(messages, config):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {__import__('json').dumps(chunk)}\n\n"
    except Exception as e:
        logger.error("_stream_response: generation failed — %s", e)
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            "error": {"message": str(e)},
        }
        yield f"data: {__import__('json').dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Final chunk
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {__import__('json').dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_tool_response(mgr: ModelManager, messages: list[dict], config: GenerationConfig):
    """Yield SSE chunks for tool-calling workflow with status updates."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = mgr.loaded_model_name or "unknown"

    async for event in mgr.generate_with_tools(messages, _mcp, config):
        event_type = event["type"]

        if event_type == "status":
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"status": event["content"]},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {__import__('json').dumps(chunk)}\n\n"

        elif event_type == "thinking":
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"thinking": event["content"]},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {__import__('json').dumps(chunk)}\n\n"

        elif event_type == "token":
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": event["content"]},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {__import__('json').dumps(chunk)}\n\n"

        elif event_type == "error":
            error_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "error",
                    }
                ],
                "error": {"message": event["content"]},
            }
            yield f"data: {__import__('json').dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        elif event_type == "done":
            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {__import__('json').dumps(final)}\n\n"
            yield "data: [DONE]\n\n"


# ── Model management endpoints ────────────────────────────────────


class LoadModelRequest(BaseModel):
    model: str
    device_map: str = "auto"
    dtype: str | None = None
    use_flash_attn: bool = False


class LoadGgufRequest(BaseModel):
    path: str
    n_ctx: int = 128000
    n_threads: int | None = None
    n_gpu_layers: int = 0


@app.post("/load")
async def load_model(request: LoadModelRequest):
    mgr = get_manager()
    try:
        spec = get_model_spec(request.model)
    except KeyError as e:
        raise HTTPException(404, str(e))

    mgr.load(
        request.model,
        device_map=request.device_map,
        dtype=request.dtype,
        use_flash_attn=request.use_flash_attn,
    )
    return {"status": "loaded", "model": spec.name}


@app.post("/load-gguf")
async def load_gguf_model(request: LoadGgufRequest):
    mgr = get_manager()
    try:
        mgr.load_gguf(
            request.path,
            n_ctx=request.n_ctx,
            n_threads=request.n_threads,
            n_gpu_layers=request.n_gpu_layers,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"status": "loaded", "model": mgr.loaded_model_name, "backend": "llamacpp"}


@app.post("/unload")
async def unload_model():
    mgr = get_manager()
    mgr.unload()
    return {"status": "unloaded"}


class DownloadModelRequest(BaseModel):
    model: str
    force: bool = False


@app.post("/download")
async def download_model(request: DownloadModelRequest):
    mgr = get_manager()
    try:
        path = mgr.download(request.model, force=request.force)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {"status": "downloaded", "path": str(path)}


# ── MCP management endpoints ───────────────────────────────────────


@app.get("/mcp/status")
async def mcp_status():
    if not _mcp:
        return {"connected": False, "servers": [], "tools": []}
    return {
        "connected": _mcp.is_connected,
        "tools": [
            {"name": t.name, "description": t.description, "server": t.server_name}
            for t in _mcp.tools
        ],
    }


@app.post("/mcp/reconnect")
async def mcp_reconnect():
    """Disconnect and reconnect to all MCP servers (re-reads config file)."""
    global _mcp
    if _mcp:
        await _mcp.disconnect_all()
    _mcp = McpClientManager()
    await _mcp.connect_all()
    return {
        "status": "reconnected",
        "tools": [t.name for t in _mcp.tools],
    }


@app.get("/mcp/api-key")
async def mcp_get_api_key():
    """Return the current MCP API key (masked for display)."""
    if not _mcp:
        return {"api_key": None, "masked": None}
    key = _mcp.api_key
    if not key:
        return {"api_key": None, "masked": None}
    masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "****"
    return {"api_key": key, "masked": masked}


class ApiKeyRequest(BaseModel):
    api_key: str


@app.post("/mcp/api-key")
async def mcp_set_api_key(request: ApiKeyRequest):
    """Update the MCP API key and reconnect."""
    global _mcp
    if _mcp:
        await _mcp.disconnect_all()
    _mcp = McpClientManager()
    await _mcp.connect_all(api_key=request.api_key)
    return {
        "status": "reconnected",
        "tools": [t.name for t in _mcp.tools],
        "tool_count": len(_mcp.tools),
    }


# ── Tool debugger endpoints ────────────────────────────────────────


@app.get("/tools", include_in_schema=False)
async def tools_page():
    return FileResponse(str(_STATIC_DIR / "tools.html"))


@app.get("/api/tools")
async def list_tools():
    """List all available MCP tools with their schemas."""
    if not _mcp or not _mcp.is_connected:
        return {"tools": []}
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
                "server": t.server_name,
            }
            for t in _mcp.tools
        ]
    }


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict


@app.post("/api/tools/call")
async def call_tool(request: ToolCallRequest):
    """Execute a tool call and return the raw result."""
    if not _mcp or not _mcp.is_connected:
        raise HTTPException(503, "MCP not connected")
    result = await _mcp.call_tool(request.name, request.arguments)
    return {"tool": request.name, "arguments": request.arguments, "result": result}


# ── Training data management endpoints ─────────────────────────────


class TrainingDataConfig(BaseModel):
    hf_repo: str | None = None
    hf_token: str | None = None


import os as _os

_training_config: TrainingDataConfig = TrainingDataConfig(
    hf_repo=_os.environ.get("TRAINING_HF_REPO"),
    hf_token=_os.environ.get("HF_TOKEN"),
)


@app.get("/training", include_in_schema=False)
async def training_page():
    return FileResponse(str(_STATIC_DIR / "training.html"))


@app.get("/api/training/config")
async def get_training_config():
    return {
        "hf_repo": _training_config.hf_repo,
        "configured": bool(_training_config.hf_repo and _training_config.hf_token),
    }


@app.post("/api/training/config")
async def set_training_config(config: TrainingDataConfig):
    _training_config.hf_repo = config.hf_repo
    _training_config.hf_token = config.hf_token
    return {"status": "ok", "hf_repo": config.hf_repo}


@app.get("/api/training/examples")
async def list_training_examples():
    """Pull training examples from HF dataset repo."""
    if not _training_config.hf_repo or not _training_config.hf_token:
        raise HTTPException(400, "HF repo and token not configured. POST to /api/training/config first.")

    from datasets import load_dataset

    try:
        ds = load_dataset(
            _training_config.hf_repo,
            split="train",
            token=_training_config.hf_token,
        )
    except Exception as e:
        raise HTTPException(502, f"Failed to load dataset: {e}")

    examples = []
    for i, row in enumerate(ds):
        msgs = row.get("messages", [])
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        has_tools = any("<tool_call>" in m.get("content", "") for m in msgs if m["role"] == "assistant")
        user_turns = sum(1 for m in msgs if m["role"] == "user")
        examples.append({
            "id": i,
            "preview": user_msg[:120],
            "messages": msgs,
            "num_messages": len(msgs),
            "num_turns": user_turns,
            "has_tools": has_tools,
        })

    return {"total": len(examples), "examples": examples}


class UpdateExampleRequest(BaseModel):
    id: int
    messages: list[dict]


class AddExampleRequest(BaseModel):
    messages: list[dict]


class DeleteExampleRequest(BaseModel):
    id: int


class SyncRequest(BaseModel):
    examples: list[dict]


@app.post("/api/training/sync")
async def sync_training_data(request: SyncRequest):
    """Push the full set of examples back to HF Hub."""
    if not _training_config.hf_repo or not _training_config.hf_token:
        raise HTTPException(400, "HF repo and token not configured.")

    import json as _json
    from datasets import Dataset

    records = []
    for ex in request.examples:
        records.append({"messages": ex["messages"]})

    ds = Dataset.from_list(records)
    try:
        ds.push_to_hub(
            _training_config.hf_repo,
            split="train",
            private=False,
            token=_training_config.hf_token,
        )
    except Exception as e:
        raise HTTPException(502, f"Failed to push dataset: {e}")

    return {"status": "synced", "total": len(records)}


# ── Factory ────────────────────────────────────────────────────────


async def _startup_mcp(mcp_config: str | None = None) -> None:
    global _mcp
    _mcp = McpClientManager(config_path=mcp_config)
    await _mcp.connect_all()


def create_app(
    cache_dir: str | None = None,
    preload_model: str | None = None,
    preload_model_path: str | None = None,
    preload_gguf: str | None = None,
    adapter_path: str | None = None,
    device_map: str = "auto",
    dtype: str | None = None,
    use_flash_attn: bool = False,
    mcp_config: str | None = None,
    n_ctx: int = 128000,
    n_gpu_layers: int = 0,
) -> FastAPI:
    """Create and configure the app with an optional preloaded model."""
    global _manager
    _manager = ModelManager(cache_dir=cache_dir)

    if preload_gguf:
        _manager.load_gguf(preload_gguf, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
    elif preload_model:
        if not preload_model_path:
            _manager.download(preload_model)
        _manager.load(
            preload_model,
            model_path=preload_model_path,
            device_map=device_map,
            dtype=dtype,
            use_flash_attn=use_flash_attn,
            adapter_path=adapter_path,
        )

    @app.on_event("startup")
    async def on_startup():
        await _startup_mcp(mcp_config)

    @app.on_event("shutdown")
    async def on_shutdown():
        if _mcp:
            await _mcp.disconnect_all()

    return app
