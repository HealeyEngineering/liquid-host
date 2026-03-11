"""Model downloading, loading, and lifecycle management.

Supports two backends:
  - transformers (HuggingFace native weights)
  - llama.cpp    (GGUF quantized weights via llama-cpp-python)
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time as _time
from pathlib import Path
from threading import Thread

from liquid_host.config import (
    GenerationConfig,
    ModelSpec,
    get_model_spec,
    resolve_cache_dir,
)

logger = logging.getLogger(__name__)

# Sentinel for backend type
BACKEND_TRANSFORMERS = "transformers"
BACKEND_LLAMACPP = "llamacpp"


class ModelManager:
    """Handles downloading, caching, loading, and running Liquid AI models."""

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = resolve_cache_dir(cache_dir)
        self._backend: str | None = None
        self._model_name: str | None = None

        # transformers backend state
        self._hf_model = None
        self._hf_tokenizer = None
        self._loaded_spec: ModelSpec | None = None

        # llama.cpp backend state
        self._llm = None  # LlamaCppModel instance

    # ── Download ──────────────────────────────────────────────────────

    def download(
        self,
        model_key: str,
        *,
        revision: str | None = None,
        force: bool = False,
    ) -> Path:
        """Download a model from HuggingFace Hub to the local cache."""
        from huggingface_hub import snapshot_download

        spec = get_model_spec(model_key)
        logger.info("Downloading %s from %s ...", spec.name, spec.repo_id)

        path = Path(
            snapshot_download(
                repo_id=spec.repo_id,
                revision=revision,
                cache_dir=str(self.cache_dir),
                force_download=force,
            )
        )
        logger.info("Downloaded %s to %s", spec.name, path)
        return path

    # ── Local inventory ───────────────────────────────────────────────

    def list_downloaded(self) -> list[dict]:
        """List models already present in the local cache."""
        from huggingface_hub import scan_cache_dir
        from liquid_host.config import MODEL_REGISTRY

        results = []

        # HuggingFace cache
        if self.cache_dir.exists():
            cache_info = scan_cache_dir(str(self.cache_dir))
            for repo in cache_info.repos:
                matched_spec: ModelSpec | None = None
                for spec in MODEL_REGISTRY.values():
                    if spec.repo_id == repo.repo_id:
                        matched_spec = spec
                        break
                for rev in repo.revisions:
                    results.append({
                        "repo_id": repo.repo_id,
                        "name": matched_spec.name if matched_spec else repo.repo_id,
                        "revision": rev.commit_hash[:12],
                        "size_gb": round(rev.size_on_disk / (1024**3), 2),
                        "path": str(rev.snapshot_path),
                        "backend": BACKEND_TRANSFORMERS,
                        "spec": matched_spec,
                    })

        # GGUF cache
        gguf_dir = self.cache_dir / "gguf"
        if gguf_dir.exists():
            for model_dir in gguf_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                ggufs = list(model_dir.glob("*.gguf"))
                if ggufs:
                    total_size = sum(f.stat().st_size for f in ggufs)
                    results.append({
                        "repo_id": f"gguf/{model_dir.name}",
                        "name": model_dir.name,
                        "revision": "local",
                        "size_gb": round(total_size / (1024**3), 2),
                        "path": str(model_dir),
                        "backend": BACKEND_LLAMACPP,
                        "spec": None,
                    })

        return results

    def delete(self, model_key: str) -> bool:
        """Remove a downloaded model from cache."""
        spec = get_model_spec(model_key)
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir(str(self.cache_dir))
        for repo in cache_info.repos:
            if repo.repo_id == spec.repo_id:
                for rev in repo.revisions:
                    shutil.rmtree(rev.snapshot_path, ignore_errors=True)
                repo_path = self.cache_dir / f"models--{repo.repo_id.replace('/', '--')}"
                if repo_path.exists():
                    shutil.rmtree(repo_path, ignore_errors=True)
                logger.info("Deleted %s from cache", spec.name)
                return True
        return False

    # ── Loading ───────────────────────────────────────────────────────

    def load(
        self,
        model_key: str,
        *,
        model_path: str | None = None,
        device_map: str = "auto",
        dtype: str | None = None,
        use_flash_attn: bool = False,
        adapter_path: str | Path | None = None,
    ) -> None:
        """Load a transformers model into memory.

        Args:
            model_key: Key in MODEL_REGISTRY to look up model spec.
            model_path: Optional local path to load weights from (e.g. /repository
                        on HF Inference Endpoints). Falls back to spec.repo_id.
            adapter_path: Optional path to a LoRA adapter directory to merge on load.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        spec = get_model_spec(model_key)
        load_from = model_path or spec.repo_id

        if self._loaded_spec and self._loaded_spec.repo_id == spec.repo_id and model_path is None and not adapter_path:
            logger.info("%s is already loaded", spec.name)
            return

        self.unload()

        resolved_dtype = dtype or spec.recommended_dtype
        torch_dtype = getattr(torch, resolved_dtype, torch.bfloat16)

        logger.info("Loading %s via transformers from %s (dtype=%s, device_map=%s) ...", spec.name, load_from, resolved_dtype, device_map)

        kwargs: dict = {
            "pretrained_model_name_or_path": load_from,
            "device_map": device_map,
            "torch_dtype": torch_dtype,
        }
        if not model_path:
            kwargs["cache_dir"] = str(self.cache_dir)
        if use_flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"

        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            load_from, cache_dir=None if model_path else str(self.cache_dir)
        )
        self._hf_model = AutoModelForCausalLM.from_pretrained(**kwargs)

        if adapter_path:
            import os
            from peft import PeftModel
            adapter_str = str(adapter_path)
            logger.info("Loading LoRA adapter from %s ...", adapter_str)
            # If adapter_path looks like a Hub repo ID (contains '/'), pass token for private repos
            peft_kwargs: dict = {}
            if "/" in adapter_str and not os.path.exists(adapter_str):
                hf_token = os.environ.get("HF_TOKEN")
                if hf_token:
                    peft_kwargs["token"] = hf_token
            self._hf_model = PeftModel.from_pretrained(self._hf_model, adapter_str, **peft_kwargs)
            self._hf_model = self._hf_model.merge_and_unload()
            logger.info("LoRA adapter merged successfully")

        self._loaded_spec = spec
        self._backend = BACKEND_TRANSFORMERS
        self._model_name = spec.name

        logger.info("%s loaded successfully (transformers backend)", spec.name)

    def load_gguf(
        self,
        path: str | Path,
        *,
        n_ctx: int = 128000,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> None:
        """Load a GGUF model via llama.cpp."""
        from liquid_host.models.llamacpp_backend import LlamaCppModel, find_gguf_model

        path = Path(path)
        gguf_path = find_gguf_model(path)
        if not gguf_path:
            raise FileNotFoundError(f"No .gguf file found at {path}")

        if self._llm and self._llm.model_path == gguf_path:
            logger.info("%s is already loaded", gguf_path.stem)
            return

        self.unload()

        self._llm = LlamaCppModel(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        self._backend = BACKEND_LLAMACPP
        self._model_name = gguf_path.stem

        logger.info("%s loaded successfully (llama.cpp backend)", self._model_name)

    def unload(self) -> None:
        """Unload the current model from memory."""
        if self._backend == BACKEND_TRANSFORMERS and self._hf_model is not None:
            import torch
            del self._hf_model
            del self._hf_tokenizer
            self._hf_model = None
            self._hf_tokenizer = None
            self._loaded_spec = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif self._backend == BACKEND_LLAMACPP and self._llm is not None:
            del self._llm
            self._llm = None

        self._backend = None
        self._model_name = None
        logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    @property
    def loaded_model_name(self) -> str | None:
        return self._model_name

    @property
    def backend(self) -> str | None:
        return self._backend

    # ── Tokenization (transformers only) ──────────────────────────────

    def _tokenize(self, messages: list[dict[str, str]], tools: list[dict] | None = None):
        """Apply the chat template and return input_ids as a tensor (transformers backend)."""
        import torch
        logger.debug("Tokenizing %d messages (tools=%s)", len(messages), "yes" if tools else "no")
        kwargs: dict = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "tokenize": True,
        }
        if tools:
            kwargs["tools"] = tools
        result = self._hf_tokenizer.apply_chat_template(messages, **kwargs)
        if hasattr(result, "input_ids"):
            result = result.input_ids
        input_ids = result.to(self._hf_model.device)
        logger.debug("Tokenized to %d tokens on device %s", input_ids.shape[1], input_ids.device)
        return input_ids

    # ── Inference ─────────────────────────────────────────────────────

    def _generate_raw(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig,
        tools: list[dict] | None = None,
        skip_special_tokens: bool = True,
    ) -> str:
        """Generate full response text."""
        if self._backend == BACKEND_LLAMACPP:
            logger.info("_generate_raw [llamacpp]: starting (max_tokens=%d)", config.max_new_tokens)
            return self._llm.generate(
                messages=messages,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                min_p=config.min_p,
                repetition_penalty=config.repetition_penalty,
                top_p=config.top_p,
                tools=tools,
            )

        # transformers backend
        logger.info("_generate_raw [transformers]: starting (max_new_tokens=%d, skip_special=%s)", config.max_new_tokens, skip_special_tokens)
        input_ids = self._tokenize(messages, tools=tools)
        logger.info("_generate_raw: calling model.generate (%d input tokens) ...", input_ids.shape[1])
        t0 = _time.perf_counter()
        output_ids = self._hf_model.generate(input_ids=input_ids, **config.to_dict())
        elapsed = _time.perf_counter() - t0
        new_token_count = output_ids.shape[1] - input_ids.shape[1]
        tps = new_token_count / elapsed if elapsed > 0 else 0
        logger.info("_generate_raw: done — %d new tokens in %.1fs (%.1f tok/s)", new_token_count, elapsed, tps)
        new_tokens = output_ids[0][input_ids.shape[1]:]
        decoded = self._hf_tokenizer.decode(new_tokens, skip_special_tokens=skip_special_tokens)
        logger.debug("_generate_raw: output preview: %.200s", decoded)
        return decoded

    def generate(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate a response from a list of chat messages."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load() or load_gguf() first.")
        config = config or GenerationConfig()
        return self._generate_raw(messages, config)

    def _generate_stream_raw(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig,
        tools: list[dict] | None = None,
        skip_special_tokens: bool = True,
    ):
        """Stream tokens. Yields (token, full_text_so_far)."""
        if self._backend == BACKEND_LLAMACPP:
            logger.info("_generate_stream_raw [llamacpp]: starting (max_tokens=%d)", config.max_new_tokens)
            t0 = _time.perf_counter()
            full = []
            token_count = 0
            for token in self._llm.generate_stream(
                messages=messages,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                min_p=config.min_p,
                repetition_penalty=config.repetition_penalty,
                top_p=config.top_p,
                tools=tools,
            ):
                full.append(token)
                token_count += 1
                yield token, "".join(full)
            elapsed = _time.perf_counter() - t0
            tps = token_count / elapsed if elapsed > 0 else 0
            logger.info("_generate_stream_raw [llamacpp]: done — %d chunks in %.1fs (%.1f tok/s)", token_count, elapsed, tps)
            return

        # transformers backend
        from transformers import TextIteratorStreamer
        from threading import Thread

        logger.info("_generate_stream_raw [transformers]: starting (max_new_tokens=%d)", config.max_new_tokens)
        input_ids = self._tokenize(messages, tools=tools)
        logger.info("_generate_stream_raw: calling model.generate (%d input tokens) ...", input_ids.shape[1])

        streamer = TextIteratorStreamer(
            self._hf_tokenizer, skip_prompt=True, skip_special_tokens=skip_special_tokens,
        )
        gen_kwargs = {**config.to_dict(), "streamer": streamer, "input_ids": input_ids}
        thread = Thread(target=self._hf_model.generate, kwargs=gen_kwargs)
        t0 = _time.perf_counter()
        thread.start()

        full = []
        token_count = 0
        for token in streamer:
            full.append(token)
            token_count += 1
            yield token, "".join(full)

        thread.join()
        elapsed = _time.perf_counter() - t0
        tps = token_count / elapsed if elapsed > 0 else 0
        logger.info("_generate_stream_raw [transformers]: done — %d tokens in %.1fs (%.1f tok/s)", token_count, elapsed, tps)

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig | None = None,
    ):
        """Stream tokens as they are generated. Yields decoded token strings."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load() or load_gguf() first.")
        config = config or GenerationConfig()
        for token, _ in self._generate_stream_raw(messages, config):
            yield token

    # ── Async streaming helpers ──────────────────────────────────────

    async def _async_stream_raw(
        self,
        messages: list[dict[str, str]],
        config: GenerationConfig,
        tools: list[dict] | None = None,
        skip_special_tokens: bool = True,
    ):
        """Run _generate_stream_raw in a thread, yielding (token, full) asynchronously."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def _run():
            try:
                for item in self._generate_stream_raw(
                    messages, config, tools=tools, skip_special_tokens=skip_special_tokens,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        thread = Thread(target=_run, daemon=True)
        thread.start()

        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        thread.join()

    # ── Tool-calling loop ─────────────────────────────────────────────

    _THINK_BATCH_SIZE = 20  # Emit thinking tokens in batches of this size

    async def generate_with_tools(
        self,
        messages: list[dict[str, str]],
        mcp_manager,
        config: GenerationConfig | None = None,
        max_tool_rounds: int = 5,
    ):
        """Generate with an agentic tool-calling loop.

        Async generator that yields event dicts:
            {"type": "status",   "content": "..."}  — progress updates
            {"type": "thinking", "content": "..."}   — batched thinking tokens
            {"type": "token",    "content": "..."}   — streamed answer tokens
            {"type": "done"}                         — generation complete
            {"type": "error",    "content": "..."}   — error
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load() or load_gguf() first.")

        from liquid_host.mcp_client import McpClientManager

        config = config or GenerationConfig()
        tools_schema = mcp_manager.get_llm_tools_schema() if mcp_manager and mcp_manager.is_connected else None
        working_messages = list(messages)

        tool_count = len(tools_schema) if tools_schema else 0
        logger.info("generate_with_tools [%s]: starting — %d MCP tools, max_rounds=%d", self._backend, tool_count, max_tool_rounds)
        yield {"type": "status", "content": f"Connected to MCP ({tool_count} tools available)"}

        for _round in range(max_tool_rounds):
            logger.info("generate_with_tools: round %d/%d — %d messages in context", _round + 1, max_tool_rounds, len(working_messages))
            yield {"type": "status", "content": f"Generating response (round {_round + 1})..."}

            # Stream the response, collecting full text for tool-call parsing.
            # Thinking tokens are emitted in batches as they arrive.
            # Content tokens are buffered until we know whether tool calls exist.
            full_tokens: list[str] = []
            thinking_buffer: list[str] = []
            content_buffer: list[str] = []
            in_thinking = False
            thinking_done = False

            try:
                skip = self._backend != BACKEND_LLAMACPP

                async for token, _full in self._async_stream_raw(
                    working_messages, config,
                    tools=tools_schema,
                    skip_special_tokens=skip,
                ):
                    full_tokens.append(token)

                    # Detect <think> / </think> boundaries
                    if not thinking_done:
                        combined = "".join(full_tokens)
                        if not in_thinking and "<think>" in combined:
                            in_thinking = True
                            thinking_buffer = []
                            continue
                        if in_thinking:
                            if "</think>" in token:
                                # Flush remaining thinking buffer
                                if thinking_buffer:
                                    yield {"type": "thinking", "content": "".join(thinking_buffer)}
                                    thinking_buffer = []
                                in_thinking = False
                                thinking_done = True
                                continue
                            thinking_buffer.append(token)
                            if len(thinking_buffer) >= self._THINK_BATCH_SIZE:
                                yield {"type": "thinking", "content": "".join(thinking_buffer)}
                                thinking_buffer = []
                            continue

                    # Buffer content tokens (don't emit yet — may contain tool calls)
                    content_buffer.append(token)

            except Exception as e:
                logger.error("generate_with_tools: streaming failed — %s", e)
                yield {"type": "error", "content": str(e)}
                return

            # Flush any remaining thinking buffer
            if thinking_buffer:
                yield {"type": "thinking", "content": "".join(thinking_buffer)}

            raw_output = "".join(full_tokens)
            logger.info("generate_with_tools: raw output (%d chars): %.500s", len(raw_output), raw_output)

            # Check for tool calls in the collected output
            tool_calls = McpClientManager.parse_tool_calls(raw_output)
            logger.info("generate_with_tools: round %d — found %d tool call(s)", _round + 1, len(tool_calls))
            for tc in tool_calls:
                logger.info("generate_with_tools: parsed tool call — %s(%s)", tc["name"], tc["arguments"])

            if not tool_calls:
                # No tool calls — emit all buffered content tokens
                for token in content_buffer:
                    yield {"type": "token", "content": token}
                yield {"type": "done"}
                return

            # Append the assistant's raw response (strip tool call markup and special tokens)
            import re as _re
            clean_output = raw_output.replace("<|im_end|>", "")
            clean_output = _re.sub(r"<tool_call>\s*\{.*?\}\s*</tool_call>", "", clean_output, flags=_re.DOTALL)
            clean_output = _re.sub(r"\<\|tool_call_start\|>.*?<\|tool_call_end\|>", "", clean_output, flags=_re.DOTALL)
            working_messages.append({"role": "assistant", "content": clean_output.strip()})

            # Extract model-generated status lines preceding each tool call.
            # The model writes a line like "Looking up NFLX earnings..." before the tool call.
            # Supports both [func_name(...)] and <tool_call>...</tool_call> formats.
            post_think = raw_output
            if "</think>" in post_think:
                post_think = post_think.split("</think>", 1)[1]
            logger.info("generate_with_tools: post-think text: %.300s", post_think)
            tool_status_lines = {}
            for i, call in enumerate(tool_calls):
                # Try <tool_call> format first, then bracket format
                markers = [f"<tool_call>", f"[{call['name']}("]
                pos = -1
                for marker in markers:
                    pos = post_think.find(marker)
                    if pos >= 0:
                        break
                logger.info("generate_with_tools: looking for tool call marker in post_think, pos=%d", pos)
                if pos > 0:
                    preceding = post_think[:pos].strip().rsplit("\n", 1)
                    status_line = preceding[-1].strip() if preceding else ""
                    logger.info("generate_with_tools: extracted status line: %r", status_line)
                    if status_line and not status_line.startswith(("[", "<")):
                        tool_status_lines[i] = status_line

            # Execute each tool call and append results
            for i, call in enumerate(tool_calls):
                logger.info("generate_with_tools: round %d, call %d/%d — %s(%s)", _round + 1, i + 1, len(tool_calls), call["name"], call["arguments"])
                status_msg = tool_status_lines.get(i, f"Calling tool: {call['name']}...")
                logger.info("generate_with_tools: status message for tool %d: %r", i, status_msg)
                yield {"type": "status", "content": status_msg}
                result_text = await mcp_manager.call_tool(call["name"], call["arguments"])
                logger.info("generate_with_tools: got result from %s (%d chars)", call["name"], len(result_text))
                logger.info("generate_with_tools: tool result preview: %.500s", result_text)
                working_messages.append({"role": "tool", "content": result_text})
                yield {"type": "status", "content": f"Got result from {call['name']}"}

        # Exhausted rounds — final streaming answer
        logger.warning("generate_with_tools: reached max tool rounds (%d), generating final answer", max_tool_rounds)
        yield {"type": "status", "content": "Generating final answer..."}
        yield {"type": "status", "content": ""}
        async for token, _ in self._async_stream_raw(working_messages, config):
            yield {"type": "token", "content": token}
        yield {"type": "done"}
