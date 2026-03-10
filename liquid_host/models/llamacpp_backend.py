"""llama.cpp backend for GGUF-quantized Liquid AI models."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import Thread

from llama_cpp import Llama

logger = logging.getLogger(__name__)


class ContextOverflowError(RuntimeError):
    """Raised when the input exceeds the model's context window."""


class LlamaCppModel:
    """Wraps llama-cpp-python for GGUF model inference."""

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 128000,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ):
        self.model_path = Path(model_path)
        logger.info("Loading GGUF model: %s", self.model_path)
        t0 = time.perf_counter()

        kwargs: dict = {
            "model_path": str(self.model_path),
            "n_ctx": n_ctx,
            "verbose": verbose,
        }
        if n_threads is not None:
            kwargs["n_threads"] = n_threads
        if n_gpu_layers > 0:
            kwargs["n_gpu_layers"] = n_gpu_layers

        self._llm = Llama(**kwargs)
        elapsed = time.perf_counter() - t0
        logger.info("GGUF model loaded in %.1fs", elapsed)

    @property
    def name(self) -> str:
        return self.model_path.stem

    @property
    def n_ctx(self) -> int:
        """Return the context window size for this model."""
        return self._llm.n_ctx()

    def _reset_cache(self) -> None:
        """Reset the KV cache to avoid stale state between requests."""
        self._llm.reset()

    @staticmethod
    def _inject_tools_into_messages(
        messages: list[dict[str, str]],
        tools: list[dict] | None,
    ) -> list[dict[str, str]]:
        """Inject tool definitions into the system prompt.

        The LFM2.5 chat template expects tools in the system message as:
            List of tools: [{...}, {...}]
        We inject them here rather than passing via llama-cpp's `tools`
        parameter, which triggers incompatible constrained decoding.
        """
        if not tools:
            return messages

        tool_strs = [json.dumps(t) for t in tools]
        tool_block = "List of tools: [" + ", ".join(tool_strs) + "]"

        messages = list(messages)  # shallow copy
        if messages and messages[0]["role"] == "system":
            messages[0] = {
                **messages[0],
                "content": messages[0]["content"] + "\n" + tool_block,
            }
        else:
            messages.insert(0, {"role": "system", "content": tool_block})

        logger.info("_inject_tools: %d tools injected into system message", len(tools))
        logger.info("_inject_tools: system message (%d chars): %.500s", len(messages[0]["content"]), messages[0]["content"])
        logger.info("_inject_tools: total messages after injection: %d", len(messages))
        for i, m in enumerate(messages):
            logger.info("_inject_tools: msg[%d] role=%s len=%d", i, m["role"], len(m["content"]))

        return messages

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.3,
        min_p: float = 0.15,
        repetition_penalty: float = 1.05,
        top_p: float = 1.0,
        top_k: int = 0,
        tools: list[dict] | None = None,
    ) -> str:
        """Generate a non-streaming response."""
        logger.info("llamacpp generate: %d messages, max_tokens=%d, temp=%.2f, tools=%d",
                     len(messages), max_tokens, temperature, len(tools) if tools else 0)
        messages = self._inject_tools_into_messages(messages, tools)
        self._reset_cache()
        t0 = time.perf_counter()

        try:
            result = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                min_p=min_p,
                repeat_penalty=repetition_penalty,
                top_p=top_p,
                top_k=top_k,
            )
        except RuntimeError as e:
            if "llama_decode returned" in str(e):
                logger.error(
                    "llama_decode failed (n_ctx=%d, %d messages): %s",
                    self.n_ctx, len(messages), e,
                )
                raise ContextOverflowError(
                    f"llama_decode failed (n_ctx={self.n_ctx}). "
                    "The input may exceed the context window."
                ) from e
            raise

        elapsed = time.perf_counter() - t0
        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        tps = completion_tokens / elapsed if elapsed > 0 else 0
        logger.info("llamacpp generate: done — %d tokens in %.1fs (%.1f tok/s)", completion_tokens, elapsed, tps)

        return text

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.3,
        min_p: float = 0.15,
        repetition_penalty: float = 1.05,
        top_p: float = 1.0,
        top_k: int = 0,
        tools: list[dict] | None = None,
    ):
        """Stream tokens as they are generated. Yields token strings."""
        logger.info("llamacpp stream: %d messages, max_tokens=%d, temp=%.2f, tools=%d",
                     len(messages), max_tokens, temperature, len(tools) if tools else 0)
        messages = self._inject_tools_into_messages(messages, tools)
        self._reset_cache()
        t0 = time.perf_counter()
        token_count = 0

        stream = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            min_p=min_p,
            repeat_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            stream=True,
        )

        try:
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    token_count += 1
                    yield content
        except RuntimeError as e:
            if "llama_decode returned" in str(e):
                logger.error(
                    "llama_decode failed during streaming (n_ctx=%d, %d messages). "
                    "This may indicate stale KV cache state or context overflow.",
                    self.n_ctx, len(messages),
                )
                raise ContextOverflowError(
                    f"llama_decode failed during streaming (n_ctx={self.n_ctx}). "
                    "The input may exceed the context window or the model is in a bad state. "
                    "Try starting a new conversation."
                ) from e
            raise

        elapsed = time.perf_counter() - t0
        tps = token_count / elapsed if elapsed > 0 else 0
        logger.info("llamacpp stream: done — %d chunks in %.1fs (%.1f tok/s)", token_count, elapsed, tps)

    def generate_raw_for_tool_detection(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.3,
        min_p: float = 0.15,
        repetition_penalty: float = 1.05,
    ) -> str:
        """Generate a response and return raw text (for tool call detection)."""
        # llama-cpp-python handles chat templates internally, so tool call
        # tokens should appear in the response if the model emits them
        return self.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
        )


def find_gguf_model(path: str | Path) -> Path | None:
    """Find the .gguf file in a directory (or return the path if it's a file)."""
    path = Path(path)
    if path.is_file() and path.suffix == ".gguf":
        return path
    if path.is_dir():
        ggufs = list(path.glob("*.gguf"))
        if ggufs:
            return ggufs[0]
    return None


def load_manifest(path: str | Path) -> dict | None:
    """Load a LEAP bundle manifest JSON if present."""
    path = Path(path)
    if path.is_file() and path.suffix == ".json":
        return json.loads(path.read_text())
    if path.is_dir():
        jsons = list(path.glob("*.json"))
        if jsons:
            return json.loads(jsons[0].read_text())
    return None
