"""Entrypoint for HF Inference Endpoints custom container.

Loads the model from /repository (HF-mounted) and starts the server.
Supports both transformers (GPU) and GGUF (llama.cpp) backends.
"""

import logging
import os

import uvicorn

# Ensure writable cache directory for HF/datasets libraries
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("entrypoint")

MODEL_PATH = os.environ.get("MODEL_PATH", "/repository")
MODEL_KEY = os.environ.get("MODEL_KEY", "lfm2-24b-a2b")
PORT = int(os.environ.get("PORT", "80"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "99"))
N_CTX = int(os.environ.get("N_CTX", "128000"))
DEVICE_MAP = os.environ.get("DEVICE_MAP", "auto")
DTYPE = os.environ.get("DTYPE", None)
USE_FLASH_ATTN = os.environ.get("USE_FLASH_ATTN", "false").lower() == "true"
MCP_CONFIG = os.environ.get("MCP_CONFIG", None)
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", None)


def detect_backend(model_path: str) -> str:
    """Detect whether /repository contains GGUF files or transformers weights."""
    from pathlib import Path

    p = Path(model_path)
    if p.is_file() and p.suffix == ".gguf":
        return "gguf"
    if p.is_dir():
        ggufs = list(p.glob("*.gguf"))
        if ggufs:
            return "gguf"
    return "transformers"


def main():
    from liquid_host.server.app import create_app

    backend = detect_backend(MODEL_PATH)
    logger.info("Detected backend: %s", backend)
    logger.info("Model path: %s", MODEL_PATH)

    if backend == "gguf":
        logger.info("Loading GGUF model (n_ctx=%d, n_gpu_layers=%d)", N_CTX, N_GPU_LAYERS)
        create_app(
            preload_gguf=MODEL_PATH,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            mcp_config=MCP_CONFIG,
        )
    else:
        logger.info("Loading transformers model: %s from %s (device_map=%s, dtype=%s)", MODEL_KEY, MODEL_PATH, DEVICE_MAP, DTYPE)
        if ADAPTER_PATH:
            logger.info("LoRA adapter: %s", ADAPTER_PATH)
        create_app(
            preload_model=MODEL_KEY,
            preload_model_path=MODEL_PATH,
            device_map=DEVICE_MAP,
            dtype=DTYPE,
            use_flash_attn=USE_FLASH_ATTN,
            adapter_path=ADAPTER_PATH,
            mcp_config=MCP_CONFIG,
        )

    logger.info("Starting server on port %d", PORT)
    uvicorn.run(
        "liquid_host.server.app:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
