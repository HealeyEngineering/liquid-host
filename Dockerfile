FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System dependencies for llama-cpp-python build (if needed) and general use
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY liquid_host/ liquid_host/

# Install the package and dependencies
RUN pip install --no-cache-dir ".[flash-attn]" 2>/dev/null || pip install --no-cache-dir .
RUN pip install --no-cache-dir "peft>=0.18.0"
RUN pip install --no-cache-dir "datasets>=3.0.0"

# HF Inference Endpoints mounts the model at /repository
# These can be overridden via endpoint env vars
ENV MODEL_PATH=/repository
ENV MODEL_KEY=lfm2-24b-a2b
ENV N_GPU_LAYERS=99
ENV PORT=80
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache

EXPOSE 80

# MCP server configuration for tool calling
COPY mcp_servers.json .
ENV MCP_CONFIG=/app/mcp_servers.json

# Entrypoint script that loads the model and starts the server
COPY deploy/entrypoint.py .

CMD ["python", "entrypoint.py"]
