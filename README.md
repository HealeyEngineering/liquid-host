# Liquid Host

A framework for downloading and hosting [Liquid AI](https://www.liquid.ai/) LFM models locally. Provides a CLI for model management, a FastAPI-based inference server with an OpenAI-compatible API, a built-in chat UI, and MCP tool integration.

Supports two inference backends:
- **Transformers** — HuggingFace native weights (full precision or automatic dtype)
- **llama.cpp** — GGUF quantized weights via `llama-cpp-python` (faster on CPU, smaller footprint)

## Prerequisites

- Python 3.10+
- (Optional) A CUDA-capable GPU for faster inference — CPU works fine for smaller models

## Setup

Clone the repo and create a virtual environment:

```bash
cd liquid-host
python3 -m venv .env
source .env/bin/activate
```

Install the package:

```bash
pip install -e .
```

For development (adds pytest and httpx):

```bash
pip install -e ".[dev]"
```

For GPU-accelerated attention (requires a compatible NVIDIA GPU):

```bash
pip install -e ".[flash-attn]"
```

Verify the installation:

```bash
liquid-host --help
```

## Quick Start

### 1. Browse available models

```bash
liquid-host list
```

This shows all 15 supported models across the LFM2 and LFM2.5 families. Filter by family with `--family lfm2` or `--family lfm2.5`.

### 2. Download a model

**Option A: HuggingFace native weights (transformers backend)**

```bash
liquid-host download lfm2.5-1.2b-instruct
```

Models are cached to `~/.cache/liquid-host/models/` by default. Use `--cache-dir /path/to/dir` to change the location.

**Option B: GGUF quantized weights (llama.cpp backend — recommended)**

```bash
leap-bundle download LFM2.5-1.2B-Thinking --quantization=Q4_0 \
    --output-path ~/.cache/liquid-host/models/gguf/LFM2.5-1.2B-Thinking-Q4_0
```

Available quantizations: `Q4_0`, `Q4_K_M`, `Q5_K_M` (default), `Q8_0`. Smaller quantizations are faster and use less memory at the cost of some quality.

To see what you've already downloaded:

```bash
liquid-host downloaded
```

### 3. Start the inference server

**With a GGUF model (llama.cpp backend):**

```bash
liquid-host serve \
    --gguf ~/.cache/liquid-host/models/gguf/LFM2.5-1.2B-Thinking-Q4_0
```

**With a HuggingFace model (transformers backend):**

```bash
liquid-host serve --model lfm2.5-1.2b-instruct
```

This loads the model into memory and starts an HTTP server on `http://localhost:8000`.

**Server options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | None | HuggingFace model to preload (transformers backend) |
| `--gguf` | None | Path to GGUF file or directory to preload (llama.cpp backend) |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Port |
| `--device-map` | `auto` | Device placement — `auto`, `cpu`, `cuda:0`, etc. (transformers only) |
| `--dtype` | model default | Override dtype — `float16`, `bfloat16`, `float32` (transformers only) |
| `--flash-attn` | off | Enable Flash Attention 2 (transformers only, requires the `flash-attn` extra) |
| `--n-ctx` | `4096` | Context window size (GGUF only) |
| `--n-gpu-layers` | `0` | Number of layers to offload to GPU (GGUF only) |
| `--cache-dir` | `~/.cache/liquid-host/models/` | Custom cache directory |
| `--mcp-config` | `./mcp_servers.json` | Path to MCP server config |
| `--workers` | `1` | Number of uvicorn workers |

### 4. Send requests

The server exposes an OpenAI-compatible chat completions endpoint, so any OpenAI SDK client works out of the box.

**curl:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is a liquid neural network?"}],
    "max_tokens": 256
  }'
```

**Python (openai SDK):**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="LFM2.5-1.2B-Instruct",
    messages=[{"role": "user", "content": "What is a liquid neural network?"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

**Streaming:**

```python
stream = client.chat.completions.create(
    model="LFM2.5-1.2B-Instruct",
    messages=[{"role": "user", "content": "Explain mixture of experts."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 5. Web UI

The server includes a built-in chat interface. Once the server is running, open your browser to:

```
http://localhost:8000
```

Features:

- **Streaming chat** — tokens appear in real time as the model generates them
- **Multi-turn conversation** — full message history is maintained and sent with each request
- **Settings panel** — click "Settings" in the top-right to adjust temperature, max tokens, min_p, repetition penalty, system prompt, and MCP tool toggle
- **MCP status badge** — shows the number of connected MCP tools in the header
- **Status indicator** — the header shows the loaded model name and a green/red dot for connection state (auto-refreshes every 5 seconds)
- **Keyboard shortcuts** — Enter to send, Shift+Enter for a newline

When MCP tools are enabled and connected, the UI shows real-time status messages as the model calls tools (e.g., "Calling tool: find_events..."), then streams the final answer token-by-token.

### 6. Interactive chat (no server needed)

For quick experimentation without running a server:

```bash
liquid-host run lfm2.5-1.2b-instruct
```

Pass a system prompt with `--system "You are a helpful assistant."`. Type your messages and press Enter. Ctrl+C to quit.

## Inference Backends

### Transformers (HuggingFace)

Uses full-precision or automatic dtype weights from the HuggingFace Hub. Best for GPU inference or when you need the exact model weights.

```bash
liquid-host serve --model lfm2.5-1.2b-instruct --device-map auto
```

### llama.cpp (GGUF)

Uses quantized GGUF weights via `llama-cpp-python`. Significantly faster on CPU, smaller memory footprint, and supports various quantization levels.

```bash
# Download a quantized model
leap-bundle download LFM2.5-1.2B-Thinking --quantization=Q4_0 \
    --output-path ~/.cache/liquid-host/models/gguf/LFM2.5-1.2B-Thinking-Q4_0

# Serve it
liquid-host serve \
    --gguf ~/.cache/liquid-host/models/gguf/LFM2.5-1.2B-Thinking-Q4_0 \
    --n-ctx 4096
```

Available quantizations via `leap-bundle`:

| Quantization | Description |
|---|---|
| `Q4_0` | 4-bit, smallest size, fastest inference |
| `Q4_K_M` | 4-bit k-quant, better quality |
| `Q5_K_M` | 5-bit k-quant (default) |
| `Q8_0` | 8-bit, highest quality |

To offload layers to GPU for faster inference:

```bash
liquid-host serve \
    --gguf ~/.cache/liquid-host/models/gguf/LFM2.5-1.2B-Thinking-Q4_0 \
    --n-gpu-layers 32
```

## MCP Tool Integration

Liquid Host can connect to [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) servers, giving the model access to external tools and data sources at runtime.

### Configuration

Create or edit `mcp_servers.json` in the project root:

```json
{
  "servers": [
    {
      "name": "aiera",
      "url": "https://mcp-pub.aiera.com?api_key=YOUR_API_KEY",
      "transport": "streamable_http",
      "enabled": true,
      "description": "Aiera financial data and research platform"
    }
  ]
}
```

Supported transports: `sse`, `streamable_http` (or `streamable-http`, `http`).

Set `"enabled": false` to disable a server without removing it. Add multiple entries to connect to several servers simultaneously.

### How it works

1. On server startup, all enabled MCP servers are connected and their tools are discovered
2. When a chat request comes in with tools enabled, the model receives the tool definitions and can emit tool calls
3. Tool calls are executed against the MCP server, results are fed back, and the model generates a final answer (up to 5 rounds)
4. The entire workflow streams status updates to the UI in real time

### MCP endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mcp/status` | GET | Shows connected servers and available tools |
| `/mcp/reconnect` | POST | Re-reads `mcp_servers.json` and reconnects (no restart needed) |

### Custom config path

```bash
liquid-host serve --gguf /path/to/model --mcp-config /path/to/mcp_servers.json
```

## Server API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web chat UI |
| `/health` | GET | Health check |
| `/status` | GET | Server status, loaded model, and MCP info |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completions (OpenAI-compatible, supports streaming) |
| `/load` | POST | Load a HuggingFace model: `{"model": "lfm2.5-1.2b-instruct"}` |
| `/load-gguf` | POST | Load a GGUF model: `{"path": "/path/to/model", "n_ctx": 4096}` |
| `/unload` | POST | Unload the current model and free memory |
| `/download` | POST | Download a model: `{"model": "lfm2-1.2b"}` |
| `/mcp/status` | GET | MCP connection status and tool list |
| `/mcp/reconnect` | POST | Reconnect to MCP servers (re-reads config) |

### Hot-swapping models

You can load and unload models at runtime without restarting the server:

```bash
# Load a GGUF model
curl -X POST http://localhost:8000/load-gguf \
  -H "Content-Type: application/json" \
  -d '{"path": "~/.cache/liquid-host/models/gguf/LFM2.5-1.2B-Thinking-Q4_0", "n_ctx": 4096}'

# Load a HuggingFace model
curl -X POST http://localhost:8000/load \
  -H "Content-Type: application/json" \
  -d '{"model": "lfm2-1.2b", "device_map": "cpu"}'

# Unload to free memory
curl -X POST http://localhost:8000/unload
```

## Model Overview

### LFM2 (generation 2)

| Key | Params | Active | Architecture | Notes |
|-----|--------|--------|--------------|-------|
| `lfm2-350m` | 350M | 350M | Dense hybrid | Ultra-edge |
| `lfm2-700m` | 700M | 700M | Dense hybrid | Edge devices |
| `lfm2-1.2b` | 1.2B | 1.2B | Dense hybrid | 10 conv + 6 attn layers |
| `lfm2-2.6b` | 2.6B | 2.6B | Dense hybrid | Mid-range |
| `lfm2-2.6b-exp` | 2.6B | 2.6B | Dense hybrid | Pure RL-trained |
| `lfm2-8b-a1b` | 8B | 1B | MoE | 8B total, 1B active |
| `lfm2-24b-a2b` | 24B | 2.3B | MoE | Largest open model, fits in 32GB RAM |

### LFM2.5 (generation 2.5)

| Key | Params | Active | Architecture | Notes |
|-----|--------|--------|--------------|-------|
| `lfm2.5-1.2b-base` | 1.2B | 1.2B | Dense hybrid | Pre-trained base (28T tokens) |
| `lfm2.5-1.2b-instruct` | 1.2B | 1.2B | Dense hybrid | Instruction-tuned (recommended) |
| `lfm2.5-1.2b-thinking` | 1.2B | 1.2B | Dense hybrid | Reasoning / chain-of-thought |
| `lfm2.5-1.2b-jp` | 1.2B | 1.2B | Dense hybrid | Japanese-optimized |
| `lfm2.5-vl-1.6b` | 1.6B | 1.6B | Dense hybrid | Vision-language |
| `lfm2.5-audio-1.5b` | 1.5B | 1.5B | Dense hybrid | Audio-language |

## Managing cached models

```bash
# See what's downloaded (shows both HuggingFace and GGUF models)
liquid-host downloaded

# Remove a HuggingFace model
liquid-host delete lfm2-1.2b

# Re-download (force)
liquid-host download lfm2-1.2b --force
```

The default cache location is `~/.cache/liquid-host/models/`. GGUF models downloaded via `leap-bundle` are stored in `~/.cache/liquid-host/models/gguf/`. Override it globally with `--cache-dir` on any command.

## Verbose Logging

Add `-v` to any command for debug-level logging:

```bash
liquid-host -v serve --gguf /path/to/model
```

This shows detailed logs for every step: tokenization, model generation (with token counts and tok/s), tool calls, MCP communication, and more.
