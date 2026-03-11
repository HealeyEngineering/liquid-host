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

The server includes three web interfaces:

- **Chat** (`/`) — streaming chat with the model, multi-turn conversation, MCP tool calling with real-time status, configurable settings (temperature, max tokens, system prompt, tool toggle)
- **Tool Debugger** (`/tools`) — test any MCP tool directly with a parameter form, see raw results, copy cURL commands
- **Training Data Editor** (`/training`) — browse, edit, add, and delete training examples, sync changes back to HuggingFace Hub

All pages are cross-linked via nav in the header.

When MCP tools are enabled and connected, the chat UI shows real-time status messages as the model calls tools (e.g., "Finding recent NFLX earnings..."), then streams the final answer token-by-token.

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
| `/tools` | GET | Tool debugger UI |
| `/training` | GET | Training data editor UI |
| `/health` | GET | Health check |
| `/status` | GET | Server status, loaded model, and MCP info |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completions (OpenAI-compatible, supports streaming) |
| `/load` | POST | Load a HuggingFace model: `{"model": "lfm2-24b-a2b"}` |
| `/load-gguf` | POST | Load a GGUF model: `{"path": "/path/to/model", "n_ctx": 4096}` |
| `/unload` | POST | Unload the current model and free memory |
| `/download` | POST | Download a model: `{"model": "lfm2-1.2b"}` |
| `/mcp/status` | GET | MCP connection status and tool list |
| `/mcp/reconnect` | POST | Reconnect to MCP servers (re-reads config) |
| `/api/tools` | GET | List all MCP tools with JSON schemas |
| `/api/tools/call` | POST | Execute a tool: `{"name": "find_events", "arguments": {...}}` |
| `/api/training/config` | GET/POST | Get or set HF dataset config for training data |
| `/api/training/examples` | GET | Pull training examples from HF dataset |
| `/api/training/sync` | POST | Push edited examples back to HF Hub |

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

## Fine-Tuning

Liquid Host supports LoRA/QLoRA fine-tuning of LFM models on chat-format JSONL data. Training can run locally (requires a GPU) or remotely on HuggingFace Spaces.

### Training Data Format

Each line is a JSON object with a `messages` array. Supports `user`, `assistant`, `system`, and `tool` roles:

```jsonl
{"messages": [{"role": "user", "content": "What were AAPL's Q3 results?"}, {"role": "assistant", "content": "<think>Look up AAPL earnings.</think>\nLooking up recent AAPL earnings...\n[find_events(bloomberg_ticker='AAPL:US', event_type='earnings')]"}, {"role": "tool", "content": "{\"events\": [...]}"}, {"role": "assistant", "content": "Apple reported Q3 revenue of $85.8B..."}]}
```

See `data/training/aiera_tools_v4.jsonl` for 286 examples covering all 39 Aiera MCP tools (earnings, financials, filings, transcripts, conferences, indexes, watchlists, sectors, Third Bridge, company docs, research metadata, web search, and more).

### Install Training Dependencies

For local training:

```bash
pip install -e ".[training]"
```

For remote training (no GPU needed):

```bash
pip install -e ".[remote-training]"
```

### Local Fine-Tuning

Requires a CUDA GPU with sufficient VRAM.

```bash
liquid-host finetune lfm2-24b-a2b data/training/aiera_tools_v4.jsonl \
  --output ./finetune-output \
  --epochs 3 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --max-seq-length 2048
```

For QLoRA (4-bit quantized base model, reduces VRAM usage):

```bash
liquid-host finetune lfm2-24b-a2b data/training/aiera_tools_v4.jsonl \
  --quantize-4bit
```

The adapter is saved to `./finetune-output/adapter/`.

### Remote Fine-Tuning (HuggingFace Spaces)

No local GPU required. Launches a custom HF Space with a GPU backend that runs the training job, then pushes the adapter to the Hub.

```bash
export HF_TOKEN=hf_your_token_here

liquid-host finetune lfm2-24b-a2b data/training/aiera_tools_v4.jsonl \
  --remote \
  --backend a100-large \
  --quantize-4bit \
  --project-name my-finetune
```

Available GPU backends:

| Backend | GPU | Notes |
|---------|-----|-------|
| `t4-small` | NVIDIA T4 (16GB) | Budget option |
| `t4-medium` | NVIDIA T4 (16GB) | More CPU/RAM |
| `a10g-small` | NVIDIA A10G (24GB) | Good for 1-3B models |
| `a10g-large` | NVIDIA A10G (24GB) | More CPU/RAM |
| `l4x1` | NVIDIA L4 (24GB) | Recommended for LFM2.5 |
| `l4x4` | 4x NVIDIA L4 | Multi-GPU |
| `l40sx1` | NVIDIA L40S (48GB) | Larger models |
| `a100-large` | NVIDIA A100 (80GB) | Maximum capability |

Monitor training at the Space URL printed in the output. When complete, the adapter is pushed to `<username>/<project-name>` on the Hub.

### Fine-Tuning CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output`, `-o` | `./finetune-output` | Local output directory |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `4` | Per-device batch size |
| `--lr` | `2e-4` | Learning rate |
| `--lora-rank` | `16` | LoRA rank (higher = more capacity) |
| `--lora-alpha` | `32` | LoRA alpha scaling |
| `--lora-dropout` | `0.05` | LoRA dropout |
| `--max-seq-length` | `2048` | Maximum sequence length |
| `--quantize-4bit` | off | Use QLoRA (4-bit quantized base model) |
| `--target-modules` | auto-detect | Comma-separated LoRA target modules |
| `--gradient-accumulation` | `4` | Gradient accumulation steps |
| `--remote` | off | Train on HuggingFace Spaces |
| `--hf-token` | `$HF_TOKEN` | HuggingFace API token |
| `--hf-username` | auto-detect | HuggingFace username |
| `--project-name` | `liquid-host-finetune` | Hub project name for remote training |
| `--backend` | `l4x1` | GPU backend for remote training |

### Serving with a LoRA Adapter

**From a local adapter:**

```bash
liquid-host serve --model lfm2-24b-a2b --adapter ./finetune-output/adapter
```

**From a Hub adapter (after remote training):**

```bash
liquid-host serve --model lfm2-24b-a2b --adapter username/my-finetune
```

The adapter is merged into the base model at startup with no inference overhead.

## Deployment to HuggingFace Inference Endpoints

Deploy Liquid Host as a custom Docker container on HuggingFace Inference Endpoints with GPU support.

### Prerequisites

1. [HuggingFace account](https://huggingface.co/) with a valid API token
2. A container registry (GitHub Container Registry, Docker Hub, etc.)
3. Docker installed locally

### Build and Push the Docker Image

```bash
# Authenticate with your container registry
echo $GHCR_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Build for linux/amd64
    docker build --platform linux/amd64 -t ghcr.io/healeyengineering/liquid-host:latest .

# Push
    docker push ghcr.io/healeyengineering/liquid-host:latest
```

### Deploy with the Script

```bash
# Authenticate with HuggingFace
pip install huggingface-hub
huggingface-cli login

# Deploy (base model only)
python deploy/deploy_hf.py \
  --image ghcr.io/healeyengineering/liquid-host:latest \
  --instance-type nvidia-a100 --instance-size x1

# Deploy with a fine-tuned LoRA adapter and training data editor
python deploy/deploy_hf.py \
  --image ghcr.io/healeyengineering/liquid-host:latest \
  --instance-type nvidia-a100 --instance-size x1 \
  --adapter YOUR_USERNAME/my-finetune \
  --hf-token $HF_TOKEN \
  --training-repo YOUR_USERNAME/my-finetune-data
```

### Deploy Script Options

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | (required) | Docker image URL |
| `--name` | `liquid-host-lfm` | Endpoint name |
| `--repo` | `LiquidAI/LFM2-24B-A2B` | HF model repo (mounted at `/repository`) |
| `--instance-type` | `nvidia-l4` | GPU type (`nvidia-t4`, `nvidia-l4`, `nvidia-a10g`, `nvidia-a100`) |
| `--instance-size` | `x1` | Instance size (`x1`, `x2`, `x4`) |
| `--region` | `us-east-1` | Cloud region |
| `--vendor` | `aws` | Cloud vendor (`aws`, `azure`) |
| `--adapter` | None | HF Hub adapter repo to load at startup |
| `--hf-token` | None | HF token (needed for private adapter repos) |
| `--training-repo` | None | HF dataset repo for the training data editor |
| `--scale-to-zero` | `15` | Minutes before scaling to zero (0 to disable) |
| `--namespace` | your username | HF namespace or org |

### Environment Variables

The container accepts these environment variables (set via HF portal or deploy script):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/repository` | Path to model weights |
| `MODEL_KEY` | `lfm2-24b-a2b` | Model registry key |
| `ADAPTER_PATH` | None | HF Hub adapter repo ID or local path |
| `PORT` | `80` | Server port |
| `DEVICE_MAP` | `auto` | Device placement |
| `DTYPE` | None | Override torch dtype |
| `USE_FLASH_ATTN` | `false` | Enable Flash Attention 2 |
| `MCP_CONFIG` | None | Path to MCP server config |
| `N_GPU_LAYERS` | `99` | GPU layers for GGUF backend |
| `N_CTX` | `128000` | Context window for GGUF backend |
| `HF_HOME` | `/tmp/hf_cache` | HuggingFace cache directory |
| `TRAINING_HF_REPO` | None | HF dataset repo for training data editor |

### GPU Recommendations

| Model | Min GPU | Recommended |
|-------|---------|-------------|
| LFM2.5-1.2B (no MCP tools) | T4 (16GB) | T4 |
| LFM2.5-1.2B (with MCP tools) | L4 (24GB) | L4 |
| LFM2-24B-A2B (with MCP tools) | A100 (80GB) | A100 (80GB) |

The 29 Aiera MCP tool schemas add ~13K tokens to the system prompt. The 24B MoE model requires an A100 (80GB) when MCP tools are enabled. Smaller models (1.2B) work on L4 (24GB).

### Accessing the Deployed Endpoint

Once deployed, the endpoint URL hosts both the API and the web UI:

- **Web UI:** `https://YOUR_ENDPOINT_URL/`
- **Health check:** `https://YOUR_ENDPOINT_URL/health`
- **Chat API:** `https://YOUR_ENDPOINT_URL/v1/chat/completions`

```bash
curl https://YOUR_ENDPOINT_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

## Training Data Management

The CLI includes commands to manage training data on HuggingFace Hub:

```bash
# Push local JSONL to HF Hub
liquid-host data push data/training/aiera_tools_v4.jsonl --repo username/my-dataset

# Pull from HF Hub to local file
liquid-host data pull --repo username/my-dataset --output data/training/pulled.jsonl

# List/preview a dataset
liquid-host data list --repo username/my-dataset

# Validate a local JSONL file
liquid-host data validate data/training/aiera_tools_v4.jsonl
```

The web-based training data editor (`/training`) provides a UI for browsing, editing, and syncing training examples. Set `TRAINING_HF_REPO` and `HF_TOKEN` environment variables (or pass `--training-repo` and `--hf-token` to the deploy script) to auto-configure it.

## Verbose Logging

Add `-v` to any command for debug-level logging:

```bash
liquid-host -v serve --gguf /path/to/model
```

This shows detailed logs for every step: tokenization, model generation (with token counts and tok/s), tool calls, MCP communication, and more.
