"""CLI for managing and serving Liquid AI models."""

from __future__ import annotations

import logging

import click
from rich.console import Console
from rich.table import Table

from liquid_host.config import MODEL_REGISTRY, get_model_spec

console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """Liquid Host — download and serve Liquid AI models locally."""
    _setup_logging(verbose)


# ── list ───────────────────────────────────────────────────────────


@cli.command("list")
@click.option("--family", type=click.Choice(["lfm2", "lfm2.5", "all"]), default="all")
def list_models(family: str) -> None:
    """List all available Liquid AI models."""
    table = Table(title="Available Liquid AI Models")
    table.add_column("Key", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Params")
    table.add_column("Active")
    table.add_column("Arch")
    table.add_column("Description")
    table.add_column("Tags", style="dim")

    for key, spec in MODEL_REGISTRY.items():
        if family != "all" and spec.family != family:
            continue
        table.add_row(
            key,
            spec.name,
            spec.params,
            spec.active_params,
            spec.architecture,
            spec.description,
            ", ".join(spec.tags),
        )

    console.print(table)


# ── download ───────────────────────────────────────────────────────


@cli.command("download")
@click.argument("model")
@click.option("--cache-dir", default=None, help="Custom cache directory")
@click.option("--force", is_flag=True, help="Re-download even if cached")
def download(model: str, cache_dir: str | None, force: bool) -> None:
    """Download a model to the local cache."""
    from liquid_host.models.manager import ModelManager

    try:
        spec = get_model_spec(model)
    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    console.print(f"Downloading [bold]{spec.name}[/bold] ({spec.repo_id}) ...")
    mgr = ModelManager(cache_dir=cache_dir)
    path = mgr.download(model, force=force)
    console.print(f"[green]Done![/green] Saved to {path}")


# ── downloaded ─────────────────────────────────────────────────────


@cli.command("downloaded")
@click.option("--cache-dir", default=None)
def downloaded(cache_dir: str | None) -> None:
    """Show models present in the local cache."""
    from liquid_host.models.manager import ModelManager

    mgr = ModelManager(cache_dir=cache_dir)
    items = mgr.list_downloaded()

    if not items:
        console.print("[dim]No models downloaded yet.[/dim]")
        return

    table = Table(title="Downloaded Models")
    table.add_column("Name", style="bold")
    table.add_column("Repo ID")
    table.add_column("Revision", style="dim")
    table.add_column("Size (GB)")

    for item in items:
        table.add_row(item["name"], item["repo_id"], item["revision"], str(item["size_gb"]))

    console.print(table)


# ── delete ─────────────────────────────────────────────────────────


@cli.command("delete")
@click.argument("model")
@click.option("--cache-dir", default=None)
@click.confirmation_option(prompt="Are you sure you want to delete this model?")
def delete(model: str, cache_dir: str | None) -> None:
    """Delete a downloaded model from cache."""
    from liquid_host.models.manager import ModelManager

    mgr = ModelManager(cache_dir=cache_dir)
    if mgr.delete(model):
        console.print(f"[green]Deleted {model}[/green]")
    else:
        console.print(f"[yellow]Model {model} not found in cache[/yellow]")


# ── serve ──────────────────────────────────────────────────────────


@cli.command("serve")
@click.option("--model", default=None, help="HuggingFace model to preload (transformers backend)")
@click.option("--gguf", default=None, help="Path to GGUF file or directory to preload (llama.cpp backend)")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory")
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--port", default=8000, type=int, help="Port")
@click.option("--cache-dir", default=None)
@click.option("--device-map", default="auto")
@click.option("--dtype", default=None, help="Override torch dtype (e.g. float16, bfloat16)")
@click.option("--flash-attn", is_flag=True, help="Use Flash Attention 2")
@click.option("--n-ctx", default=128000, type=int, help="Context window size (GGUF only)")
@click.option("--n-gpu-layers", default=0, type=int, help="Layers to offload to GPU (GGUF only)")
@click.option("--workers", default=1, type=int)
@click.option("--mcp-config", default=None, help="Path to mcp_servers.json (default: ./mcp_servers.json)")
def serve(
    model: str | None,
    gguf: str | None,
    adapter: str | None,
    host: str,
    port: int,
    cache_dir: str | None,
    device_map: str,
    dtype: str | None,
    flash_attn: bool,
    n_ctx: int,
    n_gpu_layers: int,
    workers: int,
    mcp_config: str | None,
) -> None:
    """Start the inference server."""
    import uvicorn
    from liquid_host.server.app import create_app

    console.print("[bold]Starting Liquid Host server...[/bold]")
    if gguf:
        console.print(f"  Preloading GGUF: [cyan]{gguf}[/cyan] (llama.cpp backend)")
    elif model:
        console.print(f"  Preloading model: [cyan]{model}[/cyan] (transformers backend)")
    if adapter:
        console.print(f"  LoRA adapter: [cyan]{adapter}[/cyan]")
    console.print(f"  MCP config: [cyan]{mcp_config or 'mcp_servers.json (default)'}[/cyan]")
    console.print(f"  Listening on: [cyan]{host}:{port}[/cyan]")
    console.print()

    create_app(
        cache_dir=cache_dir,
        preload_model=model,
        preload_gguf=gguf,
        adapter_path=adapter,
        device_map=device_map,
        dtype=dtype,
        use_flash_attn=flash_attn,
        mcp_config=mcp_config,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
    )

    # Match uvicorn log level to our logging config
    uvi_level = "debug" if logging.getLogger().level <= logging.DEBUG else "info"

    uvicorn.run(
        "liquid_host.server.app:app",
        host=host,
        port=port,
        workers=workers,
        log_level=uvi_level,
    )


# ── run (interactive chat) ─────────────────────────────────────────


@cli.command("run")
@click.argument("model")
@click.option("--cache-dir", default=None)
@click.option("--device-map", default="auto")
@click.option("--dtype", default=None)
@click.option("--system", default=None, help="System prompt")
def run(model: str, cache_dir: str | None, device_map: str, dtype: str | None, system: str | None) -> None:
    """Interactive chat with a model in the terminal."""
    from liquid_host.models.manager import ModelManager
    from liquid_host.config import GenerationConfig

    mgr = ModelManager(cache_dir=cache_dir)

    console.print(f"Loading [bold]{model}[/bold] ...")
    mgr.download(model)
    mgr.load(model, device_map=device_map, dtype=dtype)
    console.print(f"[green]Ready![/green] Type your messages (Ctrl+C to quit).\n")

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})

    config = GenerationConfig()

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        console.print("[bold green]Assistant:[/bold green] ", end="")
        full_response = []
        for token in mgr.generate_stream(messages, config):
            console.print(token, end="", highlight=False)
            full_response.append(token)
        console.print()

        messages.append({"role": "assistant", "content": "".join(full_response)})


# ── finetune ──────────────────────────────────────────────────────


@cli.command("finetune")
@click.argument("model")
@click.argument("data", type=click.Path(exists=True))
@click.option("--output", "-o", default="./finetune-output", help="Output directory for adapter")
@click.option("--epochs", default=3, type=int, help="Number of training epochs")
@click.option("--batch-size", default=4, type=int, help="Per-device batch size")
@click.option("--lr", default=2e-4, type=float, help="Learning rate")
@click.option("--lora-rank", default=16, type=int, help="LoRA rank")
@click.option("--lora-alpha", default=32, type=int, help="LoRA alpha")
@click.option("--lora-dropout", default=0.05, type=float, help="LoRA dropout")
@click.option("--max-seq-length", default=2048, type=int, help="Maximum sequence length")
@click.option("--quantize-4bit", is_flag=True, help="Use QLoRA (4-bit quantized base model)")
@click.option("--target-modules", default=None, help="Comma-separated LoRA target modules (auto-detected if omitted)")
@click.option("--gradient-accumulation", default=4, type=int, help="Gradient accumulation steps")
@click.option("--cache-dir", default=None)
@click.option("--remote", is_flag=True, help="Train remotely on HuggingFace Spaces via AutoTrain")
@click.option("--hf-token", default=None, help="HuggingFace API token (or set HF_TOKEN env var)")
@click.option("--hf-username", default=None, help="HuggingFace username (auto-detected from token if omitted)")
@click.option("--project-name", default="liquid-host-finetune", help="Remote project name on HF Hub")
@click.option(
    "--backend",
    default="l4x1",
    type=click.Choice([
        "t4-small", "t4-medium",
        "a10g-small", "a10g-large",
        "l4x1", "l4x4",
        "l40sx1", "a100-large",
    ]),
    help="GPU backend for remote training",
)
def finetune(
    model: str,
    data: str,
    output: str,
    epochs: int,
    batch_size: int,
    lr: float,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    max_seq_length: int,
    quantize_4bit: bool,
    target_modules: str | None,
    gradient_accumulation: int,
    cache_dir: str | None,
    remote: bool,
    hf_token: str | None,
    hf_username: str | None,
    project_name: str,
    backend: str,
) -> None:
    """Fine-tune a model using LoRA on chat JSONL data.

    By default, trains locally. Use --remote to train on HuggingFace Spaces
    with a GPU backend (no local GPU required).
    """
    from liquid_host.training.finetune import TrainingConfig, LoraConfig

    try:
        spec = get_model_spec(model)
    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    tc = TrainingConfig(
        output_dir=output,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=lr,
        max_seq_length=max_seq_length,
        quantize_4bit=quantize_4bit,
    )
    lc = LoraConfig(
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=target_modules.split(",") if target_modules else None,
    )

    if remote:
        from liquid_host.training.finetune import (
            finetune_remote,
            RemoteConfig,
        )

        rc = RemoteConfig(
            hf_token=hf_token,
            hf_username=hf_username,
            project_name=project_name,
            backend=backend,
        )

        console.print(f"[bold]Remote fine-tuning [cyan]{spec.name}[/cyan] on {backend}[/bold]")
        console.print(f"  Data: [cyan]{data}[/cyan]")
        console.print(f"  Project: [cyan]{project_name}[/cyan]")
        console.print(f"  LoRA rank={lora_rank}, alpha={lora_alpha}")
        console.print(f"  QLoRA (4-bit): {'yes' if quantize_4bit else 'no'}")
        console.print()

        hub_model_id = finetune_remote(
            model_key=model,
            data_path=data,
            training_config=tc,
            lora_config=lc,
            remote_config=rc,
        )

        username_part = hub_model_id.split("/")[0]
        console.print(f"\n[green bold]Training launched![/green bold]")
        console.print(f"  Monitor: [cyan]https://huggingface.co/spaces/{username_part}/{project_name}-training[/cyan]")
        console.print(f"  Adapter will be at: [cyan]https://huggingface.co/{hub_model_id}[/cyan]")
        console.print(f"\nOnce training completes, serve with:")
        console.print(f"  liquid-host serve --model {model} --adapter {hub_model_id}")

    else:
        from liquid_host.training.finetune import finetune as run_finetune

        console.print(f"[bold]Fine-tuning [cyan]{spec.name}[/cyan] with LoRA (local)[/bold]")
        console.print(f"  Data: [cyan]{data}[/cyan]")
        console.print(f"  Output: [cyan]{output}[/cyan]")
        console.print(f"  LoRA rank={lora_rank}, alpha={lora_alpha}")
        console.print(f"  QLoRA (4-bit): {'yes' if quantize_4bit else 'no'}")
        console.print()

        adapter_dir = run_finetune(
            model_key=model,
            data_path=data,
            training_config=tc,
            lora_config=lc,
            cache_dir=cache_dir,
        )

        console.print(f"\n[green bold]Done![/green bold] Adapter saved to [cyan]{adapter_dir}[/cyan]")
        console.print(f"\nTo serve with this adapter:")
        console.print(f"  liquid-host serve --model {model} --adapter {adapter_dir}")


# ── data (training data management) ────────────────────────────────


@cli.group("data")
def data_group() -> None:
    """Manage training data on HuggingFace Hub."""
    pass


@data_group.command("push")
@click.argument("path", type=click.Path(exists=True))
@click.option("--repo", required=True, help="HF dataset repo (e.g. user/my-training-data)")
@click.option("--hf-token", envvar="HF_TOKEN", default=None, help="HuggingFace API token")
@click.option("--private/--public", default=True, help="Make the dataset repo private (default) or public")
@click.option("--split", default="train", help="Dataset split name (default: train)")
def data_push(path: str, repo: str, hf_token: str | None, private: bool, split: str) -> None:
    """Push local JSONL training data to a HuggingFace dataset repo.

    Validates the data format, uploads it, and prints a summary.
    """
    from pathlib import Path as P
    from liquid_host.training.finetune import load_chat_dataset

    if not hf_token:
        console.print("[red]Error:[/red] HF token required. Set HF_TOKEN or pass --hf-token.")
        raise SystemExit(1)

    local_path = P(path)
    console.print(f"Loading data from [cyan]{local_path}[/cyan]...")
    dataset = load_chat_dataset(local_path)
    console.print(f"  Loaded [bold]{len(dataset)}[/bold] examples")

    console.print(f"Pushing to [cyan]{repo}[/cyan] (split={split}, private={private})...")
    dataset.push_to_hub(repo, split=split, private=private, token=hf_token)
    console.print(f"[green bold]Done![/green bold] Dataset uploaded to https://huggingface.co/datasets/{repo}")


@data_group.command("pull")
@click.argument("repo")
@click.option("--output", "-o", default=None, help="Output file path (default: data/training/<repo-name>.jsonl)")
@click.option("--hf-token", envvar="HF_TOKEN", default=None, help="HuggingFace API token")
@click.option("--split", default="train", help="Dataset split to download (default: train)")
def data_pull(repo: str, output: str | None, hf_token: str | None, split: str) -> None:
    """Pull training data from a HuggingFace dataset repo to local JSONL.

    Downloads the dataset and saves it as JSONL for local editing.
    """
    import json
    from pathlib import Path as P
    from datasets import load_dataset

    console.print(f"Downloading [cyan]{repo}[/cyan] (split={split})...")
    ds = load_dataset(repo, split=split, token=hf_token)

    if output is None:
        repo_name = repo.split("/")[-1]
        out_dir = P("data/training")
        out_dir.mkdir(parents=True, exist_ok=True)
        output = str(out_dir / f"{repo_name}.jsonl")

    out_path = P(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out_path, "w") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")
            count += 1

    console.print(f"[green bold]Done![/green bold] Saved {count} examples to [cyan]{out_path}[/cyan]")


@data_group.command("list")
@click.argument("repo")
@click.option("--hf-token", envvar="HF_TOKEN", default=None, help="HuggingFace API token")
@click.option("--split", default="train", help="Dataset split (default: train)")
@click.option("--show", type=int, default=3, help="Number of examples to preview (default: 3)")
def data_list(repo: str, hf_token: str | None, split: str, show: int) -> None:
    """List and preview training data in a HuggingFace dataset repo."""
    import json
    from datasets import load_dataset

    console.print(f"Loading [cyan]{repo}[/cyan] (split={split})...")
    ds = load_dataset(repo, split=split, token=hf_token)

    console.print(f"  Total examples: [bold]{len(ds)}[/bold]")
    console.print()

    # Show role distribution stats
    role_counts: dict[str, int] = {}
    tool_call_count = 0
    for row in ds:
        msgs = row.get("messages", [])
        for msg in msgs:
            role = msg.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
            if role == "assistant" and "<tool_call>" in msg.get("content", ""):
                tool_call_count += 1

    table = Table(title="Message Role Distribution")
    table.add_column("Role", style="cyan")
    table.add_column("Count", justify="right")
    for role, count in sorted(role_counts.items()):
        table.add_row(role, str(count))
    table.add_row("tool_calls (in assistant)", str(tool_call_count), style="dim")
    console.print(table)

    if show > 0:
        console.print(f"\n[bold]First {min(show, len(ds))} examples:[/bold]")
        for i, row in enumerate(ds):
            if i >= show:
                break
            msgs = row.get("messages", [])
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "?")
            preview = user_msg[:100] + ("..." if len(user_msg) > 100 else "")
            n_turns = sum(1 for m in msgs if m["role"] == "user")
            console.print(f"  [{i+1}] ({n_turns} turn{'s' if n_turns > 1 else ''}, {len(msgs)} msgs) {preview}")


@data_group.command("validate")
@click.argument("path", type=click.Path(exists=True))
def data_validate(path: str) -> None:
    """Validate a local JSONL training data file.

    Checks JSON parsing, required fields, and message structure.
    """
    import json
    from pathlib import Path as P

    local_path = P(path)
    errors = []
    warnings = []
    total = 0
    multi_turn = 0
    tool_examples = 0

    with open(local_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"  Line {line_num}: invalid JSON — {e}")
                continue

            if "messages" not in obj:
                errors.append(f"  Line {line_num}: missing 'messages' key")
                continue

            msgs = obj["messages"]
            if len(msgs) < 2:
                errors.append(f"  Line {line_num}: fewer than 2 messages")
                continue

            if msgs[0].get("role") != "user":
                warnings.append(f"  Line {line_num}: first message is not 'user' role")

            user_turns = sum(1 for m in msgs if m.get("role") == "user")
            if user_turns > 1:
                multi_turn += 1

            if any("<tool_call>" in m.get("content", "") for m in msgs if m.get("role") == "assistant"):
                tool_examples += 1

    console.print(f"[bold]Validation: [cyan]{local_path}[/cyan][/bold]")
    console.print(f"  Total examples: {total}")
    console.print(f"  Multi-turn: {multi_turn}")
    console.print(f"  With tool calls: {tool_examples}")

    if errors:
        console.print(f"\n[red bold]{len(errors)} error(s):[/red bold]")
        for e in errors:
            console.print(f"  [red]{e}[/red]")
    else:
        console.print(f"\n  [green bold]No errors found.[/green bold]")

    if warnings:
        console.print(f"\n[yellow]{len(warnings)} warning(s):[/yellow]")
        for w in warnings:
            console.print(f"  [yellow]{w}[/yellow]")


if __name__ == "__main__":
    cli()
