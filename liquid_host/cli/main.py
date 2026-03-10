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
    console.print(f"  MCP config: [cyan]{mcp_config or 'mcp_servers.json (default)'}[/cyan]")
    console.print(f"  Listening on: [cyan]{host}:{port}[/cyan]")
    console.print()

    create_app(
        cache_dir=cache_dir,
        preload_model=model,
        preload_gguf=gguf,
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


if __name__ == "__main__":
    cli()
