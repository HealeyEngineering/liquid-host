"""CLI command to pull logs from HF Inference Endpoints."""

from __future__ import annotations

import re
import time
from datetime import datetime

import click
import httpx
from rich.console import Console

console = Console()

_LOGS_API = "https://api.endpoints.huggingface.cloud/v2/endpoint/{namespace}/{name}/logs"

# Parse log lines: "- 2026-03-17T16:50:25.779+00:00 2026-03-17 16:50:25,778 [INFO] source: message"
_LOG_RE = re.compile(
    r"^- (\S+)\s+"  # HF timestamp
    r"(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ )?"  # optional python timestamp
    r"(?:\[(\w+)\]\s*)?"  # optional [LEVEL]
    r"(.*)$"  # rest of line
)

_LEVEL_STYLE = {
    "ERROR": "bold red",
    "WARNING": "yellow",
    "INFO": "",
    "DEBUG": "dim",
}


def _resolve_token(hf_token: str | None) -> str:
    if hf_token:
        return hf_token
    import os
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass
    raise click.ClickException("HF token required. Set HF_TOKEN or pass --hf-token.")


def _resolve_namespace(namespace: str | None, token: str) -> str:
    if namespace:
        return namespace
    from huggingface_hub import whoami
    return whoami(token=token)["name"]


def _fetch_logs(namespace: str, name: str, token: str) -> list[str]:
    url = _LOGS_API.format(namespace=namespace, name=name)
    resp = httpx.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    resp.raise_for_status()
    return resp.text.strip().splitlines()


def _parse_line(line: str) -> dict:
    m = _LOG_RE.match(line)
    if m:
        return {"timestamp": m.group(1), "level": m.group(2) or "INFO", "message": m.group(3)}
    # Bare lines (uvicorn output, progress bars, etc.)
    clean = line.lstrip("- ").strip()
    return {"timestamp": "", "level": "INFO", "message": clean}


def _should_show(entry: dict, level: str | None, keyword: str | None) -> bool:
    levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    if level and entry["level"] in levels:
        if levels.index(entry["level"]) < levels.index(level):
            return False
    if keyword and keyword.lower() not in entry["message"].lower():
        return False
    return True


def _print_entry(entry: dict) -> None:
    level = entry["level"]
    style = _LEVEL_STYLE.get(level, "")
    ts = entry["timestamp"]
    # Escape any rich markup in log message content
    msg = entry["message"].replace("[", "\\[")
    if ts:
        ts_short = ts[:19]  # trim timezone for display
        if style:
            console.print(f"[cyan]{ts_short}[/cyan] [{style}]{level:7s}[/{style}] {msg}")
        else:
            console.print(f"[cyan]{ts_short}[/cyan] {level:7s} {msg}")
    else:
        console.print(f"  {msg}")


@click.command("logs")
@click.option("--name", default="liquid-host-lfm", help="Endpoint name")
@click.option("--namespace", default=None, help="HF namespace (auto-detected)")
@click.option("--hf-token", envvar="HF_TOKEN", default=None, help="HF API token")
@click.option("--tail", "tail_n", type=int, default=None, help="Show last N lines")
@click.option("--follow", "-f", is_flag=True, help="Poll for new logs")
@click.option("--level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default=None, help="Minimum log level")
@click.option("--grep", "keyword", default=None, help="Filter by keyword")
@click.option("--poll-interval", type=float, default=3.0, help="Seconds between polls in follow mode")
def logs(
    name: str,
    namespace: str | None,
    hf_token: str | None,
    tail_n: int | None,
    follow: bool,
    level: str | None,
    keyword: str | None,
    poll_interval: float,
) -> None:
    """Pull logs from an HF Inference Endpoint."""
    token = _resolve_token(hf_token)
    ns = _resolve_namespace(namespace, token)

    try:
        raw_lines = _fetch_logs(ns, name, token)
    except httpx.HTTPStatusError as e:
        raise click.ClickException(f"Failed to fetch logs: {e.response.status_code} {e.response.text[:200]}")

    entries = [_parse_line(line) for line in raw_lines]
    entries = [e for e in entries if _should_show(e, level, keyword)]

    if tail_n:
        entries = entries[-tail_n:]

    for entry in entries:
        _print_entry(entry)

    if not follow:
        return

    # Follow mode: poll and print new lines
    seen_count = len(raw_lines)
    console.print(f"\n[dim]--- following (poll every {poll_interval}s, Ctrl+C to stop) ---[/dim]\n")
    try:
        while True:
            time.sleep(poll_interval)
            try:
                raw_lines = _fetch_logs(ns, name, token)
            except Exception:
                continue
            if len(raw_lines) > seen_count:
                new_lines = raw_lines[seen_count:]
                for line in new_lines:
                    entry = _parse_line(line)
                    if _should_show(entry, level, keyword):
                        _print_entry(entry)
                seen_count = len(raw_lines)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")
