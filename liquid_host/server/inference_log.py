"""Persistent structured inference logging.

Writes JSONL records for each inference request, including tool calls,
timing, and errors. Logs are stored locally with daily rotation and can
optionally be pushed to a HF dataset repo.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_LOG_DIR = Path.home() / ".cache" / "liquid-host" / "logs"


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result_length: int = 0
    result_preview: str = ""
    duration_ms: float = 0
    error: str | None = None


@dataclass
class RoundRecord:
    """Details for a single tool-calling round."""
    round: int = 0
    raw_output: str = ""
    tool_calls_found: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


@dataclass
class InferenceRecord:
    """A single inference request/response cycle."""

    id: str = ""
    timestamp: str = ""
    model: str = ""
    prompt: str = ""
    response_text: str = ""
    messages_in: int = 0
    temperature: float = 0.0
    max_tokens: int = 0
    stream: bool = False
    use_tools: bool = False
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    rounds: list[RoundRecord] = field(default_factory=list)
    total_rounds: int = 0
    response_length: int = 0
    finish_reason: str = ""
    duration_ms: float = 0
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop empty lists for cleaner output
        if not d["tool_calls"]:
            del d["tool_calls"]
        if not d["rounds"]:
            del d["rounds"]
        return d


class InferenceLogger:
    """Non-blocking JSONL inference logger with daily file rotation."""

    def __init__(self, log_dir: str | Path | None = None):
        self._log_dir = Path(log_dir or os.environ.get("INFERENCE_LOG_DIR", _DEFAULT_LOG_DIR))
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=10_000)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        logger.info("Inference logger started (dir=%s)", self._log_dir)

    def _current_file(self) -> Path:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._log_dir / f"inference_{date_str}.jsonl"

    def _writer_loop(self) -> None:
        while True:
            try:
                record = self._queue.get(timeout=5)
            except queue.Empty:
                continue
            try:
                path = self._current_file()
                with open(path, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            except Exception:
                logger.exception("Failed to write inference log")

    def log(self, record: InferenceRecord) -> None:
        if not record.timestamp:
            record.timestamp = datetime.now(timezone.utc).isoformat()
        try:
            self._queue.put_nowait(record.to_dict())
        except queue.Full:
            logger.warning("Inference log queue full, dropping record")

    def list_files(self) -> list[Path]:
        return sorted(self._log_dir.glob("inference_*.jsonl"))

    def read_recent(self, n: int = 100) -> list[dict]:
        """Read the last N records from the most recent log file."""
        files = self.list_files()
        if not files:
            return []
        records = []
        for f in reversed(files):
            with open(f) as fh:
                lines = fh.readlines()
            for line in reversed(lines):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                if len(records) >= n:
                    break
            if len(records) >= n:
                break
        records.reverse()
        return records

    def push_to_hub(self, repo: str, token: str) -> str:
        """Push all local log files to a HF dataset repo. Returns commit URL."""
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)

        files = self.list_files()
        if not files:
            raise ValueError("No log files to push")

        api.upload_folder(
            folder_path=str(self._log_dir),
            repo_id=repo,
            repo_type="dataset",
            path_in_repo="logs",
            allow_patterns="inference_*.jsonl",
            commit_message=f"Push inference logs ({len(files)} files)",
        )

        logger.info("Pushed %d log files to %s", len(files), repo)
        return f"https://huggingface.co/datasets/{repo}"


# Module-level singleton, initialized lazily
_logger: InferenceLogger | None = None


def get_inference_logger(log_dir: str | Path | None = None) -> InferenceLogger:
    global _logger
    if _logger is None:
        _logger = InferenceLogger(log_dir=log_dir)
    return _logger
