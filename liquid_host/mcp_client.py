"""MCP client manager — connects to configured MCP servers and exposes their tools."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "mcp_servers.json"


@dataclass
class McpToolInfo:
    """A tool discovered from an MCP server."""

    name: str
    description: str
    input_schema: dict
    server_name: str

    def to_llm_schema(self) -> dict:
        """Convert to the JSON schema format expected by LFM tool calling."""
        return {
            "name": self.name,
            "description": self.description or "",
            "parameters": self.input_schema or {"type": "object", "properties": {}},
        }


@dataclass
class _ServerConnection:
    name: str
    session: ClientSession
    tools: list[McpToolInfo] = field(default_factory=list)


class McpClientManager:
    """Manages connections to one or more MCP servers."""

    def __init__(self, config_path: str | Path | None = None):
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._connections: dict[str, _ServerConnection] = {}
        self._exit_stack: AsyncExitStack | None = None
        # Maps tool name → server name for routing calls
        self._tool_routing: dict[str, str] = {}
        # Override API key (applied to all server URLs via ?api_key= param)
        self._api_key_override: str | None = None

    @property
    def api_key(self) -> str | None:
        """Return the current API key (override, env var, or from config)."""
        import os
        if self._api_key_override:
            return self._api_key_override
        env_key = os.environ.get("MCP_API_KEY")
        if env_key:
            return env_key
        # Read from config file
        if self._config_path.exists():
            config = json.loads(self._config_path.read_text())
            for entry in config.get("servers", []):
                url = entry.get("url", "")
                if "api_key=" in url:
                    from urllib.parse import parse_qs, urlparse
                    parsed = urlparse(url)
                    params = parse_qs(parsed.query)
                    keys = params.get("api_key", [])
                    if keys:
                        return keys[0]
        return None

    # ── Lifecycle ──────────────────────────────────────────────────

    async def connect_all(self, api_key: str | None = None) -> None:
        """Read the config file and connect to all enabled servers.

        If *api_key* is provided it overrides the api_key query param in all
        server URLs.  Falls back to the ``MCP_API_KEY`` environment variable.
        """
        import os
        if api_key:
            self._api_key_override = api_key
        elif not self._api_key_override:
            # Fall back to environment variable
            env_key = os.environ.get("MCP_API_KEY")
            if env_key:
                self._api_key_override = env_key

        if not self._config_path.exists():
            logger.warning("MCP config not found at %s — no servers loaded", self._config_path)
            return

        config = json.loads(self._config_path.read_text())
        servers = config.get("servers", [])

        self._exit_stack = AsyncExitStack()

        for entry in servers:
            if not entry.get("enabled", True):
                logger.info("Skipping disabled MCP server: %s", entry.get("name"))
                continue
            # Apply API key to the URL
            if self._api_key_override:
                from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
                parsed = urlparse(entry["url"])
                params = parse_qs(parsed.query, keep_blank_values=True)
                params["api_key"] = [self._api_key_override]
                new_query = urlencode(params, doseq=True)
                entry = {**entry, "url": urlunparse(parsed._replace(query=new_query))}
            try:
                await self._connect_server(entry)
            except Exception:
                logger.exception("Failed to connect to MCP server '%s'", entry.get("name"))

    async def _connect_server(self, entry: dict) -> None:
        name = entry["name"]
        url = entry["url"]
        transport = entry.get("transport", "sse")

        logger.info("Connecting to MCP server '%s' at %s (transport=%s) ...", name, url, transport)

        if transport == "sse":
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                sse_client(url)
            )
        elif transport in ("streamable_http", "streamable-http", "http"):
            read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
                streamable_http_client(url)
            )
        else:
            logger.error("Unsupported transport '%s' for server '%s'", transport, name)
            return

        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        tools_result = await session.list_tools()
        tools: list[McpToolInfo] = []
        for t in tools_result.tools:
            tool_info = McpToolInfo(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema if hasattr(t, "inputSchema") else {},
                server_name=name,
            )
            tools.append(tool_info)
            self._tool_routing[t.name] = name

        conn = _ServerConnection(name=name, session=session, tools=tools)
        self._connections[name] = conn
        logger.info(
            "Connected to '%s' — discovered %d tools: %s",
            name,
            len(tools),
            [t.name for t in tools],
        )

    async def disconnect_all(self) -> None:
        """Close all MCP server connections."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._connections.clear()
        self._tool_routing.clear()
        logger.info("Disconnected from all MCP servers")

    # ── Tool discovery ─────────────────────────────────────────────

    @property
    def tools(self) -> list[McpToolInfo]:
        """All tools across all connected servers."""
        result = []
        for conn in self._connections.values():
            result.extend(conn.tools)
        return result

    def get_llm_tools_schema(self) -> list[dict]:
        """Get tool definitions formatted for the LLM chat template."""
        return [t.to_llm_schema() for t in self.tools]

    @property
    def is_connected(self) -> bool:
        return len(self._connections) > 0

    # ── Tool execution ─────────────────────────────────────────────

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool by name and return the result as a string."""
        server_name = self._tool_routing.get(tool_name)
        if not server_name:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        conn = self._connections.get(server_name)
        if not conn:
            return json.dumps({"error": f"Server '{server_name}' not connected"})

        logger.info("Calling tool '%s' on server '%s' with args: %s", tool_name, server_name, arguments)

        try:
            result = await conn.session.call_tool(name=tool_name, arguments=arguments)
        except Exception as e:
            logger.exception("Tool call '%s' failed", tool_name)
            return json.dumps({"error": str(e)})

        # Flatten content blocks into a single string
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))

        return "\n".join(parts)

    # ── Parse model output ─────────────────────────────────────────

    @staticmethod
    def parse_tool_calls(text: str) -> list[dict]:
        """Parse tool calls from model output.

        Handles three formats:
            1. <|tool_call_start|>[func_name(arg1="val1")]<|tool_call_end|>  (special tokens)
            2. [func_name(arg1="val1")]                                      (bare bracket)
            3. <tool_call>{"name":"func","arguments":{...}}</tool_call>      (XML+JSON, fine-tuned)

        Returns a list of {"name": ..., "arguments": {...}} dicts.
        """
        import json as _json

        calls = []

        # Format 3: <tool_call> JSON </tool_call> (fine-tuned model format)
        xml_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        xml_matches = re.findall(xml_pattern, text, re.DOTALL)
        if xml_matches:
            for m in xml_matches:
                try:
                    parsed = _json.loads(m)
                    name = parsed.get("name")
                    arguments = parsed.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = _json.loads(arguments)
                    if name:
                        calls.append({"name": name, "arguments": arguments})
                except (_json.JSONDecodeError, TypeError):
                    continue
            if calls:
                return calls

        # Format 1: special tokens
        pattern = r"<\|tool_call_start\|>\s*\[(.+?)\]\s*<\|tool_call_end\|>"
        matches = re.findall(pattern, text, re.DOTALL)

        # Format 2: bare bracket (after </think> block if present)
        if not matches:
            cleaned = text
            if "</think>" in cleaned:
                cleaned = cleaned.split("</think>", 1)[1]
            bare_pattern = r"\[(\w+\(.*?\))\]"
            matches = re.findall(bare_pattern, cleaned, re.DOTALL)

        for match in matches:
            func_match = re.match(r"(\w+)\((.*)?\)", match.strip(), re.DOTALL)
            if not func_match:
                continue
            name = func_match.group(1)
            args_str = func_match.group(2) or ""
            arguments = McpClientManager._parse_python_args(args_str)
            calls.append({"name": name, "arguments": arguments})
        return calls

    @staticmethod
    def _parse_python_args(args_str: str) -> dict:
        """Parse Python-style keyword arguments into a dict.

        Handles: arg1="val1", arg2=123, arg3=True, arg4=[1,2,3]
        """
        if not args_str.strip():
            return {}
        # Use a safe eval approach by constructing a dict
        try:
            # Wrap in a dict constructor call and eval safely
            expr = f"dict({args_str})"
            result = eval(expr, {"__builtins__": {"dict": dict, "True": True, "False": False, "None": None}}, {})  # noqa: S307
            return result
        except Exception:
            logger.warning("Failed to parse tool call arguments: %s", args_str)
            return {"raw": args_str}

    @staticmethod
    def strip_tool_calls(text: str) -> str:
        """Remove tool call tokens from text to get the plain assistant message."""
        cleaned = re.sub(
            r"<\|tool_call_start\|>.*?<\|tool_call_end\|>",
            "",
            text,
            flags=re.DOTALL,
        )
        return cleaned.strip()
