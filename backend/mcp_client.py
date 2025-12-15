"""MCP Client for connecting LangChain agent to DSPy MCP Server.

This client spawns the MCP server as a subprocess and communicates via stdio.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


class MCPClient:
    """Client for communicating with DSPy MCP Server.
    Uses subprocess with stdio for communication.
    """

    def __init__(self, server_script: str = None):
        self.server_script = server_script or str(
            Path(__file__).parent / "dspy_mcp_server.py"
        )
        self.process: subprocess.Popen | None = None
        self.tools_cache: list[dict] = []
        self._request_id = 0

    async def start(self):
        """Start the MCP server process."""
        if self.process is not None:
            return

        # Start server as subprocess
        self.process = subprocess.Popen(
            [sys.executable, self.server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Initialize connection
        await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "dspy-langchain-agent", "version": "1.0.0"},
            },
        )

        # Send initialized notification
        await self._send_notification("notifications/initialized", {})

        # Get available tools
        response = await self._send_request("tools/list", {})
        if response and "tools" in response:
            self.tools_cache = response["tools"]

    async def stop(self):
        """Stop the MCP server process."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None

    async def _send_request(self, method: str, params: dict) -> dict | None:
        """Send JSON-RPC request to server."""
        if not self.process:
            return None

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()

            # Read response
            response_line = self.process.stdout.readline()
            if response_line:
                response = json.loads(response_line)
                return response.get("result")
        except Exception as e:
            print(f"MCP request error: {e}")

        return None

    async def _send_notification(self, method: str, params: dict):
        """Send JSON-RPC notification (no response expected)."""
        if not self.process:
            return

        notification = {"jsonrpc": "2.0", "method": method, "params": params}

        try:
            self.process.stdin.write(json.dumps(notification) + "\n")
            self.process.stdin.flush()
        except Exception as e:
            print(f"MCP notification error: {e}")

    async def list_tools(self) -> list[dict]:
        """Get list of available tools."""
        if self.tools_cache:
            return self.tools_cache

        response = await self._send_request("tools/list", {})
        if response and "tools" in response:
            self.tools_cache = response["tools"]
        return self.tools_cache

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        """Call a tool on the MCP server."""
        response = await self._send_request(
            "tools/call", {"name": name, "arguments": arguments}
        )

        if response and "content" in response:
            # Parse text content
            for item in response["content"]:
                if item.get("type") == "text":
                    try:
                        return json.loads(item["text"])
                    except:
                        return {"result": item["text"]}

        return {"error": "No response from tool"}

    def get_tools_for_langchain(self) -> list[dict]:
        """Get tools formatted for LangChain."""
        tools = []
        for tool in self.tools_cache:
            tools.append(
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool.get("inputSchema", {}),
                }
            )
        return tools


class InProcessMCPServer:
    """In-process MCP server for direct tool execution.
    Implements DSPy tools directly without external MCP dependency.
    """

    def __init__(self):
        # Import tools from dspy_tools module
        from dspy_tools import TOOLS, TOOLS_SCHEMA, reset_session

        self._tools = TOOLS
        self._reset_session = reset_session
        self.tools_schema = TOOLS_SCHEMA

    def reset(self):
        """Reset session state."""
        self._reset_session()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        """Call a tool directly."""
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}"}

        try:
            result = await self._tools[name](**arguments)
            return result
        except Exception as e:
            import traceback

            traceback.print_exc()
            return {"error": str(e)}

    def list_tools(self) -> list[dict]:
        """Get list of available tools."""
        return self.tools_schema
