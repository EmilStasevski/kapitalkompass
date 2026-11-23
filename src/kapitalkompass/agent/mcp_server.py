"""The agent's five tools as an MCP server (stdio). Every call goes through ``dispatch``, so the
allowlist and the Pydantic argument validation are the same as for the local agent.

    uv run python -m kapitalkompass.agent.mcp_server
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from kapitalkompass.agent.tools import TOOLS, OptimiserConstraints, Scenario, Segment, dispatch


def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    c = dispatch(name, args)
    if not c.ok:
        raise ValueError(c.error)
    return c.result


def get_raroc(segment: Segment) -> dict[str, Any]:
    return _call("get_raroc", {"segment": segment})


def get_limits(state: str = "current") -> dict[str, Any]:
    return _call("get_limits", {"state": state})


def run_shock(name: Scenario) -> dict[str, Any]:
    return _call("run_shock", {"name": name})


def run_optimiser(constraints: OptimiserConstraints) -> dict[str, Any]:
    return _call("run_optimiser", {"constraints": constraints.model_dump(exclude_none=True)})


def search_rules(query: str) -> dict[str, Any]:
    return _call("search_rules", {"query": query})


def build() -> MCPServer:
    server = MCPServer("kapitalkompass")
    for fn in (get_raroc, get_limits, run_shock, run_optimiser, search_rules):
        server.add_tool(fn, name=fn.__name__, description=TOOLS[fn.__name__].description)
    return server


if __name__ == "__main__":
    build().run("stdio")
