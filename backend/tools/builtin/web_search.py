"""Web Search Tool - Searches the web for information."""

import os
from typing import Any

from ..base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    """Web search tool using DuckDuckGo or Tavily API.

    Falls back to DuckDuckGo if Tavily API key is not set.
    """

    name = "web_search"
    description = "Searches the web for current information. Use for questions about recent events, facts, or when you need up-to-date information."

    def __init__(self, max_results: int = 5):
        """Initialize web search tool.

        Args:
            max_results: Maximum number of search results to return
        """
        self.max_results = max_results
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        }

    def _search_tavily(self, query: str) -> list[dict[str, str]]:
        """Search using Tavily API."""
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self.tavily_api_key)
            response = client.search(query, max_results=self.max_results)

            results = []
            for item in response.get("results", []):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "snippet": item.get("content", ""),
                        "url": item.get("url", ""),
                    }
                )
            return results

        except ImportError:
            raise ImportError(
                "tavily-python required. Install with: pip install tavily-python"
            )

    def _search_duckduckgo(self, query: str) -> list[dict[str, str]]:
        """Search using DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=self.max_results):
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "snippet": r.get("body", ""),
                            "url": r.get("href", ""),
                        }
                    )
            return results

        except ImportError:
            raise ImportError(
                "duckduckgo-search required. Install with: pip install duckduckgo-search"
            )

    def run(self, query: str = "", **kwargs) -> ToolResult:
        """Search the web.

        Args:
            query: Search query

        Returns:
            ToolResult with search results
        """
        if not query:
            query = kwargs.get("input", "")

        if not query:
            return ToolResult(
                success=False, output="", error="No search query provided"
            )

        try:
            if self.tavily_api_key:
                results = self._search_tavily(query)
            else:
                results = self._search_duckduckgo(query)

            if not results:
                return ToolResult(
                    success=True,
                    output="No results found.",
                    metadata={"query": query, "num_results": 0},
                )

            output_lines = []
            for i, r in enumerate(results, 1):
                output_lines.append(f"{i}. {r['title']}")
                output_lines.append(f"   {r['snippet'][:200]}...")
                output_lines.append(f"   URL: {r['url']}")
                output_lines.append("")

            return ToolResult(
                success=True,
                output="\n".join(output_lines),
                metadata={
                    "query": query,
                    "num_results": len(results),
                    "results": results,
                },
            )

        except ImportError as e:
            return ToolResult(success=False, output="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Search error: {str(e)}")
