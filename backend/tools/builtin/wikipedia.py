"""Wikipedia Tool - Searches and retrieves Wikipedia articles."""

from typing import Any

from ..base import BaseTool, ToolResult


class WikipediaTool(BaseTool):
    """Wikipedia search and retrieval tool.

    Searches Wikipedia and returns article summaries.
    """

    name = "wikipedia"
    description = "Searches Wikipedia for information about a topic. Returns article summaries. Good for factual information about people, places, events, concepts."

    def __init__(self, max_chars: int = 2000):
        """Initialize Wikipedia tool.

        Args:
            max_chars: Maximum characters to return from article
        """
        self.max_chars = max_chars

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic to search for on Wikipedia",
                }
            },
            "required": ["query"],
        }

    def run(self, query: str = "", **kwargs) -> ToolResult:
        """Search Wikipedia and return article summary.

        Args:
            query: Topic to search for

        Returns:
            ToolResult with article summary
        """
        if not query:
            query = kwargs.get("input", "")

        if not query:
            return ToolResult(
                success=False, output="", error="No search query provided"
            )

        try:
            import wikipedia

            wikipedia.set_lang("en")

            try:
                page = wikipedia.page(query, auto_suggest=True)
                summary = page.summary[: self.max_chars]

                if len(page.summary) > self.max_chars:
                    summary += "..."

                return ToolResult(
                    success=True,
                    output=f"**{page.title}**\n\n{summary}\n\nURL: {page.url}",
                    metadata={"title": page.title, "url": page.url, "query": query},
                )

            except wikipedia.DisambiguationError as e:
                options = e.options[:5]
                return ToolResult(
                    success=True,
                    output=f"Multiple results found for '{query}'. Did you mean:\n"
                    + "\n".join(f"- {opt}" for opt in options),
                    metadata={
                        "query": query,
                        "disambiguation": True,
                        "options": options,
                    },
                )

            except wikipedia.PageError:
                search_results = wikipedia.search(query, results=5)

                if search_results:
                    return ToolResult(
                        success=True,
                        output=f"No exact match for '{query}'. Related topics:\n"
                        + "\n".join(f"- {r}" for r in search_results),
                        metadata={"query": query, "suggestions": search_results},
                    )
                else:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"No Wikipedia articles found for '{query}'",
                    )

        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error="wikipedia package required. Install with: pip install wikipedia",
            )
        except Exception as e:
            return ToolResult(
                success=False, output="", error=f"Wikipedia error: {str(e)}"
            )
