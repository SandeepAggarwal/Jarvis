from duckduckgo_api_haystack import DuckduckgoApiWebSearch
from typing import Any
from ..proxies import proxies


# Reuse one search component rather than constructing it for every call.
search_engine = DuckduckgoApiWebSearch(
    top_k=10,
    max_results=10,
    region="wt-wt",
    safesearch="moderate",
    timeout=10,
    proxy=proxies["http"]
)

DUCK_WEB_SEARCH_TOOL = {
    "type": "function",
    "name": "duck_web_search",
    "description": (
        "Search the web using DuckDuckGo. "
        "Use this tool when current or externally verifiable "
        "information is needed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The web search query.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}

def duck_web_search(query: str,
                    cancellation_token=None,) -> dict[str, Any]:
    """Execute the DuckDuckGo search"""

    if not query.strip():
        return {
            "query": query,
            "results": [],
        }

    if cancellation_token:
        cancellation_token.raise_if_cancelled()

    result = search_engine.run(query=query.strip())

    if cancellation_token:
        cancellation_token.raise_if_cancelled()

    documents = result.get("documents", [])
    links = result.get("links", [])

    results = []

    for i, document in enumerate(documents):
        meta = getattr(document, "meta", {}) or {}

        results.append({
            "title": meta.get("title", f"Result {i + 1}"),
            "url": links[i] if i < len(links) else meta.get("url"),
            "snippet": getattr(document, "content", "") or "",
        })

    return {
        "query": query,
        "results": results,
    }
