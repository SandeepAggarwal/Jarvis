import requests
import json
import re
from ..proxies import proxies

WIKIPEDIA_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "wikipedia_search",
        "description": "Search Wikipedia for information about a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    }
}

def wikipedia_search(query: str, cancellation_token=None) -> str:
    #console.rule("Step 3 - Search Tool")

    if cancellation_token:
        cancellation_token.raise_if_cancelled()

    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "opensearch",
            "search": query,
            "limit": 5,
            "namespace": 0,
            "format": "json",
        },
        proxies=proxies,
        verify=False, # for enabling proxyman requests
        timeout=30,
        headers={"User-Agent": "LocalMLXAgentDemo/1.0"},
    )

    if cancellation_token:
        cancellation_token.raise_if_cancelled()

    response.raise_for_status()

    data = response.json()
    titles = data[1] if len(data) > 1 else []
    snippets = data[2] if len(data) > 2 else []
    urls = data[3] if len(data) > 3 else []

    results = []
    for title, snippet, url in zip(titles, snippets, urls):
        clean_snippet = re.sub("<.*?>", "", snippet)
        clean_snippet = html.unescape(clean_snippet)
        results.append({"title": title, "snippet": clean_snippet, "url": url})

    result_text = json.dumps(results, indent=2, ensure_ascii=False)
    #console.print(Panel(result_text, title="Search API Result"))

    return result_text
