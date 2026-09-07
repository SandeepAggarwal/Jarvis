import requests
import json
from bs4 import BeautifulSoup
from ..proxies import proxies

FETCH_WEBPAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_webpage",
        "description": "Fetch and return the readable text from a webpage.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the webpage to fetch."
                }
            },
            "required": ["url"]
        }
    }
}

def fetch_webpage(url: str,
                  cancellation_token=None) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    if cancellation_token:
        cancellation_token.raise_if_cancelled()

    response = requests.get(
        url,
        headers=headers,
        proxies=proxies,
        verify=False, # for enabling proxyman requests
        timeout=15,
    )

    if cancellation_token:
        cancellation_token.raise_if_cancelled()

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove things that aren't useful to the LLM
    for tag in soup([
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "aside",
        "noscript",
    ]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Clean whitespace
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)