#!/usr/bin/env python3

import subprocess
import sys
import time

# Note: You must enable 'Allow JavaScript from Apple Events' in the Developer section of Safari Settings to use 'do JavaScript'

GOOGLE_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "google_search_safari",
        "description": (
            "Open Google in a new Safari tab, switch to that tab, "
            "enter a search query, submit the search, and return the "
            "text and metadata visible on the Google results page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The Google search query to perform."
                },
                "wait_seconds": {
                    "type": "number",
                    "description": (
                        "Seconds to wait for the Google results page "
                        "to load before extracting the page data."
                    ),
                    "default": 3,
                    "minimum": 0
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}


def run_applescript(script: str, 
                    *args: str,
                    cancellation_token=None) -> str:
    """Run AppleScript and return stdout."""
    result = subprocess.run(
        ["osascript", "-e", script, *args],
        capture_output=True,
        text=True,
        timeout=20,
    )

    if cancellation_token:
        cancellation_token.raise_if_cancelled()

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

    return result.stdout.strip()


def google_search(query: str, 
                    wait_seconds: float = 3,
                    cancellation_token=None) -> str:
    """
    Open Google in a new Safari tab, search for query,
    and return the visible text from the results page.
    """

    script = '''
on run argv
    set searchText to item 1 of argv

    tell application "Safari"
        activate

        -- Make sure Safari has a window
        if (count of windows) = 0 then
            make new document
        end if

        set safariWindow to front window

        -- Create new tab
        tell safariWindow
            set newTab to make new tab at end of tabs
            set URL of newTab to "https://www.google.com"
            set current tab to newTab
        end tell

        activate
    end tell

    -- Wait for Google
    delay 2

    tell application "System Events"
        tell process "Safari"
            -- Focus address/search bar
            keystroke "l" using command down

            -- Go to Google
            keystroke "https://www.google.com"
            key code 36

            delay 2

            -- Type search query
            keystroke searchText

            -- Submit search
            key code 36
        end tell
    end tell

    -- Wait for search results
    delay 3

    -- Get page text
    tell application "Safari"
        tell front window
            set resultText to do JavaScript "
                document.body ? document.body.innerText : ''
            " in current tab
        end tell
    end tell

    return resultText
end run
'''

    return run_applescript(script, query, cancellation_token)


if __name__ == "__main__":
    query = "OpenAI ChatGPT"

    try:
        data = google_search(query)

        print("===== SEARCH RESULTS =====")
        print(data)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)