#!/usr/bin/env python3

import argparse
import subprocess
import sys
import time


OPEN_SAFARI_TAB_TOOL = {
    "type": "function",
    "function": {
        "name": "open_safari_tab",
        "description": "Open a new Safari tab and navigate to a specified URL (macOS Safari).",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to open (e.g. https://example.com)."
                },
                "wait_seconds": {
                    "type": "integer",
                    "description": "Seconds to wait after opening the tab.",
                    "default": 2
                }
            },
            "required": ["url"],
            "additionalProperties": False
        }
    }
}


#!/usr/bin/env python3

import argparse
import subprocess
import sys
import time


OPEN_SAFARI_TAB_TOOL = {
    "type": "function",
    "function": {
        "name": "open_safari_tab",
        "description": "Open a new Safari tab and navigate to a specified URL (macOS Safari).",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to open (e.g. https://example.com)."
                },
                "wait_seconds": {
                    "type": "integer",
                    "description": "Seconds to wait after opening the tab.",
                    "default": 2
                }
            },
            "required": ["url"],
            "additionalProperties": False
        }
    }
}


def open_safari_tab(url: str, 
                    wait_seconds: int = 2,
                    cancellation_token=None,) -> bool:
    script = '''
on run argv
    set targetURL to item 1 of argv

    tell application "Safari"
        activate

        if (count of windows) = 0 then
            make new document
        end if

        set safariWindow to front window

        tell safariWindow
            set newTab to make new tab at end of tabs
            set URL of newTab to targetURL

            -- Force the newly created tab to become the selected tab
            set current tab to newTab

            -- Also explicitly set its index as the current tab
            set current tab to tab (index of newTab)
        end tell

        activate
    end tell
end run
'''

    try:
        subprocess.run(
            ["osascript", "-e", script, url],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if cancellation_token:
            cancellation_token.raise_if_cancelled()

        if wait_seconds:
            time.sleep(wait_seconds)

        if cancellation_token:
            cancellation_token.raise_if_cancelled()

        return True

    except subprocess.CalledProcessError as e:
        print(f"Safari error: {e.stderr.strip()}", file=sys.stderr)
        return False



def main():
    parser = argparse.ArgumentParser(
        description="Open a new tab in Safari and visit any URL"
    )

    parser.add_argument(
        "url",
        nargs="?",
        default="https://www.google.com",
        help="URL to navigate to (default: https://www.google.com)",
    )

    parser.add_argument(
        "-w",
        "--wait",
        type=int,
        default=2,
        help="Seconds to wait after opening the tab (default: 2)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if not args.url:
        parser.error("URL cannot be empty")

    if args.wait < 0:
        parser.error("--wait must be >= 0")

    success = open_safari_tab(args.url, args.wait)

    if args.verbose:
        print(
            f"Safari tab opened "
            f"{'successfully' if success else 'unsuccessfully'}"
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


def main():
    parser = argparse.ArgumentParser(
        description="Open a new tab in Safari and visit any URL"
    )

    parser.add_argument(
        "url",
        nargs="?",
        default="https://www.google.com",
        help="URL to navigate to (default: https://www.google.com)",
    )

    parser.add_argument(
        "-w",
        "--wait",
        type=int,
        default=2,
        help="Seconds to wait after opening the tab (default: 2)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if not args.url:
        parser.error("URL cannot be empty")

    if args.wait < 0:
        parser.error("--wait must be >= 0")

    success = open_safari_tab(args.url, args.wait)

    if args.verbose:
        print(
            f"Safari tab opened "
            f"{'successfully' if success else 'unsuccessfully'}"
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()