import inspect
import json
import os
import random
import re
import signal
import subprocess
import time
import uuid
import requests
from rich.console import Console
from .tools.get_weather import get_weather, GET_WEATHER_TOOL
from .tools.duck_web_search import duck_web_search, DUCK_WEB_SEARCH_TOOL
from .tools.fetch_webpage import fetch_webpage, FETCH_WEBPAGE_TOOL
from .tools.wikipedia_search import wikipedia_search, WIKIPEDIA_SEARCH_TOOL
from .tools.execute_command_tool import execute_command, EXECUTE_COMMAND_TOOL
from .tools.file_editor_tool import (
    file_exists,
    read_file,
    write_file,
    edit_file,
    apply_patch,
)
from .tools.file_editor_tool import (
    FILE_EXISTS_TOOL,
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
    EDIT_FILE_TOOL,
    APPLY_PATCH_TOOL,
)
from .tools.web_automation import web_automation, WEB_AUTOMATION_TOOL
from .tools.open_safari_tab import open_safari_tab, OPEN_SAFARI_TAB_TOOL
from .tools.google_search import google_search, GOOGLE_SEARCH_TOOL
from .proxies import proxies
from .tools.text_to_speech.speech import speak_async, SPEECH_TOOL
from .memory_monitor import get_current_rss_bytes, DEFAULT_MEM_LIMIT_BYTES
from .cancelTask import TaskCancelled, CancellationToken

console = Console()

MLX_CHAT_URL = os.getenv(
    "MLX_CHAT_URL",
    "http://localhost:8080/v1/chat/completions",
)

LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "").strip()

MEM_LIMIT_BYTES = int(
    os.getenv(
        "AGENT_MEM_LIMIT_BYTES",
        str(DEFAULT_MEM_LIMIT_BYTES),
    )
)

SYSTEM_PROMPT = """
You are a helpful orchestrator agent with access to tools.

            First, try to answer the user's request using your own knowledge and the registered tools.
            Use tools responsibly and only when necessary.
            Create a new tool only if existing tools are insufficient and the user's request cannot be fulfilled with existing tools, and use subagents to run tasks that require the new tool.
            Since your knowledge cut-off is in past, always use system date to know the current time using tool `execute_command_tool`.
            IF IT IS COMPLEX TO ACHEIVE THE TASK, SIMPLIFY IT BY BREAKING DOWN INTO SMALLER STEPS AND THEN SOLVE IT. 
            DO NOT GIVE UP.

            To determine if a new tool is needed, consider:
            - If the user's request requires information or actions that cannot be fulfilled by existing tools.
            - Perform a focused web search (using `google_search`, `duck_web_search` or `fetch_webpage`) to verify whether a new tool is required and how it might be implemented.
            - If the user's request involves a specific domain or task that is not covered by existing tools, and you have a clear implementation plan for a new tool.
            - If the user's request requires interacting with external systems or APIs not currently supported, and you can describe a safe implementation.

            Tool creation rule:
            - Only create a new tool when you determine the existing tools and knowledge are insufficient to fulfill the user's request.
            - If you decide to create a tool, follow the Create-and-Register workflow described in the `skills/create_tool.md` document to implement the tool under `./tools/` and save the files.
            - Use the `file_editor` tools (`write_file`, `edit_file`, or `apply_patch`) to create and save the new tool files under `./tools/` before spawning a subagent.
            - Note: the newly created tool cannot be executed by this running agent instance.

            Trigger rule:
            - If you attempted reasonable calls to existing tools (search, fetch, and other pertinent tools) and still cannot satisfy the user's request, produce a TOOL_CREATION_PROPOSAL (see format below) instead of continuing to iterate on the same toolset.

            Tool creation proposal format:
            When you determine a new tool is required, emit a `TOOL_CREATION_PROPOSAL` block with the following fields:
            - name: short_tool_name
            - description: one-line description of intent and when to use
            - inputs: JSON-schema or list of required inputs
            - files: a list of files to write under `./tools/` with exact file contents
            - registration: the import statements and the entries to add to `TOOLS` and `TOOL_FUNCTIONS`
            - subagent_task: the explicit command or task the subagent should run, the tools it should use, and the expected deliverable

            The agent must output this proposal exactly once before performing file edits; after the user or orchestrator accepts it, use `file_editor` tools to write files and then spawn a subagent to run the assigned `subagent_task`.

Subagent orchestration rule:
- After creating and saving a new tool, spawn a subagent process that runs in the same environment so it can import and use the newly added tool.
- The spawned subagent has access to all existing registered tools and may use them when running its assigned task.
- To spawn a subagent, run the same driver used to start the main agent, for example:
  python demo_tool.py --task "<task describing what the subagent should do>"
- The main agent acts as the orchestrator: it may spawn one or more subagents, wait for each to finish, and then incorporate each subagent's final response into the overall answer.
- A spawned subagent MUST NOT spawn further subagents. Only the main (original) agent may spawn subagents.
- The subagent should run the task using the updated codebase (including the new tool) and return a final result which the main agent will accept as the subagent's output.

Tool usage rules:
- Carefully inspect tool results before deciding next steps.
- If insufficient, you may retry the same tool at most once with a better query.
- If a tool result contains links, use the `fetch_webpage` tool to fetch and parse pages as needed. Ensure links are safe to parse.
- Batch independent tool calls together to reduce latency and redundant requests.
- If the tool is not suficient to complete the task, consider creating a new tool and spawning a subagent to run it.

Orchestration hints:
- When creating a tool, make the task for the subagent explicit: include what the subagent must run, which tool it should use, and the expected deliverable.
- Use the subagent's final output as authoritative for work done using the new tool; validate it briefly before including it in the main answer.

Post-integration code review and patching:
- After producing and applying the patch that creates or updates the tool files, perform a focused code review of only the files changed by that patch (do not code review or modify unrelated files).
- Produce a short diff summary of the changes and run lightweight checks (syntax, imports, obvious name mismatches) limited to those files.
- If the code review finds issues, produce a follow-up patch that only modifies the files included in the original diff and apply it via `apply_patch`. Repeat code review once; do not cascade to unrelated files.
- MAKE SURE THE CODE REVIEW IS DONE BY A SPAWNING SUBAGENT WITH THE NEW TOOL AVAILABLE, NOT BY THE MAIN AGENT. The main agent should only orchestrate the subagent and incorporate its final output.

Dependencies and subagent execution:
- If the tool creation requires dependencies, install those dependencies in the current environment before spawning the subagent. Use the `execute_command` tool to run each dependency installation as:
    - `python -m pip install <dependency>`
    - After successful installs, run `python -m pip freeze > requirements.txt` to update the repo's `requirements.txt` so the environment is reproducible. DO NOT MANUALLY UPDATE IT.
- If installations fail do not spawn the subagent until resolved.

After writing files (and installing dependencies if any), spawn the subagent to validate and run the new tool.

To quickly open a website, use the `open_safari_tab` tool.

In case you need to control a web browser to perform more complex tasks like filling out forms
or navigating websites consider makin javascript automation tools using selenium and web_automation tool or
making python scripts that can be executed using execute_command_tool.

If the user expects an audio response, use speak_async once to deliver the final response.
After speak_async succeeds, the current user request is complete and no further tool calls
should be made for that request to acknowledge it.

If the user is expecting text, then respond in text format.
"""

TERMINAL_TOOLS = {
    "speak_async",
}

def parse_legacy_tool_call(text):
    function_match = re.search(
        r"<tool_call>\s*<function=(.*?)>(.*?)</function>\s*</tool_call>",
        text,
        re.DOTALL,
    )
    if not function_match:
        return []
    tool_name = function_match.group(1).strip()
    body = function_match.group(2)
    parameters = {}
    for match in re.finditer(
        r"<parameter=(.*?)>\s*(.*?)\s*</parameter>",
        body,
        re.DOTALL,
    ):
        name = match.group(1).strip()
        value = match.group(2).strip()
        parameters[name] = value
    return [
        {
            "id": f"legacy_{uuid.uuid4().hex}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(parameters),
            },
        }
    ]

def _build_tool_configs():
    TOOLS = [
        GET_WEATHER_TOOL,
        SPEECH_TOOL,
        WIKIPEDIA_SEARCH_TOOL,
        DUCK_WEB_SEARCH_TOOL,
        FETCH_WEBPAGE_TOOL,
        EXECUTE_COMMAND_TOOL,
        FILE_EXISTS_TOOL,
        READ_FILE_TOOL,
        WRITE_FILE_TOOL,
        EDIT_FILE_TOOL,
        APPLY_PATCH_TOOL,
        WEB_AUTOMATION_TOOL,
        OPEN_SAFARI_TAB_TOOL,
        GOOGLE_SEARCH_TOOL,
    ]
    TOOL_FUNCTIONS = {
        "get_weather": get_weather,
        "speak_async": speak_async,
        "wikipedia_search": wikipedia_search,
        "fetch_webpage": fetch_webpage,
        "duck_web_search": duck_web_search,
        "execute_command": execute_command,
        "file_exists": file_exists,
        "read_file": read_file,
        "write_file": write_file,
        "edit_file": edit_file,
        "apply_patch": apply_patch,
        "web_automation": web_automation,
        "open_safari_tab": open_safari_tab,
        "google_search_safari": google_search,
    }
    return TOOLS, TOOL_FUNCTIONS

def _make_payload(messages, tools, max_tokens):
    payload = {
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if LOCAL_MODEL_NAME:
        payload["model"] = LOCAL_MODEL_NAME
    return payload

def _send_mlx_request(
    payload,
    cancellation_token: CancellationToken | None = None,
):
    """
    Send an MLX request while respecting cooperative cancellation.

    The request uses a short connection timeout and streams the response.
    Cancellation is checked between retry attempts and before returning.
    """
    max_attempts = 5
    base_delay = 1.0
    for attempt in range(1, max_attempts + 1):
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        resp = requests.post(
            MLX_CHAT_URL,
            json=payload,
            proxies=proxies,
            verify=False,
            timeout=(10, 1800),
            stream=True,
        )
        if resp.status_code != 429:
            if cancellation_token and cancellation_token.is_cancelled():
                resp.close()
                cancellation_token.raise_if_cancelled()
            return resp
        if attempt == max_attempts:
            return resp
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        delay = base_delay * (2 ** (attempt - 1))
        jitter = random.uniform(0, delay)
        sleep_time = delay + jitter
        console.print(
            "[yellow]"
            f"MLX 429 received — backing off {sleep_time:.2f}s "
            f"(attempt {attempt})"
            "[/yellow]"
        )
        if cancellation_token:
            if cancellation_token.event.wait(timeout=sleep_time):
                cancellation_token.raise_if_cancelled()
        else:
            time.sleep(sleep_time)

def _stream_response(
    response,
    cancellation_token: CancellationToken | None = None,
):
    """
    Stream an MLX response.

    Cancellation is checked between received SSE lines. If cancellation
    occurs, the HTTP response is closed and TaskCancelled is raised.
    """
    content_chunks = []
    reasoning_chunks = []
    tool_calls = {}
    try:
        for line in response.iter_lines(decode_unicode=True):
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                print(content, end="", flush=True)
                content_chunks.append(content)
            reasoning = delta.get("reasoning")
            if reasoning:
                print(reasoning, end="", flush=True)
                reasoning_chunks.append(reasoning)
            for tc in delta.get("tool_calls") or []:
                index = tc.get("index", 0)
                if index not in tool_calls:
                    tool_calls[index] = {
                        "id": "",
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": "",
                        },
                    }
                current = tool_calls[index]
                if tc.get("id"):
                    current["id"] = tc["id"]
                if tc.get("type"):
                    current["type"] = tc["type"]
                function_delta = tc.get("function", {})
                if function_delta.get("name"):
                    current["function"]["name"] += (
                        function_delta["name"]
                    )
                if function_delta.get("arguments"):
                    current["function"]["arguments"] += (
                        function_delta["arguments"]
                    )
    except TaskCancelled:
        response.close()
        raise
    finally:
        response.close()
    tool_list = list(tool_calls.values())
    if tool_list:
        console.print("[debug] received tool_calls:")
        for idx, tc in enumerate(tool_list):
            func = tc.get("function", {})
            name = func.get("name") or ""
            args = func.get("arguments") or ""
            console.print(
                f"[debug] index={idx} "
                f"id={tc.get('id')} "
                f"name={name[:80]!r} "
                f"args_len={len(args)}"
            )
    print()
    return content_chunks, reasoning_chunks, tool_list

def _summarize_tool_result(
    tool_name: str,
    tool_result,
    arguments: dict | None = None,
    cancellation_token: CancellationToken | None = None,
) -> str:
    """
    Summarize a tool result while respecting cancellation.
    """
    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    try:
        try:
            small_text = (
                tool_result
                if isinstance(tool_result, str)
                else json.dumps(
                    tool_result,
                    ensure_ascii=False,
                )
            )
        except Exception:
            small_text = str(tool_result)
        SHORT_THRESHOLD = 2000
        if isinstance(small_text, str) and len(small_text) <= SHORT_THRESHOLD:
            print(
                f"[summarizer] short-circuit returning: "
                f"{small_text!r}"
            )
            return small_text
        user_content = f"Tool: {tool_name}\n"
        if arguments:
            try:
                args_text = json.dumps(
                    arguments,
                    ensure_ascii=False,
                    indent=2,
                )
            except Exception:
                args_text = str(arguments)
            user_content += f"Args:\n{args_text}\n"
        user_content += "Output:\n"
        user_content += (
            json.dumps(
                tool_result,
                ensure_ascii=False,
                indent=2,
            )
            if not isinstance(tool_result, str)
            else tool_result
        )
        summary_messages = [
            {
                "role": "system",
                "content": """
                    You are a summarization agent managed by orchestrator.
                    This call is made by orchestrator agent to summarize the output of a tool call.
                    Given the tool name, the arguments it was called with, and the raw output below, figure out what was
                    the intent for this tool call and figure out if the output answers it, then provide the answer with
                    relevant facts on which orchestrator can act upon like this:
                    { "output": "<summary>" }
                """,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        payload = _make_payload(
            summary_messages,
            [],
            20000,
        )
        resp = _send_mlx_request(
            payload,
            cancellation_token=cancellation_token,
        )
        if resp.status_code == 200:
            chunks, _, _ = _stream_response(
                resp,
                cancellation_token=cancellation_token,
            )
            result = "".join(chunks).strip() or str(tool_result)
            print(
                f"[summarizer] returning: {result!r}"
            )
            return result
        print(
            "[summarizer] summarization failed: "
            "non-200 response"
        )
        return "tool response summarization failed"
    except TaskCancelled:
        raise
    except Exception as error:
        print(
            f"[summarizer] summarization exception: {error}"
        )
        return "tool response summarization failed"

def _tool_accepts_cancellation_token(tool_function):
    """
    Determine whether a tool explicitly supports a cancellation_token
    parameter.

    We do this dynamically so existing tools continue working unchanged.
    """
    try:
        signature = inspect.signature(tool_function)
        return (
            "cancellation_token" in signature.parameters
            or "cancel_event" in signature.parameters
        )
    except (TypeError, ValueError):
        return False

def _execute_tool_function(
    tool_function,
    arguments,
    cancellation_token: CancellationToken | None,
):
    """
    Execute a tool.

    Existing tools that don't know about cancellation continue to work
    normally.

    Tools that implement either:

        cancellation_token=CancellationToken

    or:

        cancel_event=threading.Event

    automatically receive the cancellation mechanism.
    """
    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    if cancellation_token and _tool_accepts_cancellation_token(tool_function):
        try:
            signature = inspect.signature(tool_function)
            if "cancellation_token" in signature.parameters:
                arguments = dict(arguments)
                arguments["cancellation_token"] = cancellation_token
            elif "cancel_event" in signature.parameters:
                arguments = dict(arguments)
                arguments["cancel_event"] = cancellation_token.event
        except (TypeError, ValueError):
            pass
    result = tool_function(**arguments)
    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    return result

def _execute_tool_calls(
    tool_calls,
    tool_functions,
    messages,
    cancellation_token: CancellationToken | None = None,
):
    """
    Execute all tool calls while respecting task cancellation.
    """
    terminal_tool_called = False
    for idx, tool_call in enumerate(tool_calls):
        function = tool_call["function"]
        tool_name = function["name"]
        console.print(
            f"[debug] executing tool_call "
            f"id={tool_call.get('id')} "
            f"name={tool_name!r} "
            f"args_preview="
            f"{str(function.get('arguments'))[:200]!r}"
        )
        try:
            arguments = json.loads(
                function.get("arguments", "{}")
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSON arguments for {tool_name}: "
                f"{function.get('arguments')}"
            ) from exc
        console.print(
            f"[bold yellow]Calling tool:[/bold yellow] "
            f"{tool_name}({arguments})"
        )
        if tool_name not in tool_functions:
            tool_result = f"Unknown tool: {tool_name}"
        else:
            max_attempts = 5
            base_delay = 1.0
            attempt = 1
            while True:
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()
                try:
                    tool_result = _execute_tool_function(
                        tool_functions[tool_name],
                        arguments,
                        cancellation_token,
                    )
                    break
                except TaskCancelled:
                    # Mark all remaining tool calls (including current) as cancelled
                    for remaining_idx in range(idx, len(tool_calls)):
                        remaining_call = tool_calls[remaining_idx]
                        messages.append({
                            "role": "tool",
                            "tool_call_id": remaining_call["id"],
                            "name": remaining_call["function"]["name"],
                            "content": json.dumps({
                                "cancelled": True,
                                "reason": "cancelled by user"
                            })
                        })
                    raise
                except requests.exceptions.RequestException as exc:
                    if cancellation_token:
                        cancellation_token.raise_if_cancelled()
                    resp = getattr(exc, "response", None)
                    status = getattr(resp, "status_code", None)
                    if (
                        status == 429
                        and attempt < max_attempts
                    ):
                        delay = base_delay * (
                            2 ** (attempt - 1)
                        )
                        jitter = random.uniform(0, delay)
                        sleep_time = delay + jitter
                        console.print(
                            "[yellow]"
                            f"Tool {tool_name} 429 — "
                            f"backing off {sleep_time:.2f}s "
                            f"(attempt {attempt})"
                            "[/yellow]"
                        )
                        if cancellation_token:
                            if cancellation_token.event.wait(
                                timeout=sleep_time
                            ):
                                cancellation_token.raise_if_cancelled()
                        else:
                            time.sleep(sleep_time)
                        attempt += 1
                        continue
                    tool_result = (
                        f"Tool {tool_name} failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    break
                except Exception as exc:
                    if cancellation_token:
                        cancellation_token.raise_if_cancelled()
                    tool_result = (
                        f"Tool {tool_name} failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    break
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        try:
            small_text = (
                tool_result
                if isinstance(tool_result, str)
                else json.dumps(
                    tool_result,
                    ensure_ascii=False,
                )
            )
        except Exception:
            small_text = str(tool_result)
        NEED_SUMMARY_THRESHOLD = 2000
        if tool_name == "execute_command":
            summarized = small_text
            print(
                "[summarizer] skipped summarization "
                f"for {tool_name}; returning raw result"
            )
        else:
            need_summarize = False
            if tool_name in (
                "duck_web_search",
                "fetch_webpage",
                "wikipedia_search",
            ):
                need_summarize = True
            elif (
                isinstance(small_text, str)
                and len(small_text) > NEED_SUMMARY_THRESHOLD
            ):
                need_summarize = True
            if need_summarize:
                summarized = _summarize_tool_result(
                    tool_name,
                    tool_result,
                    arguments,
                    cancellation_token,
                )
            else:
                summarized = small_text
                print(
                    "[summarizer] skipped summarization "
                    f"for {tool_name}; returning raw result"
                )
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_name,
                "content": summarized,
            }
        )
        print(
            summarized,
            end="",
            flush=True,
        )
        if tool_name in TERMINAL_TOOLS:
            terminal_tool_called = True
            console.print(
                f"[debug] terminal tool {tool_name!r} completed; "
                "ending current agent turn"
            )
            break
    return terminal_tool_called

class LocalAgent:
    """
    Encapsulates agent lifecycle, payload, request, streaming,
    tool execution, and cooperative task cancellation.
    """
    def __init__(self):
        self.tools, self.tool_functions = _build_tool_configs()
        self.messages = []

    def make_payload(self, messages, max_tokens):
        return _make_payload(
            messages,
            self.tools,
            max_tokens,
        )

    def send_request(
        self,
        payload,
        cancellation_token: CancellationToken | None = None,
    ):
        return _send_mlx_request(
            payload,
            cancellation_token=cancellation_token,
        )

    def stream_response(
        self,
        response,
        cancellation_token: CancellationToken | None = None,
    ):
        return _stream_response(
            response,
            cancellation_token=cancellation_token,
        )

    def execute_tool_calls(
        self,
        tool_calls,
        messages,
        cancellation_token: CancellationToken | None = None,
    ):
        return _execute_tool_calls(
            tool_calls,
            self.tool_functions,
            messages,
            cancellation_token=cancellation_token,
        )

    def _check_memory_and_maybe_exit(self):
        """
        Check current RSS and terminate process if it exceeds
        MEM_LIMIT_BYTES.
        """
        try:
            rss = get_current_rss_bytes()
            if rss is None:
                console.print(
                    "[yellow]"
                    "Could not determine current memory usage; "
                    "continuing without enforcement"
                    "[/yellow]"
                )
                return
            if rss > MEM_LIMIT_BYTES:
                console.print(
                    "[red]"
                    f"Memory limit exceeded: "
                    f"{rss / 1024**3:.2f}GB > "
                    f"{MEM_LIMIT_BYTES / 1024**3:.2f}GB — "
                    "terminating agent"
                    "[/red]"
                )
                os.kill(
                    os.getpid(),
                    signal.SIGTERM,
                )
            console.print(
                "[green]"
                f"Current memory usage: "
                f"{rss / 1024**3:.2f}GB / "
                f"{MEM_LIMIT_BYTES / 1024**3:.2f}GB"
                "[/green]"
            )
        except Exception as error:
            console.print(
                "[yellow]"
                f"Memory monitoring failed: {error}; "
                "continuing without enforcement"
                "[/yellow]"
            )

    def run(
        self,
        user_prompt: str,
        max_tokens: int = 20000,
        cancellation_token: CancellationToken | None = None,
    ) -> str:
        """
        Execute one task.

        cancellation_token belongs exclusively to this task.

        If cancellation is requested, TaskCancelled propagates to the
        caller. The caller can then decide whether to discard the task,
        report cancellation, or move to another task.
        """
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        print(
            f"[agent] Running with user prompt: "
            f"{user_prompt!r}"
        )
        if not self.messages:
            self.messages.append(
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            )
        self.messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )
        print(
            f"[agent] Running with "
            f"{len(self.messages)} messages in context"
        )
        while True:
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            self._check_memory_and_maybe_exit()
            cancellation_token.raise_if_cancelled()
            payload = self.make_payload(
                self.messages,
                max_tokens,
            )
            response = self.send_request(
                payload,
                cancellation_token=cancellation_token,
            )
            try:
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()
                if response.status_code != 200:
                    raise RuntimeError(
                        f"MLX server error: "
                        f"{response.status_code}\n"
                        f"{response.text}"
                    )
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()
                (
                    content_chunks,
                    reasoning_chunks,
                    tool_calls,
                ) = self.stream_response(
                    response,
                    cancellation_token=cancellation_token,
                )
            finally:
                response.close()
            cancellation_token.raise_if_cancelled()
            if not tool_calls:
                reasoning = "".join(
                    reasoning_chunks
                )
                if "<tool_call>" in reasoning:
                    parsed = parse_legacy_tool_call(
                        reasoning
                    )
                    if parsed:
                        tool_calls = parsed
                    else:
                        return "".join(content_chunks)
                else:
                    return "".join(content_chunks)
            cancellation_token.raise_if_cancelled()
            self.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "".join(content_chunks)
                        or None
                    ),
                    "tool_calls": tool_calls,
                }
            )
            task_completed = self.execute_tool_calls(
                tool_calls,
                self.messages,
                cancellation_token=cancellation_token,
            )
            if task_completed:
                return ""