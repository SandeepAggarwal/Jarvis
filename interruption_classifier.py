import json
import os
from typing import List, Optional

import requests

from Agent.proxies import proxies

MLX_CHAT_URL = os.getenv(
    "MLX_CHAT_URL",
    "http://localhost:8080/v1/chat/completions",
)
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "").strip()


CLASSIFIER_SYSTEM_PROMPT = """
You are Jarvis's interruption classifier.

Jarvis may currently be executing a task when the user speaks again. There may also
be other tasks waiting in the queue.

Classify the NEW user utterance into exactly ONE of:

IGNORE
QUEUE
CANCEL_AND_RUN
MERGE
MODIFY_QUEUED

Definitions:

IGNORE:
- The utterance is not a meaningful user request.
- Background speech, accidental speech, filler, acknowledgement, noise-like transcription,
  or something that should not cause Jarvis to perform another task.
- Examples:
  "okay"
  "yeah"
  "hmm"
  "never mind"
  "that's fine"

QUEUE:
- A meaningful new task that should happen after the current task finishes.
- It does NOT require stopping the current task.
- Examples:
  "after that, remind me to call John"
  "then check the weather"
  "also search for flights to Tokyo"

CANCEL_AND_RUN:
- The new request clearly changes the user's priority.
- The current task should be stopped as soon as safely possible.
- The new request should become the active task. If there is nothing new task to do but just cancel the task, then new
  request should be empty string "".
- Examples:
  "stop that"
  "cancel what you're doing"
  "forget that, search for Tokyo hotels instead"
  "no, do this instead"
  "nevermind, I will do it"

MERGE:
- The new request is an addition, refinement, correction, or continuation of the
  CURRENT task and should be incorporated into the same task rather than treated
  as an independent later task.
- Examples:
  Current: "Find me flights to London."
  New: "Make that business class."
  Current: "Search for restaurants in Delhi."
  New: "Only vegetarian ones."
  Current: "Open that website."
  New: "And click the second result."

MODIFY_QUEUED:
- The user wants to change or replace a specific task that is already waiting in the queue.
- This is for modifying a task that is NOT the currently running one, but is somewhere
  in the pending queue.
- You MUST output:
    "target_index": the 0‑based index of the task in the queued list (as shown in the prompt)
    "new_task_text": the new content for that task
    "task": a short description (optional)
- Examples:
  * User says: "Actually, for that weather request I made earlier, change the city to Paris."
    → Find the "weather" task in the queue, set its index, and new text = "Get weather for Paris".
  * User says: "Cancel the calendar check I asked for."
    → Set new_task_text = "" or "cancel" to effectively remove it, or you can simply replace it with an empty task.

Important guidelines:
- Use the CURRENT TASK and the QUEUED TASKS (with indices) to decide.
- If the new request clearly refers to a queued task (by content or time), use MODIFY_QUEUED.
- If it modifies the current task, prefer MERGE.
- If it is independent, prefer QUEUE.
- If it tells Jarvis to stop/change direction, prefer CANCEL_AND_RUN.
- If it isn't meaningful, use IGNORE.

Return ONLY valid JSON with these keys:
- "classification": one of the above
- "reason": short explanation
- "task": cleaned version of the new request (for QUEUE, CANCEL_AND_RUN, MERGE, or optional for MODIFY_QUEUED)
- "target_index": integer (only for MODIFY_QUEUED)
- "new_task_text": string (only for MODIFY_QUEUED)
"""


class InterruptionClassifier:
    def __init__(self):
        pass

    def classify(self, current_task: str, queued_tasks: List[str], new_task: str) -> dict:
        """
        Classify an interruption given the current running task, the list of queued tasks,
        and the new user utterance.

        Returns a dict with at least:
            classification: str
            reason: str
            task: str
        For MODIFY_QUEUED, also includes:
            target_index: int
            new_task_text: str
        """
        # Build a readable list of queued tasks with indices
        queued_display = "\n".join(
            f"{i}: {task}" for i, task in enumerate(queued_tasks)
        ) or "(empty queue)"

        messages = [
            {
                "role": "system",
                "content": CLASSIFIER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "current_task": current_task,
                        "queued_tasks": queued_display,
                        "new_task": new_task,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        payload = {
            "messages": messages,
            "tools": [],
            "tool_choice": "none",
            "temperature": 0.0,
            "max_tokens": 2000,
            "stream": False,
        }

        if LOCAL_MODEL_NAME:
            payload["model"] = LOCAL_MODEL_NAME

        response = requests.post(
            MLX_CHAT_URL,
            json=payload,
            proxies=proxies,
            verify=False,
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Interruption classifier failed: "
                f"{response.status_code}: {response.text}"
            )

        data = response.json()

        content = (
            data["choices"][0]["message"]
            .get("content", "")
            .strip()
        )

        result = self._parse_result(content)

        return result

    @staticmethod
    def _parse_result(content: str) -> dict:
        # Models sometimes surround JSON with ```json ... ```
        if "```" in content:
            content = content.replace("```json", "")
            content = content.replace("```", "")
            content = content.strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Safe fallback: treat as QUEUE to avoid accidental cancellation
            return {
                "classification": "QUEUE",
                "reason": "Classifier returned invalid JSON.",
                "task": content,
            }

        # Extract and validate classification
        classification = result.get("classification", "").upper()
        valid_classifications = {
            "IGNORE", "QUEUE", "CANCEL_AND_RUN", "MERGE", "MODIFY_QUEUED"
        }
        if classification not in valid_classifications:
            classification = "QUEUE"

        # Build base response
        response = {
            "classification": classification,
            "reason": result.get("reason", ""),
            "task": result.get("task", "").strip(),
        }

        # For MODIFY_QUEUED, we require target_index and new_task_text
        if classification == "MODIFY_QUEUED":
            target_index = result.get("target_index")
            new_task_text = result.get("new_task_text")
            if target_index is not None and new_task_text is not None:
                # Ensure index is integer
                try:
                    target_index = int(target_index)
                except (ValueError, TypeError):
                    target_index = -1  # invalid
                response["target_index"] = target_index
                response["new_task_text"] = new_task_text
            else:
                # Missing required fields – fallback to QUEUE
                response["classification"] = "QUEUE"
                response["reason"] = "MODIFY_QUEUED missing target_index or new_task_text, falling back to QUEUE."
                # Keep the task as the new_task_text if provided
                if new_task_text:
                    response["task"] = new_task_text
                else:
                    response["task"] = result.get("task", "").strip()

        return response