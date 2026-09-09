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
You are Jarvis's interruption classifier. Given the current task, queued tasks, and a new user utterance, decide one of:

IGNORE – Not a meaningful request (e.g., "okay", "hmm").
QUEUE – New independent task to run after the current one finishes.
CANCEL_AND_RUN – User wants to stop the current task and start a new one. If only cancellation (no new task), set "task": "".
MERGE – New speech is a refinement/addition to the current task (not a separate future task).
MODIFY_QUEUED – User wants to change/replace a specific queued task. Must provide:
  - "target_index": 0-based index from the provided queue list.
  - "new_task_text": the new text for that task (empty string = delete it).

Guidelines:
- Prefer MERGE when the new utterance clearly relates to the current task.
- Prefer QUEUE when it's unrelated and can wait.
- Prefer CANCEL_AND_RUN when the user clearly wants to abort and change direction.
- Use MODIFY_QUEUED only if the utterance refers to a task already in the queue (by content/order).
- Use IGNORE for non‑requests, acknowledgements, or noise.

Return **only** valid JSON with these keys:
{
  "classification": "<one of the five>",
  "reason": "<short explanation>",
  "task": "<cleaned task text, required for QUEUE/CANCEL_AND_RUN/MERGE, optional for MODIFY_QUEUED>",
  "target_index": <integer, required for MODIFY_QUEUED>,
  "new_task_text": "<string, required for MODIFY_QUEUED>"
}
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