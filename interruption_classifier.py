import json
import os
import requests

from Agent.proxies import proxies

MLX_CHAT_URL = os.getenv(
    "MLX_CHAT_URL",
    "http://localhost:8080/v1/chat/completions",
)
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "").strip()


CLASSIFIER_SYSTEM_PROMPT = """
You are Jarvis's interruption classifier.

Jarvis may currently be executing a task when the user speaks again.

Classify the NEW user utterance into exactly ONE of:

IGNORE
QUEUE
CANCEL_AND_RUN
MERGE

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
- The new request should become the active task.
- Examples:
  "stop that"
  "cancel what you're doing"
  "forget that, search for Tokyo hotels instead"
  "no, do this instead"

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

Important:
- Use the CURRENT TASK when deciding.
- If the new request is clearly independent, prefer QUEUE.
- If it explicitly tells Jarvis to stop/change direction, prefer CANCEL_AND_RUN.
- If it modifies the current task, prefer MERGE.
- If it isn't a meaningful request, use IGNORE.

Return ONLY valid JSON:

{
  "classification": "IGNORE|QUEUE|CANCEL_AND_RUN|MERGE",
  "reason": "short explanation",
  "task": "cleaned version of the new request"
}
"""


class InterruptionClassifier:
    def __init__(self):
        pass

    def classify(self, current_task: str, new_task: str) -> dict:
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
            # Safe fallback: don't accidentally cancel a running task.
            return {
                "classification": "QUEUE",
                "reason": "Classifier returned invalid JSON.",
                "task": content,
            }

        classification = result.get("classification", "").upper()

        if classification not in {
            "IGNORE",
            "QUEUE",
            "CANCEL_AND_RUN",
            "MERGE",
        }:
            classification = "QUEUE"

        task = result.get("task", "").strip()

        if not task:
            task = content

        return {
            "classification": classification,
            "reason": result.get("reason", ""),
            "task": task,
        }
