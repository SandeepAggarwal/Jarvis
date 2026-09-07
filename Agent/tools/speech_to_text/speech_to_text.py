"""
Global Speech-to-Text interface.

Both:

    SpeechRecognizer()

and:

    speech_to_text()

use the SAME AudioRuntime.

There is only one microphone,
one speaker and one AEC instance.
"""

from __future__ import annotations


SPEECH_TO_TEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "speech_to_text",
        "description": (
            "Listen to the microphone and convert "
            "the user's speech to text."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}


class SpeechRecognizer:
    """
    Backwards-compatible STT wrapper.
    Creating multiple SpeechRecognizer objects does NOT
    create multiple microphones.
    They all point to the same AudioRuntime.
    """

    def __init__(self):
        from .audio_runtime import (get_audio_runtime)
        self._runtime = (get_audio_runtime())

    def listen(self) -> str:
        return self._runtime.listen()

    def reset(self):
        self._runtime.reset()

    def close(self):
        from .audio_runtime import (shutdown_audio_runtime)
        shutdown_audio_runtime()


def speech_to_text(cancellation_token=None) -> str:
    """
    Listen to the microphone and return text.
    Automatically initializes the global AudioRuntime.
    """
    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    from .audio_runtime import (get_audio_runtime)
    runtime = get_audio_runtime()

    if cancellation_token:
        cancellation_token.raise_if_cancelled()
    return runtime.listen()


def shutdown_speech():
    from .audio_runtime import (shutdown_audio_runtime)
    shutdown_audio_runtime()


__all__ = [
    "SpeechRecognizer",
    "speech_to_text",
    "shutdown_speech",
]