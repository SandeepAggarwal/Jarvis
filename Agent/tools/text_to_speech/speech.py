"""
Global asynchronous TTS interface.
Usage from anywhere:
    from tools.text_to_speech.speech import speak_async
    speak_async("Hello!")
The first call initializes the shared AudioRuntime.
All subsequent calls reuse it.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

SPEECH_TOOL = {
    "type": "function",
    "function": {
        "name": "speak_async",
        "description": (
            "Speak text asynchronously using Piper TTS."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to speak.",
                }
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
}

_speech_queue: queue.Queue[Optional[str]] = queue.Queue()
_worker: Optional[threading.Thread] = None
_tts = None
_lock = threading.RLock()

def _ensure_speaker():
    global _worker
    global _tts

    if _tts is not None:
        return

    with _lock:
        if _tts is not None:
            return

        # ----------------------------------------------------
        # Get global audio runtime.
        # ----------------------------------------------------

        from ..speech_to_text.audio_runtime import (get_audio_runtime)
        runtime = get_audio_runtime()

        # ----------------------------------------------------
        # Create Piper once.
        # ----------------------------------------------------

        from .tts import TTS
        _tts = TTS(runtime)

        # ----------------------------------------------------
        # Start worker.
        # ----------------------------------------------------

        _worker = threading.Thread(
            target=_worker_loop,
            daemon=True,
            name="tts-worker",
        )

        _worker.start()
        print("TTS worker started.")


def _worker_loop():
    while True:
        text = _speech_queue.get()
        try:
            if text is None:
                return

            if _tts is None:
                continue

            _tts.speak(text)
        except Exception as e:
            print("TTS error:",repr(e))
        finally:
            _speech_queue.task_done()


def speak_async(text: str,
                cancellation_token=None):
    """
    Queue text for asynchronous speech.
    Safe to call from any module.
    """

    if cancellation_token:
        cancellation_token.raise_if_cancelled()

    if not text:
        return

    text = text.strip()

    if not text:
        return

    _ensure_speaker()
    _speech_queue.put(text)


def wait_for_speech() -> None:
    """
    Wait until all queued speech has finished.
    """
    _ensure_speaker()
    _speech_queue.join()


def stop_speaker() -> None:
    """
    Stop TTS worker and shared audio runtime.
    """
    global _worker
    global _tts

    with _lock:
        if _tts is None:
            return

        try:
            _speech_queue.put(None)

            if _worker is not None:
                _worker.join(timeout=5.0)

        except Exception as e:
            print("TTS shutdown error:", repr(e))

        finally:
            _worker = None
            _tts = None


__all__ = [
    "speak_async",
    "wait_for_speech",
    "stop_speaker",
]