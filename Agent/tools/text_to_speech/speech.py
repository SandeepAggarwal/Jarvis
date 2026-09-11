"""
Global asynchronous TTS interface.
Usage from anywhere:
    from tools.text_to_speech.speech import speak
    speak("Hello!")
The first call initializes the shared AudioRuntime.
All subsequent calls reuse it.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional

SPEECH_TOOL = {
    "type": "function",
    "function": {
        "name": "speak_sync",
        "description": (
            "Speak text synchronously using Piper TTS."
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


@dataclass
class _SpeechItem:
    """A single unit of work for the TTS worker."""
    text: str
    done: threading.Event


# Queue holds either a _SpeechItem or None (the poison pill for shutdown).
_speech_queue: "queue.Queue[Optional[_SpeechItem]]" = queue.Queue()
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
        item = _speech_queue.get()
        try:
            if item is None:
                # Poison pill: shut down the worker.
                return

            if _tts is None:
                continue

            _tts.speak(item.text)
        except Exception as e:
            print("TTS error:", repr(e))
        finally:
            # Signal completion for this specific item (if any).
            if item is not None:
                item.done.set()
            _speech_queue.task_done()


def speak_sync(text: str,
                cancellation_token=None,
                wait: bool = True):
    """
    Queue text for asynchronous speech.

    Safe to call from any module.

    Parameters
    ----------
    text:
        The text to speak. Empty/whitespace-only text is ignored.
    cancellation_token:
        Optional token with .is_cancelled() and .raise_if_cancelled().
    wait:
        If True (default), this call returns only after *this* text has
        finished being spoken. If False, it returns immediately after
        enqueueing (fire-and-forget).

    Note
    ----
    The TTS worker still processes items one at a time in FIFO order, so
    even with wait=True, a call may block behind previously queued speech.
    The difference is that it returns as soon as its *own* item is done,
    not when the entire queue is drained.
    """

    if cancellation_token and cancellation_token.is_cancelled():
        print("cancelling speech")
        cancellation_token.raise_if_cancelled()

    if not text:
        return

    text = text.strip()

    if not text:
        return

    _ensure_speaker()

    item = _SpeechItem(text=text, done=threading.Event())
    _speech_queue.put(item)

    if wait:
        # Block until the worker has finished speaking this item.
        # Poll in small increments so we can honor cancellation.
        while not item.done.wait(timeout=0.1):
            if cancellation_token and cancellation_token.is_cancelled():
                cancellation_token.raise_if_cancelled()


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
            _speech_queue.put(None)  # poison pill

            if _worker is not None:
                _worker.join(timeout=5.0)

        except Exception as e:
            print("TTS shutdown error:", repr(e))

        finally:
            _worker = None
            _tts = None


__all__ = [
    "speak_sync",
    "wait_for_speech",
    "stop_speaker",
]