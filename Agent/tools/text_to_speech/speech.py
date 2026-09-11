"""
Global asynchronous TTS interface.
Usage from anywhere:
    from tools.text_to_speech.speech import speak
    speak("Hello!")
The first call initializes the shared AudioRuntime.
All subsequent calls reuse it.
"""

from __future__ import annotations

import inspect
import queue
import threading
from dataclasses import dataclass
from typing import Optional, Any

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
    cancellation_token: Any = None


# Queue holds either a _SpeechItem or None (the poison pill for shutdown).
_speech_queue: "queue.Queue[Optional[_SpeechItem]]" = queue.Queue()
_worker: Optional[threading.Thread] = None
_tts = None
_lock = threading.RLock()


# ---------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------

def _caller_location(depth: int = 2) -> str:
    """
    Return a 'file:line in func' string for the caller `depth` frames up.

    depth=1 -> the function that called _caller_location
    depth=2 -> its caller (default; used from _log_cancel)
    """
    try:
        frame = inspect.currentframe()
        for _ in range(depth):
            if frame is None:
                break
            frame = frame.f_back
        if frame is None:
            return "<unknown>"
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        func = frame.f_code.co_name
        return f"{filename}:{lineno} in {func}()"
    except Exception:
        return "<unknown>"
    finally:
        # Avoid reference cycles.
        del frame


def _log_cancel(reason: str, *, depth: int = 2) -> None:
    """
    Print a uniform cancellation log line including the source location.

    `depth` is passed to _caller_location so that the reported location
    points at the caller of _log_cancel, not at _log_cancel itself.
    """
    where = _caller_location(depth=depth)
    print(f"[TTS][CANCEL] {reason} | at {where}")


def _is_cancelled(token) -> bool:
    """Best-effort check for a cancellation token."""
    if token is None:
        return False
    try:
        return bool(token.is_cancelled())
    except Exception:
        return False


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

            # Skip items that were cancelled before we even started.
            if _is_cancelled(item.cancellation_token):
                _log_cancel(
                    f"worker skipped queued item before speaking "
                    f"text={item.text!r}",
                    depth=2,
                )
                continue

            # Prefer to pass the token through so TTS can abort
            # mid-playback (e.g. between audio chunks).
            _tts.speak(
                item.text,
                cancellation_token=item.cancellation_token,
            )

        except Exception as e:
            # Swallow CancelledError-style exceptions as expected,
            # print anything else.
            if item is not None and _is_cancelled(item.cancellation_token):
                _log_cancel(
                    f"worker aborted while speaking text={item.text!r} "
                    f"(exception={e!r})",
                    depth=2,
                )
            else:
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
    Queue text for synchronous speech.

    Safe to call from any module.

    Parameters
    ----------
    text:
        The text to speak. Empty/whitespace-only text is ignored.
    cancellation_token:
        Optional token with .is_cancelled() and .raise_if_cancelled().
        If the token is cancelled:
          * before this call, the call raises immediately;
          * while this call is waiting, the call raises immediately,
            and the worker will skip the item if it hasn't started;
          * while the worker is actually speaking, the item is only
            aborted if TTS.speak() itself checks the token.
    wait:
        If True (default), this call returns only after *this* text has
        finished being spoken. If False, it returns immediately after
        enqueueing (fire-and-forget).
    """

    # Fast path: already cancelled before we do anything.
    if cancellation_token and cancellation_token.is_cancelled():
        _log_cancel(
            f"speak_sync called with pre-cancelled token, "
            f"text={text!r}",
            depth=2,
        )
        cancellation_token.raise_if_cancelled()

    if not text:
        return

    text = text.strip()

    if not text:
        return

    _ensure_speaker()

    item = _SpeechItem(
        text=text,
        done=threading.Event(),
        cancellation_token=cancellation_token,
    )
    _speech_queue.put(item)

    if not wait:
        return

    # Block until the worker has finished this item, OR until the
    # caller's token is cancelled — whichever happens first.
    while not item.done.wait(timeout=0.1):
        if cancellation_token and cancellation_token.is_cancelled():
            # The item is still in the queue (or in-flight). The worker
            # will see the token on its next check and either skip the
            # item or let TTS abort it. We just stop waiting here.
            _log_cancel(
                f"speak_sync stopped waiting due to cancellation, "
                f"text={text!r}",
                depth=2,
            )
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