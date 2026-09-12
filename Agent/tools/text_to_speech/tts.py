from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import resample_poly

from piper import PiperVoice

from .voice_manager import get_voice


TARGET_SAMPLE_RATE = 16000


@dataclass
class SpokenResult:
    """What was actually heard from a TTS.speak() call."""
    full_text: str
    text_heard: str
    cancelled: bool


def _split_sentences(text: str) -> List[str]:
    """Cheap sentence splitter. Piper splits similarly internally."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


class TTS:
    """
    Piper TTS. AudioRuntime owns the actual speaker device.
    """

    def __init__(self, audio_runtime):
        self.audio_runtime = audio_runtime
        self.voice = PiperVoice.load(str(get_voice()))

    # ------------------------------------------------------------

    @staticmethod
    def _resample(audio: np.ndarray, source_rate: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.int16).reshape(-1)
        if source_rate == TARGET_SAMPLE_RATE:
            return audio
        converted = resample_poly(
            audio.astype(np.float32),
            TARGET_SAMPLE_RATE,
            source_rate,
        )
        converted = np.clip(converted, -32768, 32767)
        return converted.astype(np.int16)

    # ------------------------------------------------------------

    def speak(
        self,
        text: str,
        cancellation_token=None,
    ) -> SpokenResult:

        if not text or not text.strip():
            return SpokenResult(text or "", "", False)

        full_text = text.strip()

        def _cancelled() -> bool:
            if cancellation_token is None:
                return False
            try:
                return bool(cancellation_token.is_cancelled())
            except Exception:
                return False

        # Frame offset representing "how much the speaker has played
        # before we started this call". All measurements are relative.
        base_played = self.audio_runtime.get_played_samples()

        # (sentence_text, start_frame, end_frame) in local frame coords.
        segments: List[Tuple[str, int, int]] = []
        local_cursor = 0

        def _heard() -> str:
            played = self.audio_runtime.get_played_samples() - base_played
            print(f"[TTS][DBG] played_samples={played} segments="
      f"{[(s, a, b) for s, a, b in segments]}")
            heard: List[str] = []
            for sentence, start, end in segments:
                if played < end:
                    # Partially heard — include and stop.
                    break
                heard.append(sentence)
            return " ".join(heard)

        # Pre-flight cancel.
        if _cancelled():
            return SpokenResult(full_text, "", True)

        # ---- synthesis loop ---------------------------------------

        for sentence in _split_sentences(full_text):

            if _cancelled():
                self.audio_runtime.clear_tts()
                return SpokenResult(full_text, _heard(), True)

            pcm_chunks = []
            for audio in self.voice.synthesize(sentence):
                if _cancelled():
                    self.audio_runtime.clear_tts()
                    return SpokenResult(full_text, _heard(), True)
                pcm = np.frombuffer(audio.audio_int16_bytes, dtype=np.int16)
                pcm = self._resample(pcm, audio.sample_rate)
                pcm_chunks.append(pcm)

            if pcm_chunks:
                full_pcm = np.concatenate(pcm_chunks)
            else:
                full_pcm = np.zeros(0, dtype=np.int16)

            n_samples = self.audio_runtime.enqueue_tts(full_pcm)
            segments.append((sentence, local_cursor, local_cursor + n_samples))
            local_cursor += n_samples

        # ---- drain wait -------------------------------------------

        while True:
            if _cancelled():
                heard = _heard()               # capture BEFORE clearing
                self.audio_runtime.clear_tts()
                return SpokenResult(full_text, heard, True)

            if self.audio_runtime.wait_for_tts(timeout=0.05):
                break

        return SpokenResult(full_text, " ".join(s for s, _, _ in segments), False)