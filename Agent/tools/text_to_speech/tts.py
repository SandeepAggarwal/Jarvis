from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from piper import PiperVoice

from .voice_manager import get_voice


TARGET_SAMPLE_RATE = 16000


class TTS:
    def __init__(self, audio_runtime):
        self.audio_runtime = audio_runtime
        model_path = get_voice()
        self.voice = PiperVoice.load(str(model_path))

    @staticmethod
    def _resample(audio, source_rate):
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

    def speak(self, text: str, cancellation_token=None) -> None:

        if not text:
            return
        text = text.strip()
        if not text:
            return

        # ---- cancellation helpers -------------------------------

        def _cancelled() -> bool:
            if cancellation_token is None:
                return False
            try:
                return bool(cancellation_token.is_cancelled())
            except Exception:
                return False

        def _abort():
            # Purge queued audio; in-flight frame finishes its 10 ms.
            self.audio_runtime.clear_tts()

        # Pre-flight check.
        if _cancelled():
            _abort()
            cancellation_token.raise_if_cancelled()

        # ---- synthesis loop (checks token per chunk) ------------

        for audio in self.voice.synthesize(text):

            if _cancelled():
                _abort()
                cancellation_token.raise_if_cancelled()

            pcm = np.frombuffer(audio.audio_int16_bytes, dtype=np.int16)
            pcm = self._resample(pcm, audio.sample_rate)
            self.audio_runtime.enqueue_tts(pcm)

        # ---- drain wait (checks token every ~50 ms) -------------

        while True:
            if _cancelled():
                _abort()
                cancellation_token.raise_if_cancelled()

            drained = self.audio_runtime.wait_for_tts(timeout=0.05)
            if drained:
                break