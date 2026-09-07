from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from piper import PiperVoice

from .voice_manager import get_voice


TARGET_SAMPLE_RATE = 16000


class TTS:
    """
    Piper TTS.

    Piper generates audio.

    AudioRuntime owns the actual speaker device.
    """

    def __init__(self, audio_runtime):

        self.audio_runtime = audio_runtime

        model_path = get_voice()

        self.voice = PiperVoice.load(
            str(model_path)
        )

    @staticmethod
    def _resample(
        audio: np.ndarray,
        source_rate: int,
    ) -> np.ndarray:

        audio = np.asarray(
            audio,
            dtype=np.int16,
        ).reshape(-1)

        if source_rate == TARGET_SAMPLE_RATE:

            return audio

        # ----------------------------------------------------
        # Piper may output 22050 Hz.
        #
        # Convert to the same 16 kHz format used by:
        #
        #     speaker
        #     AEC
        #     microphone
        #     STT
        # ----------------------------------------------------

        converted = resample_poly(
            audio.astype(np.float32),
            TARGET_SAMPLE_RATE,
            source_rate,
        )

        converted = np.clip(
            converted,
            -32768,
            32767,
        )

        return converted.astype(
            np.int16
        )

    def speak(
        self,
        text: str,
    ) -> None:

        if not text:
            return

        text = text.strip()

        if not text:
            return

        for audio in self.voice.synthesize(text):

            pcm = np.frombuffer(
                audio.audio_int16_bytes,
                dtype=np.int16,
            )

            pcm = self._resample(
                pcm,
                audio.sample_rate,
            )

            self.audio_runtime.enqueue_tts(
                pcm
            )

        # ----------------------------------------------------
        # Wait until the actual speaker has consumed all
        # TTS audio.
        # ----------------------------------------------------

        self.audio_runtime.wait_for_tts()