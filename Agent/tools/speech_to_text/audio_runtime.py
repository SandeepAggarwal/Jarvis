"""
Shared audio runtime.

ONE microphone
ONE speaker
ONE AEC
ONE RealtimeSTT instance

Both TTS and STT access this runtime through:

    get_audio_runtime()
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from pywebrtc_audio import AudioProcessor
from RealtimeSTT import AudioToTextRecorder


# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 16000
CHANNELS = 1

# 10 ms at 16 kHz.
FRAME_SIZE = 160

DTYPE = np.int16


class AudioRuntime:

    def __init__(self):
        self._closed = False

        # ====================================================
        # Speaker queue
        #
        # These are EXACTLY the samples that will be sent
        # to the speaker.
        # ====================================================

        self._speaker_queue: queue.Queue[np.ndarray] = (
            queue.Queue(maxsize=1000)
        )

        # ====================================================
        # AEC processing queue
        #
        # (microphone, speaker_reference)
        # ====================================================

        self._processing_queue: queue.Queue[
            tuple[np.ndarray, np.ndarray]
        ] = queue.Queue(maxsize=1000)

        # ====================================================
        # WebRTC AEC
        # ====================================================

        self.aec = AudioProcessor(
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            echo_cancellation=True,
            noise_suppression=True,
            auto_gain_control=False,
            stream_delay_ms=10,
        )

        print("AEC configured:")
        print(f"  sample_rate={SAMPLE_RATE}")
        print(f"  frame_size={FRAME_SIZE}")
        print("  echo_cancellation=True")
        print("  noise_suppression=True")
        print("  stream_delay_ms=10")

        # ====================================================
        # RealtimeSTT
        #
        # VERY IMPORTANT:
        #
        # RealtimeSTT must NOT open the microphone.
        #
        # Our sounddevice stream owns the microphone.
        # ====================================================

        self.recorder = AudioToTextRecorder(
            model="tiny",
            transcription_engine="faster_whisper",
            use_microphone=False,
            sample_rate=SAMPLE_RATE,
            spinner=False,
            post_speech_silence_duration=0.6,
            min_length_of_recording=0.5,
            faster_whisper_vad_filter=True,
            normalize_audio=False,
        )

        # ====================================================
        # Background AEC -> STT worker
        # ====================================================

        self._processing_thread = threading.Thread(
            target=self._processing_worker,
            daemon=True,
            name="aec-stt-worker",
        )

        self._processing_thread.start()

        # ====================================================
        # ONE DUPLEX STREAM
        #
        # Input:
        #     Mac microphone
        #
        # Output:
        #     Mac speaker
        # ====================================================

        self.stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            channels=CHANNELS,
            dtype=DTYPE,
            latency="low",
            callback=self._audio_callback,
        )

        self.stream.start()

        print("AudioRuntime started.")
        print(f"  {SAMPLE_RATE} Hz")
        print(f"  {FRAME_SIZE} samples/frame")
        print(f"  {FRAME_SIZE / SAMPLE_RATE * 1000:.1f} ms/frame")

    # ========================================================
    # TTS
    # ========================================================

    def enqueue_tts(self, audio: np.ndarray) -> None:
        """
        Queue 16 kHz mono int16 audio for playback.

        Audio is split into 10 ms frames.
        """

        if audio is None:
            return

        audio = np.asarray(
            audio,
            dtype=np.int16,
        ).reshape(-1)

        if audio.size == 0:
            return

        position = 0

        while position < len(audio):
            frame = audio[
                position:
                position + FRAME_SIZE
            ]

            # Pad final frame.
            if len(frame) < FRAME_SIZE:
                padded = np.zeros(
                    FRAME_SIZE,
                    dtype=np.int16,
                )
                padded[:len(frame)] = frame
                frame = padded

            self._speaker_queue.put(
                frame.copy()
            )
            position += FRAME_SIZE

    def wait_for_tts(self) -> None:
        self._speaker_queue.join()

    def clear_tts(self) -> None:
        """
        Clear queued TTS.

        Useful for future barge-in.
        """

        while True:
            try:
                self._speaker_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._speaker_queue.task_done()

    # ========================================================
    # sounddevice callback
    # ========================================================

    def _audio_callback(
        self,
        indata,
        outdata,
        frames,
        time_info,
        status,
    ):
        """
        Real-time callback.

        NEVER perform Whisper/Piper work here.
        """

        if self._closed:
            outdata.fill(0)
            return

        if status:
            print(f"Audio status: {status}")

        # ====================================================
        # SPEAKER
        # ====================================================

        try:
            far = self._speaker_queue.get_nowait()
        except queue.Empty:
            far = np.zeros(frames, dtype=np.int16)

        # Ensure exact callback size.

        if len(far) < frames:
            padded = np.zeros(frames, dtype=np.int16)
            padded[:len(far)] = far
            far = padded

        elif len(far) > frames:
            far = far[:frames]

        # ====================================================
        # Send exact far-end signal to speaker.
        # ====================================================

        outdata[:, 0] = far

        try:
            self._speaker_queue.task_done()
        except ValueError:
            pass

        # ====================================================
        # MICROPHONE
        # ====================================================

        near = np.asarray(indata[:, 0], dtype=np.int16).copy()

        # ====================================================
        # AEC reference
        #
        # Copy the EXACT signal sent to speaker.
        # ====================================================

        far_reference = np.asarray(far, dtype=np.int16).copy()

        # ====================================================
        # Give near + far to background AEC worker.
        #
        # NEVER block audio callback.
        # ====================================================

        try:
            self._processing_queue.put_nowait((near, far_reference))
        except queue.Full:
            # Never block CoreAudio.
            pass

    # ========================================================
    # AEC worker
    # ========================================================

    def _processing_worker(self):
        while not self._closed:
            try:
                near, far = (
                    self._processing_queue.get(
                        timeout=0.1
                    )
                )
            except queue.Empty:
                continue

            try:
                # =================================================
                # ACOUSTIC ECHO CANCELLATION
                # =================================================

                clean = self.aec.process(
                    near,
                    far,
                )

                clean = np.asarray(
                    clean,
                    dtype=np.int16,
                ).reshape(-1)

                # =================================================
                # DEBUG
                #
                # Uncomment this while diagnosing AEC.
                # =================================================

                # print(
                #     "AEC:",
                #     "near=",
                #     int(np.max(np.abs(near))),
                #     "far=",
                #     int(np.max(np.abs(far))),
                #     "clean=",
                #     int(np.max(np.abs(clean))),
                # )

                # =================================================
                # Feed CLEAN audio to RealtimeSTT.
                # =================================================

                self.recorder.feed_audio(
                    clean.tobytes(),
                    original_sample_rate=SAMPLE_RATE,
                )

            except Exception as e:
                print("AEC processing error:",repr(e))

            finally:
                self._processing_queue.task_done()

    # ========================================================
    # STT
    # ========================================================

    def listen(self) -> str:
        print("Listening...")

        try:
            text = self.recorder.text()

        except ValueError as e:
            if "No speech detected" in str(e):
                print("No speech detected.")
                return ""
            raise

        if not text:
            return ""

        return text.strip()

    # ========================================================
    # Diagnostics
    # ========================================================

    def status(self) -> dict:
        return {
            "sample_rate": SAMPLE_RATE,
            "frame_size": FRAME_SIZE,
            "echo_cancellation": True,
            "noise_suppression": True,
            "stream_delay_ms": (
                self.aec.stream_delay_ms
            ),
            "speech_probability": (
                self.aec.speech_probability
            ),
        }

    def reset(self) -> None:
        """
        Reset the current STT/AEC session safely.

        The underlying RealtimeSTT recorder does not expose a reliable
        "clear internal state" API; calling abort()/flush helpers while the
        transcription loop is active can deadlock or hang. The safer approach is
        to clear our local queues and rebuild the recorder instance with the same
        configuration.
        """

        print("Resetting STT/AEC session...")

        # ---------------------------------------------------------
        # 1. Stop new audio from piling up in the AEC queue.
        # ---------------------------------------------------------
        while True:
            try:
                self._processing_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._processing_queue.task_done()

        # ---------------------------------------------------------
        # 2. Reset WebRTC AEC adaptive filter.
        # ---------------------------------------------------------
        try:
            self.aec.reset()
        except Exception as e:
            print(f"AEC reset warning: {e}")

        # ---------------------------------------------------------
        # 3. Recreate the STT recorder so its internal state is fresh.
        #
        # This is safer than calling abort()/flush methods on a live recorder.
        # ---------------------------------------------------------
        try:
            if getattr(self, "recorder", None) is not None:
                try:
                    self.recorder.shutdown()
                except Exception as e:
                    print(f"STT shutdown warning: {e}")

            self.recorder = AudioToTextRecorder(
                model="tiny",
                transcription_engine="faster_whisper",
                use_microphone=False,
                sample_rate=SAMPLE_RATE,
                spinner=False,
                post_speech_silence_duration=0.6,
                min_length_of_recording=0.5,
                faster_whisper_vad_filter=True,
                normalize_audio=False,
            )

        except Exception as e:
            print(f"STT recorder recreation warning: {e}")

        print("STT/AEC session reset complete.")


    # ========================================================
    # Shutdown
    # ========================================================

    def close(self):
        if self._closed:
            return

        print("Stopping AudioRuntime...")
        self._closed = True

        # Stop sounddevice.
        try:
            self.stream.stop()
            self.stream.close()
        except Exception as e:
            print("Audio stream shutdown error:", repr(e))

        # Stop worker.
        try:
            self._processing_thread.join(
                timeout=2.0
            )
        except Exception:
            pass

        # Stop STT.
        # RealtimeSTT shutdown is fragile when the shared runtime is still
        # processing queued TTS/STT callbacks. If shutdown is already in-flight or
        # the backend is tearing down, do not let it abort the process.
        try:
            if getattr(self, "recorder", None) is not None:
                self.recorder.shutdown()
        except BaseException as e:
            print("STT shutdown error:", repr(e))

        # Reset AEC.
        try:
            self.aec.reset()
        except Exception:
            pass

        print("AudioRuntime stopped.")


# ============================================================
# GLOBAL SINGLETON
# ============================================================

_runtime: Optional[AudioRuntime] = None
_runtime_lock = threading.RLock()

def get_audio_runtime() -> AudioRuntime:
    """
    Get the ONE AudioRuntime used by the entire application.

    Safe to call from any module.
    """
    global _runtime

    if _runtime is not None:
        return _runtime

    with _runtime_lock:
        if _runtime is None:
            print("Creating global AudioRuntime...")
            _runtime = AudioRuntime()
        return _runtime


def shutdown_audio_runtime() -> None:
    """
    Shutdown the global runtime.
    """
    global _runtime

    with _runtime_lock:
        if _runtime is None:
            return

        try:
            _runtime.close()
        finally:
            _runtime = None


__all__ = [
    "AudioRuntime",
    "get_audio_runtime",
    "shutdown_audio_runtime",
    "reset"
]