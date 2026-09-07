import queue
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from Agent.local_agent import (
    LocalAgent,
    CancellationToken,
    TaskCancelled,
)

from interruption_classifier import InterruptionClassifier

from Agent.tools.speech_to_text.speech_to_text import SpeechRecognizer
from Agent.tools.text_to_speech.speech import speak_async, stop_speaker


console = Console()


# ============================================================================
# DEPENDENCIES
# ============================================================================

@dataclass
class AppDependencies:
    agent: LocalAgent
    classifier: InterruptionClassifier
    stt: SpeechRecognizer


def create_dependencies() -> AppDependencies:
    return AppDependencies(
        agent=LocalAgent(),
        classifier=InterruptionClassifier(),
        stt=SpeechRecognizer(),
    )


# ============================================================================
# QUEUES
# ============================================================================

class TaskQueue:
    """
    Thread-safe FIFO task queue.

    Queue operations are bounded by a timeout so callers cannot become
    permanently blocked during application shutdown.
    """

    GET_TIMEOUT = 0.25

    def __init__(self, name: str):
        self.name = name
        self._queue: queue.Queue[str] = queue.Queue()

    def add(self, task: str) -> None:
        task = self._clean(task)

        if not task:
            return

        self._queue.put(task)

    def add_many(self, tasks: list[str]) -> None:
        for task in tasks:
            self.add(task)

    def get(self, stop_event: Optional[threading.Event] = None) -> Optional[str]:
        """
        Wait for a task while periodically checking stop_event.

        Returns None if stop_event is set before a task arrives.
        """
        while True:
            if stop_event is not None and stop_event.is_set():
                return None

            try:
                return self._queue.get(timeout=self.GET_TIMEOUT)
            except queue.Empty:
                continue

    def get_nowait(self) -> Optional[str]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def drain(self) -> list[str]:
        """
        Atomically-ish drains currently available items.

        This should only be used when the caller explicitly intends to move
        tasks between queues. It should NOT be used merely for display.
        """
        tasks: list[str] = []

        while True:
            task = self.get_nowait()

            if task is None:
                break

            tasks.append(task)

        return tasks

    def empty(self) -> bool:
        return self._queue.empty()

    def pending_count(self) -> int:
        return self._queue.qsize()

    @staticmethod
    def _clean(task: str) -> str:
        return task.strip()


# ============================================================================
# SPEECH LISTENER
# ============================================================================

class ContinuousSpeechListener:
    """
    Continuously captures speech in a background thread.

    Responsibilities:
        microphone -> STT -> speech queue

    It does NOT:
        - classify speech
        - execute tasks
        - decide priority
        - cancel tasks

    Important:
        SpeechRecognizer is owned by the application and must not be used
        concurrently by this listener and WakeWordController.
    """

    THREAD_NAME = "jarvis-stt-listener"

    def __init__(self, stt: SpeechRecognizer):
        self.stt = stt
        self.tasks = TaskQueue("speech")

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()

    # ---------------------------------------------------------------------

    def start(self) -> None:
        with self._state_lock:
            if self.is_running:
                return

            self._stop_event.clear()

            thread = threading.Thread(
                target=self._listen_loop,
                name=self.THREAD_NAME,
                daemon=True,
            )

            self._thread = thread
            thread.start()

        console.print(
            "[green]Continuous speech listener started.[/green]"
        )

    # ---------------------------------------------------------------------

    def stop(self, timeout: float = 2.0) -> None:
        with self._state_lock:
            thread = self._thread

            if thread is None:
                return

            self._stop_event.set()

        if thread is not threading.current_thread():
            thread.join(timeout=timeout)

        if thread.is_alive():
            console.print(
                "[yellow]"
                "Speech listener did not stop within the timeout. "
                "STT may still be blocking."
                "[/yellow]"
            )
            return

        with self._state_lock:
            if self._thread is thread:
                self._thread = None

        console.print(
            "[yellow]Continuous speech listener stopped.[/yellow]"
        )

    # ---------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        thread = self._thread

        return (
            thread is not None
            and thread.is_alive()
        )

    # ---------------------------------------------------------------------

    def get(
        self,
        stop_event: Optional[threading.Event] = None,
    ) -> Optional[str]:
        return self.tasks.get(stop_event=stop_event)

    def get_nowait(self) -> Optional[str]:
        return self.tasks.get_nowait()

    def drain(self) -> list[str]:
        return self.tasks.drain()

    def pending_count(self) -> int:
        return self.tasks.pending_count()

    # ---------------------------------------------------------------------

    def _listen_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                speech = self.stt.listen()

                if self._stop_event.is_set():
                    return

                if not speech:
                    continue

                speech = speech.strip()

                if not speech:
                    continue

                self._handle_speech(speech)

            except Exception as error:
                if self._stop_event.is_set():
                    return

                self._show_error(
                    title="STT Listener Error",
                    error=error,
                )

                # Prevent a tight exception loop if the STT implementation
                # repeatedly fails immediately.
                time.sleep(0.1)

    # ---------------------------------------------------------------------

    def _handle_speech(self, speech: str) -> None:
        console.print(
            Panel(
                speech,
                title="Speech captured",
                border_style="cyan",
            )
        )

        self.tasks.add(speech)

    # ---------------------------------------------------------------------

    @staticmethod
    def _show_error(title: str, error: Exception) -> None:
        console.print(
            Panel(
                f"{type(error).__name__}: {error}",
                title=title,
            )
        )


# ============================================================================
# INTERRUPTION TYPES
# ============================================================================

class InterruptionType(Enum):
    IGNORE = auto()
    QUEUE = auto()
    CANCEL_AND_RUN = auto()
    MERGE = auto()


@dataclass
class Interruption:
    type: InterruptionType
    task: str
    reason: str = ""


# ============================================================================
# INTERRUPTION CLASSIFIER
# ============================================================================

class InterruptionPolicy:
    """
    Converts raw classifier output into a strongly typed interruption.
    """

    DEFAULT_TYPE = InterruptionType.QUEUE

    def __init__(self, classifier: InterruptionClassifier):
        self.classifier = classifier

    def classify(
        self,
        current_task: str,
        new_task: str,
    ) -> Interruption:
        result = self.classifier.classify(
            current_task=current_task,
            new_task=new_task,
        )

        if not isinstance(result, dict):
            raise ValueError(
                "Interruption classifier must return a dictionary."
            )

        classification = result.get(
            "classification",
            self.DEFAULT_TYPE.name,
        )

        if not isinstance(classification, str):
            classification = self.DEFAULT_TYPE.name

        classification = classification.strip().upper()

        task = result.get("task") or new_task

        if not isinstance(task, str):
            task = new_task

        task = task.strip()

        if not task:
            task = new_task.strip()

        reason = result.get("reason", "")

        if not isinstance(reason, str):
            reason = str(reason)

        interruption_type = self._parse_type(
            classification
        )

        return Interruption(
            type=interruption_type,
            task=task,
            reason=reason.strip(),
        )

    # ---------------------------------------------------------------------

    @classmethod
    def _parse_type(
        cls,
        value: str,
    ) -> InterruptionType:
        try:
            return InterruptionType[value]
        except KeyError:
            console.print(
                "[yellow]"
                f"Unknown interruption type {value!r}; "
                f"defaulting to {cls.DEFAULT_TYPE.name}."
                "[/yellow]"
            )

            return cls.DEFAULT_TYPE


# ============================================================================
# INTERRUPTION MONITOR
# ============================================================================

class InterruptionMonitor:
    """
    Watches speech while a task is executing.

    Every execution gets a unique generation ID. A monitor belonging to an
    older generation can never apply a classification to a newer task.
    """

    THREAD_NAME = "jarvis-interruption-monitor"
    POLL_INTERVAL = 0.05
    JOIN_TIMEOUT = 2.0

    def __init__(
        self,
        speech_listener: ContinuousSpeechListener,
        policy: InterruptionPolicy,
        pending_tasks: TaskQueue,
        urgent_tasks: TaskQueue,
    ):
        self.speech_listener = speech_listener
        self.policy = policy

        self.pending_tasks = pending_tasks
        self.urgent_tasks = urgent_tasks

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._current_task = ""
        self._cancellation_token: Optional[CancellationToken] = None

        self._lock = threading.Lock()
        self._generation = 0

    # ---------------------------------------------------------------------

    def start(
        self,
        current_task: str,
        cancellation_token: CancellationToken,
    ) -> None:
        self.stop()

        with self._lock:
            self._generation += 1
            generation = self._generation

            self._current_task = current_task
            self._cancellation_token = cancellation_token

            stop_event = threading.Event()
            self._stop_event = stop_event

            thread = threading.Thread(
                target=self._monitor_loop,
                args=(generation, stop_event),
                name=self.THREAD_NAME,
                daemon=True,
            )

            self._thread = thread

        thread.start()

        console.print(
            "[green]Interruption monitor started.[/green]"
        )

    # ---------------------------------------------------------------------

    def stop(self, timeout: float = JOIN_TIMEOUT) -> None:
        with self._lock:
            thread = self._thread
            stop_event = self._stop_event

            self._generation += 1
            self._current_task = ""
            self._cancellation_token = None

            stop_event.set()

        if thread and thread.is_alive():
            if thread is not threading.current_thread():
                thread.join(timeout=timeout)

        if thread and thread.is_alive():
            console.print(
                "[yellow]"
                "Interruption monitor did not stop within the timeout."
                "[/yellow]"
            )
            return

        with self._lock:
            if self._thread is thread:
                self._thread = None

        console.print(
            "[yellow]Interruption monitor stopped.[/yellow]"
        )

    # ---------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        thread = self._thread

        return (
            thread is not None
            and thread.is_alive()
        )

    # ---------------------------------------------------------------------

    def _monitor_loop(
        self,
        generation: int,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set():
            speech = self.speech_listener.get_nowait()

            if speech is None:
                stop_event.wait(self.POLL_INTERVAL)
                continue

            context = self._get_context(
                generation=generation,
                stop_event=stop_event,
            )

            if context is None:
                continue

            current_task, cancellation_token = context

            self._process_speech(
                current_task=current_task,
                speech=speech,
                cancellation_token=cancellation_token,
                generation=generation,
                stop_event=stop_event,
            )

    # ---------------------------------------------------------------------

    def _get_context(
        self,
        generation: int,
        stop_event: threading.Event,
    ) -> Optional[tuple[str, CancellationToken]]:
        with self._lock:
            if stop_event.is_set():
                return None

            if generation != self._generation:
                return None

            if (
                not self._current_task
                or self._cancellation_token is None
            ):
                return None

            return (
                self._current_task,
                self._cancellation_token,
            )

    # ---------------------------------------------------------------------

    def _is_current_generation(
        self,
        generation: int,
        stop_event: threading.Event,
    ) -> bool:
        with self._lock:
            return (
                not stop_event.is_set()
                and generation == self._generation
            )

    # ---------------------------------------------------------------------

    def _process_speech(
        self,
        current_task: str,
        speech: str,
        cancellation_token: CancellationToken,
        generation: int,
        stop_event: threading.Event,
    ) -> None:
        try:
            interruption = self.policy.classify(
                current_task=current_task,
                new_task=speech,
            )

        except Exception as error:
            self._handle_classifier_error(
                speech=speech,
                error=error,
            )
            return

        # Classification may have taken time. Do not allow a stale monitor
        # to mutate queues or cancel a newer task.
        if not self._is_current_generation(
            generation=generation,
            stop_event=stop_event,
        ):
            return

        self._show_interruption(interruption)

        if interruption.type == InterruptionType.IGNORE:
            self._ignore(interruption)

        elif interruption.type == InterruptionType.QUEUE:
            self._queue(interruption)

        elif interruption.type == InterruptionType.CANCEL_AND_RUN:
            self._cancel_and_run(
                interruption=interruption,
                cancellation_token=cancellation_token,
            )

        elif interruption.type == InterruptionType.MERGE:
            self._merge(
                current_task=current_task,
                interruption=interruption,
                cancellation_token=cancellation_token,
            )

    # ---------------------------------------------------------------------

    @staticmethod
    def _ignore(
        interruption: Interruption,
    ) -> None:
        console.print(
            f"[dim]Ignoring interruption: "
            f"{interruption.task}[/dim]"
        )

    # ---------------------------------------------------------------------

    def _queue(
        self,
        interruption: Interruption,
    ) -> None:
        self.pending_tasks.add(
            interruption.task
        )

    # ---------------------------------------------------------------------

    def _cancel_and_run(
        self,
        interruption: Interruption,
        cancellation_token: CancellationToken,
    ) -> None:
        console.print(
            "[yellow]"
            "Cancelling current task for higher-priority request."
            "[/yellow]"
        )

        self.urgent_tasks.add(
            interruption.task
        )

        cancellation_token.cancel()

    # ---------------------------------------------------------------------

    def _merge(
        self,
        current_task: str,
        interruption: Interruption,
        cancellation_token: CancellationToken,
    ) -> None:
        merged_task = self._build_merged_task(
            current_task=current_task,
            new_task=interruption.task,
        )

        console.print(
            "[yellow]"
            "Merging interruption into current task and "
            "restarting it."
            "[/yellow]"
        )

        self.urgent_tasks.add(merged_task)
        cancellation_token.cancel()

    # ---------------------------------------------------------------------

    @staticmethod
    def _build_merged_task(
        current_task: str,
        new_task: str,
    ) -> str:
        return (
            f"{current_task}\n\n"
            f"User refinement/addition:\n"
            f"{new_task}"
        )

    # ---------------------------------------------------------------------

    def _handle_classifier_error(
        self,
        speech: str,
        error: Exception,
    ) -> None:
        console.print(
            Panel(
                (
                    f"{type(error).__name__}: {error}\n\n"
                    f"Speech: {speech}"
                ),
                title="Interruption Classifier Error",
                border_style="red",
            )
        )

        # Safe fallback.
        self.pending_tasks.add(speech)

    # ---------------------------------------------------------------------

    @staticmethod
    def _show_interruption(
        interruption: Interruption,
    ) -> None:
        console.print(
            Panel(
                (
                    f"Classification: {interruption.type.name}\n"
                    f"Task: {interruption.task}\n"
                    f"Reason: {interruption.reason}"
                ),
                title="Interruption",
                border_style="magenta",
            )
        )


# ============================================================================
# WAKE WORD
# ============================================================================

class WakeWordController:
    """
    Handles synchronous wake-word detection.

    The returned value is the command portion of the wake-word utterance.

    Examples:

        "Jarvis"
            -> None

        "Jarvis, what's the weather?"
            -> "what's the weather?"
    """

    def __init__(
        self,
        stt: SpeechRecognizer,
        wakeword: str,
    ):
        self.stt = stt
        self.wakeword = wakeword.strip()

        if not self.wakeword:
            raise ValueError("Wake word cannot be empty.")

    # ---------------------------------------------------------------------

    def wait(self) -> Optional[str]:
        while True:
            try:
                speech = self._listen()

            except KeyboardInterrupt:
                console.print("\nStopped by user.")
                raise

            except Exception as error:
                self._show_error(error)
                continue

            command = self._extract_command(speech)

            if command is not None:
                if not command:
                    speak_async(
                        "Yes, how can I help you?"
                    )

                return command

    # ---------------------------------------------------------------------

    def _listen(self) -> str:
        speech = self.stt.listen()

        if not speech:
            raise ValueError("No speech detected.")

        console.print(
            Panel(
                speech,
                title="Wake Word",
            )
        )

        return speech.strip()

    # ---------------------------------------------------------------------

    def _extract_command(
        self,
        speech: str,
    ) -> Optional[str]:
        pattern = rf"\b{re.escape(self.wakeword)}\b"

        match = re.search(
            pattern,
            speech,
            flags=re.IGNORECASE,
        )

        if not match:
            return None

        command = speech[match.end():].strip()

        # Remove common punctuation after wake word.
        command = command.lstrip(
            " ,.!?:;-"
        ).strip()

        if not command:
            return ""

        return command

    # ---------------------------------------------------------------------

    @staticmethod
    def _show_error(error: Exception) -> None:
        Panel(
            f"{type(error).__name__}: {error}",
            title="Wake Word Error",
        )

        console.print(
            Panel(
                f"{type(error).__name__}: {error}",
                title="Wake Word Error",
            )
        )


# ============================================================================
# JARVIS EXECUTOR
# ============================================================================

class ExecutionResult(Enum):
    COMPLETED = auto()
    CANCELLED = auto()
    FAILED = auto()


class JarvisExecutor:
    """
    Responsible only for executing one task against LocalAgent.
    """

    def __init__(self, agent: LocalAgent):
        self.agent = agent

    def execute(
        self,
        task: str,
        cancellation_token: CancellationToken,
    ) -> ExecutionResult:
        prompt = self._build_prompt(task)

        try:
            self.agent.run(
                prompt,
                cancellation_token=cancellation_token,
            )

            return ExecutionResult.COMPLETED

        except TaskCancelled:
            console.print(
                "[yellow]Current task cancelled.[/yellow]"
            )

            return ExecutionResult.CANCELLED

        except Exception as error:
            self._show_error(error)

            return ExecutionResult.FAILED

    @staticmethod
    def _build_prompt(task: str) -> str:
        return f"""
You are Jarvis, a voice-controlled personal assistant.

Answer the user's request using the available tools.

When a task requires speaking to the user, call speak_async exactly once
with the final response.

After calling speak_async, consider the task complete.
Do not call speak_async again for the same task.
Do not repeat previous tool calls unless the user explicitly asks you
to retry or provides new information.

For actions such as opening or playing a file:
1. Perform the action.
2. Verify the action if possible.
3. Call speak_async once with the result.
4. Stop processing the current task.

User task:
{task}
""".strip()

    @staticmethod
    def _show_error(error: Exception) -> None:
        console.print(
            Panel(
                (
                    "The external tools ran, but local model synthesis "
                    "failed.\n\n"
                    f"Error: {type(error).__name__}: {error}"
                ),
                title="Jarvis Error",
            )
        )


# ============================================================================
# TASK SCHEDULER
# ============================================================================

class TaskScheduler:
    """
    Decides which task should execute next.

    Priority:

        1. urgent tasks
        2. normal pending tasks
        3. newly captured speech
    """

    def __init__(
        self,
        urgent_tasks: TaskQueue,
        pending_tasks: TaskQueue,
        speech_listener: ContinuousSpeechListener,
        stop_event: threading.Event,
    ):
        self.urgent_tasks = urgent_tasks
        self.pending_tasks = pending_tasks
        self.speech_listener = speech_listener
        self.stop_event = stop_event

    # ---------------------------------------------------------------------

    def next_task(self) -> Optional[str]:
        task = self.urgent_tasks.get_nowait()

        if task:
            return task

        task = self.pending_tasks.get_nowait()

        if task:
            return task

        return self.speech_listener.get(
            stop_event=self.stop_event
        )

    # ---------------------------------------------------------------------

    def preserve_captured_speech(self) -> None:
        captured = self.speech_listener.drain()

        if not captured:
            return

        self.pending_tasks.add_many(captured)

        console.print(
            "[yellow]"
            f"Preserved {len(captured)} captured task(s)."
            "[/yellow]"
        )


# ============================================================================
# GOODBYE HANDLER
# ============================================================================

class GoodbyeHandler:
    """
    Handles returning Jarvis to wake-word mode.

    Lifecycle ownership remains with JarvisApplication.
    """

    def __init__(
        self,
        wakeword_controller: WakeWordController,
        speech_listener: ContinuousSpeechListener,
        interruption_monitor: InterruptionMonitor,
    ):
        self.wakeword_controller = wakeword_controller
        self.speech_listener = speech_listener
        self.interruption_monitor = interruption_monitor

    def handle(self) -> Optional[str]:
        self.interruption_monitor.stop()

        speak_async("Goodbye!")

        self.speech_listener.stop()

        return self.wakeword_controller.wait()


# ============================================================================
# DISPLAY
# ============================================================================

class TaskDisplay:
    """Console presentation only."""

    @staticmethod
    def executing(task: str) -> None:
        console.print(
            Panel(
                task,
                title="Executing task",
                border_style="green",
            )
        )

    @staticmethod
    def queue_state(
        urgent_tasks: TaskQueue,
        pending_tasks: TaskQueue,
    ) -> None:
        urgent_count = urgent_tasks.pending_count()
        pending_count = pending_tasks.pending_count()

        if urgent_count:
            console.print(
                f"[red]Urgent tasks waiting: "
                f"{urgent_count}[/red]"
            )

        if pending_count:
            console.print(
                f"[yellow]Pending tasks waiting: "
                f"{pending_count}[/yellow]"
            )


# ============================================================================
# APPLICATION
# ============================================================================

class JarvisApplication:
    """
    Top-level application coordinator.

    This class owns lifecycle and orchestration.
    """

    WAKEWORD = "Jarvis"

    def __init__(
        self,
        dependencies: AppDependencies,
    ):
        self.dependencies = dependencies

        self._shutdown_event = threading.Event()

        self.speech_listener = ContinuousSpeechListener(
            dependencies.stt
        )

        self.pending_tasks = TaskQueue("pending")
        self.urgent_tasks = TaskQueue("urgent")

        self.policy = InterruptionPolicy(
            dependencies.classifier
        )

        self.interruption_monitor = InterruptionMonitor(
            speech_listener=self.speech_listener,
            policy=self.policy,
            pending_tasks=self.pending_tasks,
            urgent_tasks=self.urgent_tasks,
        )

        self.scheduler = TaskScheduler(
            urgent_tasks=self.urgent_tasks,
            pending_tasks=self.pending_tasks,
            speech_listener=self.speech_listener,
            stop_event=self._shutdown_event,
        )

        self.executor = JarvisExecutor(
            dependencies.agent
        )

        self.wakeword = WakeWordController(
            stt=dependencies.stt,
            wakeword=self.WAKEWORD,
        )

        self.goodbye_handler = GoodbyeHandler(
            wakeword_controller=self.wakeword,
            speech_listener=self.speech_listener,
            interruption_monitor=self.interruption_monitor,
        )

    # ---------------------------------------------------------------------

    def run(self) -> None:
        speak_async("I am on...")

        try:
            command = self._run_wakeword_mode()

            if command:
                self.pending_tasks.add(command)

            # Start listening before the scheduler can wait for speech.
            self.speech_listener.start()

            self._run_task_loop()

        finally:
            self.shutdown()


    # ---------------------------------------------------------------------

    def _run_wakeword_mode(self) -> Optional[str]:
        """
        Wait for the wake word.

        The continuous listener is guaranteed to be stopped while the same
        SpeechRecognizer is used synchronously here.
        """
        self.speech_listener.stop()

        return self.wakeword.wait()

    # ---------------------------------------------------------------------

    def _run_task_loop(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                self._run_next_task()

            except KeyboardInterrupt:
                console.print("\nStopped by user.")
                return

            except Exception as error:
                self._handle_task_error(error)

    # ---------------------------------------------------------------------

    def _run_next_task(self) -> None:
        task = self.scheduler.next_task()

        if task is None:
            return

        task = task.strip()

        if not task:
            return

        TaskDisplay.executing(task)

        if is_goodbye(task):
            command = self.goodbye_handler.handle()

            if command:
                self.pending_tasks.add(command)

            self.speech_listener.start()

            return

        # Continuous speech must be running while the task executes so
        # interruptions can be detected.
        self.speech_listener.start()

        cancellation_token = CancellationToken()

        self.interruption_monitor.start(
            current_task=task,
            cancellation_token=cancellation_token,
        )

        try:
            self.executor.execute(
                task=task,
                cancellation_token=cancellation_token,
            )

        finally:
            self.interruption_monitor.stop()

            # Preserve speech captured during execution that was not consumed
            # by the interruption monitor.
            self.scheduler.preserve_captured_speech()

        TaskDisplay.queue_state(
            urgent_tasks=self.urgent_tasks,
            pending_tasks=self.pending_tasks,
        )

    # ---------------------------------------------------------------------

    def _handle_task_error(
        self,
        error: Exception,
    ) -> None:
        console.print(
            Panel(
                f"{type(error).__name__}: {error}",
                title="Application Error",
            )
        )

        self.interruption_monitor.stop()
        self.scheduler.preserve_captured_speech()

    # ---------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._shutdown_event.is_set():
            return

        self._shutdown_event.set()

        self.interruption_monitor.stop()
        self.speech_listener.stop()

        try:
            self.dependencies.stt.close()
        except Exception as error:
            console.print(
                Panel(
                    (
                        f"{type(error).__name__}: {error}"
                    ),
                    title="STT Shutdown Error",
                )
            )

        try:
            stop_speaker()
        except Exception as error:
            console.print(
                Panel(
                    (
                        f"{type(error).__name__}: {error}"
                    ),
                    title="Speaker Shutdown Error",
                )
            )


# ============================================================================
# HELPERS
# ============================================================================

GOODBYE_PHRASES = {
    "bye",
    "goodbye",
}


def is_goodbye(task: str) -> bool:
    """
    Detect explicit goodbye commands instead of matching arbitrary
    occurrences of the substring 'bye'.
    """
    normalized = re.sub(
        r"[^\w\s]",
        " ",
        task.lower(),
    )

    normalized = " ".join(
        normalized.split()
    )

    return any(phrase in normalized for phrase in GOODBYE_PHRASES)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    dependencies = create_dependencies()

    application = JarvisApplication(
        dependencies=dependencies,
    )

    application.run()


if __name__ == "__main__":
    main()
