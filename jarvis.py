import asyncio
import functools
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List

from rich.console import Console
from rich.panel import Panel

from Agent.local_agent import LocalAgent
from Agent.cancelTask import TaskCancelled, CancellationToken
from interruption_classifier import InterruptionClassifier
from Agent.tools.speech_to_text.speech_to_text import SpeechRecognizer
from Agent.tools.text_to_speech.speech import speak_sync, stop_speaker

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
# INTERRUPTION TYPES
# ============================================================================

class InterruptionType(Enum):
    IGNORE = auto()
    QUEUE = auto()
    CANCEL_AND_RUN = auto()
    MERGE = auto()
    MODIFY_QUEUED = auto()

@dataclass
class Interruption:
    type: InterruptionType
    task: str                       # The task to run / merge / etc.
    reason: str = ""
    # For MODIFY_QUEUED:
    target_index: Optional[int] = None
    new_task_text: Optional[str] = None

# ============================================================================
# INTERRUPTION POLICY (synchronous LLM call)
# ============================================================================

class InterruptionPolicy:
    DEFAULT_TYPE = InterruptionType.QUEUE

    def __init__(self, classifier: InterruptionClassifier):
        self.classifier = classifier

    def classify(self, current_task: str, queued_tasks: List[str], new_task: str) -> Interruption:
        """
        Calls the classifier with the full queue. Expects a dict with at least:
            classification: str (one of IGNORE, QUEUE, CANCEL_AND_RUN, MERGE, MODIFY_QUEUED)
            task: str
            reason: str (optional)
            target_index: int (only for MODIFY_QUEUED)
            new_task_text: str (only for MODIFY_QUEUED)
        """
        result = self.classifier.classify(
            current_task=current_task,
            queued_tasks=queued_tasks,
            new_task=new_task,
        )
        if not isinstance(result, dict):
            raise ValueError("Classifier must return a dict")

        classification = result.get("classification", self.DEFAULT_TYPE.name).upper()
        task = result.get("task", "").strip()
        reason = result.get("reason", "")
        target_index = result.get("target_index")
        new_task_text = result.get("new_task_text")

        try:
            typ = InterruptionType[classification]
        except KeyError:
            console.print(f"[yellow]Unknown type {classification!r}, using {self.DEFAULT_TYPE.name}[/yellow]")
            typ = self.DEFAULT_TYPE

        return Interruption(
            type=typ,
            task=task,
            reason=str(reason).strip(),
            target_index=target_index if typ == InterruptionType.MODIFY_QUEUED else None,
            new_task_text=new_task_text if typ == InterruptionType.MODIFY_QUEUED else None,
        )

# ============================================================================
# GOODBYE DETECTOR
# ============================================================================

class GoodbyeDetector:
    GOODBYE_PHRASES = {"bye", "goodbye"}

    def is_goodbye(self, text: str) -> bool:
        normalized = re.sub(r"[^\w\s]", " ", text.lower())
        normalized = " ".join(normalized.split())
        return any(phrase in normalized for phrase in self.GOODBYE_PHRASES)

# ============================================================================
# WAKE WORD DETECTOR
# ============================================================================

class WakeWordDetector:
    def __init__(self, stt: SpeechRecognizer, wakeword: str = "Jarvis"):
        self.stt = stt
        self.wakeword = wakeword.strip()
        if not self.wakeword:
            raise ValueError("Wake word cannot be empty.")

    async def wait_for_wakeword(self) -> Optional[str]:
        while True:
            speech = await asyncio.to_thread(self.stt.listen)
            if not speech:
                continue
            console.print(Panel(speech, title="Wake Word", border_style="cyan"))
            command = self._extract_command(speech)
            if command is not None:
                if not command:
                    await asyncio.to_thread(speak_sync, "Yes, how can I help you?")
                return command

    def _extract_command(self, text: str) -> Optional[str]:
        match = re.search(rf"\b{re.escape(self.wakeword)}\b", text, re.IGNORECASE)
        if not match:
            return None
        command = text[match.end():].strip().lstrip(" ,.!?:;-").strip()
        return command if command else ""

# ============================================================================
# CONTINUOUS SPEECH LISTENER
# ============================================================================

class SpeechListener:
    """Background listener that pushes all captured speech into a queue."""
    def __init__(self, stt: SpeechRecognizer, speech_queue: asyncio.Queue):
        self.stt = stt
        self.speech_queue = speech_queue
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._listen_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _listen_loop(self) -> None:
        while self._running:
            try:
                speech = await asyncio.to_thread(self.stt.listen)
                if speech:
                    speech = speech.strip()
                    console.print(Panel(speech, title="Speech captured", border_style="cyan"))
                    await self.speech_queue.put(speech)
            except asyncio.CancelledError:
                break
            except Exception as e:
                console.print(f"[red]SpeechListener error: {e}[/red]")
                await asyncio.sleep(0.1)

# ============================================================================
# TASK PROCESSOR
# ============================================================================

class TaskProcessor:
    def __init__(self, agent: LocalAgent):
        self.agent = agent
        self.is_busy = False

    async def process(self, task: str, token: CancellationToken) -> None:
        self.is_busy = True
        try:
            prompt = self._build_prompt(task)
            console.print(f"[blue]Executing task: {task}[/blue]")
            await asyncio.to_thread(
                functools.partial(self.agent.run, prompt, cancellation_token=token)
            )
        except TaskCancelled:
            console.print("[yellow]Task was cancelled.[/yellow]")
        except Exception as e:
            console.print(f"[red]Agent error: {e}[/red]")
        finally:
            self.is_busy = False

    @staticmethod
    def _build_prompt(task: str) -> str:
        return f"""
You are Jarvis, a voice-controlled personal assistant.

Answer the user's request using the available tools.

When a task requires speaking to the user, call speak_sync exactly once
with the final response.

After calling speak_sync, consider the task complete.
Do not call speak_sync again for the same task.
Do not repeat previous tool calls unless the user explicitly asks you
to retry or provides new information.

For actions such as opening or playing a file:
1. Perform the action.
2. Verify the action if possible.
3. Call speak_sync once with the result.
4. Stop processing the current task.

User task:
{task}
""".strip()

# ============================================================================
# TASK MANAGER (pushes tasks to processor when idle)
# ============================================================================

class TaskManager:
    def __init__(self, processor: TaskProcessor):
        self.processor = processor
        self.urgent: asyncio.Queue = asyncio.Queue()
        self.pending: asyncio.Queue = asyncio.Queue()
        self._current_token: Optional[CancellationToken] = None
        self._current_task: Optional[str] = None
        self._processing = False

    def current_task(self) -> Optional[str]:
        return self._current_task

    def cancel_current(self) -> None:
        if self._current_token is not None:
            # Log the task that is about to be cancelled
            task_desc = (
                self._current_task
                if self._current_task is not None
                else "Unknown task"
            )

            console.print(
                f"[bold yellow]Cancelling current task: {task_desc}[/bold yellow]"
            )

            # Request cancellation from the running agent.
            self._current_token.cancel()

            # IMPORTANT:
            # Immediately remove the task from the logical "current" state.
            #
            # CancellationToken.cancel() only requests cancellation.
            # The actual processor may take some time to notice it.
            # Without clearing this here, a new interruption can still
            # incorrectly think the OLD task is running.
            self._current_task = None

        else:
            console.print("[dim]No current task to cancel.[/dim]")

    # ------------------------------------------------------------------------
    # Public methods for adding tasks (called by TaskHandler or InterruptionHandler)
    # ------------------------------------------------------------------------

    async def add_task(self, task: str, urgent: bool = False) -> None:
        if not task.strip():
            console.print(f"[yellow]Task is empty, so not adding it in queue.")
            return
        else:
            console.print(f"[green]Task is not empty: {task}")
        if urgent:
            await self.urgent.put(task)
        else:
            await self.pending.put(task)
        await self._try_process()

    async def queue_interruption(self, task: str) -> None:
        await self.add_task(task, urgent=False)

    async def cancel_and_run(self, task: str) -> None:
        # Cancel the currently running task.
        #
        # cancel_current() now immediately clears _current_task, so
        # subsequent interruptions will correctly see the replacement
        # task once it becomes current.
        self.cancel_current()

        # Put the replacement at the front of the execution priority.
        await self.add_task(task, urgent=True)

    async def merge(self, task: str, current: str) -> None:
        # A MERGE is treated as a refinement/replacement of the current task.
        #
        # Example:
        #
        # Current:
        #   Find weather in Delhi
        #
        # New:
        #   Actually, find the weather in Mumbai
        #
        # We don't concatenate both instructions because that would leave
        # the agent responsible for figuring out which request supersedes
        # the other. Instead, the classifier has already interpreted the
        # new request, so the new task becomes the current task.
        self.cancel_current()

        await self.add_task(task, urgent=True)

    async def get_queued_tasks(self) -> List[str]:
        """
        Returns a copy of all queued tasks (urgent first, then pending)
        without modifying the queue.
        """
        urgent_items = []
        while not self.urgent.empty():
            urgent_items.append(self.urgent.get_nowait())
        pending_items = []
        while not self.pending.empty():
            pending_items.append(self.pending.get_nowait())

        # Refill immediately
        for item in urgent_items:
            await self.urgent.put(item)
        for item in pending_items:
            await self.pending.put(item)

        return urgent_items + pending_items

    async def replace_task_at_index(self, index: int, new_task: str) -> bool:
        """
        Replaces the task at the given global index (0 = first urgent, then pending).
        Returns True if successful, False if index out of range.
        """
        # Drain both queues
        urgent_items = []
        while not self.urgent.empty():
            urgent_items.append(self.urgent.get_nowait())
        pending_items = []
        while not self.pending.empty():
            pending_items.append(self.pending.get_nowait())

        all_tasks = urgent_items + pending_items
        if index < 0 or index >= len(all_tasks):
            # Refill and return False
            for item in urgent_items:
                await self.urgent.put(item)
            for item in pending_items:
                await self.pending.put(item)
            return False

        # Replace
        all_tasks[index] = new_task

        # Refill with modified list
        urgent_count = len(urgent_items)
        for item in all_tasks[:urgent_count]:
            await self.urgent.put(item)
        for item in all_tasks[urgent_count:]:
            await self.pending.put(item)

        return True

    # ------------------------------------------------------------------------
    # Internal: start next task when idle
    # ------------------------------------------------------------------------

    async def _try_process(self) -> None:
        if self._processing or self.processor.is_busy:
            return
        if self.urgent.empty() and self.pending.empty():
            return

        self._processing = True
        # Get next task (urgent first)
        if not self.urgent.empty():
            task = await self.urgent.get()
        elif not self.pending.empty():
            task = await self.pending.get()
        else:
            self._processing = False
            return

        self._current_task = task
        token = CancellationToken(console=console)
        self._current_token = token

        asyncio.create_task(self._run_and_continue(task, token))

    async def _run_and_continue(self, task: str, token: CancellationToken) -> None:
        try:
            await self.processor.process(task, token)
        finally:
            # Only clear the current task if this task is still the
            # current task.
            #
            # This protects us from accidentally clearing state belonging
            # to a replacement task if the old cancelled task finishes
            # after the replacement has started.
            if self._current_token is token:
                self._current_task = None
                self._current_token = None
                self._processing = False
            await self._try_process()

    def clear_queues(self) -> None:
        while not self.urgent.empty():
            self.urgent.get_nowait()
        while not self.pending.empty():
            self.pending.get_nowait()

# ============================================================================
# INTERRUPTION HANDLER
# ============================================================================

class InterruptionHandler:
    def __init__(self, policy: InterruptionPolicy, task_manager: TaskManager):
        self.policy = policy
        self.task_manager = task_manager

    async def handle(self, speech: str) -> None:
        current = self.task_manager.current_task()
        if current is None:
            # No task is running: just queue the speech
            await self.task_manager.add_task(speech, urgent=False)
            return

        # Get a snapshot of the current queue
        queued_tasks = await self.task_manager.get_queued_tasks()

        interruption = await asyncio.to_thread(
            self.policy.classify,
            current,
            queued_tasks,
            speech
        )

        console.print(
            Panel(
                f"Classification: {interruption.type.name}\n"
                f"Task: {interruption.task}\n"
                f"Reason: {interruption.reason}",
                title="Interruption",
                border_style="magenta",
            )
        )

        if interruption.type == InterruptionType.IGNORE:
            return

        elif interruption.type == InterruptionType.QUEUE:
            await self.task_manager.queue_interruption(interruption.task)

        elif interruption.type == InterruptionType.CANCEL_AND_RUN:
            await self.task_manager.cancel_and_run(interruption.task)

        elif interruption.type == InterruptionType.MERGE:
            # MERGE means the new request supersedes/refines the
            # currently running task.
            #
            # Example:
            #   Current: Find weather in Delhi
            #   New:     Actually, find weather in Mumbai
            #
            # Result:
            #   Current: Find weather in Mumbai
            #
            # Existing queued tasks remain unchanged.
            await self.task_manager.merge(interruption.task, current)

        elif interruption.type == InterruptionType.MODIFY_QUEUED:
            target_idx = interruption.target_index
            new_text = interruption.new_task_text
            if target_idx is not None and new_text:
                replaced = await self.task_manager.replace_task_at_index(
                    target_idx,
                    new_text
                )

                if replaced:
                    console.print(
                        f"[green]"
                        f"Modified queued task at index {target_idx}: "
                        f"-> {new_text}"
                        f"[/green]"
                    )
                else:
                    console.print(
                        f"[red]"
                        f"Index {target_idx} out of range."
                        f"[/red]"
                    )

            else:
                console.print(
                    "[red]"
                    "MODIFY_QUEUED missing target_index or new_task_text."
                    "[/red]"
                )

# ============================================================================
# TASK HANDLER (main orchestrator)
# ============================================================================

class TaskHandler:
    def __init__(
        self,
        wakeword_detector: WakeWordDetector,
        goodbye_detector: GoodbyeDetector,
        task_manager: TaskManager,
        interruption_handler: InterruptionHandler,
        speech_listener: SpeechListener,
        speech_queue: asyncio.Queue,
        stt: SpeechRecognizer,
    ):
        self.wakeword_detector = wakeword_detector
        self.goodbye_detector = goodbye_detector
        self.task_manager = task_manager
        self.interruption_handler = interruption_handler
        self.speech_listener = speech_listener
        self.speech_queue = speech_queue
        self.stt = stt
        self.running = True

    async def run(self) -> None:
        await asyncio.to_thread(speak_sync, "I am on...")
        console.print("[green]Jarvis is ready.[/green]")

        while self.running:
            # --- 1. Wait for wake word ---
            command = await self.wakeword_detector.wait_for_wakeword()
            if command is not None:
                await self.task_manager.add_task(command, urgent=False)

            # --- 2. Start continuous speech listener ---
            await self.speech_listener.start()

            # --- 3. Process incoming speech until goodbye ---
            while self.running:
                speech = await self.speech_queue.get()  # blocks until speech arrives

                # Check for goodbye
                if self.goodbye_detector.is_goodbye(speech):
                    console.print("[yellow]Goodbye detected – returning to wake‑word mode.[/yellow]")
                    await self.speech_listener.stop()
                    self.task_manager.cancel_current()
                    self.task_manager.clear_queues()
                    await asyncio.to_thread(speak_sync, "Goodbye!")
                    break  # exit inner loop

                # Interruption handling – we pass the whole task_manager to the handler
                await self.interruption_handler.handle(speech)

            # After goodbye, the listener is stopped and we go back to wake‑word loop

        # Shutdown
        await self.shutdown()

    async def shutdown(self) -> None:
        self.running = False
        await self.speech_listener.stop()
        self.task_manager.cancel_current()
        self.task_manager.clear_queues()
        await asyncio.to_thread(self.stt.close)
        await asyncio.to_thread(stop_speaker)
        console.print("[yellow]Jarvis shut down.[/yellow]")

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================


    # ============================================================================
# MOCK STT FOR TESTING
# ============================================================================

class MockSpeechRecognizer:
    """Simulates speech input for testing without a microphone."""
    def __init__(self, responses: list):
        self.responses = responses.copy()
        self.index = 0

    def listen(self) -> str:
        """Return the next simulated utterance, sleeping 5 seconds between each."""
        if self.index >= len(self.responses):
            # Return empty to stop (or loop)
            return ""
        response = self.responses[self.index]
        self.index += 1
        # Simulate the gap between utterances
        import time
        time.sleep(4.4)
        return response

    def close(self) -> None:
        pass


def main(test_mode: bool = False):
    deps = create_dependencies()

    if test_mode:
        console.print("[bold yellow]=== RUNNING IN TEST MODE ===[/bold yellow]")
        # Create a mock STT that returns:
        # 1) Wake word + first command
        # 2) Follow‑up 1
        # 3) Follow‑up 2
        # 4) Goodbye to end the session
        mock_stt = MockSpeechRecognizer([
            "Hey Jarvis, tell me about yourself in your voice",
            "Stop, be quiet"
        ])
        # Override dependencies: we use the mock instead of the real STT
        deps.stt = mock_stt   # replace with mock


    stt = deps.stt
    agent = deps.agent
    classifier = deps.classifier

    # Build components
    wakeword_detector = WakeWordDetector(stt)
    goodbye_detector = GoodbyeDetector()
    processor = TaskProcessor(agent)
    task_manager = TaskManager(processor)
    policy = InterruptionPolicy(classifier)
    interruption_handler = InterruptionHandler(policy, task_manager)

    speech_queue: asyncio.Queue = asyncio.Queue()
    speech_listener = SpeechListener(stt, speech_queue)

    task_handler = TaskHandler(
        wakeword_detector=wakeword_detector,
        goodbye_detector=goodbye_detector,
        task_manager=task_manager,
        interruption_handler=interruption_handler,
        speech_listener=speech_listener,
        speech_queue=speech_queue,
        stt=stt,
    )

    asyncio.run(task_handler.run())

if __name__ == "__main__":
    # Set test_mode=True to run the simulation, False for real microphone
    main(test_mode=True)