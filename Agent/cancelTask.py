from rich.console import Console
import threading

class TaskCancelled(Exception):
    """
    Raised when the currently executing agent task has been cancelled.
    """
    pass

class CancellationToken:
    """
    Cooperative cancellation token for a single agent task.

    The token is thread-safe and can be shared between the task controller,
    LocalAgent, HTTP streaming, and tools that support cancellation.
    """
    def __init__(self, console: Console):
        self._event = threading.Event()
        self.console = console

    def cancel(self):
        """
        Request cancellation.

        This does not forcibly kill any Python thread. Code currently
        performing work must observe the token and exit cooperatively.
        """
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self):
        if self.is_cancelled():
            self.console.print("[yellow]Task cancellation requested. Aborting operation.[/yellow]")
            raise TaskCancelled("Task cancelled by user.")

    @property
    def event(self):
        """
        Expose the underlying Event for tools that want to use it directly.
        """
        return self._event