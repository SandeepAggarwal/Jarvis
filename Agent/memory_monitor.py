import os
import subprocess

# Default memory limit (3 GB)
DEFAULT_MEM_LIMIT_BYTES = 3 * 1024 ** 3


def get_current_rss_bytes() -> int | None:
    """Return current process RSS in bytes, or None on failure.

    Tries to use psutil when available, otherwise falls back to `ps`.
    """
    try:
        import psutil

        try:
            return psutil.Process(os.getpid()).memory_info().rss
        except Exception:
            return None
    except Exception:
        try:
            out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
            if isinstance(out, bytes):
                out = out.decode("utf-8")
            rss_kb = int(out.strip().split()[0])
            return rss_kb * 1024
        except Exception:
            return None
