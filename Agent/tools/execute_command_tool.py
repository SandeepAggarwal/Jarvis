from typing import Optional
import subprocess
import os
import shutil
import shlex
import logging


EXECUTE_COMMAND_TOOL = {
    "name": "execute_command",
    "description": (
        "Execute an operating-system command. Supports ordinary command "
        "arguments and safe pipelines using |. Supports stderr redirection "
        "to /dev/null using 2>/dev/null. Commands are never executed through "
        "a shell."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "Executable name/path with optional arguments. "
                    "Safe pipeline syntax using | and 2>/dev/null is supported."
                ),
            },
            "args": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Arguments to pass to the executable.",
            },
            "cwd": {
                "type": ["string", "null"],
                "description": "Optional working directory.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 300,
                "description": "Maximum execution time.",
            },
        },
        "required": ["command"],
    },
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

allowlist_env = os.getenv(
    "EXECUTE_COMMAND_ALLOWLIST",
    "",
).strip()

ALLOWED_COMMANDS = None

if allowlist_env:
    ALLOWED_COMMANDS = {
        command.strip()
        for command in allowlist_env.split(",")
        if command.strip()
    }


# Commands that should never be executed.
DENIED_COMMANDS = {
    "rm",
    "shutdown",
    "reboot",
    "poweroff",
    "halt",
    "init",
    "mkfs",
    "dd",
    "passwd",
    "useradd",
    "userdel",
    "chpasswd",
    "chown",
    "chmod",
    "su",
    "sudo",
    "systemctl",
    "fdisk",
    "parted",
    "mkfs.ext4",
    "mkfs.xfs",
    "mkfs.vfat",
    "shred",
    "iptables",
    "ip6tables",
    "mount",
    "umount",
    "ddrescue",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

if not logger.handlers:
    os.makedirs("logs", exist_ok=True)

    file_handler = logging.FileHandler("logs/execute_command.log")

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def expand_argument(value: str) -> str:
    """
    Expand ~ and environment variables.

    Examples:
        ~/Desktop
        ~/Music
        $HOME/Music
        ${HOME}/Downloads

    This does not execute shell expressions.
    """

    return os.path.expandvars(os.path.expanduser(value))


def validate_executable(
    executable: str,
) -> Optional[str]:
    """
    Validate an executable against the denylist,
    PATH, and optional allowlist.
    """

    command_basename = os.path.basename(executable).lower()

    # Security denylist.
    if command_basename in DENIED_COMMANDS:
        return f"Execution blocked for security: {command_basename}"

    # Check PATH.
    if shutil.which(executable) is None:
        return f"Command not found in PATH: {executable}"

    # Optional allowlist.
    if ALLOWED_COMMANDS is not None:
        if (command_basename not in ALLOWED_COMMANDS and executable not in ALLOWED_COMMANDS):
            return f"Command not permitted: {executable}"

    return None


def tokenize_command(
    command: str,
) -> list[str]:
    """
    Tokenize a command using shlex.

    Quoted arguments are preserved as individual arguments.

    Example:

        find /home -name "*.mp3"

    becomes:

        ["find", "/home", "-name", "*.mp3"]
    """

    return shlex.split(command, posix=True)


def parse_pipeline(
    command: str,
) -> tuple[list[list[str]], bool]:
    """
    Parse a command into pipeline stages.

    Supported shell-like syntax:

        |

        2>/dev/null

    Examples:

        ls -la

        find /home -name "*.mp3" | head -20

        find / -name "*.mp3" 2>/dev/null | head -20

    Unsupported shell syntax includes:

        >
        >>
        <
        <<
        &&
        ||
        ;
        &
        $(...)
        backticks
    """

    tokens = tokenize_command(command)

    if not tokens:
        raise ValueError("No command specified")

    stages: list[list[str]] = []
    current_stage: list[str] = []

    stderr_to_null = False

    for token in tokens:

        # ---------------------------------------------------------------
        # Pipeline
        # ---------------------------------------------------------------

        if token == "|":
            if not current_stage:
                raise ValueError("Invalid pipeline: empty command before |")
            stages.append(current_stage)
            current_stage = []
            continue

        # ---------------------------------------------------------------
        # Supported stderr redirection
        # ---------------------------------------------------------------

        if token == "2>/dev/null":
            stderr_to_null = True
            continue

        # ---------------------------------------------------------------
        # Unsupported shell syntax
        # ---------------------------------------------------------------

        unsupported_tokens = {
            ">",
            ">>",
            "<",
            "<<",
            "&&",
            "||",
            ";",
            "&",
            "2>",
            "2>>",
            "1>",
            "1>>",
            "&>",
        }

        if token in unsupported_tokens:
            raise ValueError(f"Unsupported shell syntax: {token!r}")

        # Catch forms such as:
        #
        #   2>/tmp/error.log
        #   1>/tmp/output.log
        #   &>/tmp/output.log

        if (token.startswith("2>") or token.startswith("1>") or token.startswith("&>")):
            raise ValueError(f"Unsupported shell syntax: {token!r}")

        # Command substitution.
        if token.startswith("$("):
            raise ValueError(f"Unsupported shell syntax: {token!r}")

        # Backtick command substitution.
        if "`" in token:
            raise ValueError(f"Unsupported shell syntax: {token!r}")

        current_stage.append(
            token
        )

    # A trailing pipe is invalid.
    if not current_stage:
        raise ValueError("Invalid pipeline: empty command after |")

    stages.append(current_stage)
    return stages, stderr_to_null


def prepare_stage(
    stage: list[str],
) -> tuple[str, list[str]]:
    """
    Expand arguments and separate executable from arguments.
    """

    if not stage:
        raise ValueError("Empty command stage")

    expanded_stage = [expand_argument(argument) for argument in stage]

    executable = expanded_stage[0]
    arguments = expanded_stage[1:]

    return executable, arguments


# ---------------------------------------------------------------------------
# Process execution
# ---------------------------------------------------------------------------

def execute_single(
    executable: str,
    arguments: list[str],
    cwd: Optional[str],
    stdin=None,
    stdout=None,
    stderr=None,
):
    """
    Start one process.

    shell=False is intentionally hard-coded.
    """

    validation_error = validate_executable(executable)

    if validation_error:
        raise PermissionError(validation_error)

    return subprocess.Popen([
        executable,
        *arguments,
    ], cwd=cwd, stdin=stdin, stdout=stdout, stderr=stderr, text=True, shell=False)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def execute_pipeline(
    stages: list[list[str]],
    cwd: Optional[str],
    timeout_seconds: int,
    stderr_to_null: bool,
) -> dict:
    """
    Execute pipeline stages without shell=True.

    Example:

        find /home -name "*.mp3" | head -20

    is implemented using two Popen processes connected
    directly through a pipe.
    """
    processes = []

    previous_stdout = None

    try:
        for index, stage in enumerate(stages):
            executable, arguments = prepare_stage(stage)

            is_last_stage = index == len(stages) - 1

            stdout_target = subprocess.PIPE

            if stderr_to_null:
                stderr_target = subprocess.DEVNULL
            else:
                stderr_target = subprocess.PIPE

            process = execute_single(
                executable=executable,
                arguments=arguments,
                cwd=cwd,
                stdin=previous_stdout,
                stdout=stdout_target,
                stderr=stderr_target,
            )

            processes.append(process)

            if previous_stdout is not None:
                previous_stdout.close()

            previous_stdout = process.stdout

        try:
            final_stdout, final_stderr = processes[-1].communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            logger.warning("Pipeline timed out")
            for process in processes:
                if process.poll() is None:
                    try:
                        process.kill()
                    except Exception:
                        pass

            for process in processes:
                try:
                    process.wait(timeout=2)
                except Exception:
                    pass

            return {
                "success": False,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": True,
                "error": f"Command timed out after {timeout_seconds} seconds",
            }

        for process in processes[:-1]:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass

                process.wait()

        stderr_parts = []

        if final_stderr:
            stderr_parts.append(final_stderr)

        if not stderr_to_null:
            for process in processes[:-1]:
                try:
                    if process.stderr:
                        process_stderr = process.stderr.read()
                        if process_stderr:
                            stderr_parts.append(process_stderr)
                except Exception:
                    pass

        stderr_output = "".join(stderr_parts)

        final_process = processes[-1]
        exit_code = final_process.returncode

        return {
            "success": exit_code == 0,
            "exit_code": exit_code,
            "stdout": final_stdout or "",
            "stderr": stderr_output,
            "timed_out": False,
        }

    except Exception:
        for process in processes:
            if process.poll() is None:
                try:
                    process.kill()
                except Exception:
                    pass

        for process in processes:
            try:
                process.wait(timeout=2)
            except Exception:
                pass

        raise
# ---------------------------------------------------------------------------
# Public execute_command function
# ---------------------------------------------------------------------------

def execute_command(
    command: str,
    args: Optional[list[str]] = None,
    cwd: Optional[str] = None,
    timeout_seconds: int = 300,
    cancellation_token=None
) -> dict:
    """
    Execute an operating-system command.

    Examples:

        execute_command(
            "ls",
            ["-la", "~/Desktop"]
        )

    Or:

        execute_command(
            'find /home -name "*.mp3" | head -20'
        )

    Or:

        execute_command(
            'find / -name "*.mp3" 2>/dev/null | head -20'
        )

    Supported:

        |
        2>/dev/null
        ~
        $HOME
        ${HOME}

    Not supported:

        >
        >>
        <
        <<
        &&
        ||
        ;
        &
        $(...)
        `...`

    Commands are NEVER executed using shell=True.
    """

    if cancellation_token:
        cancellation_token.raise_if_cancelled()

    # ------------------------------------------------------------------
    # Validate timeout
    # ------------------------------------------------------------------

    if timeout_seconds < 1 or timeout_seconds > 300:
        raise ValueError("timeout_seconds must be between 1 and 300")

    # ------------------------------------------------------------------
    # Normalize cwd
    # ------------------------------------------------------------------

    if cwd:
        cwd = expand_argument(cwd)
        if not os.path.isdir(cwd):
            return {
                "success": False,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "error": f"Working directory does not exist: {cwd}",
            }

    # ------------------------------------------------------------------
    # Build command string
    # ------------------------------------------------------------------

    if args:
        command_parts = [command, *args]
        command_string = " ".join(shlex.quote(str(part)) for part in command_parts)
    else:
        command_string = command

    # ------------------------------------------------------------------
    # Parse pipeline
    # ------------------------------------------------------------------

    try:
        stages, stderr_to_null = parse_pipeline(command_string)
    except (ValueError, TypeError) as e:
        logger.info("command_parse_failed: %s error=%s", command_string, e)
        return {"success": False, "exit_code": None, "stdout": "", "stderr": "", "timed_out": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Normalize all arguments.
    # ------------------------------------------------------------------

    normalized_stages = []
    try:
        for stage in stages:
            normalized_stage = [expand_argument(argument) for argument in stage]
            normalized_stages.append(normalized_stage)
    except Exception as e:
        return {"success": False, "exit_code": None, "stdout": "", "stderr": "", "timed_out": False, "error": f"Failed to process command: {e}"}

    # ------------------------------------------------------------------
    # Log request.
    # ------------------------------------------------------------------

    logger.info("command_requested: %s cwd=%s timeout=%s", normalized_stages, cwd, timeout_seconds)

    # ------------------------------------------------------------------
    # Execute.
    # ------------------------------------------------------------------

    try:
        result = execute_pipeline(stages=normalized_stages, cwd=cwd, timeout_seconds=timeout_seconds, stderr_to_null=stderr_to_null)
        logger.info("command_completed: %s result=%s", normalized_stages, result)
        return result

    # ------------------------------------------------------------------
    # Security / executable errors.
    # ------------------------------------------------------------------

    except PermissionError as e:
        logger.info("command_blocked_or_permission_denied: %s error=%s", normalized_stages, e)
        return {"success": False, "exit_code": None, "stdout": "", "stderr": "", "timed_out": False, "error": str(e)}

    except FileNotFoundError as e:
        logger.info("command_not_found: %s error=%s", normalized_stages, e)
        return {"success": False, "exit_code": None, "stdout": "", "stderr": "", "timed_out": False, "error": str(e)}

    except OSError as e:
        logger.exception("command_os_error: %s", e)
        return {"success": False, "exit_code": None, "stdout": "", "stderr": "", "timed_out": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Unexpected errors.
    # ------------------------------------------------------------------

    except Exception as e:
        logger.exception("command_failed: %s", e)
        return {"success": False, "exit_code": None, "stdout": "", "stderr": "", "timed_out": False, "error": str(e)}
