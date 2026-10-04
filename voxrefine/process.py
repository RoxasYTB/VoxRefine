import subprocess

from .audio import VoxRefineError


def run_checked(command: list[str]) -> str:
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=1800
        )
    except subprocess.TimeoutExpired as error:
        raise VoxRefineError("Process exceeded the 30-minute timeout.") from error
    if process.returncode:
        detail = (process.stderr or process.stdout).strip()
        raise VoxRefineError(
            f"Process exited with code {process.returncode}: {detail}"
        )
    return process.stdout.strip()
