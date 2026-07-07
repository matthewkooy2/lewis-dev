"""Validation for the Windows deployment scripts.

The scripts run with highest privileges from the "Lewis Always On" scheduled
task, so they get the same treatment as code: every script must parse, and
the restart-request protocol's ordering guarantees (PID file survives a failed
stop; the restart marker is cleared only after a healthy replacement) are
asserted against the script text.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "windows"
RESTART_MARKER = "lewis-restart-requested"


def _script_text(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8-sig")


@pytest.mark.parametrize(
    "name",
    sorted(path.name for path in SCRIPTS_DIR.glob("*.ps1")),
)
def test_windows_scripts_parse_cleanly(name: str) -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is not available in this environment")
    probe = (
        "$tokens = $null; $errors = $null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{SCRIPTS_DIR / name}', [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }"
    )
    completed = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", probe],
        capture_output=True,
        text=True,
        check=False,
        shell=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_request_restart_script_only_writes_the_marker() -> None:
    text = _script_text("request-lewis-restart.ps1")
    assert RESTART_MARKER in text
    assert "Set-Content" in text
    # The request script must never try to stop or kill anything itself; the
    # elevated scheduled task owns process replacement.
    assert "Stop-Process" not in text
    assert "taskkill" not in text.casefold()
    assert "Remove-Item" not in text


def test_start_script_clears_marker_only_after_health_gate() -> None:
    text = _script_text("start-lewis.ps1")
    assert RESTART_MARKER in text
    marker_removal = text.index("Remove-Item $RestartMarker")
    health_gate = text.index("if (-not $lewisReady)")
    start_failure = text.index('throw "Lewis failed to start')
    assert text.count("Remove-Item $RestartMarker") == 1
    assert marker_removal > health_gate
    assert marker_removal > start_failure


def test_start_script_keeps_pid_file_when_stop_fails() -> None:
    text = _script_text("start-lewis.ps1")
    restart_block = text[
        text.index("$RestartRequested = Test-Path") : text.index(
            "# Start LM Studio"
        )
    ]
    kept = restart_block.index("could not be stopped")
    removed = restart_block.index("Remove-Item $PidFile")
    # The unable-to-stop throw fires before the stale-PID-file cleanup can be
    # reached, so a surviving process always keeps its PID file.
    assert kept < removed
    assert "keeping the PID file" in restart_block


def test_stop_script_never_deletes_pid_file_for_survivor() -> None:
    text = _script_text("stop-lewis.ps1")
    survivor_guard = text.index("still running; keeping the PID file")
    exit_guard = text.index("exit 1")
    removal = text.index("Remove-Item $PidFile")
    assert survivor_guard < removal
    assert survivor_guard < exit_guard < removal
