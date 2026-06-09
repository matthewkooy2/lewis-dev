from __future__ import annotations

from pathlib import Path


PROJECT_TEST_MODULES = (
    "ruff check .",
    "pytest",
)


def normalize_suggested_test_commands(
    suggested_commands: list[str], project_root: Path
) -> list[str]:
    del suggested_commands
    python_command = _project_python_command(project_root)
    return [f"{python_command} -m {module}" for module in PROJECT_TEST_MODULES]


def _project_python_command(project_root: Path) -> str:
    root = Path(project_root)
    candidates = (
        (root / ".venv" / "Scripts" / "python.exe", r".\.venv\Scripts\python.exe"),
        (root / ".venv" / "bin" / "python", ".venv/bin/python"),
    )
    for path, command in candidates:
        if path.is_file():
            return command
    return "python"
