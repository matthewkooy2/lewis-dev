from __future__ import annotations

from pathlib import Path


PROJECT_TEST_MODULES = (
    "ruff format .",
    "ruff check .",
    "pytest",
)


def normalize_suggested_test_commands(
    suggested_commands: list[str], project_root: Path
) -> list[str]:
    del suggested_commands, project_root
    python_command = r".\.venv\Scripts\python.exe"
    return [f"{python_command} -m {module}" for module in PROJECT_TEST_MODULES]
