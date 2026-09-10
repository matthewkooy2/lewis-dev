"""Repository-level pytest configuration.

Default the pytest base temporary directory to a repository-local path.

The suite creates many per-test fixture repositories through ``tmp_path``.
On Windows the default user temp root (``%TEMP%/pytest-of-<user>``) can be
left unreadable by interrupted or elevated runs, which makes every
``tmp_path`` test error at setup for reasons unrelated to Lewis. A
repository-local base temp under the ignored ``runtime/`` directory keeps the
plain repo-root ``pytest`` invocation reliable; an explicit ``--basetemp``
still wins.
"""

from __future__ import annotations

from pathlib import Path


def pytest_configure(config) -> None:
    if getattr(config.option, "basetemp", None):
        return
    basetemp = Path(config.rootpath) / "runtime" / "pytest-temp" / "current"
    # pytest clears and recreates the basetemp leaf itself but does not create
    # missing parent directories.
    basetemp.parent.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = basetemp
