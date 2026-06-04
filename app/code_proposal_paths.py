from __future__ import annotations

from pathlib import Path, PurePosixPath

from app.code_proposal_file_structure import path_has_valid_project_structure


WEB_UI_PATHS = (
    "app/web.py",
    "app/web_routes.py",
    "app/web_pages.py",
    "app/web_components.py",
    "app/static/lewis.css",
    "app/static/chat.js",
)


def align_relevant_files_with_project_structure(
    files: list[str],
    *,
    goal: str,
    approach: str,
    target_surface: str,
    project_root: Path,
) -> list[str]:
    """Keep proposed paths in Lewis's real package layout."""
    root = Path(project_root)
    aligned = _without_missing_src_paths(files, root)
    aligned = _prefer_app_package_for_new_modules(aligned, goal, approach, root)

    if _mentions_config(goal, approach, *aligned):
        aligned = _prefer_config_path(aligned, root)

    if target_surface == "command center web UI":
        aligned = _prefer_web_ui_paths(aligned, root)

    return _dedupe(aligned)


def _without_missing_src_paths(files: list[str], project_root: Path) -> list[str]:
    if _exists(project_root, "src"):
        return files

    aligned: list[str] = []
    for path in files:
        parts = PurePosixPath(path).parts
        if parts and parts[0] == "src":
            replacement = _app_path_for_src_path(path, project_root)
            if replacement is not None:
                aligned.append(replacement)
            continue
        aligned.append(path)
    return aligned


def _prefer_app_package_for_new_modules(
    files: list[str], goal: str, approach: str, project_root: Path
) -> list[str]:
    if not _exists(project_root, "app") or not _mentions_new_python_module(
        goal, approach
    ):
        return files

    aligned: list[str] = []
    for path in files:
        posix_path = PurePosixPath(path)
        if (
            len(posix_path.parts) == 1
            and posix_path.suffix == ".py"
            and not _exists(project_root, path)
        ):
            aligned.append(PurePosixPath("app", posix_path.name).as_posix())
        else:
            aligned.append(path)
    return aligned


def _app_path_for_src_path(path: str, project_root: Path) -> str | None:
    if not _exists(project_root, "app"):
        return None
    parts = PurePosixPath(path).parts
    if len(parts) < 2:
        return None
    app_path = PurePosixPath("app", *parts[1:]).as_posix()
    if _reasonable_app_file(app_path):
        return app_path
    return None


def _prefer_config_path(files: list[str], project_root: Path) -> list[str]:
    preferred = _preferred_config_path(project_root)
    if preferred is None:
        return files

    non_config = [path for path in files if not _is_config_path(path)]
    return _dedupe([*non_config, preferred])


def _preferred_config_path(project_root: Path) -> str | None:
    candidates = (
        "app/config.py",
        "config.py",
    )
    for path in candidates:
        if _exists(project_root, path):
            return path
    if _exists(project_root, "app"):
        return "app/config.py"
    return None


def _prefer_web_ui_paths(files: list[str], project_root: Path) -> list[str]:
    web_files = [
        path
        for path in files
        if _is_web_ui_path(path)
        and path_has_valid_project_structure(path, project_root)
    ]
    if web_files:
        return web_files
    existing = [path for path in WEB_UI_PATHS if _exists(project_root, path)]
    return existing or files


def _is_config_path(path: str) -> bool:
    normalized = path.lower()
    return normalized.endswith("/config.py") or normalized == "config.py"


def _is_web_ui_path(path: str) -> bool:
    normalized = path.lower()
    return (
        normalized.startswith("app/web")
        or normalized.startswith("app/static/")
        or normalized in WEB_UI_PATHS
    )


def _reasonable_app_file(path: str) -> bool:
    suffix = PurePosixPath(path).suffix.lower()
    return path.startswith("app/") and suffix in {".py", ".css", ".js", ".html"}


def _mentions_config(*values: object) -> bool:
    text = " ".join(str(value).lower() for value in values if value)
    return any(
        marker in text
        for marker in (
            "config",
            "configuration",
            "environment variable",
            "env var",
            ".env.example",
        )
    )


def _mentions_new_python_module(*values: object) -> bool:
    text = " ".join(str(value).lower() for value in values if value)
    return "new" in text and any(
        marker in text
        for marker in (
            "module",
            "python file",
            "helper",
            "service",
        )
    )


def _exists(project_root: Path, path: str) -> bool:
    return (project_root / PurePosixPath(path)).exists()


def _dedupe(files: list[str]) -> list[str]:
    deduped: list[str] = []
    for path in files:
        if path not in deduped:
            deduped.append(path)
    return deduped
