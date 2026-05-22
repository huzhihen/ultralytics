# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Shared helpers for standalone YOLO command-line scripts.

Usage:
    Import `resolve_project` in scripts that expose a `--project` argument before calling `model.train()`,
    `model.val()`, or `model.predict()`.

Why this exists:
    Ultralytics resolves relative `project` paths under the global settings `runs_dir` by default. These standalone
    scripts are typically launched from a shell in the repository root, so users expect paths such as
    `--project runs/train-seg` to be relative to the current working directory. `resolve_project` converts relative
    project paths to absolute paths before passing them into the engine, preserving that command-line behavior.

Example:
    project = resolve_project("runs/predict-seg")
    # If the current working directory is /repo, project becomes /repo/runs/predict-seg.
"""

from pathlib import Path


def resolve_project(project: str) -> str:
    """Resolve relative project paths from the current working directory."""
    return str(Path(project).resolve()) if project and not Path(project).is_absolute() else project
