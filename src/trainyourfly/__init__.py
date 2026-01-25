import sys
import os
from pathlib import Path

from trainyourfly.config import Config
from trainyourfly.utils.downloader import download_connectome

__all__ = ["Config", "download_connectome", "setup_notebook"]


def setup_notebook(use_project_root_as_cwd=False):
    """
    Setup function for notebooks to properly resolve paths.

    Args:
        use_project_root_as_cwd: If True, changes the working directory to the project root.

    Returns:
        Path object pointing to the project root.
    """
    # Find project root
    notebook_dir = Path.cwd()
    project_root = notebook_dir

    while (
        not (project_root / "setup.py").exists() and project_root != project_root.parent
    ):
        project_root = project_root.parent

    if project_root == project_root.parent:
        raise RuntimeError("Could not find project root (directory with setup.py)")

    # Change working directory if requested
    if use_project_root_as_cwd:
        os.chdir(project_root)
        print(f"Changed working directory to {project_root}")

    print(f"Project root: {project_root}")
    return project_root
