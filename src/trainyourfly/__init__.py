import importlib
import sys
import os
from pathlib import Path


def get_config():
    """Import and return the config module."""
    try:
        # Try direct import first (if installed properly)
        return importlib.import_module("configs.config")
    except ImportError:
        # Find project root as fallback
        current_dir = Path(os.getcwd())
        project_root = current_dir
        while (
            not (project_root / "setup.py").exists()
            and project_root != project_root.parent
        ):
            project_root = project_root.parent

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Try again after modifying path
        return importlib.import_module("configs.config")


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
