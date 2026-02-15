"""
Utility functions for downloading connectome data.

This module provides functions to download the connectome data required
for training models. The data is hosted on GitHub releases.
"""

import logging
import os
import sys
import zipfile
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

# GitHub release URL for connectome data
CONNECTOME_DATA_URL = (
    "https://github.com/ecorreig/train-your-fly/releases/latest/download/data.zip"
)

# Files that must exist for the connectome to be considered valid
REQUIRED_FILES = [
    "classification.csv",
    "right_visual_positions_all_neurons.csv",
]

# Approximate size of the download in bytes (for user information)
DOWNLOAD_SIZE_MB = 1300


def _is_interactive() -> bool:
    """Check if running in an interactive environment."""
    return sys.stdin.isatty()


def _prompt_user(message: str) -> bool:
    """Prompt user for yes/no confirmation."""
    if not _is_interactive():
        return False

    try:
        response = input(f"{message} [y/n]: ").strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _print_progress(downloaded: int, total_size: int) -> None:
    """Print a progress bar to stderr (interactive, overwrites the line)."""
    if total_size <= 0:
        return
    percent = (downloaded / total_size) * 100
    bar_len = 40
    filled = int(bar_len * downloaded / total_size)
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stderr.write(f"\r  [{bar}] {percent:5.1f}%")
    sys.stderr.flush()


def download_connectome(
    target_dir: str = "connectome_data",
    force: bool = False,
    interactive: bool = True,
) -> bool:
    """Download connectome data from GitHub releases.

    Args:
        target_dir: Directory where connectome data will be extracted.
        force: If True, download even if data already exists.
        interactive: If True, prompt user before downloading.

    Returns:
        True if download was successful or data already exists, False otherwise.
    """
    # Check if already exists
    if not force and connectome_exists(target_dir):
        return True

    # Prompt user if interactive
    if interactive:
        logger.info("Connectome data not found in '%s'.", target_dir)
        logger.info(
            "This data is required for training and is approximately %d MB.",
            DOWNLOAD_SIZE_MB,
        )

        if not _prompt_user("Download now?"):
            logger.info("Download cancelled. You can download manually from:")
            logger.info("  %s", CONNECTOME_DATA_URL)
            logger.info("Extract the contents to '%s'", target_dir)
            return False

    # Download
    logger.info("Downloading connectome data from GitHub...")

    zip_path = os.path.join(target_dir, "_temp_data.zip")
    os.makedirs(target_dir, exist_ok=True)

    try:
        request = Request(
            CONNECTOME_DATA_URL,
            headers={"User-Agent": "trainyourfly-downloader/1.0"},
        )

        with urlopen(request, timeout=30) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 8192

            with open(zip_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    _print_progress(downloaded, total_size)

            sys.stderr.write("\n")  # newline after progress bar

    except (URLError, HTTPError) as e:
        logger.error("Error downloading data: %s", e)
        logger.error("You can download manually from: %s", CONNECTOME_DATA_URL)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False

    # Extract
    logger.info("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                parts = member.split("/", 1)
                if len(parts) > 1 and parts[1]:
                    target_path = os.path.join(target_dir, parts[1])

                    if member.endswith("/"):
                        os.makedirs(target_path, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        with zf.open(member) as src, open(target_path, "wb") as dst:
                            dst.write(src.read())

        os.remove(zip_path)
        logger.info("Connectome data extracted to '%s'.", target_dir)
        return True

    except zipfile.BadZipFile as e:
        logger.error("Error extracting data: %s", e)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False


def connectome_exists(connectome_dir: str) -> bool:
    """Check if connectome data exists and is valid.

    Args:
        connectome_dir: Path to the connectome data directory.

    Returns:
        True if all required files exist, False otherwise.
    """
    if not os.path.isdir(connectome_dir):
        return False

    for filename in REQUIRED_FILES:
        if not os.path.isfile(os.path.join(connectome_dir, filename)):
            return False

    return True


def ensure_connectome_exists(connectome_dir: str) -> None:
    """Ensure connectome data exists, prompting to download if needed.

    Args:
        connectome_dir: Path to the connectome data directory.

    Raises:
        FileNotFoundError: If data doesn't exist and user declined to download
            or download failed.
    """
    if connectome_exists(connectome_dir):
        return

    success = download_connectome(target_dir=connectome_dir, interactive=True)

    if not success or not connectome_exists(connectome_dir):
        raise FileNotFoundError(
            f"Connectome data not found in '{connectome_dir}'.\n"
            f"Download it from: {CONNECTOME_DATA_URL}\n"
            f"Extract the contents to '{connectome_dir}'"
        )
