import logging

from trainyourfly.config import Config
from trainyourfly.train import train, evaluate, TrainResult
from trainyourfly.utils.downloader import download_connectome
from trainyourfly.integrations.base import NullTracker

__all__ = ["Config", "train", "evaluate", "TrainResult", "download_connectome", "NullTracker"]


# ---------------------------------------------------------------------------
# Package-level logger setup
# ---------------------------------------------------------------------------
# All modules use ``logging.getLogger(__name__)`` so everything flows
# through the ``trainyourfly`` root logger.  We configure it once here
# with a coloured formatter so that users get readable output out of
# the box.  If the user configures their own logging before importing
# the package, this is a no-op (we only add a handler when there are
# none).
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    """Configure the ``trainyourfly`` logger with a coloured console handler."""
    root = logging.getLogger("trainyourfly")
    if root.handlers:
        return  # already configured by the user

    root.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(_ColorFormatter())
    root.addHandler(handler)

    # Don't propagate to the root logger to avoid duplicate messages
    root.propagate = False

    # Silence noisy third-party loggers whose DEBUG/INFO output leaks
    # through the root logger when libraries like wandb call basicConfig.
    for name in ("PIL", "urllib3", "matplotlib", "git", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)


class _ColorFormatter(logging.Formatter):
    """Compact coloured formatter for console output."""

    _GREY = "\x1b[38;20m"
    _GREEN = "\x1b[32;20m"
    _YELLOW = "\x1b[33;20m"
    _ORANGE = "\x1b[38;5;208m"
    _RED = "\x1b[31;20m"
    _BOLD_RED = "\x1b[31;1m"
    _RESET = "\x1b[0m"

    _BASE = "%(asctime)s | %(levelname)-8s | %(message)s"
    _DATEFMT = "%H:%M:%S"

    _FORMATS = {
        logging.DEBUG: _GREY + _BASE + _RESET,
        logging.INFO: _GREEN + _BASE + _RESET,
        logging.WARNING: _ORANGE + _BASE + _RESET,
        logging.ERROR: _RED + _BASE + _RESET,
        logging.CRITICAL: _BOLD_RED + _BASE + _RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        fmt = self._FORMATS.get(record.levelno, self._BASE)
        formatter = logging.Formatter(fmt, datefmt=self._DATEFMT)
        return formatter.format(record)


_setup_logging()
