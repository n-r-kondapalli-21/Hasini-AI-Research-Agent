"""
Centralized logging configuration for Hasini AI Research Agent.

Separates detailed application logs (file) from interactive UI output (console).
Suppresses noisy third-party progress bars and debug output.

IMPORTANT: import this module BEFORE importing transformers / sentence_transformers /
chromadb so the environment variables below are picked up.
"""

import contextlib
import logging
import os
import sys
import warnings
from functools import partialmethod
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment (must be set before third-party imports)
# ---------------------------------------------------------------------------
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ---------------------------------------------------------------------------
# Warnings (targeted, not blanket-ignoring every UserWarning/FutureWarning)
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*IncompleteFieldDefinitionWarning.*")
warnings.filterwarnings("ignore", message=".*LOAD REPORT.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"(transformers|sentence_transformers|huggingface_hub|torch)(\..*)?")
warnings.filterwarnings("ignore", category=UserWarning, module=r"(transformers|sentence_transformers|huggingface_hub|torch|pydantic_settings)(\..*)?")

# ---------------------------------------------------------------------------
# Disable tqdm properly.
# Patch the real class's __init__ (instead of replacing it with a stub) so that
#  - `for x in tqdm(iterable)` still iterates the real iterable
#  - modules that already did `from tqdm import tqdm` are affected too
#  - subclasses (tqdm.auto, tqdm.notebook) inherit the behaviour
# ---------------------------------------------------------------------------
try:
    from tqdm.std import tqdm as _tqdm_std

    _tqdm_std.__init__ = partialmethod(_tqdm_std.__init__, disable=True)
except ImportError:
    pass

try:
    from huggingface_hub.utils import disable_progress_bars as _hf_disable_bars

    _hf_disable_bars()
except Exception:  # huggingface_hub not installed or API changed
    pass

# ---------------------------------------------------------------------------
# stderr suppression helpers
# ---------------------------------------------------------------------------
_original_stderr = sys.stderr
_devnull = None


def suppress_progress_bars() -> None:
    """Redirect stderr to /dev/null (does not accumulate memory like StringIO)."""
    global _devnull
    if _devnull is None or _devnull.closed:
        _devnull = open(os.devnull, "w", encoding="utf-8")
    sys.stderr = _devnull


def restore_stderr() -> None:
    """Restore the original stderr."""
    sys.stderr = _original_stderr


@contextlib.contextmanager
def quiet_stderr():
    """Context manager: silence stderr inside the block, always restore after."""
    suppress_progress_bars()
    try:
        yield
    finally:
        restore_stderr()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_NOISY_CONSOLE_PATTERNS = (
    "Processing request of type",
    "Secure MCP Filesystem Server",
    "Client does not support MCP Roots",
    "IncompleteFieldDefinitionWarning",
    "Loading weights:",
    "LOAD REPORT",
)

# Third-party loggers: capped at WARNING everywhere (console AND file).
_NOISY_THIRD_PARTY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "openai",
    "openai._base_client",
    "chromadb",
    "huggingface_hub",
    "datasets",  # "PyTorch version ... available" DEBUG line
    "mcp",
)

# Own loggers whose detail is redundant in the FILE log (applies to all handlers).
# Examples from real logs:
#  - mcps.mcp_registry logs one DEBUG line per tool (63 lines) and then an INFO summary.
#  - bm25_index / fusion / reranker repeat what hasini.rag.retriever already logs per stage.
# Set a name to logging.DEBUG here when you actually need to debug that component.
_LEVEL_OVERRIDES = {
    "mcps.mcp_registry": logging.INFO,
    "hasini.rag.bm25_index": logging.WARNING,
    "hasini.rag.fusion": logging.WARNING,
    "hasini.rag.reranker": logging.WARNING,
}


class DuplicateFilter(logging.Filter):
    """Drops a record identical (logger, level, message) to the one right before it."""

    def __init__(self) -> None:
        super().__init__()
        self._last = None

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            key = (record.name, record.levelno, record.getMessage())
        except Exception:
            return True
        if key == self._last:
            return False
        self._last = key
        return True

# Loggers silenced completely.
_SILENCED_LOGGERS = {
    "pydantic_settings": logging.ERROR,
    "sentence_transformers": logging.CRITICAL,
    "transformers": logging.CRITICAL,
}

# Own internal loggers: hidden from the CONSOLE only, still fully logged to file.
_QUIET_ON_CONSOLE_PREFIXES = (
    "hasini.agent_runtime",
    "hasini.rag.embedding",
    "hasini.rag.vector_store",
    "hasini.rag.bm25_index",
    "hasini.rag.reranker",
    "hasini.rag.fusion",
)


class ConsoleFilter(logging.Filter):
    """Drops noisy messages and internal chatter from console output only."""

    def filter(self, record: logging.LogRecord) -> bool:
        if "pydantic_settings" in record.name:
            return False
        if record.levelno < logging.ERROR and record.name.startswith(_QUIET_ON_CONSOLE_PREFIXES):
            return False
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(p in msg for p in _NOISY_CONSOLE_PATTERNS)


def configure_logging(
    log_file: str = "logs/hasini.log",
    console_level: int = logging.WARNING,
    file_level: int = logging.DEBUG,
    clear_previous: bool = True,
) -> None:
    """
    Configure application-wide logging with file and console handlers.

    Args:
        log_file: Path to log file (relative to this file's folder, or absolute).
        console_level: Logging level for console output (default WARNING).
        file_level: Logging level for file output (default DEBUG).
        clear_previous: If True (default), wipe the old log file and its rotated
            backups so every run starts with a clean log.
    """
    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = Path(__file__).resolve().parent / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Reset existing handlers (close them so file handles aren't leaked)
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    root_logger.setLevel(logging.DEBUG)

    # Clear previous logs (current file + rotated backups like hasini.log.1 ... .5)
    if clear_previous:
        for old in log_path.parent.glob(log_path.name + "*"):
            try:
                old.unlink()
            except OSError:
                pass  # e.g. locked by another process; "w" mode below truncates the main file

    # File handler - detailed logs
    file_handler = RotatingFileHandler(
        log_path,
        mode="w" if clear_previous else "a",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    file_handler.addFilter(DuplicateFilter())
    root_logger.addHandler(file_handler)

    # Console handler - only important messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    console_handler.addFilter(ConsoleFilter())
    root_logger.addHandler(console_handler)

    # Third-party noise
    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name, level in _LEVEL_OVERRIDES.items():
        logging.getLogger(name).setLevel(level)
    for name, level in _SILENCED_LOGGERS.items():
        logging.getLogger(name).setLevel(level)

    logging.getLogger("hasini.logging_config").info(
        "Logging configured: file=%s, console_level=%s, file_level=%s",
        log_path,
        logging.getLevelName(console_level),
        logging.getLevelName(file_level),
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)