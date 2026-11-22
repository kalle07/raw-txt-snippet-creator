"""Shared logging configuration for the text-search application.

Available levels are ``error``, ``warning`` and ``info``. The shipped default
is ``error``; change ``LOG_LEVEL`` below to increase diagnostic detail.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path

LOG_LEVEL = "error"
LOG_LEVEL_CHOICES = ("error", "warning", "info")
LOG_FILE = Path("search_debug.log")
_LOG_LEVELS = {
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
}
_CONFIG_LOCK = threading.RLock()
_LOGGER_NAME = "text_search"
_HANDLER_MARKER = "_text_search_shared_handler"


def configure_logging(level: str | None = None) -> logging.Logger:
    """Configure one shared file handler and return the application logger.

    Configuration is safe to call from both ``gui.py`` and ``engine.py``. The
    handler is installed once, preventing duplicate entries and leaked handles.
    """
    selected = LOG_LEVEL if level is None else str(level)
    normalized = selected.strip().casefold()
    if normalized not in _LOG_LEVELS:
        choices = ", ".join(LOG_LEVEL_CHOICES)
        raise ValueError(f"Logging level must be one of: {choices}.")

    numeric_level = _LOG_LEVELS[normalized]
    app_logger = logging.getLogger(_LOGGER_NAME)
    with _CONFIG_LOCK:
        app_logger.setLevel(numeric_level)
        app_logger.propagate = False

        handler = next(
            (item for item in app_logger.handlers if getattr(item, _HANDLER_MARKER, False)),
            None,
        )
        if handler is None:
            candidates = (
                Path(LOG_FILE),
                Path.home() / "search_debug.log",
                Path(tempfile.gettempdir()) / "text_search_debug.log",
            )
            seen_paths: set[str] = set()
            handler = None
            failed_paths: list[tuple[Path, OSError]] = []
            for candidate in candidates:
                try:
                    candidate_key = str(candidate.absolute()).casefold()
                    if candidate_key in seen_paths:
                        continue
                    seen_paths.add(candidate_key)
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    handler = logging.FileHandler(candidate, encoding="utf-8")
                    break
                except OSError as exc:
                    failed_paths.append((candidate, exc))

            if handler is None:
                # Logging must not prevent the UI/database from starting.
                handler = logging.StreamHandler()

            setattr(handler, _HANDLER_MARKER, True)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
                )
            )
            app_logger.addHandler(handler)
            if failed_paths:
                first_path, first_error = failed_paths[0]
                app_logger.error(
                    "Could not open primary log path %s (%s); using %s",
                    first_path,
                    first_error,
                    getattr(handler, "baseFilename", "stderr"),
                )
        handler.setLevel(numeric_level)

    return app_logger


def get_app_logger(component: str) -> logging.Logger:
    """Return a component logger routed through the shared application log."""
    configure_logging()
    logger = logging.getLogger(f"{_LOGGER_NAME}.{component}")
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    return logger
