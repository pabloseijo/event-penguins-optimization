"""Filesystem and config guards used by the entry-point scripts."""

import os
import logging

logger = logging.getLogger(__name__)


def check_key_and_bool(config: dict, key: str) -> bool:
    """Return True when ``key`` is present in ``config`` and holds a truthy value.

    Used to read the opt-in flags of the pipeline without forcing every config
    file to list every switch.
    """
    return key in config and bool(config[key])


def check_file_exists(filename: str) -> bool:
    """Return whether ``filename`` exists, logging a warning when it does not."""
    logger.debug(f"Comprobando {filename}")
    exists = os.path.exists(filename)
    if not exists:
        logger.warning(f"{filename} non existe!")
    return exists


def uniquify_dir(path: str) -> str:
    """Append a numeric suffix to ``path`` until it names a free location.

    Runs never overwrite each other's outputs, which is what lets a result
    directory be trusted months later.
    """
    base, ext = os.path.splitext(path)
    counter   = 1
    while os.path.exists(path):
        path = f"{base}-{counter}{ext}"
        counter += 1
    return path
