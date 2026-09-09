import os
import logging

logger = logging.getLogger(__name__)


def check_key_and_bool(config: dict, key: str) -> bool:
    """Return True only if *key* exists in *config* and its value is truthy.

    Args:
        config: Configuration dictionary.
        key: Key to check.
    """
    return key in config and bool(config[key])


def check_file_exists(filename: str) -> bool:
    """Return True if *filename* exists on disk; log a warning otherwise.

    Args:
        filename: Path to check.
    """
    logger.debug(f"Check {filename}")
    exists = os.path.exists(filename)
    if not exists:
        logger.warning(f"{filename} does not exist!")
    return exists


def uniquify_dir(path: str) -> str:
    """Append a counter suffix to *path* until it does not exist on disk.

    Args:
        path: Desired directory path.

    Returns:
        A path that does not yet exist: either *path* unchanged, or
        *path*-1, *path*-2, … (extension preserved if any).
    """
    base, ext = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = f"{base}-{counter}{ext}"
        counter += 1
    return path
