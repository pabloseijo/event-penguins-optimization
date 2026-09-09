import os
import shutil
import subprocess

import yaml
from absl import logging

from .misc import uniquify_dir


def get_config(config_path: str, root: str) -> dict:
    """Load a YAML config, create a unique output directory, and save run metadata.

    Args:
        config_path: Path to the YAML config file.
        root: Root directory relative to which *config["output_dir"]* is resolved.

    Returns:
        Config dict with *output_dir* updated to the created unique directory.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir = uniquify_dir(os.path.join(root, config["output_dir"]))
    logging.info(f"Log dir: {output_dir}")
    os.makedirs(output_dir)
    config["output_dir"] = output_dir

    _save_config(output_dir, config_path)
    return config


def _save_config(save_dir: str, config_file: str) -> None:
    """Copy the config file and save git commit info to *save_dir*."""
    shutil.copy(config_file, os.path.join(save_dir, "config.yaml"))
    with open(os.path.join(save_dir, "runtime.yaml"), "w") as f:
        yaml.dump({"commit": _fetch_commit_id()}, f)


def _fetch_commit_id() -> str:
    """Return the current git HEAD commit hash, or 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).strip().decode("utf-8")
    except subprocess.CalledProcessError:
        return "unknown"
