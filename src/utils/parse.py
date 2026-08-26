"""YAML config loading with per-run output directories and provenance stamping.

Every run writes its own configuration and the git commit it was launched from
next to its results, so a number found months later can be traced back to the
exact code that produced it.
"""

import os
import shutil
import subprocess

import yaml
from absl import logging

from .misc import uniquify_dir


def get_config(config_path: str, root: str) -> dict:
    """Carga o YAML de configuración e crea un directorio de saída único."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir = uniquify_dir(os.path.join(root, config["output_dir"]))
    logging.info(f"Directorio de saída: {output_dir}")
    os.makedirs(output_dir)
    config["output_dir"] = output_dir

    _save_config(output_dir, config_path)
    return config


def _save_config(save_dir: str, config_file: str) -> None:
    shutil.copy(config_file, os.path.join(save_dir, "config.yaml"))
    with open(os.path.join(save_dir, "runtime.yaml"), "w") as f:
        yaml.dump({"commit": _fetch_commit_id()}, f)


def _fetch_commit_id() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).strip().decode("utf-8")
    except subprocess.CalledProcessError:
        return "unknown"
