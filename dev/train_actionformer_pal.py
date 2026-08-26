#!/usr/bin/env python3
"""Register PAL extensions and delegate to the official ActionFormer trainer."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    actionformer_root = Path.cwd().resolve()
    sys.path.insert(0, str(actionformer_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import actionformer_pal_dataset  # noqa: F401
    import actionformer_pal_meta_arch  # noqa: F401

    runpy.run_path(str(actionformer_root / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
