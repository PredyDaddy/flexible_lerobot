from __future__ import annotations

import os
from pathlib import Path
import sys

from openpi_so101 import paths


def bootstrap() -> None:
    openpi_src = paths.OPENPI_ROOT / "src"
    openpi_client_src = paths.OPENPI_ROOT / "packages" / "openpi-client" / "src"
    for path in (paths.PROJECT_ROOT, paths.OPENPI_ROOT, openpi_src, openpi_client_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OPENPI_DATA_HOME", str(paths.DEFAULT_OPENPI_DATA_HOME))
    os.environ.setdefault("OPENPI_SO101_DATASET_ROOT", str(paths.DEFAULT_SOURCE_DATASET))
    os.environ.setdefault("OPENPI_SO101_REPO_ID", paths.DEFAULT_REPO_ID)
    os.environ.setdefault("OPENPI_SO101_ASSETS_BASE_DIR", str(paths.DEFAULT_ASSETS_BASE_DIR))
    os.environ.setdefault("OPENPI_SO101_CHECKPOINT_BASE_DIR", str(paths.DEFAULT_CHECKPOINT_BASE_DIR))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def dataset_root() -> Path:
    return Path(os.environ["OPENPI_SO101_DATASET_ROOT"]).expanduser().resolve()


def converted_dataset_root() -> Path:
    return Path(os.environ.get("OPENPI_SO101_V21_ROOT", str(paths.DEFAULT_CONVERTED_DATA_ROOT))).expanduser().resolve()
