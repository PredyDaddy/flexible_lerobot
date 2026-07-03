from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = PROJECT_ROOT / "openpi-main"
DEFAULT_SOURCE_DATASET = Path(
    "/data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_cup_multi_task"
)
DEFAULT_REPO_ID = "desk_cleanup_v1/eraser_cup_multi_task"
DEFAULT_LOCAL_DATA_HOME = PROJECT_ROOT / "data" / "lerobot_home"
DEFAULT_CONVERTED_DATA_ROOT = PROJECT_ROOT / "data" / "lerobot_v21_pilot" / "desk_cleanup_v1" / "eraser_cup_multi_task"
DEFAULT_OPENPI_DATA_HOME = PROJECT_ROOT / "assets" / "openpi_cache"
DEFAULT_ASSETS_BASE_DIR = PROJECT_ROOT / "assets" / "openpi_assets"
DEFAULT_CHECKPOINT_BASE_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
