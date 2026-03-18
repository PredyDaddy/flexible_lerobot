from __future__ import annotations

from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "lerobot").is_dir():
            return candidate
    raise RuntimeError(f"Unable to locate repository root from {start}")


SCRIPT_DIR = Path(__file__).resolve().parent
NEW_ACT_TRT_DIR = SCRIPT_DIR.parent
DOCS_DIR = NEW_ACT_TRT_DIR / "docs"
REFERENCE_DIR = NEW_ACT_TRT_DIR / "reference" / "act_trt"
REPO_ROOT = _find_repo_root(SCRIPT_DIR)
SRC_DIR = REPO_ROOT / "src"
DEPLOY_ROOT = REPO_ROOT / "outputs" / "deploy" / "act_trt"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "train"
    / "20260305_190147_act_grasp_block_in_bin1_e15"
    / "checkpoints"
    / "last"
    / "pretrained_model"
)
