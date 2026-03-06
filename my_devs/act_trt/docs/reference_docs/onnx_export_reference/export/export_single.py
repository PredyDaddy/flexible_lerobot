#!/usr/bin/env python3
"""单臂 ACT 模型 ONNX 导出脚本

本脚本将 ACT 核心网络（`policy.model`）导出为 ONNX 格式，不包含 Python 侧的动作队列逻辑
（`ACTPolicy.select_action`）。导出的 ONNX 模型接收 *已归一化* 的输入，输出 *归一化* 的动作序列。

输入输出规格 (normalized in → normalized out)

输入 (已归一化, FP32):
  - obs_state_norm: (1, state_dim)
  - img0_norm:      (1, 3, H, W)
  - img1_norm:      (1, 3, H, W)

输出 (归一化, FP32):
  - actions_norm:   (1, chunk_size, action_dim)

重要：相机顺序 (img0_norm, img1_norm) 必须与 config.json 中 input_features 的 VISUAL 顺序一致！


python trt_act/export/export_single.py --checkpoint outputs/act_agilex_left_box/checkpoints/last/pretrained_model --output trt_act/export_models
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.onnx import TrainingMode

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.utils import is_torch_device_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class ActChunkOnnxWrapper(nn.Module):
    """Wrap ACT core model to expose a flat ONNX-friendly signature."""

    def __init__(self, act_model: nn.Module, *, image_feature_keys: list[str]) -> None:
        super().__init__()
        self.act = act_model.eval()
        self.image_feature_keys = list(image_feature_keys)
        if len(self.image_feature_keys) != 2:
            raise ValueError(f"Expected 2 cameras, got {self.image_feature_keys}")

    def forward(self, obs_state_norm: Tensor, img0_norm: Tensor, img1_norm: Tensor) -> Tensor:
        batch: dict[str, Any] = {
            OBS_STATE: obs_state_norm,
            OBS_IMAGES: [img0_norm, img1_norm],
        }
        actions, _ = self.act(batch)
        return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export single-arm ACT core model to ONNX (normalized in/out).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory (pretrained_model).")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output ONNX file path (recommended, endswith .onnx) or a directory path. "
            "Defaults to <checkpoint>/act_single.onnx."
        ),
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used for export forward pass.",
    )
    parser.add_argument("--dynamic-batch", action="store_true", help="Export with dynamic batch axis.")
    parser.add_argument("--dynamo", action="store_true", help="Use torch ONNX dynamo exporter.")
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify PyTorch output vs ONNXRuntime output (requires onnxruntime).",
    )
    parser.add_argument(
        "--verify-max-abs-diff",
        type=float,
        default=1e-4,
        help="Max abs diff threshold for verification (FP32).",
    )
    parser.add_argument(
        "--keep-backbone-pretrained-weights",
        action="store_true",
        help="Keep torchvision pretrained backbone weights config (may trigger download if not cached).",
    )
    parser.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simplify ONNX model using onnx-simplifier (requires onnxsim).",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_visual_feature_keys_in_order(config_dict: dict[str, Any]) -> list[str]:
    input_features = config_dict.get("input_features", {})
    if not isinstance(input_features, dict):
        raise ValueError("Invalid config.json: input_features is not a dict.")
    return [k for k, v in input_features.items() if isinstance(v, dict) and v.get("type") == "VISUAL"]


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def _require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")


def _resolve_export_device(device_str: str) -> str:
    if is_torch_device_available(device_str):
        return device_str
    logger.warning("Requested device '%s' not available; falling back to 'cpu'.", device_str)
    return "cpu"


def _load_policy(checkpoint: Path, *, device: str, keep_backbone_pretrained_weights: bool) -> ACTPolicy:
    cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    cfg.device = device
    if not keep_backbone_pretrained_weights and hasattr(cfg, "pretrained_backbone_weights"):
        setattr(cfg, "pretrained_backbone_weights", None)

    policy = ACTPolicy.from_pretrained(str(checkpoint), config=cfg)
    policy.eval()
    return policy


def _create_dummy_inputs(
    *,
    state_dim: int,
    image_hw: tuple[int, int],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    h, w = image_hw
    g = torch.Generator(device=device.type)
    g.manual_seed(0)
    obs_state_norm = torch.randn((1, state_dim), dtype=torch.float32, generator=g, device=device)
    img0_norm = torch.randn((1, 3, h, w), dtype=torch.float32, generator=g, device=device)
    img1_norm = torch.randn((1, 3, h, w), dtype=torch.float32, generator=g, device=device)
    return obs_state_norm, img0_norm, img1_norm


def _export_onnx(
    *,
    wrapper: nn.Module,
    dummy_inputs: tuple[Tensor, Tensor, Tensor],
    output_path: Path,
    opset: int,
    dynamic_batch: bool,
    dynamo: bool,
) -> None:
    try:
        import onnx  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("onnx is required for export. Install it first (e.g., `pip install onnx`).") from e

    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper.eval()
    input_names = ["obs_state_norm", "img0_norm", "img1_norm"]
    output_names = ["actions_norm"]
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {name: {0: "batch"} for name in input_names}
        dynamic_axes["actions_norm"] = {0: "batch"}

    logger.info("Exporting ONNX to %s", output_path)
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        external_data=True,
        dynamo=dynamo,
        training=TrainingMode.EVAL,
    )

    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)


def _simplify_onnx(onnx_path: Path) -> dict[str, Any]:
    try:
        import onnx
        import onnxsim
    except ImportError as e:
        raise RuntimeError("onnxsim is required for simplification. Install it: `pip install onnxsim`") from e

    logger.info("Simplifying ONNX model...")

    original_model = onnx.load(str(onnx_path))
    original_nodes = len(original_model.graph.node)

    try:
        simplified_model, check = onnxsim.simplify(original_model)
        if not check:
            logger.warning("onnxsim.simplify() returned check=False, model may not be fully simplified.")
    except Exception as e:
        logger.warning("ONNX simplification failed: %s", e)
        return {"enabled": True, "success": False, "error": str(e)}

    simplified_nodes = len(simplified_model.graph.node)
    onnx.save(simplified_model, str(onnx_path))

    reduction = original_nodes - simplified_nodes
    reduction_pct = (reduction / original_nodes * 100) if original_nodes > 0 else 0
    logger.info(
        "Simplified: %d -> %d nodes (reduced %d, %.1f%%)",
        original_nodes,
        simplified_nodes,
        reduction,
        reduction_pct,
    )

    return {
        "enabled": True,
        "success": True,
        "original_nodes": original_nodes,
        "simplified_nodes": simplified_nodes,
        "reduction": reduction,
        "reduction_percent": round(reduction_pct, 2),
    }


def _verify_with_onnxruntime(
    *,
    wrapper: nn.Module,
    dummy_inputs: tuple[Tensor, Tensor, Tensor],
    onnx_path: Path,
) -> dict[str, float]:
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "onnxruntime is required for verification. Install it (e.g., `pip install onnxruntime`)."
        ) from e

    wrapper.eval()
    with torch.inference_mode():
        pt_out = wrapper(*dummy_inputs).detach().to("cpu").numpy()

    obs_state_norm, img0_norm, img1_norm = dummy_inputs
    ort_inputs = {
        "obs_state_norm": obs_state_norm.detach().to("cpu").numpy(),
        "img0_norm": img0_norm.detach().to("cpu").numpy(),
        "img1_norm": img1_norm.detach().to("cpu").numpy(),
    }

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"])
    ort_out = session.run(["actions_norm"], ort_inputs)[0]

    diff = np.abs(pt_out - ort_out)
    return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    _require_dir(checkpoint)
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    _require_file(config_path)
    _require_file(weights_path)

    export_device_str = _resolve_export_device(args.device)
    export_device = torch.device(export_device_str)

    cfg_dict = _load_json(config_path)
    visual_keys = _get_visual_feature_keys_in_order(cfg_dict)
    if len(visual_keys) != 2:
        raise ValueError(f"Expected 2 VISUAL inputs in config.json, got: {visual_keys}")

    if "observation.state" not in cfg_dict.get("input_features", {}):
        raise ValueError("Expected 'observation.state' in input_features for single-arm ACT export.")

    state_dim = int(cfg_dict["input_features"]["observation.state"]["shape"][0])
    c0, h0, w0 = cfg_dict["input_features"][visual_keys[0]]["shape"]
    c1, h1, w1 = cfg_dict["input_features"][visual_keys[1]]["shape"]
    if int(c0) != 3 or int(c1) != 3:
        raise ValueError(
            f"Expected 3-channel images, got C=({c0},{c1}) for ({visual_keys[0]},{visual_keys[1]})"
        )
    if (int(h0), int(w0)) != (int(h1), int(w1)):
        raise ValueError(
            "Expected both cameras to have the same (H,W), got "
            f"{visual_keys[0]}=({h0},{w0}) and {visual_keys[1]}=({h1},{w1})"
        )
    h, w = int(h0), int(w0)

    chunk_size = int(cfg_dict.get("chunk_size", 0))
    action_dim = int(cfg_dict.get("output_features", {}).get("action", {}).get("shape", [0])[0])

    if args.output:
        user_output = Path(args.output)
        if user_output.exists() and user_output.is_dir():
            output_path = user_output / "act_single.onnx"
        elif user_output.suffix.lower() == ".onnx":
            output_path = user_output
        else:
            # Treat as output directory (even if it doesn't exist yet) for better UX.
            output_path = user_output / "act_single.onnx"
    else:
        output_path = checkpoint / "act_single.onnx"
    metadata_path = output_path.parent / "export_metadata.json"

    logger.info("Checkpoint: %s", checkpoint)
    logger.info("Device: %s", export_device_str)
    logger.info("Camera order (img0,img1): %s", visual_keys)
    logger.info(
        "Shapes: state_dim=%d, image=(%d,%d), chunk_size=%d, action_dim=%d",
        state_dim,
        h,
        w,
        chunk_size,
        action_dim,
    )

    policy = _load_policy(
        checkpoint,
        device=export_device_str,
        keep_backbone_pretrained_weights=args.keep_backbone_pretrained_weights,
    )
    wrapper = ActChunkOnnxWrapper(policy.model, image_feature_keys=visual_keys).to(export_device)
    wrapper.eval()

    dummy_inputs = _create_dummy_inputs(state_dim=state_dim, image_hw=(h, w), device=export_device)
    _export_onnx(
        wrapper=wrapper,
        dummy_inputs=dummy_inputs,
        output_path=output_path,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
        dynamo=args.dynamo,
    )

    simplify_info: dict[str, Any] = {"enabled": bool(args.simplify)}
    if args.simplify:
        try:
            simplify_info = _simplify_onnx(output_path)
        except Exception as e:
            simplify_info["success"] = False
            simplify_info["error"] = f"{type(e).__name__}: {e}"
            logger.warning("ONNX simplification skipped: %s", simplify_info["error"])
    else:
        simplify_info["skipped"] = True

    verify_info: dict[str, Any] = {"enabled": bool(args.verify), "skipped": False, "error": None}
    verify_passed: bool | None = None
    if args.verify:
        try:
            metrics = _verify_with_onnxruntime(wrapper=wrapper, dummy_inputs=dummy_inputs, onnx_path=output_path)
            verify_info.update(metrics)
            verify_info["threshold_max_abs_diff"] = float(args.verify_max_abs_diff)
            verify_passed = metrics["max_abs_diff"] <= args.verify_max_abs_diff
            verify_info["passed"] = bool(verify_passed)
            logger.info(
                "Verify: max_abs_diff=%.3e mean_abs_diff=%.3e (threshold=%.1e) passed=%s",
                metrics["max_abs_diff"],
                metrics["mean_abs_diff"],
                args.verify_max_abs_diff,
                verify_passed,
            )
        except Exception as e:
            verify_info["skipped"] = True
            verify_info["error"] = f"{type(e).__name__}: {e}"
            logger.warning("Verification skipped/failed: %s", verify_info["error"])

    metadata: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "onnx_path": str(output_path),
        "camera_order_visual_keys": visual_keys,
        "io_mapping": {
            "obs_state_norm": "observation.state (mean/std normalized)",
            "img0_norm": f"{visual_keys[0]} (mean/std normalized)",
            "img1_norm": f"{visual_keys[1]} (mean/std normalized)",
            "actions_norm": "action chunk (mean/std normalized)",
        },
        "shapes": {
            "obs_state_norm": [1, state_dim],
            "img0_norm": [1, 3, h, w],
            "img1_norm": [1, 3, h, w],
            "actions_norm": [1, chunk_size, action_dim],
        },
        "export": {
            "opset": int(args.opset),
            "dynamic_batch": bool(args.dynamic_batch),
            "dynamo": bool(args.dynamo),
            "device": export_device_str,
            "keep_backbone_pretrained_weights": bool(args.keep_backbone_pretrained_weights),
        },
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "verification": verify_info,
        "simplification": simplify_info,
        "notes": [
            "This ONNX exports ACT core network only (policy.model). Action queue logic stays in Python.",
            "Inputs/outputs are normalized (mean/std). Use policy_*_processor*.safetensors stats to (un)normalize.",
            "Camera order must match config.json VISUAL input_features order.",
        ],
    }
    try:
        import torchvision

        metadata["versions"]["torchvision"] = torchvision.__version__
    except Exception:
        metadata["versions"]["torchvision"] = None

    _write_metadata(metadata_path, metadata)
    logger.info("Wrote metadata: %s", metadata_path)

    if verify_passed is False:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
