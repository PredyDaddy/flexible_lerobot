from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from openpi_so101 import runtime


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image_to_hwc_uint8(value: Any) -> np.ndarray:
    image = _to_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape={image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image)) <= 1.5 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    return np.ascontiguousarray(image)


def _state_to_float32(value: Any) -> np.ndarray:
    state = _to_numpy(value).astype(np.float32, copy=False)
    if state.shape != (6,):
        raise ValueError(f"Expected SO101 state shape=(6,), got shape={state.shape}")
    return np.ascontiguousarray(state)


def _build_random_observation(prompt: str) -> dict[str, Any]:
    return {
        "observation.images.top": np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.wrist": np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8),
        "observation.state": np.zeros((6,), dtype=np.float32),
        "prompt": prompt,
    }


def _build_dataset_observation(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    from openpi_so101 import config as so101_config
    from openpi_so101 import dataset_v3
    from openpi_so101 import patches

    config = so101_config.make_config(
        exp_name="remote_infer_client",
        asset_id=args.asset_id
        or ("desk_cleanup_v1/eraser_cup_multi_task_v21_pilot" if args.dataset_format == "v21" else None),
    )
    if args.dataset_format == "v21":
        dataset = patches.LeRobotDatasetV21Compat(
            config.data.repo_id,
            runtime.converted_dataset_root(),
            action_horizon=config.model.action_horizon,
            max_frames=max(args.dataset_index + 1, 1),
        )
    else:
        dataset = dataset_v3.SO101LeRobotV3Dataset(
            config.data.repo_id,
            runtime.dataset_root(),
            action_horizon=config.model.action_horizon,
            max_frames=max(args.dataset_index + 1, 1),
            decode_images=True,
        )

    sample = dataset[args.dataset_index]
    return {
        "observation.images.top": _image_to_hwc_uint8(sample["observation.images.top"]),
        "observation.images.wrist": _image_to_hwc_uint8(sample["observation.images.wrist"]),
        "observation.state": _state_to_float32(sample["observation.state"]),
        "prompt": prompt,
    }


def _build_observation(args: argparse.Namespace) -> dict[str, Any]:
    prompt = args.prompt
    if args.source == "random":
        return _build_random_observation(prompt)
    return _build_dataset_observation(args, prompt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query a running SO101 OpenPI websocket policy server.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="Put the eraser into the small box")
    parser.add_argument("--source", choices=("dataset", "random"), default="dataset")
    parser.add_argument("--dataset-format", choices=("v3", "v21"), default="v21")
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--num-requests", type=int, default=1)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--api-key", default=None)
    return parser


def main() -> None:
    runtime.bootstrap()
    args = build_parser().parse_args()

    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port, api_key=args.api_key)
    print(f"server metadata={client.get_server_metadata()}")

    for request_index in range(args.num_requests):
        observation = _build_observation(args)
        start = time.perf_counter()
        result = client.infer(observation)
        elapsed_ms = (time.perf_counter() - start) * 1000
        actions = np.asarray(result["actions"], dtype=np.float32)
        print(
            f"request={request_index} actions shape={actions.shape} dtype={actions.dtype} "
            f"client_ms={elapsed_ms:.1f}"
        )
        print(f"first_action={np.array2string(actions[0], precision=4, suppress_small=True)}")
        if "server_timing" in result:
            print(f"server_timing={result['server_timing']}")
        if "policy_timing" in result:
            print(f"policy_timing={result['policy_timing']}")
        if args.sleep_s > 0 and request_index + 1 < args.num_requests:
            time.sleep(args.sleep_s)


if __name__ == "__main__":
    main()
