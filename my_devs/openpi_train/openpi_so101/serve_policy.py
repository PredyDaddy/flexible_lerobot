from __future__ import annotations

import argparse
import logging

from openpi_so101 import config as so101_config
from openpi_so101 import policy as so101_policy
from openpi_so101 import runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a trained SO101 OpenPI checkpoint.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--dataset-format", choices=("v3", "v21"), default="v3")
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--pytorch-device", default=None)
    return parser


def main() -> None:
    runtime.bootstrap()
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server

    config = so101_config.make_config(
        exp_name="serve",
        asset_id=args.asset_id
        or ("desk_cleanup_v1/eraser_cup_multi_task_v21_pilot" if args.dataset_format == "v21" else None),
    )
    so101_config.register_config(config)
    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint_dir,
        repack_transforms=so101_policy.SO101_INFERENCE_REPACK_TRANSFORMS,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
