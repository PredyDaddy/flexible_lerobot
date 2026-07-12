#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
from gr00t.policy.server_client import PolicyServer


class RtcGr00tPolicy(Gr00tPolicy):
    """GR00T policy with stateful N1.7 model-level RTC inference."""

    def __init__(
        self,
        *args: Any,
        inference_backend: str = "pytorch",
        trt_engine_path: Path | None = None,
        trt_mode: str = "n17_full_pipeline",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.inference_backend = inference_backend
        self.trt_mode = None
        if inference_backend == "tensorrt":
            if trt_engine_path is None:
                raise ValueError("TensorRT backend requires --trt-engine-path")
            engine_path = trt_engine_path.resolve(strict=True)
            metadata_path = engine_path.parent / "onnx" / "export_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("batch_size") != 1 or metadata.get("embodiment_tag") not in {
                "EmbodimentTag.NEW_EMBODIMENT",
                "new_embodiment",
            }:
                raise RuntimeError(f"TensorRT metadata does not match SO101 deployment: {metadata}")
            from so101_rtc_trt import setup_rtc_tensorrt_engines

            setup_rtc_tensorrt_engines(self, engine_path, trt_mode)
            self.trt_mode = trt_mode
        self._previous_action: dict[str, np.ndarray] | None = None
        self._request_index = 0

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self._previous_action = None
        self._request_index = 0
        return {"status": "ok", "rtc_state": "reset", "options": options or {}}

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "message": "Server is running",
            "inference_backend": self.inference_backend,
            "trt_mode": self.trt_mode,
        }

    @staticmethod
    def _rtc_model_options(options: dict[str, Any], action_horizon: int) -> dict[str, Any]:
        advance_steps = int(options["rtc_advance_steps"])
        frozen_steps = int(options["rtc_frozen_steps"])
        ramp_rate = float(options.get("rtc_ramp_rate", 2.0))
        overlap_steps = action_horizon - advance_steps
        if not 1 <= advance_steps < action_horizon:
            raise ValueError(f"rtc_advance_steps must be in [1, {action_horizon - 1}]")
        if not 0 <= frozen_steps <= overlap_steps:
            raise ValueError(f"rtc_frozen_steps must be in [0, {overlap_steps}]")
        if not np.isfinite(ramp_rate) or ramp_rate <= 0:
            raise ValueError("rtc_ramp_rate must be finite and positive")
        return {
            "action_horizon": action_horizon,
            "rtc_overlap_steps": overlap_steps,
            "rtc_frozen_steps": frozen_steps,
            "rtc_ramp_rate": ramp_rate,
        }

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        options = options or {}
        rtc_requested = bool(options.get("rtc_enabled", False))
        unbatched_observations = self._unbatch_observation(observation)
        if len(unbatched_observations) != 1:
            raise ValueError("The stateful RTC server requires batch size 1")

        action_horizon = len(self.modality_configs["action"].delta_indices)
        use_rtc = rtc_requested and self._previous_action is not None
        model_options = self._rtc_model_options(options, action_horizon) if use_rtc else None

        processed_inputs = []
        states = []
        for obs in unbatched_observations:
            actions = {}
            if use_rtc:
                assert self._previous_action is not None
                actions = {key: value[0].copy() for key, value in self._previous_action.items()}
            vla_step_data = VLAStepData(
                images=obs["video"],
                states=obs["state"],
                actions=actions,
                text=obs["language"][self.language_key][0],
                embodiment=self.embodiment_tag,
            )
            states.append(vla_step_data.states)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        collated_inputs = _rec_to_dtype(self.collate_fn(processed_inputs), dtype=torch.bfloat16)
        started = time.perf_counter()
        with torch.inference_mode():
            model_pred = self.model.get_action(**collated_inputs, options=model_options)
        model_s = time.perf_counter() - started
        normalized_action = model_pred["action_pred"].float()

        batched_states = {
            key: np.stack([state[key] for state in states], axis=0)
            for key in self.modality_configs["state"].modality_keys
        }
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )
        casted_action = {key: value.astype(np.float32) for key, value in unnormalized_action.items()}
        self._previous_action = {key: value.copy() for key, value in casted_action.items()}
        info = {
            "request_index": self._request_index,
            "rtc_applied": use_rtc,
            "rtc_requested": rtc_requested,
            "model_inference_s": model_s,
            "model_options": model_options,
            "inference_backend": self.inference_backend,
            "trt_mode": self.trt_mode,
        }
        self._request_index += 1
        return casted_action, info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stateful GR00T N1.7 RTC policy server")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--inference-backend", choices=("pytorch", "tensorrt"), default="pytorch")
    parser.add_argument("--trt-engine-path", type=Path)
    parser.add_argument(
        "--trt-mode", choices=("n17_full_pipeline", "action_head"), default="n17_full_pipeline"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy = RtcGr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag),
        model_path=str(args.model_path.resolve(strict=True)),
        device=args.device,
        strict=True,
        inference_backend=args.inference_backend,
        trt_engine_path=args.trt_engine_path,
        trt_mode=args.trt_mode,
    )
    print(
        f"[RTC SERVER] ready on {args.host}:{args.port} backend={args.inference_backend}",
        flush=True,
    )
    server = PolicyServer(policy=policy, host=args.host, port=args.port)
    server.register_endpoint("ping", policy.health, requires_input=False)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[RTC SERVER] stopped", flush=True)


if __name__ == "__main__":
    main()
