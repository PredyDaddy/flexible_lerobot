from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any
from unittest.mock import patch

import torch

from lerobot import policies  # noqa: F401  # Register policy config classes.
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.constants import ACTION

from my_devs.groot_trt.trt_utils import TrtSession


@dataclass(frozen=True)
class _TrtEnginePaths:
    vit: Path
    llm: Path
    vlln: Path
    state_encoder: Path
    action_encoder: Path
    dit: Path
    action_decoder: Path

    def values(self) -> tuple[Path, ...]:
        return (
            self.vit,
            self.llm,
            self.vlln,
            self.state_encoder,
            self.action_encoder,
            self.dit,
            self.action_decoder,
        )


@dataclass(frozen=True)
class BackendSelfCheckResult:
    name: str
    ok: bool
    detail: str
    category: str = "behavior"


def _resolve_policy_dir(pretrained_name_or_path: str | Path) -> Path:
    policy_path = Path(pretrained_name_or_path).expanduser().resolve()
    candidates = (
        policy_path,
        policy_path / "pretrained_model",
        policy_path / "checkpoints" / "last" / "pretrained_model",
    )
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Policy directory must contain `config.json` (the LeRobot `pretrained_model/` folder).\n"
        f"You passed: {policy_path}\n"
        "Searched:\n"
        f"{searched}\n"
    )


def _resolve_cuda_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"GrootTrtPolicyBackend requires a CUDA device, got {resolved!s}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but the GR00T TensorRT backend requires CUDA.")
    return resolved


def _num_patches(backbone: torch.nn.Module) -> int:
    vision_model = backbone.eagle_model.vision_model
    if hasattr(vision_model, "vision_model") and hasattr(vision_model.vision_model, "embeddings"):
        return int(vision_model.vision_model.embeddings.num_patches)
    if hasattr(vision_model, "embeddings"):
        return int(vision_model.embeddings.num_patches)
    raise AttributeError("Cannot determine num_patches from backbone vision model.")


def _postprocess_vit(backbone: torch.nn.Module, vit_embeds: torch.Tensor) -> torch.Tensor:
    # Keep the same post-ViT glue that was used during TRT export and validation.
    vit_embeds = vit_embeds.view(1, -1, vit_embeds.shape[-1])
    if getattr(backbone.eagle_model, "use_pixel_shuffle", False):
        token_count = int(vit_embeds.shape[1])
        side = int(token_count**0.5)
        if side * side != token_count:
            raise ValueError(f"Pixel-shuffle expects square token layout, got token_count={token_count}.")
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], side, side, -1)
        pixel_shuffle = getattr(backbone, "pixel_shuffle", None) or getattr(backbone.eagle_model, "pixel_shuffle", None)
        downsample_ratio = getattr(backbone, "downsample_ratio", None) or getattr(
            backbone.eagle_model, "downsample_ratio", None
        )
        if pixel_shuffle is None or downsample_ratio is None:
            raise RuntimeError("Eagle pixel shuffle is enabled but pixel_shuffle/downsample_ratio is missing.")
        vit_embeds = pixel_shuffle(vit_embeds, scale_factor=downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
    return backbone.eagle_model.mlp1(vit_embeds)


def _build_inputs_embeds_from_vit(
    backbone: torch.nn.Module,
    input_ids: torch.Tensor,
    vit_embeds: torch.Tensor,
) -> torch.Tensor:
    embedding_layer = backbone.eagle_model.language_model.get_input_embeddings()
    image_token_index = int(backbone.eagle_model.image_token_index)

    input_ids = input_ids.to(device=vit_embeds.device, dtype=torch.int64)
    inputs_embeds = embedding_layer(input_ids).to(torch.float16)

    if vit_embeds.ndim != 3:
        raise ValueError(f"Expected vit_embeds to have shape (batch, tokens, hidden), got {tuple(vit_embeds.shape)}.")

    batch_size, _, channels = inputs_embeds.shape
    if vit_embeds.shape[0] != batch_size:
        raise ValueError(
            "ViT embeddings batch dimension must match input_ids batch dimension: "
            f"input_ids batch={batch_size}, vit_embeds batch={vit_embeds.shape[0]}."
        )
    if vit_embeds.shape[-1] != channels:
        raise ValueError(
            "ViT embeddings hidden size must match language embedding size: "
            f"language hidden={channels}, vit hidden={vit_embeds.shape[-1]}."
        )

    image_token_mask = input_ids == image_token_index
    slot_counts = image_token_mask.sum(dim=1)
    expected_slots = int(vit_embeds.shape[1])
    if not torch.all(slot_counts == expected_slots):
        slot_counts_list = [int(count) for count in slot_counts.tolist()]
        raise ValueError(
            "Image-token slot count must exactly match post-ViT token count for every batch item: "
            f"slots_per_batch={slot_counts_list}, vit_tokens_per_batch={expected_slots}."
        )

    vit_flat = vit_embeds.reshape(-1, channels).to(dtype=inputs_embeds.dtype)
    if vit_flat.shape[0] != int(image_token_mask.sum().item()):
        raise ValueError(
            "Flattened ViT token count must exactly match total image-token slots: "
            f"slots={int(image_token_mask.sum().item())}, vit_tokens={vit_flat.shape[0]}."
        )

    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[image_token_mask] = vit_flat
    return inputs_embeds


class _SelfCheckLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 8) -> None:
        super().__init__()
        self._embeddings = torch.nn.Embedding(vocab_size, hidden_size)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self._embeddings


class _SelfCheckEagleModel(torch.nn.Module):
    def __init__(self, image_token_index: int, vocab_size: int = 32, hidden_size: int = 8) -> None:
        super().__init__()
        self.image_token_index = image_token_index
        self.language_model = _SelfCheckLanguageModel(vocab_size=vocab_size, hidden_size=hidden_size)
        self.mlp1 = torch.nn.Identity()
        self.use_pixel_shuffle = False


class _SelfCheckBackbone(torch.nn.Module):
    def __init__(self, image_token_index: int = 7, vocab_size: int = 32, hidden_size: int = 8) -> None:
        super().__init__()
        self.eagle_model = _SelfCheckEagleModel(
            image_token_index=image_token_index,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )


@dataclass
class _BehaviorCheckActionHeadConfig:
    action_horizon: int = 2
    action_dim: int = 2
    add_pos_embed: bool = False


@dataclass
class _PredictBehaviorHarness:
    backend: GrootTrtPolicyBackend
    batch: dict[str, torch.Tensor]
    events: list[_BehaviorCheckEvent]
    lock: _BehaviorCheckLock
    policy: _BehaviorCheckPolicy
    stream: _BehaviorCheckCudaStream
    trackers: dict[str, Any]
    sessions: dict[str, _BehaviorCheckSession]


@dataclass(frozen=True)
class _BehaviorCheckEvent:
    name: str
    lock_held: bool | None = None
    session_name: str | None = None
    stream_name: str | None = None


class _BehaviorCheckLock:
    def __init__(self, *, events: list[_BehaviorCheckEvent] | None = None) -> None:
        self.enter_count = 0
        self.exit_count = 0
        self._depth = 0
        self._events = events

    @property
    def is_held(self) -> bool:
        return self._depth > 0

    def __enter__(self) -> _BehaviorCheckLock:
        self.enter_count += 1
        self._depth += 1
        if self._events is not None:
            self._events.append(_BehaviorCheckEvent(name="lock_enter", lock_held=True))
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        self._depth -= 1
        self.exit_count += 1
        if self._depth < 0:
            raise AssertionError("Behavior self-check lock depth went negative.")
        if self._events is not None:
            self._events.append(_BehaviorCheckEvent(name="lock_exit", lock_held=self.is_held))
        return False


class _BehaviorCheckCudaDeviceContext:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __enter__(self) -> _BehaviorCheckCudaDeviceContext:
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        return False


class _BehaviorCheckCudaStream:
    def __init__(
        self,
        device: torch.device,
        *,
        name: str = "predict-stream",
        lock: _BehaviorCheckLock | None = None,
        events: list[_BehaviorCheckEvent] | None = None,
    ) -> None:
        self.device = device
        self.name = name
        self.synchronize_count = 0
        self._lock = lock
        self._events = events

    def synchronize(self) -> None:
        self.synchronize_count += 1
        if self._events is not None:
            self._events.append(
                _BehaviorCheckEvent(
                    name="stream_synchronize",
                    lock_held=None if self._lock is None else self._lock.is_held,
                    stream_name=self.name,
                )
            )


class _BehaviorCheckPolicy:
    def __init__(self, lock: _BehaviorCheckLock) -> None:
        self._lock = lock
        self.reset_calls = 0
        self.eval_calls = 0

    def reset(self) -> None:
        if not self._lock.is_held:
            raise AssertionError("reset() executed outside the shared TRT lock.")
        self.reset_calls += 1

    def eval(self) -> _BehaviorCheckPolicy:
        if not self._lock.is_held:
            raise AssertionError("predict_action_chunk() called eval() outside the shared TRT lock.")
        self.eval_calls += 1
        return self


class _BehaviorCheckActionHead(torch.nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.config = _BehaviorCheckActionHeadConfig()
        self.future_tokens = torch.nn.Embedding(1, hidden_size)
        self.num_inference_timesteps = 1
        self.num_timestep_buckets = 8
        with torch.no_grad():
            self.future_tokens.weight.copy_(torch.tensor([[0.5, 1.0, 1.5, 2.0]], dtype=torch.float32))


class _BehaviorCheckSession:
    def __init__(
        self,
        name: str,
        lock: _BehaviorCheckLock,
        events: list[_BehaviorCheckEvent],
        output_key: str,
        output_fn,
        *,
        bound_stream_name: str,
        fail_with: Exception | None = None,
    ) -> None:
        self.name = name
        self._lock = lock
        self._events = events
        self._output_key = output_key
        self._output_fn = output_fn
        self._bound_stream_name = bound_stream_name
        self._fail_with = fail_with
        self.calls = 0

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.calls += 1
        if not self._lock.is_held:
            raise AssertionError(f"Session {self.name} ran outside the shared TRT lock.")
        self._events.append(
            _BehaviorCheckEvent(
                name="session_run",
                lock_held=self._lock.is_held,
                session_name=self.name,
                stream_name=self._bound_stream_name,
            )
        )
        if self._fail_with is not None:
            self._events.append(
                _BehaviorCheckEvent(
                    name="session_fail",
                    lock_held=self._lock.is_held,
                    session_name=self.name,
                    stream_name=self._bound_stream_name,
                )
            )
            raise self._fail_with
        output = self._output_fn(inputs)
        self._events.append(
            _BehaviorCheckEvent(
                name="session_complete",
                lock_held=self._lock.is_held,
                session_name=self.name,
                stream_name=self._bound_stream_name,
            )
        )
        return {self._output_key: output}


def _prime_self_check_embedding_weights(backbone: _SelfCheckBackbone) -> None:
    embedding = backbone.eagle_model.language_model.get_input_embeddings()
    with torch.no_grad():
        values = torch.arange(embedding.weight.numel(), dtype=embedding.weight.dtype).view_as(embedding.weight)
        embedding.weight.copy_(values)


def _has_lock_guard(method_name: str) -> bool:
    source = inspect.getsource(getattr(GrootTrtPolicyBackend, method_name))
    return "with self._trt_sessions_lock" in source


def _has_stream_synchronize_guard() -> bool:
    source = inspect.getsource(GrootTrtPolicyBackend.predict_action_chunk)
    return "torch.cuda.current_stream(device=self.device).synchronize()" in source


def _build_predict_action_chunk_inputs_embeds(
    backbone: torch.nn.Module,
    input_ids: torch.Tensor,
    vit_embeds: torch.Tensor,
) -> torch.Tensor:
    try:
        return _build_inputs_embeds_from_vit(backbone, input_ids, vit_embeds)
    except ValueError as exc:
        raise ValueError(
            "predict_action_chunk() rejected LLM input assembly before LLM execution: "
            f"{exc}"
        ) from exc


def _format_behavior_events(events: list[_BehaviorCheckEvent]) -> str:
    if not events:
        return "[]"

    rendered_events: list[str] = []
    for event in events:
        fields = [f"lock_held={event.lock_held}"]
        if event.session_name is not None:
            fields.append(f"session={event.session_name}")
        if event.stream_name is not None:
            fields.append(f"stream={event.stream_name}")
        rendered_events.append(f"{event.name}({', '.join(fields)})")

    return "[" + ", ".join(rendered_events) + "]"


def _session_events(events: list[_BehaviorCheckEvent]) -> list[_BehaviorCheckEvent]:
    return [event for event in events if event.name in {"session_complete", "session_fail"}]


def _stream_sync_happened_before_unlock(events: list[_BehaviorCheckEvent]) -> bool:
    sync_index = None
    unlock_index = None
    for index, event in enumerate(events):
        if sync_index is None and event.name == "stream_synchronize" and event.lock_held is True:
            sync_index = index
        if unlock_index is None and event.name == "lock_exit":
            unlock_index = index

    return sync_index is not None and unlock_index is not None and sync_index < unlock_index


def _stream_sync_proves_last_session_completion(events: list[_BehaviorCheckEvent]) -> bool:
    session_events = _session_events(events)
    if not session_events:
        return False

    sync_index = None
    unlock_index = None
    for index, event in enumerate(events):
        if sync_index is None and event.name == "stream_synchronize" and event.lock_held is True:
            sync_index = index
        if unlock_index is None and event.name == "lock_exit":
            unlock_index = index

    if sync_index is None or unlock_index is None or sync_index >= unlock_index:
        return False

    sync_event = events[sync_index]
    last_session_index = max(index for index, event in enumerate(events) if event in session_events)
    if last_session_index >= sync_index:
        return False

    if sync_event.stream_name is None:
        return False

    return all(event.stream_name == sync_event.stream_name for event in session_events)


def _run_image_token_behavior_self_checks() -> list[BackendSelfCheckResult]:
    backbone = _SelfCheckBackbone(hidden_size=4)
    _prime_self_check_embedding_weights(backbone)

    input_ids = torch.tensor([[1, 7, 7, 2]], dtype=torch.int64)
    baseline_embeds = backbone.eagle_model.language_model.get_input_embeddings()(input_ids).to(torch.float16)
    vit_embeds = torch.tensor(
        [[[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]],
        dtype=torch.float16,
    )

    checks: list[BackendSelfCheckResult] = []
    try:
        inputs_embeds = _build_inputs_embeds_from_vit(backbone, input_ids, vit_embeds)
    except Exception as exc:
        checks.append(
            BackendSelfCheckResult(
                name="image-token-content-helper-behavior",
                ok=False,
                detail=f"unexpected failure on image-token injection: {type(exc).__name__}: {exc}",
                category="behavior",
            )
        )
    else:
        injected_ok = torch.equal(inputs_embeds[0, 1], vit_embeds[0, 0]) and torch.equal(
            inputs_embeds[0, 2], vit_embeds[0, 1]
        )
        passthrough_ok = torch.equal(inputs_embeds[0, 0], baseline_embeds[0, 0]) and torch.equal(
            inputs_embeds[0, 3], baseline_embeds[0, 3]
        )
        checks.append(
            BackendSelfCheckResult(
                name="image-token-content-helper-behavior",
                ok=injected_ok and passthrough_ok,
                detail=(
                    "validated exact ViT token injection into image-token slots and preserved the non-image slots"
                ),
                category="behavior",
            )
        )

    try:
        _build_inputs_embeds_from_vit(
            backbone,
            input_ids,
            torch.randn(1, 3, 4, dtype=torch.float16),
        )
    except ValueError as exc:
        checks.append(
            BackendSelfCheckResult(
                name="image-token-slot-mismatch-helper-behavior",
                ok="Image-token slot count must exactly match" in str(exc),
                detail=str(exc),
                category="behavior",
            )
        )
    except Exception as exc:
        checks.append(
            BackendSelfCheckResult(
                name="image-token-slot-mismatch-helper-behavior",
                ok=False,
                detail=f"unexpected exception type: {type(exc).__name__}: {exc}",
                category="behavior",
            )
        )
    else:
        checks.append(
            BackendSelfCheckResult(
                name="image-token-slot-mismatch-helper-behavior",
                ok=False,
                detail="slot mismatch did not raise ValueError",
                category="behavior",
            )
        )

    return checks


def _build_predict_behavior_harness(
    *,
    fail_session: str | None = None,
    vit_embeds: torch.Tensor | None = None,
    session_stream_name: str = "predict-stream",
    sync_stream_name: str = "predict-stream",
) -> _PredictBehaviorHarness:
    device = torch.device("cpu")
    events: list[_BehaviorCheckEvent] = []
    lock = _BehaviorCheckLock(events=events)
    policy = _BehaviorCheckPolicy(lock)
    stream = _BehaviorCheckCudaStream(device=device, name=sync_stream_name, lock=lock, events=events)
    trackers: dict[str, Any] = {"image_content_validated": False}

    backbone = _SelfCheckBackbone(hidden_size=4)
    _prime_self_check_embedding_weights(backbone)
    action_head = _BehaviorCheckActionHead(hidden_size=4)
    batch = {
        "eagle_pixel_values": torch.arange(12, dtype=torch.float16).view(1, 3, 2, 2),
        "eagle_input_ids": torch.tensor([[1, 7, 7, 2]], dtype=torch.int64),
        "eagle_attention_mask": torch.ones((1, 4), dtype=torch.int64),
        "embodiment_id": torch.tensor([0], dtype=torch.int64),
        "state": torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float16),
    }
    baseline_embeds = backbone.eagle_model.language_model.get_input_embeddings()(batch["eagle_input_ids"]).to(
        torch.float16
    )
    expected_vit_embeds = (
        vit_embeds.to(dtype=torch.float16).contiguous()
        if vit_embeds is not None
        else torch.tensor(
            [[[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]],
            dtype=torch.float16,
        )
    )

    def _llm_output(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs_embeds = inputs["inputs_embeds"]
        injected_ok = torch.equal(inputs_embeds[0, 1], expected_vit_embeds[0, 0]) and torch.equal(
            inputs_embeds[0, 2], expected_vit_embeds[0, 1]
        )
        passthrough_ok = torch.equal(inputs_embeds[0, 0], baseline_embeds[0, 0]) and torch.equal(
            inputs_embeds[0, 3], baseline_embeds[0, 3]
        )
        if not injected_ok or not passthrough_ok:
            raise AssertionError("predict_action_chunk() did not inject ViT tokens into inputs_embeds as expected.")
        trackers["image_content_validated"] = True
        return inputs_embeds + torch.tensor(0.5, dtype=inputs_embeds.dtype)

    session_failures = {
        "vit": RuntimeError("intentional vit failure"),
        "llm": RuntimeError("intentional llm failure"),
        "vlln": RuntimeError("intentional vlln failure"),
        "state_encoder": RuntimeError("intentional state encoder failure"),
        "action_encoder": RuntimeError("intentional action encoder failure"),
        "dit": RuntimeError("intentional dit failure"),
        "action_decoder": RuntimeError("intentional action decoder failure"),
    }
    sessions = {
        "vit": _BehaviorCheckSession(
            "vit",
            lock,
            events,
            "vit_embeds",
            lambda inputs: expected_vit_embeds.clone(),
            bound_stream_name=session_stream_name,
            fail_with=session_failures["vit"] if fail_session == "vit" else None,
        ),
        "llm": _BehaviorCheckSession(
            "llm",
            lock,
            events,
            "embeddings",
            _llm_output,
            bound_stream_name=session_stream_name,
            fail_with=session_failures["llm"] if fail_session == "llm" else None,
        ),
        "vlln": _BehaviorCheckSession(
            "vlln",
            lock,
            events,
            "output",
            lambda inputs: inputs["backbone_features"][:, :1, :].contiguous(),
            bound_stream_name=session_stream_name,
            fail_with=session_failures["vlln"] if fail_session == "vlln" else None,
        ),
        "state_encoder": _BehaviorCheckSession(
            "state_encoder",
            lock,
            events,
            "output",
            lambda inputs: inputs["state"].unsqueeze(1).contiguous(),
            bound_stream_name=session_stream_name,
            fail_with=session_failures["state_encoder"] if fail_session == "state_encoder" else None,
        ),
        "action_encoder": _BehaviorCheckSession(
            "action_encoder",
            lock,
            events,
            "output",
            lambda inputs: torch.cat((inputs["actions"], inputs["actions"]), dim=-1).contiguous(),
            bound_stream_name=session_stream_name,
            fail_with=session_failures["action_encoder"] if fail_session == "action_encoder" else None,
        ),
        "dit": _BehaviorCheckSession(
            "dit",
            lock,
            events,
            "output",
            lambda inputs: inputs["sa_embs"][:, -2:, :].contiguous(),
            bound_stream_name=session_stream_name,
            fail_with=session_failures["dit"] if fail_session == "dit" else None,
        ),
        "action_decoder": _BehaviorCheckSession(
            "action_decoder",
            lock,
            events,
            "output",
            lambda inputs: inputs["model_output"][..., :2].contiguous(),
            bound_stream_name=session_stream_name,
            fail_with=session_failures["action_decoder"] if fail_session == "action_decoder" else None,
        ),
    }

    backend = object.__new__(GrootTrtPolicyBackend)
    backend.device = device
    backend.torch_policy = policy
    backend.backbone = backbone
    backend.action_head = action_head
    backend._num_patches = 2
    backend._original_action_dim = 2
    backend._num_denoising_steps = 1
    backend._trt_sessions_lock = lock
    backend.sess_vit = sessions["vit"]
    backend.sess_llm = sessions["llm"]
    backend.sess_vlln = sessions["vlln"]
    backend.sess_state_encoder = sessions["state_encoder"]
    backend.sess_action_encoder = sessions["action_encoder"]
    backend.sess_dit = sessions["dit"]
    backend.sess_action_decoder = sessions["action_decoder"]

    return _PredictBehaviorHarness(
        backend=backend,
        batch=batch,
        events=events,
        lock=lock,
        policy=policy,
        stream=stream,
        trackers=trackers,
        sessions=sessions,
    )


def _execute_predict_behavior_harness(
    *,
    fail_session: str | None = None,
    vit_embeds: torch.Tensor | None = None,
    session_stream_name: str = "predict-stream",
    sync_stream_name: str = "predict-stream",
) -> tuple[_PredictBehaviorHarness, torch.Tensor | None, Exception | None]:
    harness = _build_predict_behavior_harness(
        fail_session=fail_session,
        vit_embeds=vit_embeds,
        session_stream_name=session_stream_name,
        sync_stream_name=sync_stream_name,
    )
    with patch.object(torch.cuda, "device", new=lambda device: _BehaviorCheckCudaDeviceContext(device)):
        with patch.object(torch.cuda, "current_stream", new=lambda device=None: harness.stream):
            try:
                output = harness.backend.predict_action_chunk(harness.batch)
            except Exception as exc:
                return harness, None, exc
    return harness, output, None


def _run_lock_and_stream_behavior_self_checks() -> list[BackendSelfCheckResult]:
    checks: list[BackendSelfCheckResult] = []

    lock = _BehaviorCheckLock()
    policy = _BehaviorCheckPolicy(lock)
    reset_backend = object.__new__(GrootTrtPolicyBackend)
    reset_backend._trt_sessions_lock = lock
    reset_backend.torch_policy = policy
    try:
        reset_backend.reset()
    except Exception as exc:
        checks.append(
            BackendSelfCheckResult(
                name="reset-lock-behavior",
                ok=False,
                detail=f"unexpected reset() failure: {type(exc).__name__}: {exc}",
                category="behavior",
            )
        )
    else:
        checks.append(
            BackendSelfCheckResult(
                name="reset-lock-behavior",
                ok=policy.reset_calls == 1 and lock.enter_count == 1 and lock.exit_count == 1 and not lock.is_held,
                detail=f"reset_calls={policy.reset_calls}, lock_enter={lock.enter_count}, lock_exit={lock.exit_count}",
                category="behavior",
            )
        )

    harness, output, error = _execute_predict_behavior_harness()
    if error is not None:
        detail = f"predict_action_chunk() raised unexpectedly: {type(error).__name__}: {error}"
        checks.extend(
            [
                BackendSelfCheckResult(
                    name="predict-lock-behavior",
                    ok=False,
                    detail=detail,
                    category="behavior",
                ),
                BackendSelfCheckResult(
                    name="predict-stream-sync-before-unlock",
                    ok=False,
                    detail=detail,
                    category="behavior",
                ),
                BackendSelfCheckResult(
                    name="predict-image-token-content-behavior",
                    ok=False,
                    detail=detail,
                    category="behavior",
                ),
            ]
        )
    else:
        expected_sessions = {
            "vit",
            "llm",
            "vlln",
            "state_encoder",
            "action_encoder",
            "dit",
            "action_decoder",
        }
        called_sessions = {name for name, session in harness.sessions.items() if session.calls > 0}
        checks.append(
            BackendSelfCheckResult(
                name="predict-lock-behavior",
                ok=(
                    harness.policy.eval_calls == 1
                    and harness.lock.enter_count == 1
                    and harness.lock.exit_count == 1
                    and not harness.lock.is_held
                    and called_sessions == expected_sessions
                ),
                detail=(
                    f"eval_calls={harness.policy.eval_calls}, lock_enter={harness.lock.enter_count}, "
                    f"lock_exit={harness.lock.exit_count}, sessions={sorted(called_sessions)}, "
                    f"output_shape={tuple(output.shape)}, timeline={_format_behavior_events(harness.events)}"
                ),
                category="behavior",
            )
        )
        checks.append(
            BackendSelfCheckResult(
                name="predict-stream-sync-before-unlock",
                ok=(
                    harness.stream.synchronize_count == 1
                    and _stream_sync_happened_before_unlock(harness.events)
                    and _stream_sync_proves_last_session_completion(harness.events)
                ),
                detail=(
                    f"stream_synchronize_count={harness.stream.synchronize_count}, "
                    f"timeline={_format_behavior_events(harness.events)}"
                ),
                category="behavior",
            )
        )
        wrong_stream_harness, _, wrong_stream_error = _execute_predict_behavior_harness(
            session_stream_name="session-stream",
            sync_stream_name="sync-stream",
        )
        checks.append(
            BackendSelfCheckResult(
                name="predict-stream-binding-detects-wrong-stream",
                ok=(
                    wrong_stream_error is None
                    and wrong_stream_harness.stream.synchronize_count == 1
                    and not _stream_sync_proves_last_session_completion(wrong_stream_harness.events)
                ),
                detail=(
                    f"stream_synchronize_count={wrong_stream_harness.stream.synchronize_count}, "
                    f"timeline={_format_behavior_events(wrong_stream_harness.events)}"
                ),
                category="behavior",
            )
        )
        checks.append(
            BackendSelfCheckResult(
                name="predict-image-token-content-behavior",
                ok=bool(harness.trackers["image_content_validated"]),
                detail="sess_llm observed the exact ViT token payload at image-token slots during predict_action_chunk()",
                category="behavior",
            )
        )

    failed_harness, _, failed_error = _execute_predict_behavior_harness(fail_session="dit")
    checks.append(
        BackendSelfCheckResult(
            name="predict-stream-sync-before-unlock-on-error",
            ok=(
                isinstance(failed_error, RuntimeError)
                and str(failed_error) == "intentional dit failure"
                and failed_harness.stream.synchronize_count == 1
                and _stream_sync_happened_before_unlock(failed_harness.events)
                and _stream_sync_proves_last_session_completion(failed_harness.events)
            ),
            detail=(
                f"error={type(failed_error).__name__ if failed_error else 'None'}:"
                f" {failed_error}, stream_synchronize_count={failed_harness.stream.synchronize_count}, "
                f"timeline={_format_behavior_events(failed_harness.events)}"
            ),
            category="behavior",
        )
    )

    mismatch_harness, _, mismatch_error = _execute_predict_behavior_harness(
        vit_embeds=torch.arange(12, dtype=torch.float16).view(1, 3, 4)
    )
    downstream_calls = {
        name: session.calls for name, session in mismatch_harness.sessions.items() if name != "vit"
    }
    checks.append(
        BackendSelfCheckResult(
            name="predict-image-token-mismatch-entry-behavior",
            ok=(
                isinstance(mismatch_error, ValueError)
                and "predict_action_chunk() rejected LLM input assembly before LLM execution:" in str(mismatch_error)
                and "Image-token slot count must exactly match" in str(mismatch_error)
                and mismatch_harness.sessions["vit"].calls == 1
                and all(call_count == 0 for call_count in downstream_calls.values())
                and mismatch_harness.stream.synchronize_count == 1
                and _stream_sync_happened_before_unlock(mismatch_harness.events)
                and _stream_sync_proves_last_session_completion(mismatch_harness.events)
            ),
            detail=(
                f"error={type(mismatch_error).__name__ if mismatch_error else 'None'}: {mismatch_error}, "
                f"vit_calls={mismatch_harness.sessions['vit'].calls}, downstream_calls={downstream_calls}, "
                f"timeline={_format_behavior_events(mismatch_harness.events)}"
            ),
            category="behavior",
        )
    )

    return checks


def run_backend_self_checks() -> list[BackendSelfCheckResult]:
    """Lightweight checks that keep critical backend guards executable without loading engines."""

    results = [
        BackendSelfCheckResult(
            name="reset-lock-source-tripwire",
            ok=_has_lock_guard("reset"),
            detail="reset() still contains the shared TensorRT lock guard in source form",
            category="static",
        ),
        BackendSelfCheckResult(
            name="predict-lock-source-tripwire",
            ok=_has_lock_guard("predict_action_chunk"),
            detail="predict_action_chunk() still contains the shared TensorRT lock guard in source form",
            category="static",
        ),
        BackendSelfCheckResult(
            name="predict-stream-sync-source-tripwire",
            ok=_has_stream_synchronize_guard(),
            detail="predict_action_chunk() source still contains the explicit CUDA stream synchronize call",
            category="static",
        ),
    ]
    results.extend(_run_lock_and_stream_behavior_self_checks())
    results.extend(_run_image_token_behavior_self_checks())
    return results


def backend_integration_validation_gaps() -> tuple[str, ...]:
    return (
        "Real TensorRT engines and CUDA streams are not exercised here; this remains a mock-backed behavior harness.",
        "Concurrent requests across protocol/server boundaries still need integrated validation with the async server.",
        "Numerical parity and performance under real GR00T TensorRT engines remain out of scope for this self-check.",
    )


class GrootTrtPolicyBackend:
    """Service-side GR00T TensorRT backend exposing chunk-level inference."""

    def __init__(
        self,
        pretrained_name_or_path,
        device,
        engine_dir,
        tensorrt_py_dir,
        vit_dtype,
        llm_dtype,
        dit_dtype,
        num_denoising_steps,
    ) -> None:
        self.policy_path = _resolve_policy_dir(pretrained_name_or_path)
        self.device = _resolve_cuda_device(device)
        self.engine_dir = Path(engine_dir).expanduser().resolve()
        if not self.engine_dir.is_dir():
            raise FileNotFoundError(f"TensorRT engine directory does not exist: {self.engine_dir}")

        if num_denoising_steps is not None and int(num_denoising_steps) <= 0:
            raise ValueError("num_denoising_steps must be positive when provided.")
        self._num_denoising_steps = None if num_denoising_steps is None else int(num_denoising_steps)

        policy_cfg = PreTrainedConfig.from_pretrained(str(self.policy_path))
        if policy_cfg.type != "groot":
            raise ValueError(f"Expected a GR00T checkpoint, got policy type {policy_cfg.type!r}.")
        policy_class = get_policy_class(policy_cfg.type)

        with torch.cuda.device(self.device):
            self.torch_policy = policy_class.from_pretrained(str(self.policy_path), strict=False)
            self.torch_policy.to(self.device)
            self.torch_policy.eval()

        self.config = self.torch_policy.config
        self.groot_model = self.torch_policy._groot_model
        self.backbone = self.groot_model.backbone
        self.action_head = self.groot_model.action_head

        self._prepare_glue_modules()
        self._num_patches = _num_patches(self.backbone)
        self._original_action_dim = int(self.config.output_features[ACTION].shape[0])
        # TensorRT sessions mutate execution context state on every run, so shared backend access must be serialized.
        self._trt_sessions_lock = threading.Lock()
        self._engine_paths = _TrtEnginePaths(
            vit=self.engine_dir / f"vit_{vit_dtype}.engine",
            llm=self.engine_dir / f"llm_{llm_dtype}.engine",
            vlln=self.engine_dir / "vlln_vl_self_attention.engine",
            state_encoder=self.engine_dir / "state_encoder.engine",
            action_encoder=self.engine_dir / "action_encoder.engine",
            dit=self.engine_dir / f"DiT_{dit_dtype}.engine",
            action_decoder=self.engine_dir / "action_decoder.engine",
        )
        self._validate_engine_paths()

        with torch.cuda.device(self.device):
            self.sess_vit = TrtSession.load(self._engine_paths.vit, trt_py_dir=tensorrt_py_dir)
            self.sess_llm = TrtSession.load(self._engine_paths.llm, trt_py_dir=tensorrt_py_dir)
            self.sess_vlln = TrtSession.load(self._engine_paths.vlln, trt_py_dir=tensorrt_py_dir)
            self.sess_state_encoder = TrtSession.load(self._engine_paths.state_encoder, trt_py_dir=tensorrt_py_dir)
            self.sess_action_encoder = TrtSession.load(self._engine_paths.action_encoder, trt_py_dir=tensorrt_py_dir)
            self.sess_dit = TrtSession.load(self._engine_paths.dit, trt_py_dir=tensorrt_py_dir)
            self.sess_action_decoder = TrtSession.load(self._engine_paths.action_decoder, trt_py_dir=tensorrt_py_dir)

    def _prepare_glue_modules(self) -> None:
        self.backbone.eagle_model.mlp1.to(device=self.device, dtype=torch.float16)
        self.backbone.eagle_model.language_model.get_input_embeddings().to(device=self.device, dtype=torch.float16)
        self.action_head.future_tokens.to(device=self.device, dtype=torch.float16)
        if getattr(self.action_head.config, "add_pos_embed", False):
            self.action_head.position_embedding.to(device=self.device, dtype=torch.float16)

    def _validate_engine_paths(self) -> None:
        missing = [path for path in self._engine_paths.values() if not path.is_file()]
        if missing:
            missing_lines = "\n".join(f"  - {path.name}" for path in missing)
            raise FileNotFoundError(
                "Engine directory is missing required TensorRT engine files.\n"
                f"Directory: {self.engine_dir}\n"
                "Missing:\n"
                f"{missing_lines}\n"
            )

    def _require_tensor(self, batch: dict[str, torch.Tensor], key: str, dtype: torch.dtype) -> torch.Tensor:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise KeyError(f"Missing required tensor {key!r} in preprocessed batch.")
        return value.to(device=self.device, dtype=dtype).contiguous()

    def reset(self) -> None:
        with self._trt_sessions_lock:
            if hasattr(self.torch_policy, "reset"):
                self.torch_policy.reset()

    @torch.inference_mode()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with self._trt_sessions_lock:
            self.torch_policy.eval()

            with torch.cuda.device(self.device):
                try:
                    pixel_values = self._require_tensor(batch, "eagle_pixel_values", torch.float16)
                    input_ids = self._require_tensor(batch, "eagle_input_ids", torch.int64)
                    attention_mask = self._require_tensor(batch, "eagle_attention_mask", torch.int64)
                    embodiment_id = self._require_tensor(batch, "embodiment_id", torch.int64)
                    state = self._require_tensor(batch, "state", torch.float16)

                    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
                        raise ValueError(
                            "GrootTrtPolicyBackend expects a single logical batch item after preprocessing; "
                            f"got eagle_input_ids shape {tuple(input_ids.shape)}."
                        )

                    position_ids = (
                        torch.arange(self._num_patches, dtype=torch.int64, device=self.device)
                        .unsqueeze(0)
                        .expand(pixel_values.shape[0], -1)
                        .contiguous()
                    )
                    vit_embeds = self.sess_vit.run({"pixel_values": pixel_values, "position_ids": position_ids})[
                        "vit_embeds"
                    ]
                    vit_embeds = _postprocess_vit(self.backbone, vit_embeds.to(torch.float16))

                    inputs_embeds = _build_predict_action_chunk_inputs_embeds(self.backbone, input_ids, vit_embeds)
                    llm_outputs = self.sess_llm.run(
                        {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
                    )["embeddings"]
                    backbone_features = llm_outputs.to(torch.float16)

                    vl_embs = self.sess_vlln.run({"backbone_features": backbone_features})["output"].to(torch.float16)
                    state_features = self.sess_state_encoder.run(
                        {"state": state, "embodiment_id": embodiment_id}
                    )["output"].to(torch.float16)

                    batch_size = int(vl_embs.shape[0])
                    action_horizon = int(self.action_head.config.action_horizon)
                    action_dim = int(self.action_head.config.action_dim)
                    num_steps = (
                        self._num_denoising_steps
                        if self._num_denoising_steps is not None
                        else int(self.action_head.num_inference_timesteps)
                    )
                    dt = 1.0 / float(num_steps)

                    actions = torch.randn(
                        size=(batch_size, action_horizon, action_dim),
                        dtype=torch.float16,
                        device=self.device,
                    )
                    future_tokens = self.action_head.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1).to(
                        torch.float16
                    )
                    if getattr(self.action_head.config, "add_pos_embed", False):
                        pos_ids = torch.arange(action_horizon, dtype=torch.long, device=self.device)
                        pos_embs = self.action_head.position_embedding(pos_ids).unsqueeze(0).to(torch.float16)
                    else:
                        pos_embs = None

                    for step in range(num_steps):
                        t_cont = step / float(num_steps)
                        t_discretized = int(t_cont * int(self.action_head.num_timestep_buckets))
                        timesteps_tensor = torch.full((batch_size,), t_discretized, dtype=torch.int64, device=self.device)

                        action_features = self.sess_action_encoder.run(
                            {
                                "actions": actions,
                                "timesteps_tensor": timesteps_tensor,
                                "embodiment_id": embodiment_id,
                            }
                        )["output"].to(torch.float16)
                        if pos_embs is not None:
                            action_features = action_features + pos_embs

                        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1).to(torch.float16)
                        model_output = self.sess_dit.run(
                            {
                                "sa_embs": sa_embs,
                                "vl_embs": vl_embs,
                                "timesteps_tensor": timesteps_tensor,
                            }
                        )["output"].to(torch.float16)
                        pred = self.sess_action_decoder.run(
                            {"model_output": model_output, "embodiment_id": embodiment_id}
                        )["output"].to(torch.float16)
                        pred_velocity = pred[:, -action_horizon:, :]
                        actions = actions + dt * pred_velocity

                    return actions[:, :, : self._original_action_dim]
                finally:
                    torch.cuda.current_stream(device=self.device).synchronize()

__all__ = [
    "BackendSelfCheckResult",
    "GrootTrtPolicyBackend",
    "backend_integration_validation_gaps",
    "run_backend_self_checks",
]
