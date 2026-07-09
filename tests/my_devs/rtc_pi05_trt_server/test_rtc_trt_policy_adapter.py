from __future__ import annotations

from types import SimpleNamespace

import torch

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from my_devs.train.pi.so101.rtc_pi05.trt_server.rtc_trt_policy_adapter import RTCPurePI05TRTPolicyAdapter


class FakeEngine:
    def __init__(self, *, kind: str) -> None:
        self.kind = kind
        self.input_names = []
        self.output_names = []
        self.tensor_dtypes = {}
        self.calls = 0

    def __call__(self, **inputs):  # noqa: ANN001
        self.calls += 1
        if self.kind == "prefix":
            batch_size = inputs["tokens"].shape[0]
            return {"prefix_pad_masks": torch.ones(batch_size, 2, dtype=torch.bool, device=inputs["tokens"].device)}
        return {"v_t": torch.zeros_like(inputs["x_t"])}


class FakeRuntime:
    def __init__(self) -> None:
        self.prefix_engine = FakeEngine(kind="prefix")
        self.denoise_engine = FakeEngine(kind="denoise")
        self.cache_names = ["prefix_pad_masks"]

    @staticmethod
    def _cast_for_engine(_engine, inputs):  # noqa: ANN001
        return inputs


def make_adapter(*, rtc_enabled: bool = True, output_dim: int = 2) -> RTCPurePI05TRTPolicyAdapter:
    adapter = object.__new__(RTCPurePI05TRTPolicyAdapter)
    adapter.runtime = FakeRuntime()
    adapter._device = torch.device("cpu")
    adapter.config = SimpleNamespace(
        chunk_size=4,
        max_action_dim=6,
        num_inference_steps=2,
        output_features={ACTION: SimpleNamespace(shape=(output_dim,))},
        rtc_config=RTCConfig(enabled=rtc_enabled, max_guidance_weight=10.0) if rtc_enabled else None,
    )
    adapter.rtc_processor = None
    adapter.init_rtc_processor()
    adapter._action_queue = []
    return adapter


def test_rtc_guidance_changes_pure_trt_denoise_loop_when_leftover_is_present() -> None:
    adapter = make_adapter(rtc_enabled=True, output_dim=6)
    images = [torch.zeros(1, 3, 224, 224), torch.zeros(1, 3, 224, 224)]
    img_masks = [torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)]
    tokens = torch.ones(1, 3, dtype=torch.int64)
    masks = torch.ones(1, 3, dtype=torch.bool)
    noise = torch.zeros(1, adapter.config.chunk_size, adapter.config.max_action_dim)

    unguided = adapter._sample_actions_rtc_aware(
        images=images,
        img_masks=img_masks,
        tokens=tokens,
        masks=masks,
        noise=noise,
    )
    guided = adapter._sample_actions_rtc_aware(
        images=images,
        img_masks=img_masks,
        tokens=tokens,
        masks=masks,
        noise=noise,
        inference_delay=1,
        prev_chunk_left_over=torch.ones(adapter.config.chunk_size, 6),
        execution_horizon=3,
    )

    assert torch.allclose(unguided, torch.zeros_like(unguided))
    assert not torch.allclose(guided, unguided)
    assert adapter.runtime.denoise_engine.calls == 4


def test_predict_action_chunk_accepts_rtc_kwargs_and_trims_to_action_dim() -> None:
    adapter = make_adapter(rtc_enabled=True, output_dim=2)
    adapter._preprocess_images = lambda _batch: (  # noqa: SLF001
        [torch.zeros(1, 3, 224, 224), torch.zeros(1, 3, 224, 224)],
        [torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)],
    )
    batch = {
        OBS_LANGUAGE_TOKENS: torch.ones(1, 3, dtype=torch.int64),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
    }

    actions = adapter.predict_action_chunk(
        batch,
        inference_delay=1,
        prev_chunk_left_over=torch.ones(4, 2),
        execution_horizon=3,
    )

    assert actions.shape == (1, 4, 2)


def test_select_action_rejects_rtc_enabled_adapter() -> None:
    adapter = make_adapter(rtc_enabled=True)

    try:
        adapter.select_action({})
    except AssertionError as exc:
        assert "predict_action_chunk" in str(exc)
    else:
        raise AssertionError("select_action should reject RTC-enabled pure TensorRT adapter")
