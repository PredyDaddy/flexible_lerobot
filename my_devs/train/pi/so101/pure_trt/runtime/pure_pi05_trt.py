from __future__ import annotations

import sys
from pathlib import Path

from my_devs.train.pi.so101.pure_trt.runtime.paths import LEGACY_OPENPI_TRT_ROOT, REPO_ROOT


for _path in (REPO_ROOT, LEGACY_OPENPI_TRT_ROOT):
    if _path.as_posix() not in sys.path:
        sys.path.insert(0, _path.as_posix())

from runtime.pure_pi05_trt import PurePI05TRTPolicyAdapter, PurePI05TRTProfile, PurePI05TRTRuntime  # noqa: E402
from runtime.trt_engine import TorchTensorRTEngine  # noqa: E402


__all__ = [
    "PurePI05TRTPolicyAdapter",
    "PurePI05TRTProfile",
    "PurePI05TRTRuntime",
    "TorchTensorRTEngine",
]
