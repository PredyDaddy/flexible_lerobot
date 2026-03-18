from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from my_devs.act_trt.run_act_trt_mock import main as run_mock_main

INFER_ALIAS_MODULE = (
    __spec__.name if __spec__ is not None and __spec__.name else "my_devs.act_trt.run_act_trt_infer"
)


def main() -> int:
    """Compatibility entrypoint kept for CLI parity; execution stays mock-only."""
    return run_mock_main(
        cli_entrypoint_path=Path(__file__).resolve(),
        cli_entrypoint_module=INFER_ALIAS_MODULE,
    )


if __name__ == "__main__":
    raise SystemExit(main())
