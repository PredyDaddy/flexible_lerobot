#!/usr/bin/env python

from __future__ import annotations

import runpy
import sys
from pathlib import Path


LEGACY_CHECKER = (
    Path(__file__).resolve().parents[2] / "jz_robot_pin" / "data_check" / "check_3_episodes.py"
)


def _has_expected_robot_type_argument(arguments: list[str]) -> bool:
    return any(
        argument == "--expected-robot-type" or argument.startswith("--expected-robot-type=")
        for argument in arguments
    )


if __name__ == "__main__":
    if not _has_expected_robot_type_argument(sys.argv[1:]):
        sys.argv.extend(["--expected-robot-type", "jz_robot_pin_timed"])
    runpy.run_path(str(LEGACY_CHECKER), run_name="__main__")
