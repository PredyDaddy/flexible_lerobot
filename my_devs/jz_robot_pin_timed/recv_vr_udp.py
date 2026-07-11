#!/usr/bin/env python3
"""Run the shared Pin VR UDP packet diagnostic without duplicating it."""

import runpy
from pathlib import Path


TARGET = Path(__file__).resolve().parents[1] / "jz_robot_pin" / "recv_vr_udp.py"
if not TARGET.is_file():
    raise FileNotFoundError(f"Shared VR diagnostic not found: {TARGET}")

runpy.run_path(str(TARGET), run_name="__main__")

