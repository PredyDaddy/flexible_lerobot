from __future__ import annotations

import importlib
import re

import pytest


def test_known_leader_port_is_rejected_without_explicit_override() -> None:
    entrypoint = importlib.import_module("my_devs.train.pi.so101.rtc_pi05.run_rtc_pi05_infer")

    with pytest.raises(ValueError, match=re.escape("known leader/main-arm port")):
        entrypoint.validate_robot_port(
            "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123582-if00",
            allow_known_non_follower=False,
        )


def test_known_leader_port_can_only_be_used_with_override() -> None:
    entrypoint = importlib.import_module("my_devs.train.pi.so101.rtc_pi05.run_rtc_pi05_infer")

    entrypoint.validate_robot_port(
        "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123582-if00",
        allow_known_non_follower=True,
    )
