#!/usr/bin/env python3

from pathlib import Path

from get_pic import DEFAULT_CAMERA_URLS, DEFAULT_OUTPUT_DIR, output_path_for_camera


def test_default_rtsp_urls_use_orin_wired_ip() -> None:
    assert DEFAULT_CAMERA_URLS == {
        "camera_head": "rtsp://192.168.1.81:8554/robot_camera/camera_head",
        "camera_left": "rtsp://192.168.1.81:8554/robot_camera/camera_left",
        "camera_right": "rtsp://192.168.1.81:8554/robot_camera/camera_right",
    }


def test_default_output_paths_are_the_three_expected_images() -> None:
    assert DEFAULT_OUTPUT_DIR == Path(__file__).resolve().parent / "output_media"
    assert output_path_for_camera(DEFAULT_OUTPUT_DIR, "camera_head") == DEFAULT_OUTPUT_DIR / "camera_head.jpg"
    assert output_path_for_camera(DEFAULT_OUTPUT_DIR, "camera_left") == DEFAULT_OUTPUT_DIR / "camera_left.jpg"
    assert output_path_for_camera(DEFAULT_OUTPUT_DIR, "camera_right") == DEFAULT_OUTPUT_DIR / "camera_right.jpg"
