import os

from lerobot.robots.jz_robot.bridge_capture.config import CameraConfig
from lerobot.robots.jz_robot.bridge_capture.rtsp_receiver import configure_opencv_rtsp_environment


def test_bridge_capture_rtsp_receiver_configures_low_latency_tcp_options(monkeypatch):
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    camera = CameraConfig(
        rtsp_url="rtsp://192.168.50.10:8554/robot_camera/camera_head",
        width=1280,
        height=720,
        fps=30,
        transport="tcp",
    )

    configure_opencv_rtsp_environment(camera)

    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == (
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
    )
