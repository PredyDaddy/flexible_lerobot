from __future__ import annotations

from pathlib import Path

from my_devs.jz_robot import run_record_jz_three_realsense


def test_record_runner_can_disable_camera_initialization(monkeypatch) -> None:
    captured = {}

    def fake_record(cfg) -> None:
        captured["cfg"] = cfg

    monkeypatch.setattr(run_record_jz_three_realsense, "record", fake_record)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_record_jz_three_realsense.py",
            "--robot-config",
            "src/lerobot/configs/robot/jz_robot_smooth.yaml",
            "--dataset-repo-id",
            "local/test_no_cameras",
            "--dataset-root",
            "tests/outputs/test_no_cameras",
            "--num-episodes",
            "1",
            "--episode-time-s",
            "1",
            "--reset-time-s",
            "0",
            "--connect-cameras",
            "false",
            "--video",
            "false",
            "--play-sounds",
            "false",
        ],
    )

    run_record_jz_three_realsense.main()

    cfg = captured["cfg"]
    assert cfg.robot.cameras == {}
    assert cfg.dataset.root == Path("tests/outputs/test_no_cameras")
    assert cfg.dataset.video is False
