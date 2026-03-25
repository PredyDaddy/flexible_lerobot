#!/usr/bin/env python

from unittest.mock import patch
from lerobot.datasets.video_utils import VideoEncodingManager

from lerobot.teleoperators.jz_command_teleop import JZCommandTeleop, JZCommandTeleopConfig
from lerobot.teleoperators.jz_command_teleop.jz_command_teleop import LEFT, RIGHT


def make_teleop(**overrides) -> JZCommandTeleop:
    cfg = JZCommandTeleopConfig(
        id="test_jz_command_teleop",
        use_gripper=False,
        **overrides,
    )
    return JZCommandTeleop(cfg)


def test_connect_deadline_disabled_for_non_positive_timeout():
    teleop = make_teleop(connect_timeout_s=0.0)
    assert teleop._connect_deadline() is None

    teleop.config.connect_timeout_s = -1.0
    assert teleop._connect_deadline() is None


def test_connect_deadline_uses_monotonic_time_for_positive_timeout():
    teleop = make_teleop(connect_timeout_s=5.0)

    with patch("lerobot.teleoperators.jz_command_teleop.jz_command_teleop.time.monotonic", return_value=10.0):
        assert teleop._connect_deadline() == 15.0


def test_arm_command_ready_requires_first_message_and_all_joints():
    teleop = make_teleop()

    assert not teleop._arm_command_ready(LEFT)

    teleop._last_command_time[LEFT] = 1.0
    teleop._latest_joint_pos[LEFT] = {teleop.config.left_joint_names[0]: 0.0}
    assert not teleop._arm_command_ready(LEFT)

    teleop._latest_joint_pos[LEFT] = {joint: float(idx) for idx, joint in enumerate(teleop.config.left_joint_names)}
    assert teleop._arm_command_ready(LEFT)


def test_format_initial_command_status_reports_ready_and_waiting_topics():
    teleop = make_teleop()
    teleop._last_command_time[LEFT] = 1.0
    teleop._latest_joint_pos[LEFT] = {joint: float(idx) for idx, joint in enumerate(teleop.config.left_joint_names)}

    status = teleop._format_initial_command_status()

    assert f"{LEFT}_arm[{teleop.config.left_command_topic}]=ready" in status
    assert f"{RIGHT}_arm[{teleop.config.right_command_topic}]=waiting_first_message" in status

class _DummyMeta:
    video_keys = ["observation.images.chest"]


class _DummyDataset:
    def __init__(self, root):
        self.root = root
        self.meta = _DummyMeta()
        self.num_episodes = 0
        self.episodes_since_last_encoding = 0
        self.finalize_calls = 0

    def finalize(self):
        self.finalize_calls += 1

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int):
        return self.root / "images" / image_key / f"episode-{episode_index:06d}" / f"frame-{frame_index:06d}.png"


def test_video_encoding_manager_does_not_mask_original_exception_when_cleanup_fails(tmp_path):
    dataset = _DummyDataset(tmp_path)
    interrupted_dir = dataset._get_image_file_path(0, dataset.meta.video_keys[0], 0).parent
    interrupted_dir.mkdir(parents=True)
    (interrupted_dir / "frame-000000.png").write_bytes(b"test")

    manager = VideoEncodingManager(dataset)

    with patch(
        "lerobot.datasets.video_utils.shutil.rmtree",
        side_effect=OSError("Directory not empty"),
    ):
        assert manager.__exit__(RuntimeError, RuntimeError("stale command"), None) is False

    assert dataset.finalize_calls == 1
