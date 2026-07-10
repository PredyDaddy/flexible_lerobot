from __future__ import annotations

from unittest.mock import Mock

from lerobot.scripts import lerobot_record
from lerobot.teleoperators.teleoperator import Teleoperator


class FakeDataset:
    fps = 1000
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["joint.pos"],
        },
        "action": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["joint.pos"],
        },
    }
    num_episodes = 0
    episode_buffer = None

    def __init__(self) -> None:
        self.frames = []

    def create_episode_buffer(self) -> dict:
        return {"size": 0}

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)
        self.episode_buffer["size"] += 1


class FakeRobot:
    cameras = {}

    def get_observation(self) -> dict[str, float]:
        return {"joint.pos": 0.0}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        assert action == {"joint.pos": 9.0}
        return {"joint.pos": 1.0}


def test_record_loop_saves_and_displays_action_returned_by_robot(monkeypatch) -> None:
    dataset = FakeDataset()
    teleop = Mock(spec=Teleoperator)
    teleop.get_action.return_value = {"joint.pos": 9.0}
    rerun_calls = []
    monkeypatch.setattr(lerobot_record, "log_rerun_data", lambda **kwargs: rerun_calls.append(kwargs))

    lerobot_record.record_loop(
        robot=FakeRobot(),
        events={"exit_early": False},
        fps=dataset.fps,
        teleop_action_processor=lambda value: value[0],
        robot_action_processor=lambda value: value[0],
        robot_observation_processor=lambda value: value,
        dataset=dataset,
        teleop=teleop,
        control_time_s=0.0001,
        single_task="test",
        display_data=True,
    )

    assert len(dataset.frames) == 1
    assert dataset.frames[0]["action"].tolist() == [1.0]
    assert len(rerun_calls) == 1
    assert rerun_calls[0]["action"] == {"joint.pos": 1.0}
