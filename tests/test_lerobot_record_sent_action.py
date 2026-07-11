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

    def __init__(self, event_log: list[str] | None = None) -> None:
        self.frames = []
        self.event_log = event_log

    def create_episode_buffer(self) -> dict:
        return {"size": 0}

    def add_frame(self, frame: dict) -> None:
        if self.event_log is not None:
            self.event_log.append("add_frame")
        self.frames.append(frame)
        self.episode_buffer["size"] += 1


class FailingDataset(FakeDataset):
    def add_frame(self, frame: dict) -> None:
        raise RuntimeError("frame write failed")


class FakeRobot:
    cameras = {}

    def __init__(self) -> None:
        self.sent_actions = []

    def get_observation(self) -> dict[str, float]:
        return {"joint.pos": 0.0}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        assert action == {"joint.pos": 9.0}
        self.sent_actions.append(action)
        return {"joint.pos": 1.0}


def test_record_loop_saves_and_displays_action_returned_by_robot(monkeypatch) -> None:
    dataset = FakeDataset()
    teleop = Mock(spec=Teleoperator)
    teleop.get_action.return_value = {"joint.pos": 9.0}
    rerun_calls = []
    spoken_events = []
    monkeypatch.setattr(lerobot_record, "log_rerun_data", lambda **kwargs: rerun_calls.append(kwargs))
    monkeypatch.setattr(
        lerobot_record,
        "log_say",
        lambda text, play_sounds: spoken_events.append((text, play_sounds)),
    )

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
        episode_index=0,
        play_sounds=True,
    )

    assert len(dataset.frames) == 1
    assert dataset.frames[0]["action"].tolist() == [1.0]
    assert len(rerun_calls) == 1
    assert rerun_calls[0]["action"] == {"joint.pos": 1.0}
    assert spoken_events == [("Start recording episode 1", True)]


def test_record_loop_announces_only_after_first_frame_is_buffered(monkeypatch) -> None:
    event_log = []
    dataset = FakeDataset(event_log)
    robot = FakeRobot()
    teleop = Mock(spec=Teleoperator)
    events = {"exit_early": False}
    action_count = 0

    def get_action() -> dict[str, float]:
        nonlocal action_count
        action_count += 1
        if action_count == 2:
            events["exit_early"] = True
        return {"joint.pos": 9.0}

    teleop.get_action.side_effect = get_action
    monkeypatch.setattr(
        lerobot_record,
        "log_say",
        lambda text, play_sounds: event_log.append(f"announce:{text}:{play_sounds}"),
    )

    lerobot_record.record_loop(
        robot=robot,
        events=events,
        fps=1000,
        teleop_action_processor=lambda value: value[0],
        robot_action_processor=lambda value: value[0],
        robot_observation_processor=lambda value: value,
        dataset=dataset,
        teleop=teleop,
        control_time_s=10.0,
        single_task="test",
        episode_index=1,
        play_sounds=True,
    )

    assert len(dataset.frames) == 2
    assert event_log == ["add_frame", "announce:Start recording episode 2:True", "add_frame"]


def test_reset_loop_keeps_sending_teleop_actions_without_saving_frames(monkeypatch) -> None:
    robot = FakeRobot()
    teleop = Mock(spec=Teleoperator)
    teleop.get_action.return_value = {"joint.pos": 9.0}
    spoken_events = []
    monkeypatch.setattr(
        lerobot_record,
        "log_say",
        lambda text, play_sounds: spoken_events.append((text, play_sounds)),
    )

    lerobot_record.record_loop(
        robot=robot,
        events={"exit_early": False},
        fps=1000,
        teleop_action_processor=lambda value: value[0],
        robot_action_processor=lambda value: value[0],
        robot_observation_processor=lambda value: value,
        dataset=None,
        teleop=teleop,
        control_time_s=0.0001,
        single_task="test",
    )

    assert robot.sent_actions == [{"joint.pos": 9.0}]
    assert spoken_events == []


def test_record_loop_does_not_announce_when_first_frame_write_fails(monkeypatch) -> None:
    dataset = FailingDataset()
    teleop = Mock(spec=Teleoperator)
    teleop.get_action.return_value = {"joint.pos": 9.0}
    spoken_events = []
    monkeypatch.setattr(
        lerobot_record,
        "log_say",
        lambda text, play_sounds: spoken_events.append((text, play_sounds)),
    )

    try:
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
            episode_index=0,
            play_sounds=True,
        )
    except RuntimeError as error:
        assert str(error) == "frame write failed"
    else:
        raise AssertionError("record_loop should propagate the frame write failure")

    assert spoken_events == []
