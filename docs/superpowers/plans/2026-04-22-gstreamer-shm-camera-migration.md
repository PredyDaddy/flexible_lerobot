# GStreamer SHM Camera Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ROS2 image-topic camera consumption in mainline `flexible_lerobot` JZRobot recording with a GStreamer shared-memory camera backend so one upstream capture pipeline can feed both RoboWeb preview and LeRobot recording without ROS image topics.

**Architecture:** Add a dedicated `gstreamer_shm` camera backend under `src/lerobot/cameras/` that consumes `shmsrc` streams through OpenCV's GStreamer backend and exposes the same `Camera` interface used by `JZRobot.get_observation()`. Migrate the JZRobot example config from `ros2_topic` cameras to `gstreamer_shm`, keep arm and gripper state on ROS2, and document the external producer contract as an explicit prerequisite.

**Tech Stack:** Python 3.10, OpenCV, GStreamer (`shmsrc`, `shmsink`, `appsink`), draccus, pytest

---

## Scope

- In scope: `src/lerobot` camera backend, robot config example, parser wiring, tests, operator docs.
- Out of scope: RoboWeb media-bridge implementation, browser playback, and `my_devs/cobot_magic/collect_data.py`.
- Important boundary: `my_devs/cobot_magic/collect_data.py` currently synchronizes on ROS image `header.stamp`; that legacy collector needs a separate follow-up plan if it still matters operationally.

## External Producer Contract

The repository change assumes an external capture process already owns each physical camera and publishes one shared-memory stream per camera socket path.

Baseline producer contract:

```bash
gst-launch-1.0 \
  v4l2src device=/dev/video0 ! \
  video/x-raw,format=BGR,width=640,height=480,framerate=30/1 ! \
  queue leaky=downstream max-size-buffers=1 ! \
  shmsink socket-path=/tmp/lerobot_chest.sock wait-for-connection=false sync=false shm-size=67108864
```

The backend in this plan will consume either:

- the default raw BGR shared-memory contract above, or
- a caller-provided custom pipeline template when the producer differs.

## File Structure

- Create: `src/lerobot/cameras/gstreamer_shm/configuration_gstreamer_shm.py`
  Responsibility: define the typed `gstreamer_shm` camera config, validate shared-memory parameters, and support a custom pipeline template.
- Create: `src/lerobot/cameras/gstreamer_shm/camera_gstreamer_shm.py`
  Responsibility: open a `shmsrc` consumer via `cv2.CAP_GSTREAMER`, cache the latest frame, and implement `connect/read/async_read/disconnect`.
- Create: `src/lerobot/cameras/gstreamer_shm/__init__.py`
  Responsibility: export the new camera and config class.
- Modify: `src/lerobot/cameras/utils.py`
  Responsibility: instantiate `GStreamerSHMCamera` from camera configs.
- Modify: `src/lerobot/cameras/__init__.py`
  Responsibility: import and expose the new camera config so parser registration happens on normal package import paths.
- Modify: `src/lerobot/scripts/lerobot_record.py`
  Responsibility: import `GStreamerSHMCameraConfig` for CLI and config parsing parity with other camera backends.
- Modify: `src/lerobot/scripts/lerobot_teleoperate.py`
  Responsibility: import `GStreamerSHMCameraConfig` for CLI and config parsing parity.
- Modify: `src/lerobot/scripts/lerobot_calibrate.py`
  Responsibility: import `GStreamerSHMCameraConfig` for CLI and config parsing parity.
- Modify: `src/lerobot/__init__.py`
  Responsibility: add `gstreamer_shm` to `available_cameras`.
- Create: `src/lerobot/configs/robot/jz_robot_three_realsense_gstreamer_shm.yaml`
  Responsibility: provide the no-ROS-image JZRobot example config using shared-memory camera sockets.
- Create: `docs/tools/gstreamer_shm_camera.md`
  Responsibility: document the external producer contract, socket naming, and record-time usage.
- Create: `tests/cameras/test_gstreamer_shm_camera.py`
  Responsibility: cover config validation, pipeline construction, connection behavior, timeout handling, and color conversion.
- Create: `tests/robots/test_jz_robot_gstreamer_config.py`
  Responsibility: verify draccus parses JZRobot YAML with `gstreamer_shm` cameras and that the example config stays valid.

### Task 1: Add the failing camera-backend tests

**Files:**
- Create: `tests/cameras/test_gstreamer_shm_camera.py`
- Test: `tests/cameras/test_gstreamer_shm_camera.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from lerobot.cameras.utils import make_cameras_from_configs


def test_make_cameras_from_configs_builds_gstreamer_shm_camera():
    from lerobot.cameras.gstreamer_shm import GStreamerSHMCameraConfig

    cfg = GStreamerSHMCameraConfig(
        socket_path="/tmp/lerobot_head.sock",
        width=640,
        height=480,
        fps=30,
    )

    cameras = make_cameras_from_configs({"head": cfg})

    assert cameras["head"].__class__.__name__ == "GStreamerSHMCamera"


def test_default_pipeline_uses_shmsrc_and_appsink():
    from lerobot.cameras.gstreamer_shm import GStreamerSHMCamera, GStreamerSHMCameraConfig

    cfg = GStreamerSHMCameraConfig(
        socket_path="/tmp/lerobot_head.sock",
        width=640,
        height=480,
        fps=30,
    )

    camera = GStreamerSHMCamera(cfg)
    pipeline = camera._build_pipeline()

    assert "shmsrc socket-path=/tmp/lerobot_head.sock" in pipeline
    assert "appsink" in pipeline


def test_connect_uses_gstreamer_backend(monkeypatch):
    import cv2
    from lerobot.cameras.gstreamer_shm import GStreamerSHMCamera, GStreamerSHMCameraConfig

    opened = {}

    class FakeCapture:
        def __init__(self, source, backend):
            opened["source"] = source
            opened["backend"] = backend

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            opened["released"] = True

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    camera = GStreamerSHMCamera(
        GStreamerSHMCameraConfig(
            socket_path="/tmp/lerobot_head.sock",
            width=640,
            height=480,
            fps=30,
        )
    )

    camera.connect(warmup=False)

    assert opened["backend"] == cv2.CAP_GSTREAMER
    assert "shmsrc socket-path=/tmp/lerobot_head.sock" in opened["source"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/cameras/test_gstreamer_shm_camera.py -q`
Expected: FAIL because `lerobot.cameras.gstreamer_shm` does not exist yet

- [ ] **Step 3: Commit**

```bash
git add tests/cameras/test_gstreamer_shm_camera.py
git commit -m "test(camera): add gstreamer shm camera coverage"
```

### Task 2: Implement the `gstreamer_shm` camera backend

**Files:**
- Create: `src/lerobot/cameras/gstreamer_shm/configuration_gstreamer_shm.py`
- Create: `src/lerobot/cameras/gstreamer_shm/camera_gstreamer_shm.py`
- Create: `src/lerobot/cameras/gstreamer_shm/__init__.py`
- Modify: `src/lerobot/cameras/utils.py`
- Modify: `src/lerobot/cameras/__init__.py`
- Modify: `tests/cameras/test_gstreamer_shm_camera.py`

- [ ] **Step 1: Write the typed config and pipeline template builder**

```python
@CameraConfig.register_subclass("gstreamer_shm")
@dataclass
class GStreamerSHMCameraConfig(CameraConfig):
    socket_path: str
    color_mode: ColorMode = ColorMode.RGB
    timeout_ms: int = 5000
    warmup_s: float = 0.0
    pipeline_template: str | None = None

    def __post_init__(self) -> None:
        if not self.socket_path:
            raise ValueError("`socket_path` cannot be empty.")
        if self.timeout_ms <= 0:
            raise ValueError("`timeout_ms` must be positive.")
```

- [ ] **Step 2: Implement the OpenCV + GStreamer camera**

```python
DEFAULT_PIPELINE_TEMPLATE = (
    "shmsrc socket-path={socket_path} is-live=true do-timestamp=true ! "
    "video/x-raw,width={width},height={height},framerate={fps}/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)


def _build_pipeline(self) -> str:
    template = self.config.pipeline_template or DEFAULT_PIPELINE_TEMPLATE
    return template.format(
        socket_path=self.config.socket_path,
        width=self.width,
        height=self.height,
        fps=self.fps,
    )
```

- [ ] **Step 3: Match the existing `Camera` interface with latest-frame caching**

```python
def connect(self, warmup: bool = True) -> None:
    self.videocapture = cv2.VideoCapture(self._build_pipeline(), cv2.CAP_GSTREAMER)
    if not self.videocapture.isOpened():
        raise ConnectionError(f"Failed to open {self}")
    frame = self.read()
    self.height, self.width = frame.shape[:2]
    if warmup and self.warmup_s > 0:
        time.sleep(self.warmup_s)


def async_read(self, timeout_ms: float | None = None) -> NDArray[Any]:
    if not self.thread or not self.thread.is_alive():
        self._start_read_thread()
    ...
```

- [ ] **Step 4: Run targeted tests**

Run: `pytest tests/cameras/test_gstreamer_shm_camera.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/cameras/gstreamer_shm src/lerobot/cameras/utils.py src/lerobot/cameras/__init__.py tests/cameras/test_gstreamer_shm_camera.py
git commit -m "feat(camera): add gstreamer shm backend"
```

### Task 3: Wire parser registration and migrate the JZRobot example config

**Files:**
- Modify: `src/lerobot/scripts/lerobot_record.py`
- Modify: `src/lerobot/scripts/lerobot_teleoperate.py`
- Modify: `src/lerobot/scripts/lerobot_calibrate.py`
- Modify: `src/lerobot/__init__.py`
- Create: `src/lerobot/configs/robot/jz_robot_three_realsense_gstreamer_shm.yaml`
- Create: `docs/tools/gstreamer_shm_camera.md`
- Create: `tests/robots/test_jz_robot_gstreamer_config.py`

- [ ] **Step 1: Write the failing YAML parse test**

```python
from __future__ import annotations

from pathlib import Path

import draccus

from lerobot.robots.jz_robot import JZRobotConfig


def test_jz_robot_gstreamer_shm_yaml_parses(tmp_path: Path):
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(
        """
type: jz_robot
id: jz_robot_three_realsense_gstreamer_shm
left_joint_state_topic: /robot1/arm_left/joint_states
right_joint_state_topic: /robot1/arm_right/joint_states
left_position_command_topic: /robot1/telecon/arm_left/joint_commands_input
right_position_command_topic: /robot1/telecon/arm_right/joint_commands_input
cameras:
  chest:
    type: gstreamer_shm
    socket_path: /tmp/lerobot_chest.sock
    width: 640
    height: 480
    fps: 30
""".strip()
    )

    cfg = draccus.parse(config_class=JZRobotConfig, config_path=config_path, args=[])

    assert cfg.cameras["chest"].type == "gstreamer_shm"
    assert cfg.cameras["chest"].socket_path == "/tmp/lerobot_chest.sock"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/robots/test_jz_robot_gstreamer_config.py -q`
Expected: FAIL because `gstreamer_shm` is not registered on the normal parser import path yet

- [ ] **Step 3: Import the new config class in the CLI entry points and add the example YAML**

```python
from lerobot.cameras.gstreamer_shm.configuration_gstreamer_shm import GStreamerSHMCameraConfig  # noqa: F401
```

```yaml
cameras:
  chest:
    type: gstreamer_shm
    socket_path: /tmp/lerobot_chest.sock
    width: 640
    height: 480
    fps: 30
  left_arm:
    type: gstreamer_shm
    socket_path: /tmp/lerobot_left_arm.sock
    width: 848
    height: 480
    fps: 30
  right_arm:
    type: gstreamer_shm
    socket_path: /tmp/lerobot_right_arm.sock
    width: 848
    height: 480
    fps: 30
```

- [ ] **Step 4: Document the operator contract and usage**

```markdown
1. Start one producer pipeline per physical camera.
2. Point the robot YAML at the corresponding SHM socket paths.
3. Keep ROS2 only for arm, gripper, and base state topics.
4. Do not use this plan for `my_devs/cobot_magic/collect_data.py`; that script still depends on ROS image timestamps.
```

- [ ] **Step 5: Run targeted tests**

Run: `pytest tests/robots/test_jz_robot_gstreamer_config.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/lerobot/scripts/lerobot_record.py src/lerobot/scripts/lerobot_teleoperate.py src/lerobot/scripts/lerobot_calibrate.py src/lerobot/__init__.py src/lerobot/configs/robot/jz_robot_three_realsense_gstreamer_shm.yaml docs/tools/gstreamer_shm_camera.md tests/robots/test_jz_robot_gstreamer_config.py
git commit -m "feat(jz_robot): add gstreamer shm camera config"
```

### Task 4: Verify the migration end to end at the repository level

**Files:**
- Modify: `tests/cameras/test_gstreamer_shm_camera.py`
- Modify: `tests/robots/test_jz_robot_gstreamer_config.py`

- [ ] **Step 1: Add timeout and custom-pipeline coverage if it is still missing**

```python
def test_async_read_uses_configured_timeout():
    ...


def test_pipeline_template_can_override_default():
    ...
```

- [ ] **Step 2: Run the camera and robot test bundle**

Run: `pytest tests/cameras/test_gstreamer_shm_camera.py tests/robots/test_jz_robot_gstreamer_config.py tests/robots/test_jz_robot_initial_state.py -q`
Expected: PASS

- [ ] **Step 3: Run an import and syntax smoke check**

Run: `python -m py_compile src/lerobot/cameras/gstreamer_shm/configuration_gstreamer_shm.py src/lerobot/cameras/gstreamer_shm/camera_gstreamer_shm.py tests/cameras/test_gstreamer_shm_camera.py tests/robots/test_jz_robot_gstreamer_config.py`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/cameras/test_gstreamer_shm_camera.py tests/robots/test_jz_robot_gstreamer_config.py
git commit -m "chore(camera): verify gstreamer shm migration"
```
