# JZRobot Cross-Machine Capture Status

Date: 2026-06-24

## Current State

The cross-machine capture work is in progress. The implemented path currently
captures raw episodes on the collector machine and validates their basic shape.
It does not yet provide a complete raw-to-LeRobot v3 conversion workflow.

Implemented pieces:

- `jz-lerobot-vector-bridge` aggregates robot-side ROS 2 topics into fixed-order
  `/robot1/lerobot/state` and `/robot1/lerobot/action` vectors.
- `jz-lerobot-raw-recorder` records collector-side raw episodes from those ROS
  vector topics and RTSP camera streams.
- `jz-lerobot-validate-raw` performs basic validation for raw episode files.
- `src/lerobot/configs/robot/jz_bridge_capture.yaml` defines the current robot,
  camera, state, and action topic layout.

## Verified Locally

The bridge capture Python files compile with:

```bash
PYTHONPATH=src python -m compileall -q src/lerobot/robots/jz_robot/bridge_capture tests/robots/jz_robot_bridge_capture
```

Full pytest execution is currently blocked by environment dependencies before
the bridge capture tests run:

- ROS pytest plugin path requires `lark`.
- Repository `tests/conftest.py` requires `pyserial` (`serial` module).
- Importing through `lerobot.robots` requires `draccus`.

These are environment setup blockers, not confirmed bridge capture test
failures.

## Remaining Work

Required before calling the workflow complete:

1. Run ROS-only and RTSP-enabled trial captures on the real two-machine setup.
2. Run a 10-minute acceptance capture and check frame rate, invalid sample
   ratio, and state/action/image alignment.
3. Replace the current OpenCV RTSP receiver with a GStreamer appsink receiver if
   OpenCV capture is not stable enough on the real network.

## Implementation Direction

The recorder should continue to write raw episodes first. Conversion should be a
separate command that reads completed raw episode directories, rejects invalid
data, and writes a LeRobot v3 dataset with `LeRobotDataset.create`,
`add_frame`, and `save_episode`.

The offline converter command is:

```bash
jz-lerobot-convert-raw \
  --raw-root /home/test/datasets/jz_raw \
  --repo-id local/jz_bridge_capture \
  --output-root /home/test/data/lerobot/jz_bridge_capture
```

The converter should skip samples marked invalid by default, preserve the task
text, write `observation.state`, `action`, and one `observation.images.<camera>`
feature per raw camera, then finalize the resulting dataset.
