---
name: jz-timed-dataset-curation
description: Audit interrupted or retried JZ Robot Pin timed LeRobot v3 datasets, classify completed episodes, preserve every source directory, merge only validated production datasets, restore JZ timing/training-schema metadata, and verify the curated result. Use when working with tests/outputs/jz_robot_pin_timed_* recordings that may contain incomplete episodes, orphan timing sidecars, broken resume metadata, camera transport anomalies, or multiple valid datasets that need one training dataset.
---

# JZ Timed Dataset Curation

Perform all work offline in `lerobot_flex`. Never start state bridges, command executors, joystick,
armed recorder, replay, ROS services, or robot commands during curation.

## Safety and preservation rules

- Treat every source directory as immutable. Never delete, rename, move, or merge in place.
- Require a new, nonexistent output directory. Stop if it already exists.
- Do not infer validity from the directory name or requested episode count.
- Ignore orphan timing only when its `(episode_index, frame_index)` is absent from saved Parquet.
- Reject a source if saved Parquet, standard episode metadata, or referenced video cannot be mapped
  exactly. Do not reconstruct timestamps or video ownership by guessing.
- Exclude color-diagnostic/RTSP/alternate-CRF datasets from a homogeneous production ZMQ/CRF18
  merge even when they are structurally readable.
- Do not use generic `split_dataset` or `delete_episodes` on shared MP4 files without separately
  proving the output codec. The current partial-video path can re-encode to AV1 while metadata still
  describes the source codec.
- Do not use `LeRobotDataset(..., episodes=...)` directly with `merge_datasets`; the merge API reloads
  the full root and ignores the in-memory episode subset.

Read [references/quality-gates.md](references/quality-gates.md) before changing thresholds or
handling a partially valid saved dataset.

## 1. Establish the candidate list

For each requested root, record:

- `meta/info.json` total episodes/frames/FPS/robot type/video encoding;
- Parquet episode IDs, frame counts, contiguous frame indexes and timestamp grid;
- standard `meta/episodes` IDs and video location columns;
- timing sidecar keys versus saved Parquet keys;
- referenced MP4 existence, codec and decodability;
- `meta/jz_pin_training_schema.json` source semantics.

Classify every root as one of:

- `ACCEPT`: every saved episode passes all gates and standard metadata is complete;
- `ACCEPT_SAVED_ONLY`: saved episodes pass, but unsaved interrupted sidecars remain; merge only the
  standard dataset and copy timing keys that occur in Parquet;
- `EXCLUDE`: zero completed episodes, diagnostic acquisition, schema/transport mismatch, failed
  episode quality, corrupt video, or ambiguous data/meta/video ownership.

## 2. Run repository quality checks

Use the actual saved episode count rather than the directory name:

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python \
  my_devs/jz_robot_pin_timed/data_check/check_3_episodes.py \
  --dataset-root <root> \
  --expected-episodes <saved_count> \
  --expected-episode-time-s 10 \
  --expected-fps 20 \
  --max-initial-joint-delta-rad 10 \
  --max-action-joint-step-rad 10 \
  --max-lag-p95-rad 0.05 \
  --report-json /tmp/<name>.data.json
```

Run strict timing validation for production ZMQ/CRF18 data:

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python \
  my_devs/jz_robot_pin_timed/data_check/check_timing.py \
  --dataset-root <root> \
  --expected-codec h264 --expected-crf 18 \
  --expected-camera-fps 20 --expected-camera-source-fps 30 \
  --min-camera-source-fps-ratio 0.9 \
  --expected-camera-protocol jz_realsense_zmq \
  --expected-command-mode armed --expected-command-transport udp \
  --expected-action-key-count 18 --require-source-timing \
  --max-source-age-ms 50 --max-source-skew-ms 20 \
  --max-camera-age-ms 1000 --max-camera-state-skew-ms 100 \
  --report-json /tmp/<name>.timing.json
```

Run the training boundary checker with the dataset manifest. Require strict trainability; do not pass
`--allow-unavailable` for the curated training dataset.

Treat an error containing only `timing contains ... frames absent from the dataset` as an interrupted
orphan-sidecar condition only after confirming every saved Parquet key has valid timing. The merge
script enforces this asymmetry and drops only those extras.

## 3. Review color stability

Run the full-frame scanner on accepted sources:

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python \
  my_devs/jz_robot_pin_timed/skill/jz-timed-dataset-curation/scripts/scan_video_color.py \
  --dataset-root <root> \
  --max-chromaticity-step 0.008 \
  --report-json /tmp/<name>.color.json
```

`REVIEW` is not an automatic rejection. Extract the reported frame and its predecessor. Reject
whole-frame white-balance/color-state changes; retain ordinary object motion, exposure changes, or a
strongly colored object moving close to a gripper camera. Record the human decision.

## 4. Preflight and merge accepted full sources

Pass only `ACCEPT` and `ACCEPT_SAVED_ONLY` roots whose standard saved episodes all pass. First run:

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python \
  my_devs/jz_robot_pin_timed/skill/jz-timed-dataset-curation/scripts/merge_valid_datasets.py \
  --source-root <accepted-root-1> \
  --source-root <accepted-root-2> \
  --output-root <new-output-root> \
  --output-repo-id local/<output-name> \
  --expected-codec h264 --expected-crf 18 \
  --sample-frames-per-episode 3 \
  --preflight-only \
  --report-json /tmp/<output-name>.preflight.json
```

Proceed only on `PREFLIGHT_PASS`. Remove `--preflight-only` and `--report-json` to create the output.
The script invokes repository `merge_datasets`, then restores/reindexes timing sidecars, writes the
explicit training schema and curation episode map, restores verified `video_encoding`, runs ffprobe,
and decodes three frames per episode/camera. It refuses an existing output directory.

## 5. Validate the merged dataset

Run the three repository checkers again with `--expected-episodes` equal to the merged count. Then
fully decode every output MP4 without allowing ffmpeg to consume the filename pipe:

```bash
mapfile -d '' VIDEOS < <(find <output-root>/videos -type f -name '*.mp4' -print0)
for video in "${VIDEOS[@]}"; do
  ffmpeg -nostdin -v error -i "$video" -f null -
done
```

Require:

- data checker `PASS`;
- timing checker `PASS`, all data keys matched, source timing valid for every frame, sequence gaps 0;
- projection checker `PASS`, raw18 preserved, model16 shape valid, force indices 15/17 excluded;
- full video decode exit 0 for every MP4;
- `meta/jz_pin_curation_report.json` status `MERGE_PASS`;
- source directories still exist unchanged.

## 6. Report results

Return a table containing each source root, saved episodes, classification, accepted episode range,
frame count, exclusion reason, timing/projection status and color review. Report merged root, repo ID,
episode/frame totals, quality report paths, warnings, and any policy decisions such as excluding
diagnostic RTSP data.

For the 2026-07-13 curation performed with this workflow, read
[references/current-curation.md](references/current-curation.md).
