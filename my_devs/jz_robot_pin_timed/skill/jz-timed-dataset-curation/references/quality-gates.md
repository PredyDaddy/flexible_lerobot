# Quality gates

## Contents

1. Episode and metadata integrity
2. Motion/state quality
3. Timing quality
4. Video quality
5. Training schema
6. Partial-source policy

## Episode and metadata integrity

- Require `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`, data Parquet and standard episode
  metadata.
- Require saved episode IDs contiguous from zero and equal across info, Parquet and standard metadata.
- Require frame indexes contiguous from zero and timestamps within 1 ms of `frame_index/fps`.
- At 20 FPS for a nominal 10-second episode, require at least 180 frames.
- Do not treat a timing-only interrupted episode as saved data.

## Motion/state quality

- Require 18D float32 action and observation state, finite values and exact feature order.
- Use the existing data checker lag search. Require best lag MAE at most 0.01 rad and P95 at most
  0.05 rad.
- Require meaningful motion according to the existing moving-frame rule.
- Report large initial action/state delta and large action steps for human review. The historical
  recording wrapper used 10 rad guards, so these are not automatically rejected without a
  task-specific speed/transition policy.

## Timing quality

- Require a timing record for every saved Parquet frame; allow extras only for an unsaved interrupted
  episode and exclude those extras from the curated result.
- Require `source_timing v1` for every frame, source age at most 50 ms, source skew at most 20 ms,
  camera/state receive skew at most 100 ms and camera age at most 1000 ms.
- Require ZMQ protocol v1, source FPS at least 27, sequence gaps zero and state reuse zero.
- Camera observation-frame reuse may be reported as a low-rate warning; do not silently introduce a
  new hard threshold without approval.

## Video quality

- Require actual H.264 stream codec and positive duration for every referenced MP4.
- Decode at least first/middle/last frame of every episode and camera before merging.
- Fully decode every merged MP4 with `ffmpeg -nostdin -v error` after merging.
- Run RGB chromaticity scan at 0.008. Review flagged pairs visually; this detects the previously
  observed head white-balance toggle but also flags legitimate motion of strongly colored objects in
  a close gripper camera.

## Training schema

- Require raw18 features in the exact audited order.
- Require left observation source `measured_opening` and right source `commanded_opening` for this
  deployment. Do not relabel right command echo as measured.
- Require all merged manifests to have identical semantic schema.
- Require model16 projection `[0..14,16]`, dropped force indices `[15,17]`, and canonical
  `0=closed,100=open`.

## Partial-source policy

If a saved episode itself fails but shares an MP4 with good episodes, do not use the current generic
split/delete implementation without a codec-preservation patch and dedicated tests. It may re-encode
the partial file to AV1. Either exclude the full source or implement a staging dataset that copies the
source MP4 whole and keeps only valid timestamp references. Never guess missing metadata/video maps.
