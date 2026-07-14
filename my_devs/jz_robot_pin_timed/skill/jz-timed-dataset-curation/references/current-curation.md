# 2026-07-13 curation result

## Source decisions

| Source suffix | Saved episodes | Frames | Decision | Reason |
|---|---:|---:|---|---|
| `color_diag_20260712_203425` | 0 | 0 | Exclude | No completed data, standard metadata, timing or video |
| `color_diag_20260712_203708` | 1 | 200 | Exclude from production merge | Structurally valid RTSP/CRF20 diagnostic acquisition; heterogeneous with ZMQ/CRF18 production data |
| `real_10eps_20260713_170544` | 10 | 1995 | Accept | All data/timing/projection/video checks pass |
| `real_10eps_20260713_171156` | 0 | 0 | Exclude | Only an interrupted timing sidecar remains |
| `real_10eps_20260713_171325` | 3 | 599 | Accept saved only | Episodes 0-2 pass; unsaved episode 3 has 61 orphan timing records |
| `real_10eps_20260713_172850` | 15 | 2988 | Exclude | Standard metadata contains only 10-14 without video locations; MP4 covers only the earlier 0-9 batch; data/meta/video ownership is ambiguous |
| `real_10eps_20260713_173813` | 10 | 1992 | Accept | All data/timing/projection/video checks pass |
| `real_10eps_20260713_174727` | 9 | 1794 | Accept saved only | Episodes 0-8 pass; unsaved episode 9 has 147 orphan timing records |
| `real_10eps_20260713_175726` | 10 | 1990 | Accept | All data/timing/projection/video checks pass |

Production merge total: 42 episodes and 8370 frames.

## Output

```text
tests/outputs/jz_robot_pin_timed_curated_42eps_20260713
```

Repository ID:

```text
local/jz_robot_pin_timed_curated_42eps_20260713
```

Important reports:

```text
meta/jz_pin_curation_report.json
data_check_report.json
timing_check_report.json
training_projection_report.json
color_scan_report.json
color_review.json
```

The raw color scan is `REVIEW`: head has zero events, the only left event is a source-video
concatenation boundary, and sampled right events are close blue-object/normal camera motion. The
manual review is recorded as `PASS_WITH_REVIEW`; all 42 episodes are retained.

All original source directories remain in place.
