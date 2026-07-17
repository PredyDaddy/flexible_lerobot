# 2026-07-15 `data/` curation result

Canonical procedure and full source decision table:

```text
my_devs/jz_robot_pin_timed/data_check/DATASET_INTEGRITY_AND_CURATION.md
```

Output:

```text
data/jz_robot_pin_timed_curated_87eps_20260715
local/jz_robot_pin_timed_curated_87eps_20260715
```

Totals:

- 10 source roots;
- 97 saved source episodes and 23,790 source frames;
- exclude `testrightright1` episodes 2–11 because all three standard video references are null;
- retain 87 episodes and 21,398 frames;
- retain only `testrightright1` episodes 0–1 through codec-preserving partial-source staging.

Validation:

- `meta/jz_pin_curation_report.json`: `MERGE_PASS`;
- `meta/data_integrity_report.json`: 87/87 usable, zero FAIL;
- `timing_check_report.json`: `PASS`, all 21,398 timing/data keys matched;
- `training_projection_report.json`: `PASS`;
- 19 MP4 files fully decoded with ffmpeg, zero errors;
- `color_review.json`: `PASS_WITH_REVIEW`.

Known warning: one copied left-camera shared MP4 retains 239 unreferenced physical tail frames to avoid
AV1 re-encoding. Selected episode timestamp ownership remains exact and every selected frame decodes.
