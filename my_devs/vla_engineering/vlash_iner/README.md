# vlash_iner

`vlash_iner` is an isolated PI0.5 inference workspace for checkpoints trained by
`my_devs/train/pi/so101/easy_train.sh`.

The intent is to keep runtime experiments under `my_devs/vla_engineering/`
without modifying `src/lerobot`.

## First checks

Dry-run without touching robot hardware:

```bash
cd /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering
conda run -n lerobot_flex python -m vlash_iner.run_pi05_sync --dry-run true
```

Load policy weights and saved processors, still without connecting robot:

```bash
cd /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering
conda run -n lerobot_flex python -m vlash_iner.run_pi05_sync \
  --policy-path /data/cqy_workspace/flexible_lerobot/outputs/pi05_eraser_cup_multi_task_runs/<RUN_ID>/checkpoints/last/pretrained_model \
  --check-policy-load true
```

## Runtime modes

- `run_pi05_sync.py`: synchronous baseline adapted from `my_devs/train/pi/so101/run_pi05_infer.py`.
- `run_pi05_async.py`: single-threaded VLASH-style chunk manager for overlap experiments.

`future-state-aware` and `action_quant_ratio > 1` are intentionally disabled in
the first async implementation. Keep `--action-quant-ratio 1` until skipped-action
execution semantics are implemented and tested.
