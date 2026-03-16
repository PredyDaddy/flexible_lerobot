
```bash
python my_devs/train/groot/agilex/run_async_groot_single_arm.py check \
    --policy-path /data/cqy_workspace/flexible_lerobot/outputs/train/groot_agilex_first_test_right_20260315_221522/bs4_20260315_221522/checkpoints/020000/pretrained_model \
    --policy-device cuda
```

```bash
python my_devs/train/groot/agilex/run_async_groot_single_arm.py server \
    --host 0.0.0.0 \
    --port 8080 \
    --fps 30 \
    --inference-latency 0.033 \
    --obs-queue-timeout 2
```