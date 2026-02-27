# 个人使用
 /home/agilex/.cache/huggingface/lerobot/cqy_vla_dev/agilex_both_side_black_cup_new 这里有一个正确的数据格式你可以拿这个对比一下

```bash
# 
conda run -n lerobot_flex --no-capture-output uvicorn my_devs.web_collection.app:app --host 0.0.0.0 --port 8008 --log-level info
```

# web_collection

Goal: a minimal, reliable pipeline to **record ROS topics into HDF5** and **convert to LeRobot v3 dataset format**,
with a simple local web UI.

## Run (UI)

All commands should use the `lerobot_flex` conda env.

Recommended (shows uvicorn logs):
```bash
cd /home/agilex/cqy/flexible_lerobot
conda activate lerobot_flex
uvicorn my_devs.web_collection.app:app --host 0.0.0.0 --port 8008 --log-level info
```

Alternative (also shows logs; `conda run` captures output unless you add `--no-capture-output`):
```bash
cd /home/agilex/cqy/flexible_lerobot
conda run -n lerobot_flex --no-capture-output uvicorn my_devs.web_collection.app:app --host 0.0.0.0 --port 8008 --log-level info
```

Then open `http://<machine-ip>:8008/`.

Notes:
- You still need a running ROS master (`roscore`) and your sensor/robot nodes publishing topics.
- If your ROS master is not local, make sure `ROS_MASTER_URI`/`ROS_IP` (or `ROS_HOSTNAME`) are set in the shell
  that launches `uvicorn` so the subprocesses inherit them.

## Config

Default config: `my_devs/web_collection/configs/default.yaml`

Update the topic names there to match your system. The UI lets you select which config to use.

## CLI

Record:

```bash
cd /home/agilex/cqy/flexible_lerobot
conda run -n lerobot_flex python -m my_devs.web_collection.record_hdf5 \
  --config my_devs/web_collection/configs/default.yaml \
  --dataset_dir /tmp/web_collection_data \
  --task_name aloha_mobile_dummy \
  --max_frames 60 \
  --num_episodes 10
```

Convert:

```bash
cd /home/agilex/cqy/flexible_lerobot
conda run -n lerobot_flex python -m my_devs.web_collection.convert_to_lerobot \
  --input_hdf5 /tmp/web_collection_data/aloha_mobile_dummy/episode_000000.hdf5 \
  --output_dir /tmp/web_collection_lerobot \
  --repo_id local/web_collection_demo
```

Convert a whole dataset directory:

```bash
cd /home/agilex/cqy/flexible_lerobot
conda run -n lerobot_flex python -m my_devs.web_collection.convert_dataset_to_lerobot \
  --input_dataset_dir datasets/aloha_mobile_dummy \
  --output_dir /tmp/web_collection_lerobot \
  --repo_id local/aloha_mobile_dummy_all
```
