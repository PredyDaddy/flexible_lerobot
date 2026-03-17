CONDA_ENV_NAME=lerobot_flex \
  POLICY_PATH=/data/cqy_workspace/flexible_lerobot/outputs/train/groot_agilex_first_test_right_20260315_221522/bs4_20260315_221522/checkpoints/020000/pretrained_model \
  DATASET_TASK="Execute the trained AgileX GR00T task" \
  CONTROL_ARM=right \
  REMOTE_GROOT_SERVER_HOST=0.0.0.0 \
  REMOTE_GROOT_SERVER_PORT=5560 \
  bash my_devs/train/groot/agilex/run_groot_remote_policy_server.sh