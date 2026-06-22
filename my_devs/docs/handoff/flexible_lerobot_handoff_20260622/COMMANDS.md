# Commands

所有命令默认在仓库根目录执行：

```bash
cd /data/cqy_workspace/flexible_lerobot
```

## 1. 环境检查

```bash
git status --short --branch
conda run -n lerobot_flex python --version
conda run -n lerobot_flex python -c "import lerobot; print('lerobot import ok')"
conda run -n lerobot_flex python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

TensorRT 检查：

```bash
conda run -n lerobot_flex python -c "import tensorrt as trt; print(trt.__version__)"
```

如果上面失败，但文档里指定了本地 TensorRT 路径，可尝试：

```bash
TENSORRT_PY_DIR=/data/cqy_workspace/third_party/tensorrt_10_13_0_35 \
conda run -n lerobot_flex python -c "import sys, os; sys.path.insert(0, os.environ['TENSORRT_PY_DIR']); import tensorrt as trt; print(trt.__version__)"
```

## 2. 安装 / Lint / Test

开发安装：

```bash
conda run -n lerobot_flex python -m pip install -e ".[dev,test]"
```

格式化和 lint：

```bash
conda run -n lerobot_flex ruff format my_devs/<feature>
conda run -n lerobot_flex ruff check my_devs/<feature> --fix
```

全仓 pre-commit：

```bash
conda run -n lerobot_flex pre-commit run -a
```

测试：

```bash
conda run -n lerobot_flex pytest -sv tests
conda run -n lerobot_flex pytest -q tests/ -k <keyword>
```

端到端 smoke：

```bash
conda run -n lerobot_flex make test-end-to-end DEVICE=cpu
```

## 3. ACT TensorRT

当前推荐从 `my_devs/new_act_trt/README.md` 开始。

一条命令 export / build / verify：

```bash
conda run -n lerobot_flex python my_devs/new_act_trt/scripts/export_act_checkpoint_engine.py \
  --checkpoint outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model \
  --device cpu \
  --trt-device cuda:0 \
  --task "Put the block in the bin" \
  --robot-type so101_follower
```

TRT 上机入口示例：

```bash
conda run -n lerobot_flex python my_devs/new_act_trt/scripts/run_trt_act.py \
  --policy-path outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model \
  --task "Put the block in the bin"
```

历史 ACT 流程详见：

```text
my_devs/docs/act_trt/act_trt复现与参考工作流.md
```

## 4. GR00T TensorRT

准备输出目录：

```bash
RUN_ID=consistency_rerun_$(date +%Y%m%d_%H%M%S)
RUN_DIR=outputs/trt/$RUN_ID
mkdir -p "$RUN_DIR/logs"
POLICY_PATH=/path/to/pretrained_model
```

导出 backbone：

```bash
conda run -n lerobot_flex python my_devs/groot_trt/export_backbone_onnx.py \
  --policy-path "$POLICY_PATH" \
  --onnx-out-dir "$RUN_DIR/gr00t_onnx" \
  --seq-len 296 \
  --video-views 1 \
  --vit-dtype fp16 \
  --llm-dtype fp16 \
  --device cuda
```

导出 action head：

```bash
conda run -n lerobot_flex python my_devs/groot_trt/export_action_head_onnx.py \
  --policy-path "$POLICY_PATH" \
  --onnx-out-dir "$RUN_DIR/gr00t_onnx" \
  --seq-len 296 \
  --device cuda
```

构建 TensorRT engine：

```bash
ONNX_DIR="$RUN_DIR/gr00t_onnx" \
ENGINE_DIR="$RUN_DIR/gr00t_engine_api_trt1013" \
VIDEO_VIEWS=2 MAX_BATCH=2 WORKSPACE_GB=8 \
bash my_devs/groot_trt/build_engine.sh
```

完整细节见：

```text
my_devs/groot_trt/README.md
```

## 5. PI0.5 / OpenPI TensorRT

先读：

```text
my_devs/openpi_trt/README.md
my_devs/openpi_trt/docs/openpi_trt实现链路报告.md
my_devs/openpi_trt/docs/lerobot_pi05_trt难点分析.md
```

当前建议路线：

```text
prefix_cache TensorRT engine
  + denoise_step TensorRT engine
  + Python 10-step denoise loop
```

不要一开始就尝试完整 `sample_actions(...)` 单体 engine，除非用户明确要求研究这条路线。

## 6. VLA Engineering

关键入口：

```text
my_devs/vla_engineering/vlash_iner/
```

先做只读/加载检查，再做真实机器人：

```bash
conda run -n lerobot_flex python my_devs/vla_engineering/vlash_iner/run_pi05_sync.py \
  --policy-path /path/to/pretrained_model \
  --task "Put the block in the bin" \
  --check-policy-load
```

具体参数以 `--help` 和当前文档为准：

```bash
conda run -n lerobot_flex python my_devs/vla_engineering/vlash_iner/run_pi05_sync.py --help
```

## 7. AgileX Web Collection

启动 AgileX LeRobot 采集 Web backend：

```bash
bash my_devs/agilex_web_collection/run.sh
```

或：

```bash
conda run --no-capture-output -n lerobot_flex \
  uvicorn my_devs.agilex_web_collection:app --host 0.0.0.0 --port 8000
```

## 8. ROS HDF5 Web Collection

启动 UI：

```bash
conda run -n lerobot_flex --no-capture-output \
  uvicorn my_devs.web_collection.app:app --host 0.0.0.0 --port 8008 --log-level info
```

录制 HDF5：

```bash
conda run -n lerobot_flex python -m my_devs.web_collection.record_hdf5 \
  --config my_devs/web_collection/configs/default.yaml \
  --dataset_dir /tmp/web_collection_data \
  --task_name aloha_mobile_dummy \
  --max_frames 60 \
  --num_episodes 10
```

转换整个 dataset directory 到 LeRobot：

```bash
conda run -n lerobot_flex python -m my_devs.web_collection.convert_dataset_to_lerobot \
  --input_dataset_dir datasets/aloha_mobile_dummy \
  --output_dir /tmp/web_collection_lerobot \
  --repo_id local/aloha_mobile_dummy_all
```

## 9. 搜索建议

```bash
rg "prefix_cache|denoise_step|tensorrt_split|sample_actions" my_devs src/lerobot
rg "so101|so100|agilex|calibration|camera" my_devs
rg "TODO|FIXME|P0|阻断|验收" my_devs/docs my_devs
```

文件列表：

```bash
rg --files my_devs | sed -n '1,240p'
find my_devs/docs -maxdepth 3 -type f | sort
```
