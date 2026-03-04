# PI0.5 复刻训练工作报告（阶段性，已中断可续训）

## 1. 任务目标

基于文档：
- `/data/cqy_workspace/flexible_lerobot/my_devs/train/pi/pi0_pi05_modelscope_training.md`

目标：
- 在 `lerobot_flex` 环境中完成 PI0.5 训练复刻。
- 基模与相关资产优先使用 ModelScope 下载至当前仓库 `assets`。
- 训练目标步数 `15000`，中途可断点续训。

---

## 2. 环境信息

- 仓库根目录：`/data/cqy_workspace/flexible_lerobot`
- Conda 环境：`lerobot_flex`
- Python：`3.10.19`
- torch：`2.7.1+cu126`
- lerobot：`0.4.3`
- modelscope：`1.34.0`
- transformers：`4.53.3`（来自分支 `fix/lerobot_openpi`）
- GPU：`NVIDIA GeForce RTX 4090 48GB`

说明：
- PI0.5 初始化时报错要求特定 transformers 实现，已将 `transformers` 切换为 LeRobot 需要的分支版本。

---

## 3. 数据与资产路径

### 3.1 训练数据

- Dataset repo_id：`admin123/grasp_block_in_bin1`
- Dataset root：`/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1`
- 数据版本：`v3.0`（`meta/info.json`）

### 3.2 ModelScope 下载资产（已落盘）

- PI0.5 基模：
  - `/data/cqy_workspace/flexible_lerobot/assets/modelscope/lerobot/pi05_base`
  - 大小约 `14G`
- PI0 基模：
  - `/data/cqy_workspace/flexible_lerobot/assets/modelscope/lerobot/pi0_base`
  - 大小约 `14G`
- PaliGemma tokenizer（用于绕过 HF 不可达问题）：
  - `/data/cqy_workspace/flexible_lerobot/assets/modelscope/google/paligemma-3b-pt-224`
  - 大小约 `21M`

### 3.3 本地 tokenizer 路由（关键）

为了兼容 `AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")` 且避免访问 `huggingface.co`，创建了本地同名路径：

- 软链接：
  - `/data/cqy_workspace/flexible_lerobot/google/paligemma-3b-pt-224`
  - -> `/data/cqy_workspace/flexible_lerobot/assets/modelscope/google/paligemma-3b-pt-224`

注意：
- 训练命令需要在仓库根目录运行（使相对路径 `google/paligemma-3b-pt-224` 可见）。

---

## 4. 新增/使用的脚本

- ModelScope 下载脚本：
  - `/data/cqy_workspace/flexible_lerobot/my_devs/train/pi/download_modelscope_model.py`
- PI0.5 训练脚本（支持 smoke/full）：
  - `/data/cqy_workspace/flexible_lerobot/my_devs/train/pi/run_pi05_modelscope_train.sh`
- 监控脚本（阶段性使用）：
  - `/data/cqy_workspace/flexible_lerobot/my_devs/train/pi/monitor_pi_training.sh`
  - `/data/cqy_workspace/flexible_lerobot/my_devs/train/pi/monitor_pi_training.py`

---

## 5. 关键执行过程（摘要）

1. 读取训练模板文档并确定参数。  
2. 下载 `pi05_base`（ModelScope）并验证可读取。  
3. 冒烟训练第一次失败：`transformers` 版本不匹配。  
4. 安装定制 transformers：`fix/lerobot_openpi`，问题解决。  
5. 冒烟训练第二次失败：访问 `huggingface.co/google/paligemma-3b-pt-224` 网络不可达。  
6. 从 ModelScope 下载 paligemma tokenizer，并创建本地同名软链接，问题解决。  
7. 冒烟训练成功跑通至 `step 200`。  
8. 启动正式训练 `steps=15000`。  
9. 训练过程中到达 `step 10000`（并保存 `010000` checkpoint）。  
10. 本轮被人工中断，训练停在 `011000` checkpoint。

---

## 6. 本次正式训练配置

正式训练运行目录：
- `/data/cqy_workspace/flexible_lerobot/outputs/train/pi05_grasp_block_in_bin1_repro_20260303_170727/bs8_20260303_170727`

正式训练日志：
- `/data/cqy_workspace/flexible_lerobot/outputs/train/pi05_grasp_block_in_bin1_repro_20260303_170727/bs8_20260303_170727.log`

核心参数：
- `--dataset.repo_id=admin123/grasp_block_in_bin1`
- `--dataset.root=/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1`
- `--dataset.video_backend=pyav`
- `--policy.type=pi05`
- `--policy.pretrained_path=/data/cqy_workspace/flexible_lerobot/assets/modelscope/lerobot/pi05_base`
- `--policy.push_to_hub=false`
- `--policy.compile_model=false`
- `--policy.gradient_checkpointing=true`
- `--policy.dtype=bfloat16`
- `--policy.device=cuda`
- `--policy.normalization_mapping={"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}`
- `--batch_size=8`
- `--steps=15000`
- `--save_freq=1000`
- `--log_freq=200`

---

## 7. 当前结果与中断点

### 7.1 里程碑

- `step 10000` 达成时间：`2026-03-03 21:03:47`  
- 日志行：`Checkpoint policy after step 10000`

### 7.2 当前中断状态

- 训练进程：已停止（当前 `ps` 未发现对应训练进程）
- 最新完整 checkpoint：`011000`
- checkpoint 目录：
  - `/data/cqy_workspace/flexible_lerobot/outputs/train/pi05_grasp_block_in_bin1_repro_20260303_170727/bs8_20260303_170727/checkpoints/011000`
- `last` 链接当前指向：
  - `/data/cqy_workspace/flexible_lerobot/outputs/train/pi05_grasp_block_in_bin1_repro_20260303_170727/bs8_20260303_170727/checkpoints/011000`

---

## 8. 断点续训命令（直接可用）

在仓库根目录执行：

```bash
cd /data/cqy_workspace/flexible_lerobot
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_flex
export TOKENIZERS_PARALLELISM=false

python src/lerobot/scripts/lerobot_train.py \
  --config_path=/data/cqy_workspace/flexible_lerobot/outputs/train/pi05_grasp_block_in_bin1_repro_20260303_170727/bs8_20260303_170727/checkpoints/011000/pretrained_model/train_config.json \
  --resume=true
```

说明：
- 该命令会从 `011000` 继续跑到配置中的总步数（`15000`）。
- 不需要重复指定 dataset/model 参数，已在 `train_config.json` 中。

---

## 9. 额外说明（避免再次踩坑）

1. 必须在仓库根目录执行训练命令。  
原因：使用了本地路径 `google/paligemma-3b-pt-224` 兼容 tokenizer。

2. PI0.5 在当前机器上建议 `compile_model=false`。  
原因：`torch.compile` 初次编译耗时过长，不利于稳定推进。

3. 继续保持 `dataset.video_backend=pyav`（已在配置中）。  
原因：规避 `torchcodec/ffmpeg` 相关兼容问题。

---

## 10. 关键文件总览

- 文档模板：
  - `/data/cqy_workspace/flexible_lerobot/my_devs/train/pi/pi0_pi05_modelscope_training.md`
- 本工作报告：
  - `/data/cqy_workspace/flexible_lerobot/my_devs/train/pi/pi05_modelscope_repro_work_report.md`
- 正式训练日志：
  - `/data/cqy_workspace/flexible_lerobot/outputs/train/pi05_grasp_block_in_bin1_repro_20260303_170727/bs8_20260303_170727.log`
- 当前断点 checkpoint：
  - `/data/cqy_workspace/flexible_lerobot/outputs/train/pi05_grasp_block_in_bin1_repro_20260303_170727/bs8_20260303_170727/checkpoints/011000`
- PI0.5 基模：
  - `/data/cqy_workspace/flexible_lerobot/assets/modelscope/lerobot/pi05_base`
- PI0 基模：
  - `/data/cqy_workspace/flexible_lerobot/assets/modelscope/lerobot/pi0_base`
- tokenizer 资产：
  - `/data/cqy_workspace/flexible_lerobot/assets/modelscope/google/paligemma-3b-pt-224`
  - `/data/cqy_workspace/flexible_lerobot/google/paligemma-3b-pt-224`

