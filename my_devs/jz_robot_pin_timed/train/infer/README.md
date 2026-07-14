# JZ Robot Pin Timed ACT 推理

本目录使用 checkpoint 中已序列化的 processor：现场 Robot 和评估数据集保持 raw18，ACT 内部使用
model16；三路相机统一 resize 到 224x224，model16 action 在 unnormalization 后展开回 raw18。

默认 checkpoint 是 40 Epoch 正式训练的最终模型：

```text
tests/outputs/act_jz_robot_pin_timed_curated_42eps_20260713_e40_b16_r224x224/
  checkpoints/last/pretrained_model
```

训练未完成时该路径不存在，脚本会明确失败。测试其他 checkpoint 时显式设置 `POLICY_PATH`，它
必须指向 `pretrained_model/`，不能指向训练 run 根目录或 `checkpoints/last/`。

脚本会解析 `last` 的真实目标目录，并要求 checkpoint step 等于 `train_config.json` 中的总 step。
本次 40 Epoch 正式模型必须显示 `20960/20960`。训练中的第 5/10/15...Epoch checkpoint 默认拒绝；
仅离线或 dry-run 检查可以显式设置 `ALLOW_INCOMPLETE_CHECKPOINT=1`，armed 永远拒绝未完成模型。

## 1. 离线推理

离线推理不会连接机器人、state UDP、相机或 command executor：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot
bash my_devs/jz_robot_pin_timed/train/infer/offline_infer.sh
```

默认验证数据集的首帧、中间帧和末帧。指定样本并保存 JSON：

```bash
SAMPLE_INDICES=0,100,8369 \
OUTPUT_JSON=tests/outputs/jz_act_e40_offline_infer.json \
bash my_devs/jz_robot_pin_timed/train/infer/offline_infer.sh
```

使用其他 checkpoint：

```bash
POLICY_PATH=/absolute/path/checkpoints/last/pretrained_model \
bash my_devs/jz_robot_pin_timed/train/infer/offline_infer.sh
```

离线 PASS 要求：checkpoint 是 ACT model16；JZ schema 可严格训练；三路图像经 processor 变为
224x224；模型输出 16D；postprocessor 输出 18D；所有数值 finite。

## 2. 查看现场命令但不启动

下面只做参数与 checkpoint 文件检查，打印完整命令，不连接任何现场服务：

```bash
PRINT_COMMAND_ONLY=1 \
bash my_devs/jz_robot_pin_timed/train/infer/run_dry_run.sh
```

## 3. Policy dry-run

dry-run 会连接 X86 state UDP 和三路 ZMQ 相机，并生成/保存 policy evaluation 数据；action 只走
`local + dry_run`，不会向 Orin command UDP 发送。它仍要求 Orin state bridge 和三路相机服务正常，
但不需要启动 armed command executor：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

NUM_EPISODES=1 \
EPISODE_TIME_S=10 \
bash my_devs/jz_robot_pin_timed/train/infer/run_dry_run.sh
```

默认会先对参考数据集首帧执行一次离线 checkpoint 推理，然后才连接现场。默认 joint initial/step
guard 均为 0.02 rad；如果策略第一步超限会明确失败，不应为了让测试通过而直接放宽。左右夹爪
width/force 在 Robot 边界固定钳位到 `[0,100]`；离线脚本会报告钳位前的越界预测。

## 4. Armed policy evaluation

armed 会实际通过 UDP 向 Orin 发送策略动作。只有现场人员、急停、机器人姿态和 Orin executor 都
确认后才能执行。脚本要求三个确认变量：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
JZ_POLICY_INFERENCE_ARMED=1 \
NUM_EPISODES=1 \
EPISODE_TIME_S=10 \
bash my_devs/jz_robot_pin_timed/train/infer/run_armed.sh
```

armed 前还必须在 Orin 按现有现场流程启动 state bridge 和经过授权的 command executor。脚本不负责
启动、重启或修改 Orin 服务，也不包含自动 reset/choreography。

## 5. 常用覆盖

```bash
POLICY_DEVICE=cuda
POLICY_N_ACTION_STEPS=25
DISPLAY_DATA=false
PLAY_SOUNDS=true
MAX_INITIAL_JOINT_DELTA_RAD=0.02
MAX_JOINT_STEP_RAD=0.02
```

评估数据会写入新的 `tests/outputs/eval_<run>_<mode>_<timestamp>`，目录已存在时拒绝覆盖。
