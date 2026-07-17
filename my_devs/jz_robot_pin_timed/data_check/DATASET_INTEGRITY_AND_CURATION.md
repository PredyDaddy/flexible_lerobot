# JZ Timed 数据完整性、录制中断与合并规范

本文是 `jz_robot_pin_timed` 数采数据的规范入口。后续智能体在判断 episode 是否可用、清理录制残片、
修改时序阈值、合并训练数据或解释一次录制中断前，必须先阅读本文。

本文描述的是离线数据处理。禁止在检查、整理和合并期间启动 state bridge、joystick、armed recorder、
replay、ROS 服务或任何机器人控制命令。所有 Python 命令必须使用：

```bash
/home/luzhuang/miniconda3/envs/lerobot_flex/bin/python
```

## 1. 当前录制参数：采集容错 200 ms，训练质量门 100 ms

录制专用的 camera/state 本机接收时间偏差参数是：

```text
MAX_CAMERA_STATE_RECEIVE_SKEW_MS
```

当前 Web 和 `record.sh` 的录制默认值为 `200.0 ms`：

- `my_devs/jz_robot_pin_timed/record.sh`
- `my_devs/jz_robot_pin_timed/web_collection_system/server.py`
- Web 页面“高级参数 / 相机-State 对齐上限（毫秒）”

该值会传给：

```text
--robot.max_camera_state_receive_skew_ms
```

不要通过修改 `src/lerobot/robots/jz_robot_pin_timed/config_jz_robot_pin_timed.py` 的全局默认来实现
录制放宽；那会同时影响没有显式覆盖参数的回放、推理和其他调用。录制链显式传 `200 ms`，机器人类
全局默认继续保持 `100 ms`。

`STATE_ADVANCE_TIMEOUT_S=0.1` 是另一个参数，只控制等待新 robot state revision 的时间，不能解决
`Timed ZMQ camera ... did not produce an aligned frame`。不要因为相机超时把它机械地改成 `0.2`。

采集容错和训练准入故意分开：

| 阶段 | camera/state skew | 含义 |
|---|---:|---|
| 录制运行时 | 最多 200 ms | 避免一次短相机停顿立即终止整次采集 |
| 离线训练质量门 | 最多 100 ms | 超过 100 ms 的正式帧/episode 不自动进入训练 |
| 离线告警 | 超过 50 ms | 允许保留，但报告发生次数和最大值 |

因此，把录制放宽到 200 ms 不代表 100–200 ms 的样本自动合格。录制先完成，之后由检查器决定是否
保留 episode。

## 2. 录制中断时，系统实际保存什么

正常 episode 调用顺序是：

```text
record_loop
    -> 可选 reset
    -> dataset.save_episode()
    -> recorded_episodes += 1
```

### 2.1 相机 Timeout 发生在当前 `record_loop` 内

- 之前已经执行过 `dataset.save_episode()` 的 episode，其 data Parquet 和标准 episode metadata 已提交。
- 当前 episode 不会执行 `save_episode()`，不会增加 `meta/info.json.total_episodes`。
- 当前 episode 可能留下 `meta/timing/episode-XXXXXX.jsonl`，甚至临时图片/MP4；这些是未提交残片。
- timing 文件存在不等于 episode 已保存。必须以 data Parquet 和标准 episode metadata 为准。

### 2.2 reset 阶段失败

`record_loop` 已经完成但尚未正常保存时，recorder 会检查 episode buffer；如果不是“重录本条”，会
尝试 `dataset.save_episode()` 后再保留原异常退出。因此 reset 错误下，刚录完的 episode 可能已经被
补救保存，必须检查实际 metadata，不能仅凭终端最后一条日志判断。

### 2.3 `save_episode()` 或视频批量编码阶段失败

这是最危险的边界：

- data Parquet 可能已经写入；
- metadata 可能已经登记；
- 视频可能尚未编码、只编码了一部分，或视频位置字段仍为空；
- Web 当前把 `VIDEO_ENCODING_BATCH_SIZE` 设为目标 episode 数，因此前面多条视频可能推迟到退出阶段
  才统一编码。

所以“报错之前的 episode 一定可训练”不是绝对保证。更准确的规则是：

> 报错之前已经保存的 data/meta 通常保留；是否真正可训练，还必须验证三路视频和 timing 的完整映射。

`data/testrightright1` 是实际反例：12 条 data/meta/timing 已登记，但只有 episode 0、1 有完整三路视频
引用；episode 2–11 不可训练。

## 3. episode 和数据集分类

### PASS

- data/meta/video/timing 全部一致；
- 没有硬阈值违规；
- 没有 sequence 倒退或视频损坏。

### WARN（可用）

- 结构和三路视频完整；
- camera/state skew 在 50–100 ms；或
- 少量相邻 observation 复用同一相机 sequence；或
- 存在与正式 Parquet 无交集的未提交 timing 残片；或
- 为避免重新编码而复制的共享 MP4 有未引用物理尾帧，但所有选中 episode 的视频范围准确。

WARN 不等于不可用。必须把具体告警写进报告。

### FAIL（不可用）

任一情况成立即 FAIL：

- info、episode metadata、data Parquet 的 episode/frame/index 不一致；
- 正式 episode 缺少任一路相机视频引用；
- 视频文件缺失、截断、无法解码、FPS/分辨率/codec 不匹配；
- 正式帧缺 timing，或 timing/Parquet key 不一致；
- camera/state skew 超过训练质量门 100 ms；
- source age 超过 50 ms、source skew 超过 20 ms；
- state 或 camera sequence 倒退；
- raw18/schema/model16 投影语义不合格。

### UNCOMMITTED_FRAGMENT

只存在 timing、临时图片或临时 MP4，但 `(episode_index, frame_index)` 不在正式 data Parquet 中。
它不属于训练 episode。保留源目录并在合并时忽略，不要通过猜测补 metadata。

## 4. 快速完整性检查

本地只读入口：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

/home/luzhuang/miniconda3/envs/lerobot_flex/bin/python \
  data/check_data.py \
  data/<dataset>
```

它逐 episode 检查：

- `info.json`、tasks、stats、episode metadata；
- data Parquet 行数、episode/frame/global index、timestamp grid；
- 三路视频引用、文件、帧数、时长、FPS、codec、分辨率；
- timing 行数和 key；
- camera skew/age/sequence/reuse/gap；
- state/source timing；
- 未提交 timing 与临时 MP4。

完整解码视频：

```bash
/home/luzhuang/miniconda3/envs/lerobot_flex/bin/python \
  data/check_data.py data/<dataset> --deep-video --verbose
```

默认退出码：没有 FAIL 为 0；存在 FAIL 为 1。`--fail-on-warning` 可把 WARN 也变成退出码 1。

## 5. JZ timed 严格检查

### 5.1 严格 timing

```bash
PYTHONPATH=src /home/luzhuang/miniconda3/envs/lerobot_flex/bin/python \
  my_devs/jz_robot_pin_timed/data_check/check_timing.py \
  --dataset-root <root> \
  --expected-codec h264 --expected-crf 18 \
  --expected-camera-fps 20 --expected-camera-source-fps 30 \
  --min-camera-source-fps-ratio 0.9 \
  --expected-camera-protocol jz_realsense_zmq \
  --expected-command-mode armed --expected-command-transport udp \
  --expected-action-key-count 18 --require-source-timing \
  --max-source-age-ms 50 --max-source-skew-ms 20 \
  --max-camera-age-ms 1000 --max-camera-state-skew-ms 100 \
  --report-json <root>/timing_check_report.json
```

### 5.2 训练投影

```bash
PYTHONPATH=src /home/luzhuang/miniconda3/envs/lerobot_flex/bin/python \
  my_devs/jz_robot_pin_timed/data_check/check_training_projection.py \
  --dataset-root <root> \
  --manifest <root>/meta/jz_pin_training_schema.json \
  --report-json <root>/training_projection_report.json
```

正式训练集必须是 `PASS`，不能用 `--allow-unavailable` 把未知夹爪来源当作训练批准。

### 5.3 全视频解码

```bash
set -euo pipefail
while IFS= read -r -d '' video; do
  ffmpeg -nostdin -v error -i "$video" -f null -
done < <(find <root>/videos -type f -name '*.mp4' -print0)
```

### 5.4 颜色稳定性

```bash
PYTHONPATH=src /home/luzhuang/miniconda3/envs/lerobot_flex/bin/python \
  my_devs/jz_robot_pin_timed/skill/jz-timed-dataset-curation/scripts/scan_video_color.py \
  --dataset-root <root> \
  --max-chromaticity-step 0.008 \
  --report-json <root>/color_scan_report.json
```

`REVIEW` 不是自动 FAIL。必须检查事件前后帧：全画面一致变色可能是白平衡切换；橙色托盘、蓝色物体、
机械臂或场景在近距离移动，经常只因画面内容变化触发。

## 6. 合并规范

- 所有源目录只读，禁止原地 merge、rename、delete。
- 输出必须是新的、不存在的目录。
- 源的 FPS、robot type、features、训练 schema 语义必须一致。
- 只使用 JZ timed 专用合并脚本；通用 merge 不会恢复 timing/schema。
- 不要直接用 generic `split_dataset/delete_episodes` 裁共享 MP4；它可能重编码为 AV1。
- 部分源使用 `--source-episodes ROOT=SPEC`。专用脚本只过滤 Parquet/meta/timing，并把选中 episode
  引用的共享 H.264 MP4 整文件复制到临时 staging，不重新编码。
- 合并后必须再次运行第 4、5 节全部检查。

示例：

```bash
PYTHONPATH=src /home/luzhuang/miniconda3/envs/lerobot_flex/bin/python \
  my_devs/jz_robot_pin_timed/skill/jz-timed-dataset-curation/scripts/merge_valid_datasets.py \
  --source-root data/source_a \
  --source-root data/partially_valid_source \
  --source-episodes data/partially_valid_source=0-1,4,7-9 \
  --output-root data/new_curated_dataset \
  --output-repo-id local/new_curated_dataset \
  --expected-codec h264 --expected-crf 18 \
  --sample-frames-per-episode 3 \
  --preflight-only \
  --report-json /tmp/new_curated_dataset.preflight.json
```

先要求 `PREFLIGHT_PASS`，再去掉 `--preflight-only` 创建正式输出。

## 7. 2026-07-15 当前数据整理结果

源数据共 97 条正式 episode、23,790 帧。`testrightright1` episode 2–11 因三路视频 metadata 缺失
被排除，共排除 10 条、2,392 帧。最终选择 87 条、21,398 帧。

| 源目录 | 选择源 episode | 合并 episode | 帧数 | 处理 |
|---|---|---|---:|---|
| `testdownright1` | 0–11 | 0–11 | 2869 | 全部保留；忽略 timing-only 源 ep12（239帧） |
| `testdownright2` | 0–17 | 12–29 | 4303 | 全部保留 |
| `testrightright1` | 0–1 | 30–31 | 479 | 排除源 ep2–11；忽略 timing-only 源 ep12（104帧） |
| `testrightright2` | 0–3 | 32–35 | 956 | 全部保留；忽略 timing-only 源 ep4（126帧） |
| `testrightright3` | 0–9 | 36–45 | 2391 | 全部保留 |
| `testrightright4` | 0–3 | 46–49 | 956 | 全部保留；忽略 timing-only 源 ep4（46帧） |
| `testrightright5` | 0–2 | 50–52 | 717 | 全部保留；忽略 timing-only 源 ep3（45帧） |
| `testupright1` | 0–9 | 53–62 | 2990 | 全部保留 |
| `testupright2` | 0–3 | 63–66 | 957 | 全部保留；忽略 timing-only 源 ep4（30帧） |
| `testupright3` | 0–19 | 67–86 | 4780 | 全部保留 |

合并输出：

```text
data/jz_robot_pin_timed_curated_87eps_20260715
local/jz_robot_pin_timed_curated_87eps_20260715
```

验证结果：

| 检查 | 结果 |
|---|---|
| 专用合并验证 | `MERGE_PASS` |
| episode/frame | 87 / 21,398 |
| 快速完整性 | 87/87 usable，0 FAIL，整体 WARN |
| 严格 timing | `PASS`，21,398/21,398 timing 匹配 |
| 训练投影 | `PASS`，raw18 保留，model16 映射正确 |
| 视频 | 19 个 MP4 全部完整 ffmpeg 解码通过 |
| 颜色 | `PASS_WITH_REVIEW`；事件为 episode/源边界或近物体运动，未见全画面白平衡跳变 |

已知非致命告警：

- head/left/right observation-frame reuse 分别约 0.08% / 0.17% / 0.14%；
- 最大 camera/state skew 分别约 79.87 / 96.68 / 64.84 ms，均低于 100 ms；
- 为避免把 `testrightright1` 的共享 H.264 视频重编码，合并后的左相机文件保留了 239 个未引用物理
  尾帧；所有 87 条选中 episode 的 timestamp/video 引用仍准确，抽样解码和完整解码均通过。

## 8. 后续智能体的决策顺序

1. 不看目录名推断 episode 数，先读 `meta/info.json` 和 data Parquet。
2. 运行 `data/check_data.py`，逐条记录 PASS/WARN/FAIL。
3. timing-only 残片只标记，不作为正式数据，也不猜测恢复。
4. 正式 episode 缺任一路视频引用时判 FAIL。
5. 录制允许 200 ms，但训练质量门默认仍是 100 ms。
6. 部分源只通过专用 `--source-episodes` staging 合并，不用 generic split/delete 重编码共享视频。
7. 先 preflight，再 merge；输出目录必须不存在。
8. 合并后重新跑完整性、strict timing、projection、全视频 decode 和颜色扫描。
9. 保留所有源目录和 curation episode map；不要删除排除数据，除非用户明确授权。

## 9. 禁止事项

- 禁止通过 timing 文件行数直接修改 `total_episodes`。
- 禁止把孤立 timing/临时 MP4 当作正式 episode。
- 禁止为“让检查通过”而同步放宽训练质量门到 200 ms。
- 禁止猜测缺失的 video chunk/file/timestamp。
- 禁止原地修复或覆盖源目录。
- 禁止在离线检查期间启动或控制机器人。
