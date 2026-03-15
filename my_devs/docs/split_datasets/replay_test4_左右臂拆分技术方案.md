# replay_test4 左右臂拆分技术方案

## 1. 背景

当前数据集是：

- 源数据集目录：
  - `/home/agilex/cqy/flexible_lerobot/my_devs/add_robot/agilex/outputs/dummy/replay_test4`

当前这份数据我们已经确认：

1. `action.shape == [14]`
2. `observation.state.shape == [14]`
3. 图像为三路：
   - `observation.images.camera_front`
   - `observation.images.camera_left`
   - `observation.images.camera_right`
4. 数据语义满足：
   - `action_t = observation.state_(t+1)`
5. `action` / `observation.state` 的维度顺序为：
   - 前 7 维：左臂
   - 后 7 维：右臂

现在希望把这一份双臂数据，拆成两份单臂数据：

1. 左臂数据集
   - 左臂 7 维关节角
   - 头部相机 `camera_front`
   - 左臂相机 `camera_left`
2. 右臂数据集
   - 右臂 7 维关节角
   - 头部相机 `camera_front`
   - 右臂相机 `camera_right`

本方案只描述技术实现方案，不在这一步写代码。

---

## 2. 目标

本次拆分的目标不是“修改原数据”，而是：

1. 保留原始双臂数据集不动
2. 从原始数据集稳定导出两份新的单臂数据集
3. 保证拆分后的数据仍然符合 LeRobot 数据格式
4. 保证拆分后仍然保留当前已经验证正确的时序语义：
   - 左臂集：`left_action_t = left_state_(t+1)`
   - 右臂集：`right_action_t = right_state_(t+1)`

---

## 3. 输出定义

## 3.1 输出位置

建议代码开发放在：

- `my_devs/split_datasets`

建议拆分后的新数据放在：

- `my_devs/split_datasets/outputs/dummy/replay_test4_left`
- `my_devs/split_datasets/outputs/dummy/replay_test4_right`

这样做有两个好处：

1. 不污染原始录制目录
2. 后续可以反复重建，不影响原始数据

## 3.2 左臂数据集定义

左臂数据集保留以下字段：

1. `action`
   - 7 维
   - 来自原始 `action[:, 0:7]`
2. `observation.state`
   - 7 维
   - 来自原始 `observation.state[:, 0:7]`
3. `observation.images.camera_front`
4. `observation.images.camera_left`
5. `timestamp`
6. `frame_index`
7. `episode_index`
8. `index`
9. `task_index`
10. `task`

左臂数据集删除以下字段：

1. `observation.images.camera_right`
2. 原始右臂 7 维 state
3. 原始右臂 7 维 action

## 3.3 右臂数据集定义

右臂数据集保留以下字段：

1. `action`
   - 7 维
   - 来自原始 `action[:, 7:14]`
2. `observation.state`
   - 7 维
   - 来自原始 `observation.state[:, 7:14]`
3. `observation.images.camera_front`
4. `observation.images.camera_right`
5. `timestamp`
6. `frame_index`
7. `episode_index`
8. `index`
9. `task_index`
10. `task`

右臂数据集删除以下字段：

1. `observation.images.camera_left`
2. 原始左臂 7 维 state
3. 原始左臂 7 维 action

---

## 4. 关键设计决策

## 4.1 不直接原地改 parquet / mp4

第一版不建议直接原地修改：

- `data/*.parquet`
- `meta/*.json`
- `meta/*.parquet`
- `videos/*.mp4`

原因很直接：

1. LeRobot 数据集不是只有一个 parquet
2. 元数据、episode 索引、视频索引是联动的
3. 原地改容易出现 schema 改了但 metadata 没完全同步的问题
4. 这类错误表面上“文件还在”，但训练时才爆

所以第一版建议走“重导出”路线：

1. 读取源数据集
2. 构造左臂帧和右臂帧
3. 用 LeRobotDataset 重新写出两份目标数据集

这条路线更慢，但更稳，更容易验收。

## 4.2 拆分时不重新计算 action 语义

当前源数据已经满足：

- `action_t = observation.state_(t+1)`

因此拆分时不要再重新做 shift。

只需要切片即可：

1. 左臂：
   - `left_action = source_action[:7]`
   - `left_state = source_state[:7]`
2. 右臂：
   - `right_action = source_action[7:14]`
   - `right_state = source_state[7:14]`

原因：

如果源数据在 14 维上已经满足 next-state 对齐，那么对左右臂分别切片之后，这个语义会天然保留。

## 4.3 第一版不改 joint names

第一版建议：

1. 左臂数据保留 joint names：
   - `left_joint0.pos ... left_joint6.pos`
2. 右臂数据保留 joint names：
   - `right_joint0.pos ... right_joint6.pos`

不建议第一版在拆分时额外重命名成：

- `joint0.pos ... joint6.pos`

原因：

1. 保留来源信息更清楚
2. 更容易做回溯和调试
3. 少做一层元数据变换，风险更低

如果后续训练代码强依赖“左右臂共用统一命名”，那是第二阶段增强，不放进第一版拆分脚本。

## 4.4 任务文本默认复制，但要预留覆盖能力

当前 `replay_test4` 的任务文本是：

- `agilex static record test`

这个文本对 VLA 来说偏弱，甚至带有误导性。

因此拆分工具建议支持两种模式：

1. 默认模式：直接复制原始 `task`
2. 可选模式：允许调用者在导出时覆盖任务文本

建议预留参数：

- `--task-mode copy`
- `--task-text "..."` 或 `--task-json path/to/tasks.json`

第一版可以先实现 `copy`，但接口设计上最好把 override 能力预留出来。

---

## 5. 推荐实现路线

## 5.1 代码目录建议

建议在 `my_devs/split_datasets` 下使用下面这套结构：

1. `my_devs/split_datasets/__init__.py`
2. `my_devs/split_datasets/split_bimanual_lerobot_dataset.py`
3. `my_devs/split_datasets/validate_split_dataset.py`
4. `my_devs/split_datasets/README.md`
5. `my_devs/split_datasets/outputs/`

其中：

1. `split_bimanual_lerobot_dataset.py`
   - 负责读取源数据并导出左右臂数据
2. `validate_split_dataset.py`
   - 负责检查导出结果是否符合预期

## 5.2 CLI 设计建议

建议第一版命令行长这样：

```bash
conda run --no-capture-output -n lerobot_flex \
python -m my_devs.split_datasets.split_bimanual_lerobot_dataset \
  --source-root /home/agilex/cqy/flexible_lerobot/my_devs/add_robot/agilex/outputs \
  --source-repo-id dummy/replay_test4 \
  --target-root /home/agilex/cqy/flexible_lerobot/my_devs/split_datasets/outputs \
  --left-repo-id dummy/replay_test4_left \
  --right-repo-id dummy/replay_test4_right \
  --task-mode copy \
  --vcodec h264
```

可选增强参数：

```bash
--overwrite
--episode-indices 0,1
--task-text "Pick up the black cup with the left arm"
--video true
--fps 30
```

---

## 6. 数据流设计

## 6.1 输入

输入数据集：

- `dummy/replay_test4`

依赖的输入字段：

1. `action`
2. `observation.state`
3. `observation.images.camera_front`
4. `observation.images.camera_left`
5. `observation.images.camera_right`
6. `task`
7. `timestamp`
8. `frame_index`
9. `episode_index`

## 6.2 输出

输出两份数据集：

1. `dummy/replay_test4_left`
2. `dummy/replay_test4_right`

## 6.3 导出流程

建议按 episode 重建，不要全局平铺重建。

流程如下：

1. 加载源 `LeRobotDataset`
2. 读取源数据集元信息：
   - `fps`
   - `video`
   - `vcodec`
   - `episodes`
   - `task`
3. 分别创建：
   - 左臂目标数据集
   - 右臂目标数据集
4. 按 episode 遍历
5. 对每个 frame 做左右拆分
6. 把左右 frame 分别写入左右目标数据集
7. 每个 episode 结束后分别 `save_episode()`
8. 最后分别 `finalize()`

---

## 7. 帧级拆分规则

## 7.1 左臂 frame 构造规则

对源 frame：

```text
source_action: [14]
source_state:  [14]
```

构造左臂 frame：

```text
left_action = source_action[0:7]
left_state  = source_state[0:7]
```

保留图像：

1. `camera_front`
2. `camera_left`

丢弃图像：

1. `camera_right`

## 7.2 右臂 frame 构造规则

构造右臂 frame：

```text
right_action = source_action[7:14]
right_state  = source_state[7:14]
```

保留图像：

1. `camera_front`
2. `camera_right`

丢弃图像：

1. `camera_left`

## 7.3 时间字段处理

以下字段不需要重新发明逻辑，直接沿用源 episode 的顺序写入即可：

1. `timestamp`
2. `task`

而：

1. `frame_index`
2. `episode_index`
3. `index`
4. `task_index`

建议交给目标 `LeRobotDataset` 在写出时自动生成和维护，不手工拼装最终 parquet。

---

## 8. 元数据设计

## 8.1 左臂数据集元数据

左臂数据集最终应满足：

1. `action.shape = [7]`
2. `observation.state.shape = [7]`
3. 图像特征只有：
   - `observation.images.camera_front`
   - `observation.images.camera_left`

## 8.2 右臂数据集元数据

右臂数据集最终应满足：

1. `action.shape = [7]`
2. `observation.state.shape = [7]`
3. 图像特征只有：
   - `observation.images.camera_front`
   - `observation.images.camera_right`

## 8.3 episode 和 frame 数量

建议第一版保持：

1. 左臂数据集 episode 数 = 源数据集 episode 数
2. 右臂数据集 episode 数 = 源数据集 episode 数
3. 每条 episode 的 frame 数不变

对 `replay_test4` 来说，拆分后应得到：

1. 左臂集：
   - `total_episodes = 2`
   - `total_frames = 480`
2. 右臂集：
   - `total_episodes = 2`
   - `total_frames = 480`

---

## 9. 推荐实现方式

## 9.1 第一版采用“安全重建法”

第一版建议直接使用：

1. `LeRobotDataset` 读取源数据
2. `LeRobotDataset.create(...)` 创建目标数据
3. `dataset.add_frame(...)`
4. `dataset.save_episode()`

优点：

1. 简单
2. 最稳
3. 容易验证
4. 不容易把 metadata 搞坏

缺点：

1. 会重新编码视频
2. 对大数据集会比较慢

但对当前 `replay_test4` 这种规模完全可以接受。

## 9.2 第二版再考虑“视频直拷贝快路径”

如果后续要处理更大的数据集，可以再考虑优化：

1. 不重新 decode/re-encode 视频
2. 直接拷贝 `camera_front + 目标腕部相机` 对应 mp4
3. 只重建 parquet 和 metadata

但这个优化版复杂度明显更高，第一版不建议做。

---

## 10. 验收标准

拆分脚本开发完成后，验收必须至少包含以下检查。

## 10.1 左右数据集都能加载

必须能通过 `LeRobotDataset` 正常加载：

1. `dummy/replay_test4_left`
2. `dummy/replay_test4_right`

## 10.2 schema 检查

左臂集：

1. `action.shape == [7]`
2. `observation.state.shape == [7]`
3. 只有 `camera_front` 和 `camera_left`

右臂集：

1. `action.shape == [7]`
2. `observation.state.shape == [7]`
3. 只有 `camera_front` 和 `camera_right`

## 10.3 时序语义检查

必须检查：

1. 左臂集：
   - `left_action_t = left_state_(t+1)`
2. 右臂集：
   - `right_action_t = right_state_(t+1)`

这一步非常关键，不能只检查 shape。

## 10.4 结构一致性检查

必须检查：

1. `total_episodes` 是否等于源数据
2. `total_frames` 是否等于源数据
3. 每条 episode 的长度是否与源数据一致
4. `task_index` 映射是否正常
5. 视频是否可解码

## 10.5 可视化抽检

左右集各抽至少 1 条 episode 做可视化：

1. 左臂集看：
   - `camera_front`
   - `camera_left`
   - 左臂 7 维 state/action 曲线
2. 右臂集看：
   - `camera_front`
   - `camera_right`
   - 右臂 7 维 state/action 曲线

---

## 11. 已知风险

## 11.1 语言标签风险

如果直接复制 `replay_test4` 的任务文本：

- `agilex static record test`

那么左右臂拆分后的数据也会继承这个偏弱标签。

这不会破坏 imitation 数据本身，但会影响 VLA 训练质量。

建议：

1. 第一版默认复制
2. 第二步尽快支持任务文本覆盖

## 11.2 joint names 仍然不是参考语义名

拆分后第一版建议保留：

1. 左臂：
   - `left_joint0.pos ... left_joint6.pos`
2. 右臂：
   - `right_joint0.pos ... right_joint6.pos`

这不是错误，但和更语义化的参考数据集命名仍不完全一致。

## 11.3 单臂数据是否真的适合训练，要看运动分布

拆分脚本本身只负责正确拆分，不负责判断“左臂是不是足够动”“右臂是不是足够动”。

如果一侧 arm 在原始数据里几乎没动，那么拆出来的那一侧数据集也会信息量有限。

所以拆分完成后，还需要补一轮运动统计。

---

## 12. 开发边界

本方案建议第一版只做以下能力：

1. 从一个双臂 LeRobot 数据集导出左右两个单臂数据集
2. 保留 LeRobot 可加载格式
3. 保留 next-state action 语义
4. 保留目标相机
5. 支持基础校验

第一版不做以下内容：

1. joint names 重命名为语义化名称
2. 直接原地改源数据集
3. 大规模视频零拷贝优化
4. 多任务 task 文本复杂映射

---

## 13. 建议的实施顺序

建议按下面顺序落地：

1. 先写拆分脚本主流程
2. 再写验证脚本
3. 先对 `replay_test4` 做一次导出
4. 检查左右两个数据集的：
   - shape
   - 相机
   - lag+1
   - 可加载性
5. 验证通过后，再考虑是否加 task override 和 joint names 归一化

---

## 14. 一句话版本

这件事最稳妥的做法，不是去硬改 `replay_test4` 现有 parquet，而是：

1. 把 `replay_test4` 当源数据集读取
2. 左臂切 `0:7` + `camera_front/camera_left`
3. 右臂切 `7:14` + `camera_front/camera_right`
4. 重新导出为两份新的 LeRobot 数据集
5. 用 `lag+1 == 0` 作为最关键验收标准

