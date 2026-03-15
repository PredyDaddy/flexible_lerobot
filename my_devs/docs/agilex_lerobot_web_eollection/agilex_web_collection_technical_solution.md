# AgileX 数据采集网页技术方案报告

## 1. 背景与约束

目标是在 **不修改** `my_devs/add_robot/agilex/record.sh` 及其相关脚本的前提下，为 AgileX 数据采集提供一个简单、稳定、苹果风格的网页入口。

网页实际包装的核心命令为：

```bash
bash my_devs/add_robot/agilex/record.sh <dataset_name> <episode_time_s> <num_episodes> <fps> <reset_time_s> <resume>
```

本方案同时受以下硬约束约束：

- 只能包装现有脚本，不能改变脚本业务语义。
- 所有服务启动、任务执行、联调与测试都必须在 `lerobot_flex` conda 环境中完成。
- 第一版必须优先保证安全调用、状态可见、误操作可控，而不是功能铺得很广。
- 用户要求主代理只做架构收敛，因此本报告完全基于子代理脑暴文档收敛，不直接参与代码设计实现。

## 2. 子代理输入来源

本技术方案基于以下并行脑暴文档收敛：

- [01_command_interface_brainstorm.md](/home/agilex/cqy/flexible_lerobot/my_devs/docs/agilex_lerobot_web_eollection/tmp_docs/01_command_interface_brainstorm.md)
- [02_repo_web_stack_brainstorm.md](/home/agilex/cqy/flexible_lerobot/my_devs/docs/agilex_lerobot_web_eollection/tmp_docs/02_repo_web_stack_brainstorm.md)
- [03_apple_style_ui_brainstorm.md](/home/agilex/cqy/flexible_lerobot/my_devs/docs/agilex_lerobot_web_eollection/tmp_docs/03_apple_style_ui_brainstorm.md)
- [04_runtime_orchestration_brainstorm.md](/home/agilex/cqy/flexible_lerobot/my_devs/docs/agilex_lerobot_web_eollection/tmp_docs/04_runtime_orchestration_brainstorm.md)
- [05_risks_validation_brainstorm.md](/home/agilex/cqy/flexible_lerobot/my_devs/docs/agilex_lerobot_web_eollection/tmp_docs/05_risks_validation_brainstorm.md)
- [06_user_flow_brainstorm.md](/home/agilex/cqy/flexible_lerobot/my_devs/docs/agilex_lerobot_web_eollection/tmp_docs/06_user_flow_brainstorm.md)

## 3. 最终结论

### 3.1 技术路线结论

第一版最合理的路线是：

- **新建 AgileX 专用轻量 FastAPI Web 应用**
- **整体复用 `my_devs/web_collection` 的实现模式**
- **前端采用原生 HTML/CSS/JS 单页**
- **后端用固定命令模板 + 参数数组方式调用 `record.sh`**
- **运行时采用“单进程 Web 服务 + 持久化任务台账 + 单活跃录制任务互斥”**

这条路线的核心优点：

- 不侵入 `src/lerobot/` 主包。
- 不改动现有 AgileX 脚本。
- 不引入新的 Node/SPA 技术栈。
- 与仓库现有 `FastAPI + 静态页 + subprocess + 轮询日志` 模式一致。
- 能在第一版就把“启动、停止、日志、恢复、风险提示”这层壳做稳。

### 3.2 产品定位结论

第一版网页不是“机器人平台总控制台”，而是：

- 一个 **面向新手和现场采集人员的 `record.sh` 安全执行面板**

它只做四件事：

- 帮用户填写 6 个核心参数
- 帮用户理解即将执行的命令与输出位置
- 帮用户在网页中启动、观察、停止这次采集
- 帮用户在任务结束后看到结果、日志和下一步

## 4. 第一版功能边界

### 4.1 必做范围

第一版建议只支持以下内容：

- 启动一条 `record.sh` 录制任务
- 展示 6 个核心参数表单
- 展示命令预览、输出目录预览、数据集冲突状态
- 展示当前任务状态、最近日志、停止按钮
- 支持刷新页面后恢复当前任务视图
- 支持查看最近任务结果与错误摘要

### 4.2 不做范围

第一版明确不建议做：

- 不支持修改 `record.sh` 内部固定参数
- 不支持任意脚本路径或任意命令执行
- 不把 replay、vis、训练、转换等工作流混进主采集页
- 不支持真正并发录制
- 不承诺“精确进度条”
- 不暴露 topic、camera key、编码器、视频格式等底层细节到主表单
- 不做重型前端框架或复杂仪表盘

## 5. 命令接口到网页表单的映射

### 5.1 六个核心字段

网页首版建议只暴露以下 6 个字段：

| 字段 | 对应命令参数 | 控件建议 | 默认策略 | 说明 |
| --- | --- | --- | --- | --- |
| 数据集名称 | `dataset_name` | 文本输入 | 默认留空，占位提示脚本默认值 | 必须受限命名 |
| 单段录制时长 | `episode_time_s` | 数字输入 | `8` | 支持小数 |
| 录制段数 | `num_episodes` | 步进器 | `1` | 正整数 |
| 帧率 | `fps` | 步进器或常用下拉 | `10`，并提供 `30` 快捷模板 | 正整数 |
| 段间重置时间 | `reset_time_s` | 数字输入 | `0` | 非负数 |
| 继续录制 | `resume` | 开关/分段选择器 | `false` | 映射为 `true/false` |

### 5.2 表单联动规则

必须存在以下联动：

- `num_episodes = 1` 时，`reset_time_s` 自动置 `0` 并禁用。
- `resume = false` 且目标数据集已存在时，禁止提交。
- `resume = true` 且目标数据集已存在时，允许提交并明确提示“将追加到现有数据集”。
- `resume = true` 且目标数据集不存在时，允许用户看到黄色警告，不建议静默放行。
- `dataset_name` 变化时，实时更新命令预览、输出目录预览和冲突状态。

### 5.3 字段校验原则

建议页面与后端都执行同一套校验：

- `dataset_name` 仅允许 `[A-Za-z0-9_-]+`
- `episode_time_s > 0`
- `num_episodes >= 1` 且为整数
- `fps >= 1` 且为整数
- `reset_time_s >= 0`
- `resume` 只落 `true` 或 `false`

## 6. 网页信息架构与交互方案

### 6.1 交互模型

推荐采用：

- **单页表单 + 实时命令预览**
- 顶部补 **2 到 3 个预设卡片**

推荐预设：

- 首次试录：`8s / 1 episode / 30fps / reset=0 / resume=false`
- 两段录制：`8s / 2 episodes / 30fps / reset=5 / resume=false`
- 继续追加：`8s / 1 episode / 30fps / reset=0 / resume=true`

这样做的原因是：

- 命令本身只有 6 个核心参数，不值得做重型向导。
- README 已天然提供两类典型命令，很适合做成产品预设。
- 单页模型最接近用户对命令行的理解，也最易落地。

### 6.2 页面结构

页面建议按以下顺序组织：

1. 顶部轻导航
2. Hero 首屏
3. Readiness 状态条
4. 参数配置区
5. 命令与输出预览区
6. 运行状态区
7. 结果区
8. 极简帮助区

### 6.3 用户任务流

推荐的主流程是：

1. 用户进入页面，先看到环境、设备、目录等 readiness 信息
2. 用户点击一个模板或手动填写参数
3. 页面实时显示命令预览和输出目录
4. 用户点击“开始采集”后看到确认弹窗
5. 用户确认安全项后发起任务
6. 页面切换到运行中状态卡，核心表单锁定
7. 任务结束后页面展示成功或失败摘要、输出路径与下一步建议

## 7. Apple 风格设计原则

### 7.1 应保留的风格特征

页面视觉建议遵循以下方向：

- 浅色主基底，低噪声背景
- 大留白、中心稳定布局
- 大圆角卡片，边界柔和
- 信息层级极少但非常明确
- 轻玻璃感只用于顶部状态条或悬浮层
- 动效只服务于理解，不服务于炫技
- 文案简短、按钮语义清楚

### 7.2 建议的视觉关键词

推荐关键词：

- `calm`
- `precise`
- `soft depth`
- `single-task focus`
- `quiet confidence`

### 7.3 应避免的视觉方向

第一版明确避免：

- 满屏玻璃拟态
- 高饱和科技蓝紫大渐变
- 大面积黑底终端风主界面
- 复杂图表与仪表盘
- 把实时日志作为页面主视觉中心
- 多层导航、多 tab、多抽屉叠加

## 8. 运行时架构方案

### 8.1 推荐运行时模式

第一版推荐：

- **单进程 Web 服务**
- **持久化任务台账**
- **单活跃录制任务互斥**

核心原因：

- 仓库现有模式就是单体 FastAPI + 子进程任务
- 需要的是“把脚本稳定包进网页”，不是分布式调度
- 持久化任务台账可以支撑刷新恢复、日志追踪、失败归因

### 8.2 核心运行边界

第一版建议明确写死这些边界：

- 同一时刻只允许一个运行中的录制任务
- 页面只调用固定 `record.sh`
- 启动命令必须通过参数数组而不是 shell 字符串拼接
- 每个任务有独立的状态文件和日志文件
- 日志必须落盘
- 状态必须落盘
- 进度只能做估算展示

### 8.3 任务状态模型

任务建议至少具备以下状态：

- `pending`
- `starting`
- `running`
- `stopping`
- `succeeded`
- `failed`
- `stopped`

任务记录建议至少包含：

- `job_id`
- 提交时间
- 启动命令摘要
- 参数快照
- 运行环境摘要
- 输出目录
- 当前状态
- 退出码
- `stdout.log` 路径
- `stderr.log` 路径

### 8.4 刷新恢复策略

页面刷新后的推荐恢复路径：

1. 页面启动请求 `GET /api/jobs/active`
2. 若有活跃任务，则请求该任务详情
3. 从日志偏移量继续拉取最新日志
4. 前端重建运行态界面
5. 若无活跃任务，则展示最近一次任务摘要与历史入口

## 9. 建议的后端职责

后端只负责以下职责：

- 校验前端参数
- 检查数据集冲突与目录边界
- 组装固定命令模板
- 以参数数组方式拉起 `record.sh`
- 捕获 stdout/stderr 并写入日志
- 维护任务状态
- 提供状态、日志、历史任务查询 API
- 实现停止任务

后端不负责：

- 改写脚本逻辑
- 改写机器人配置逻辑
- 伪造精确进度
- 直接开放任意 shell 执行能力

## 10. 建议的前端职责

前端只负责以下职责：

- 展示 readiness 信息
- 展示模板和核心参数表单
- 执行表单联动和基础校验
- 展示命令预览和输出目录预览
- 轮询当前任务状态与日志
- 在运行态锁定表单
- 在结果态给出简明结论和下一步建议

## 11. 建议的 API 轮廓

第一版建议具备以下 API 轮廓：

- `GET /api/health`
  - 返回服务环境、是否运行于 `lerobot_flex`、基础就绪状态
- `GET /api/datasets/check?name=...`
  - 返回数据集是否存在、是否允许新建、是否允许续录
- `POST /api/jobs`
  - 提交一次采集任务
- `GET /api/jobs/active`
  - 查询当前活跃任务
- `GET /api/jobs/{job_id}`
  - 查询任务详情
- `GET /api/jobs/{job_id}/logs?offset=...`
  - 拉取增量日志
- `POST /api/jobs/{job_id}/stop`
  - 请求停止任务
- `GET /api/jobs/history`
  - 获取最近任务列表

这只是架构层接口轮廓，不代表最终必须逐字按此命名实现。

## 12. 安全、风险与运维约束

### 12.1 P0 级约束

上线前必须明确以下事项：

- 服务必须从 `lerobot_flex` 环境启动
- 同机只允许单实例、单 worker 运行
- 必须明确是否允许 `0.0.0.0` 暴露；若允许，必须补网络访问控制
- 必须明确“关闭浏览器不会自动停止后台录制”
- 必须有现场机器人安全 SOP

### 12.2 主要风险点

第一版最需要重点防的不是“shell 注入已经发生”，而是这些现实风险：

- 未授权访问
- 路径边界不清
- 参数误填
- 输出目录冲突
- 多实例并发
- 浏览器关闭后的认知偏差
- `uvicorn` 进程退出后的状态丢失
- ROS 断连后的静默坏数据
- 日志不落盘导致无法排障
- 环境漂移导致页面能开但任务不能跑

### 12.3 关键缓解策略

建议在技术方案层明确如下措施：

- 固定命令模板，禁止 shell 字符串拼接
- 路径必须白名单化，避免任意落盘
- 表单和后端双层校验
- 日志落盘并保留任务台账
- 服务部署在 `tmux`、`screen` 或 `systemd` 之下
- 首版禁止真正并发录制
- 录制完成后提供数据完整性检查入口或最小校验结果

## 13. 验收口径

第一版验收至少应覆盖：

- 服务是否由 `lerobot_flex` 启动
- 正常参数能否启动与完成录制
- 非法参数能否被前后端拦截
- 数据集存在/续录逻辑是否正确
- 浏览器刷新后能否恢复当前任务态
- 停止任务后状态是否正确
- 日志是否持久保存
- 输出目录与结果摘要是否可追溯
- 机器人安全确认是否在开始前明确展示

## 14. 推荐目录边界

第一版建议把 AgileX 网页能力与现有 `web_collection` 逻辑分开：

- 复用模式
- 不共用语义边界

推荐原则是：

- 在 `my_devs` 下新建 AgileX 专用目录承载网页应用
- 继续把 `my_devs/add_robot/agilex/` 视为底层脚本与输出根
- 文档仍集中放在 `my_devs/docs/agilex_lerobot_web_eollection/`

## 15. 最终推荐

如果只保留一句结论，本方案的最终推荐是：

- **以 AgileX 专用轻量 FastAPI 单页应用为外壳，复用 `my_devs/web_collection` 的模式，在 `lerobot_flex` 环境中以固定命令模板安全调用 `record.sh`，并通过持久化任务台账、日志落盘和单活跃录制互斥来守住第一版的稳定性、安全性与可维护性。**
