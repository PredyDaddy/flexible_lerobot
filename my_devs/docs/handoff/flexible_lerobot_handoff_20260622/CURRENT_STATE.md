# Current State

生成日期：2026-06-22  
仓库路径：`/data/cqy_workspace/flexible_lerobot`

## 1. Git 状态快照

当前分支：

```text
feature/cqy...origin/feature/cqy
```

当前 HEAD：

```text
c25e170 pi05trt后端推理
```

最近提交包括：

```text
c25e170 pi05trt后端推理
4d809b0 增加trt后端
76fe1c1 更新vlash服务器客户端
8698ad8 完成vlash结合pi05
5140cc4 测试机器人的a CD推理
e6f4a29 训练异步小改动
d7ec79b wahteaver
499163e feat(groot_trt_async_server): add async TRT server workflow
a18b0a2 feat(groot-agilex): add guarded async actuation bringup
8bf09e4 feat(groot-agilex): add async remote inference workflow
```

## 2. 当前工作区状态

生成 handoff 时观察到以下未提交内容：

```text
.gitignore
my_devs/docs/add_robot/agilex/按照_agilex_配置新机器人详解.md
my_devs/docs/vla_engineering/
my_devs/flash_RT_pi/
my_devs/openpi_trt/docs/lerobot_pi05_trt难点分析.md
my_devs/openpi_trt/docs/openpi_trt实现链路报告.md
```

注意：

- `.gitignore` 当前有未提交修改，内容是新增 FlashRT 参考源码忽略规则：
  `my_devs/flash_RT_pi/reference_source_code/FlashRT-main/`。
- 这些看起来是用户或此前 Agent 新增的工作文档。
- `my_devs/flash_RT_pi/` 是 FlashRT 参考源码/资料目录，约 42 MB。由于 `.gitignore` 规则，`reference_source_code/FlashRT-main/` 现在会显示为 ignored。
- 不要擅自删除、重命名、覆盖或回滚。
- 如果任务要继续 PI0.5 / VLA / AgileX / FlashRT 相关工作，应优先阅读这些未跟踪内容。

## 3. 本次 handoff 新增内容

本次只新增交接包：

```text
my_devs/docs/handoff/flexible_lerobot_handoff_20260622/
my_devs/docs/handoff/flexible_lerobot_handoff_20260622.zip
```

没有修改核心代码。当前工作区还存在上面列出的非 handoff 改动，处理它们前请先确认用户意图。

## 4. 环境状态

仓库要求：

```text
conda env: lerobot_flex
python: >= 3.10
```

所有脚本、测试、格式化、lint 都必须通过该环境运行，例如：

```bash
conda run -n lerobot_flex python my_devs/.../script.py
conda run -n lerobot_flex pytest -q tests/...
conda run -n lerobot_flex ruff check ...
```

TensorRT 相关工作通常还要求：

- CUDA 可用。
- TensorRT Python API 或 `trtexec` 可用，取决于具体路径。
- 部分脚本约定 TensorRT 目录为 `/data/cqy_workspace/third_party/tensorrt_10_13_0_35`。
- 运行大型模型时可能需要设置 `TMPDIR=/data/cqy_workspace/tmp`，避免 `/tmp` 空间不足。

## 5. 当前最容易误踩的坑

1. `my_devs/act_trt/` 是历史 ACT TRT 脚本区，没有 `README.md`。
   - 当前更可复现的 ACT TRT 工作区是 `my_devs/new_act_trt/`。
   - 历史复现文档在 `my_devs/docs/act_trt/act_trt复现与参考工作流.md`。

2. PI0.5 TensorRT 不建议直接从“完整 `sample_actions` 单体 engine”入手。
   - 当前文档推荐 split TRT：`prefix_cache engine + denoise_step engine + Python 10-step denoise loop`。
   - 详见 `my_devs/openpi_trt/docs/openpi_trt实现链路报告.md` 和 `my_devs/openpi_trt/docs/lerobot_pi05_trt难点分析.md`。

3. `src/lerobot/` 是主包，默认不要动。
   - 用户明确说主要开发在 `my_devs/`。
   - 需要接主包时先确认。

4. 大量目录里有生成物。
   - 不要把 `__pycache__`、ONNX、engine、checkpoint、大 zip、运行输出纳入交付，除非用户明确要。

5. 上机类命令有真实机器人风险。
   - 真机前先跑 dry-run / mock / consistency。
   - 核对串口、相机 index、calibration、任务文本、动作维度和安全阈值。

## 6. 建议下一个 Agent 开工前执行

```bash
cd /data/cqy_workspace/flexible_lerobot
git status --short --branch
find .. -name AGENTS.md -print
conda run -n lerobot_flex python --version
conda run -n lerobot_flex python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果任务涉及 TensorRT：

```bash
conda run -n lerobot_flex python -c "import tensorrt as trt; print(trt.__version__)"
```

如果 `tensorrt` import 失败，先阅读对应功能 README，很多脚本支持通过 `TENSORRT_PY_DIR` 指向本地 TensorRT Python 包。
