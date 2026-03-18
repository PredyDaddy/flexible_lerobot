# Review 01

## 结论

未发现会推翻当前技术路线的架构级矛盾。主方案与实施 Plan 基本一致，且与仓库现状相符：

- 当前 working path 的确是同步 `select_action()` 路径
- PI05/RTC 的确应切到 `predict_action_chunk()`
- `async_inference` 现状的确更适合作为第二阶段

## Findings

### 中

技术方案与 Plan 都把第一阶段拆得较细，模块化方向是对的，但实现时要警惕“先分层、后落地”的倾向。对当前仓库来说，第一阶段最重要的是尽快做出一个稳定可跑的本地 chunk runner，而不是一次性把 runtime 目录拆成过多细文件。

建议：

- 实现时允许先把 `producer_loop`、`actor_loop`、`queue_controller` 合并为较少文件
- 只要接口边界保留，后续再细分也不晚

### 低

技术方案中已经引用了 `examples/rtc/eval_with_real_robot.py`，但可以再强调一句：

- 第一阶段应明确“参考其运行时语义”，而不是“复制其实现结构”

这样能避免 reviewer 误以为必须完全按 example 的文件结构复刻。

## Residual Risks

1. 如果第一阶段为了“架构整洁”而过度拆模块，落地速度会变慢。
2. 如果实现时没有持续对照 baseline，容易出现结构更漂亮但 bring-up 退化的问题。
3. 第二阶段是否真的迁到 `async_inference`，最终还要取决于第一阶段的观测数据，而不是预设。
