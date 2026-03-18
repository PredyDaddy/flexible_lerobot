# Review 02

## 结论

当前技术方案方向正确，但文档里有一个需要补强的高优先级缺口：

- **实机安全门槛写得不够明确**

对真实机器人来说，这是比“是否优雅接入 async”更优先的事情。

## Findings

### 高

方案和 Plan 已经覆盖 queue、latency、RTC state，但没有把以下安全机制列为第一阶段明确要求：

1. actor loop 在空队列时的安全行为
2. 首个 chunk 尚未准备好时，机器人应保持什么状态
3. 单步 action 幅度或关节增量异常时是否拒绝发送
4. RTC 或 plain chunk 模式出现异常时，是否立即降级为 stop / hold / fallback
5. 是否保留显式 emergency stop / Ctrl+C 后安全停机路径

建议把这部分补到技术方案和 Plan 中，作为第一阶段必须项，而不是“实现时自行考虑”。

### 中

Plan 里虽然提到了 starvation、leftover、delay，但没有明确规定：

- 当 `ActionQueue.get()` 返回 `None` 时 actor thread 的动作策略

建议至少二选一并写死：

1. `hold-last-action`
2. `send-no-action / skip-send`

不能让它变成实现者随意决定的隐式行为。

## Residual Risks

1. 没有安全约束的实时 runner，即使观测指标完整，也可能在异常时产生危险动作。
2. RTC 开启后如果 queue 替换语义出错，缺少降级策略会把问题直接暴露到机器人上。
3. 如果 shutdown 只做线程退出、不做机器人安全收尾，实机风险会被低估。
