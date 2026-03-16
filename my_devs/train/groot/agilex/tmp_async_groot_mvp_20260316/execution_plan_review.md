# Execution Plan Review Notes

## Review Scope

审核对象：

- `execution_plan.md`

审核目标：

1. 计划是否只覆盖最小一步异步推理闭环
2. 阶段顺序是否能降低 bring-up 风险
3. 验收标准是否与代码现实一致

## Review Conclusion

执行计划当前满足 MVP 要求，原因如下：

1. 先做 `asset/schema/runtime` 三道门禁，再进入编码和联调
2. 把 one-step timestep 修正列为显式工作项，而不是调优项
3. 把 `SingleArmAgileXRobot` 作为既有边界复用，而不是重新设计机器人层
4. 把 `passive_follow` 放在真实控制之前
5. 没有把 ACT temporal ensemble、queue 调优、多臂支持混入 MVP

## Must-keep Constraints

以下约束在执行阶段不能放松：

1. `actions_per_chunk = 1`
2. `chunk_size_threshold = 1.0`
3. 只支持右臂 checkpoint
4. 不依赖内置 `async_client()` CLI
5. 服务端 checkpoint 路径必须是服务端本机可见路径

## Re-review Triggers

如果出现以下任一变化，执行计划需要重新审核：

1. 客户端改为直接读取并强依赖本地 checkpoint 内容
2. 需要支持连续 chunk 控制而非一步动作返回
3. 需要在 MVP 阶段直接进入 `command_master`
4. 想把代码改进 `src/lerobot/async_inference` 主干
