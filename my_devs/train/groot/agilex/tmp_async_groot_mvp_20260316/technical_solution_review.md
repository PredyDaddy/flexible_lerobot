# Technical Solution Review Notes

## Review Scope

审核对象：

- `technical_solution.md`

审核依据：

1. `src/lerobot/async_inference`
2. `my_devs/async_act`
3. `src/lerobot/robots/agilex`
4. `src/lerobot/policies/groot`
5. `outputs/train/groot_agilex_first_test_right_20260315_221522/.../020000/pretrained_model`

## Incorporated Findings

以下问题已在技术方案中显式纳入：

1. 服务端对 `groot` 原生可用，但内置 `async_client()` CLI 不能直接用于 Agilex
2. 当前 checkpoint 是右臂单臂契约，不可直接接整机 `AgileXRobot`
3. `actions_per_chunk = 1` 只会截断服务端返回，不会改变 `groot` 内部 chunk 推理事实
4. fine-tuned checkpoint 仍依赖：
   - `model.safetensors`
   - `base_model_path`
   - Eagle tokenizer assets
5. 客户端若在 one-step 模式下直接复用原始 timestep 语义，会在首个动作后持续丢弃后续动作
6. 跨设备场景下，`policy_path` 必须首先是服务端可见路径
7. 若本地脚本要读取 `groot` checkpoint config，必须先保证 `lerobot.policies` 已导入

## Blocking Issue Status

当前审查未发现仍留在文档中的阻塞级错误。

前提是后续实现必须严格遵守：

1. 右臂 checkpoint only
2. one-step timestep fix 必须落地
3. 不使用内置 `async_client()` CLI 作为 Agilex MVP 入口

## Follow-up Notes

后续如果技术实现改动了以下任意项，需要重新审核技术方案：

1. `actions_per_chunk` 不再固定为 `1`
2. 引入左臂或双臂支持
3. 修改 gRPC 协议
4. 不再复用 `SingleArmAgileXRobot`
