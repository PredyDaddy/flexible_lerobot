# Policies 原理文档索引

本目录按 `src/lerobot/policies/` 的策略模块拆分文档，每个策略单独一份，统一覆盖：

- 策略背景
- 核心原理
- 本仓库实现解读（`configuration_* / modeling_* / processor_*`）
- 训练与推理机制
- 关键超参数
- 适用场景、优势与局限
- 参考资料（含外部链接）

## 文档列表

- [ACT](./act.md)
- [Diffusion Policy](./diffusion.md)
- [Groot](./groot.md)
- [PI0](./pi0.md)
- [PI0 Fast](./pi0_fast.md)
- [PI0.5 (pi05)](./pi05.md)
- [RTC (Real-Time Chunking)](./rtc.md)
- [SAC](./sac.md)
- [SARM](./sarm.md)
- [SmolVLA](./smolvla.md)
- [TD-MPC](./tdmpc.md)
- [VQ-BeT](./vqbet.md)
- [Wall-X](./wall_x.md)
- [XVLA](./xvla.md)

## 说明

- 这些文档优先区分“论文/公开资料中的通用原理”和“本仓库当前代码实现”。
- 对公开资料尚不充分的策略（如部分新模型），文档中明确标注了证据边界，避免把推断写成已证实事实。
