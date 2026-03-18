# R2 TRT Backend 执行简报

更新时间：2026-03-15

本文档用于给实现子代理直接执行 `TRT backend` 整改与验证，不再重复架构背景。

## 1. 写入边界

- `my_devs/groot_trt_async_server/groot_trt_policy.py`
- 必要时可新增 `my_devs/groot_trt_async_server/` 下轻量验证脚本

## 2. 这包必须解决什么

### 2.1 保持既有保护不回退

不允许回退：

- 共享 `TrtSession` 的串行化保护
- image-token slots 与 ViT tokens 的一致性校验
- 返回前的 CUDA stream 同步

### 2.2 补最小可信验证闭环

当前问题：

- 现有修复更多是“代码上看起来合理”
- 还缺少最小的自检、mock 验证或轻量验证入口

必须达到的结果：

- 新增一个最小可信验证入口
- 能说明当前防护至少被运行过一次，不只是写在代码里

允许的实现方向：

- 轻量验证脚本
- 自检入口
- mock session 验证
- 样本契约检查

### 2.3 明确已验证 / 未验证边界

必须达到的结果：

- 明确哪些项已经被验证
- 明确哪些项仍需要真实 gRPC 并发压测或真实 processor 样本验证

## 3. 不允许做的事

- 不允许大改模型数学边界
- 不允许把“未验证”写成“已验证”
- 不允许为了凑通过而删除现有保护

## 4. 最低可接受验证

必须至少给出下面内容中的一项：

1. `lerobot_flex` 环境里的 `py_compile`
2. 新增验证脚本的实际运行结果
3. 自检入口的运行结果

如果环境限制导致无法完成更深验证，必须明确写出受限点。

## 5. 完成回报模板

1. `Implemented`
2. `Verification`
3. `Residual Risks`
4. `Files Changed`
5. `Ready For Re-Review`
