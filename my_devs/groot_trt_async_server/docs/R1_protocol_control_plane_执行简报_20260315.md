# R1 Protocol / Control-Plane 执行简报

更新时间：2026-03-15

本文档用于给实现子代理直接执行 `protocol/control-plane` 联合整改，不再重复架构背景。

## 1. 写入边界

- `my_devs/groot_trt_async_server/configs.py`
- `my_devs/groot_trt_async_server/policy_server.py`
- `my_devs/groot_trt_async_server/run_server.py`
- `my_devs/groot_trt_async_server/robot_client.py`
- `my_devs/groot_trt_async_server/run_client.py`
- `my_devs/groot_trt_async_server/run_mock_roundtrip.py`

## 2. 这包必须解决什么

### 2.1 server 资源所有权

当前问题：

- client 仍在决定 server 本地加载哪套 checkpoint / engine

必须达到的结果：

- server 端基于 server-side 资源标识、白名单或等价机制决定本地加载资源
- client 不再直接把本地路径当作 server 最终加载决策

### 2.2 sticky session 可恢复性

当前问题：

- 现在更接近基于 `context.peer()` 的硬拒绝
- 没有 release / reconnect / takeover

必须达到的结果：

- 至少具备一种可恢复策略
- 行为必须是显式的，不靠注释或文档约束

### 2.3 request / ack 协议闭环

当前问题：

- `request_id` 只存在于 client 本地
- 没有显式 ack
- 旧 chunk / 晚到 chunk / 重复 chunk 只能靠本地推断兜底

必须达到的结果：

- `request_id` 成为协议字段
- server 返回与请求对应的最小 ack 或等价信息
- client 显式丢弃旧 chunk、晚到 chunk、重复 chunk

### 2.4 纯协议级 mock

当前问题：

- `run_mock_roundtrip.py` 仍依赖真实 robot/camera

必须达到的结果：

- 提供纯协议级 mock 模式，能在没有 robot/camera 的前提下验证 client/server 交互

## 3. 不允许回退的既有修复

- `--robot-type` 必须继续真正生效
- TRT 不得再静默降级成 PyTorch
- `from_payload()` 不得恢复静默降级
- `run_server.py --dry-run` 不得恢复伪成功

## 4. 最低可接受验证

必须至少给出下面两类验证中的一类：

1. `lerobot_flex` 环境里的 `py_compile`
2. 不依赖真实设备的最小 smoke 或纯 mock roundtrip

如果受 `grpc` 阻塞无法完成更深验证，必须明确写出：

- 哪些验证已完成
- 哪些验证因为环境阻塞未完成

## 5. 完成回报模板

1. `Implemented`
2. `Verification`
3. `Residual Risks`
4. `Files Changed`
5. `Ready For Re-Review`
