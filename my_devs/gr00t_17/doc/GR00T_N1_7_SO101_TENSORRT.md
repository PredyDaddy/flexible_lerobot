# GR00T N1.7 SO101 TensorRT Backend

## 1. 实现范围

RTC policy server 的默认后端已从 PyTorch 切换为 TensorRT 10.15 full pipeline。Client、ZeroMQ 协议、16/8 RTC 时间轴和正式上机参数保持不变。

TensorRT full pipeline 包含 7 个 engine：

1. ViT（FP32 导出，engine 文件沿用参考实现的 `vit_bf16.engine` 名称）
2. LLM（BF16）
3. VL self-attention（BF16）
4. state encoder（BF16）
5. action encoder（BF16）
6. DiT（BF16）
7. action decoder（BF16）

轻量的 token embedding、masked scatter、RoPE index、VLLN、动作归一化和 RTC 控制逻辑仍由 PyTorch 执行。

## 2. RTC 修正

参考实现的 `action_head_tensorrt_forward()` 接受 `options` 参数，但没有使用上一动作块，也没有应用 frozen/overlap/ramp。项目实现 `scripts/so101_rtc_trt.py` 补回了与 N1.7 PyTorch action head 一致的逻辑：

- 从上一条 16 步动作块的后 8 步初始化当前 overlap；
- 推理延迟对应的前 2 步 velocity strength 为 0；
- overlap 的剩余 6 步使用指数 ramp；
- 每个 denoising step 使用 `velocity * velocity_strength` 更新动作。

因此 TensorRT 后端不会让 RTC 静默退化。

## 3. 构建

Engine 已生成在：

```text
artifacts/tensorrt/so101_n17_b1_bf16_full/engines
```

需要重新构建时运行：

```bash
cd /data/cqy_workspace/flexible_lerobot

/home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/build_so101_trt.sh
```

构建脚本固定使用正式 `checkpoint-63600`、转换后的本地数据、batch 1、BF16、8192 MB builder workspace。ONNX 和 engine 均写入 `my_devs/gr00t_17/artifacts/tensorrt`。

## 4. 验证结果

参考 full-pipeline 验证：

```text
ViT cosine:        0.997885
Backbone cosine:   0.999951
Final action:      0.999999
Final action L1:   0.076156
Final action Linf: 0.361862
```

项目 RTC 专项验证：

```text
Initial chunk cosine: 0.9999990
RTC chunk cosine:     0.9999990
RTC chunk L1:         0.062548
RTC chunk Linf:       0.485878
```

完整 engine 哈希和结果位于 `reports/so101_rtc_trt_verification.json`。

真实设备零动作 dry-run 位于：

```text
outputs/inference/trt_smoke/rtc_trt_full_20260712_01/reports/summary.json
```

结果为 90 步、30.0 Hz、0 queue underrun、0 电机命令。稳态模型推理约 29 ms，端到端约 35--47 ms。

## 5. Server/Client

服务端脚本现在默认 `INFERENCE_BACKEND=tensorrt`：

```bash
cd /data/cqy_workspace/flexible_lerobot

SERVER_HOST=127.0.0.1 SERVER_PORT=5556 \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_so101_rtc_policy_server.sh
```

启动完成必须看到：

```text
[RTC SERVER] ready on 127.0.0.1:5556 backend=tensorrt
```

Client 指令不变。仅在诊断 TensorRT 问题时，可显式回退 PyTorch：

```bash
INFERENCE_BACKEND=pytorch SERVER_HOST=127.0.0.1 SERVER_PORT=5556 \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_so101_rtc_policy_server.sh
```

TensorRT engine 与生成它的 TensorRT/CUDA/GPU 架构绑定。更换 GPU、TensorRT 版本或 CUDA 主版本后必须重新构建和验证，不能直接复制使用。
