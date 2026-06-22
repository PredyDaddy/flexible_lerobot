# LeRobot PI0.5 TensorRT 化难点分析

日期：2026-06-12  
仓库：`/data/cqy_workspace/flexible_lerobot`  
项目目录：`my_devs/openpi_trt`  
目标模型：LeRobot PI0.5 / SO101 top+wrist 双相机 checkpoint

## 1. 这份文档说明什么

这份文档专门解释 PI0.5 推理链路里各个模块在 TensorRT 化过程中可能遇到的困难，以及为什么当前 `openpi_trt` 最终选择：

```text
prefix_cache TensorRT engine
  + denoise_step TensorRT engine
  + Python 10-step denoise loop
```

而不是直接把完整 `sample_actions(...)` 一次性导出成一个大 TensorRT engine。

讨论的环节包括：

```text
vision encoder
language embedding
prefix attention mask
PaliGemma prefix forward
past_key_values cache
10-step denoise loop
Gemma expert transformer
RoPE
AdaRMS / adarms_cond
action_out_proj
```

## 2. 总体判断

这些环节并不是同等难度。

相对容易的部分：

```text
language embedding
action_out_proj
普通 Linear / MLP / MatMul / Add / Reshape
```

中等困难的部分：

```text
vision encoder
prefix attention mask
AdaRMS / adarms_cond
prefix_cache 输出 I/O 管理
```

最困难的部分：

```text
PaliGemma prefix forward
past_key_values cache
10-step denoise loop
Gemma expert transformer
RoPE
FP16 下的 LayerNorm / Softmax / Reduce 数值稳定性
```

原因是 TensorRT 化不是单纯“代码里调用了某个网络，就能自动变成 engine”。它必须经过两道门：

```text
PyTorch wrapper
  -> torch.onnx.export 能不能捕获成 ONNX

ONNX graph
  -> TensorRT parser/builder 能不能解析并构建成 engine
```

所以困难通常来自四类问题：

```text
1. PyTorch -> ONNX 导出困难
2. ONNX -> TensorRT build 困难
3. TensorRT runtime I/O 管理困难
4. Torch / ONNX / TensorRT 数值一致性困难
```

## 3. vision encoder

### 3.1 它做什么

PI0.5 输入有 top/wrist 两路图像。图像会先进入视觉编码器，变成 image token embedding。

对应源码概念：

```text
self.paligemma_with_expert.embed_image(img)
```

输出会和 language token embedding 一起组成 prefix。

### 3.2 TRT 化难点

视觉编码器本身通常由比较规则的算子组成：

```text
Conv / Linear / MatMul / LayerNorm / Reshape / Add
```

所以单看算子，它不是最难的部分。

但实际仍然有几个难点。

第一，输入 shape 必须固定。

真实相机帧通常是：

```text
[480, 640, 3]
```

但 TensorRT prefix_cache engine 看到的是预处理后的图像：

```text
image_0: [1, 3, 224, 224]
image_1: [1, 3, 224, 224]
```

也就是说，resize、pad、normalize 等图像预处理没有放进 TensorRT engine，而是在 LeRobot policy/preprocessor 侧完成。

第二，dtype 要和模型导出策略一致。

当前主线为了稳定，使用：

```text
model_dtype: float32
prefix_cache engine: FP32
```

如果改成 BF16/FP16，需要重新验证。

第三，prefix_cache engine 体积会很大。

视觉编码器进入 `prefix_cache` 后，prefix_cache engine 会包含大量 prefix 侧权重。当前已有 engine 大约：

```text
pi05_so101_prefix_cache_b1_fp32.engine: 约 11 GB
```

### 3.3 难点总结

```text
vision encoder 算子本身不是最难；
主要难在输入预处理边界、固定 shape、dtype 和 engine 体积。
```

## 4. language embedding

### 4.1 它做什么

PI0.5 会把 task 文本和 state 处理成 prompt，再 tokenize 成：

```text
tokens: [1, 200]
masks:  [1, 200]
```

language embedding 负责：

```text
tokens -> token embeddings
```

对应源码概念：

```text
self.paligemma_with_expert.embed_language_tokens(tokens)
```

### 4.2 TRT 化难点

language embedding 本身通常可以导成：

```text
Gather / embedding lookup
```

难度不高。

但要注意两个边界。

第一，tokenizer 不在 TensorRT engine 里。

真实流程是：

```text
task 文本
observation.state
  -> policy preprocessor
  -> state normalize / discretize
  -> prompt 拼接
  -> tokenizer
  -> tokens / masks
```

TensorRT engine 只看到：

```text
tokens
masks
```

它不会处理原始字符串，也不会处理原始 state。

第二，token 长度固定。

当前导出假设：

```text
tokens: [1, 200]
```

如果 `tokenizer_max_length` 或 checkpoint 配置变化，需要重新导出和构建。

### 4.3 难点总结

```text
language embedding 算子容易；
难点在于 tokenizer/preprocessor 不属于 TRT，且 token length 固定。
```

## 5. prefix attention mask

### 5.1 它做什么

prefix 由图像 token 和语言 token 组成。模型需要构造 attention mask，决定哪些 token 可以互相 attend。

prefix_cache wrapper 里会做：

```text
prefix_att_2d_masks = make_att_2d_masks_for_onnx(prefix_pad_masks, prefix_att_masks)
prefix_position_ids = cumsum(prefix_pad_masks) - 1
prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
```

### 5.2 TRT 化难点

这部分不是最重的计算，但非常容易导致导出和构建问题。

第一，bool tensor 的 `cumsum` 对 ONNX/ORT 不友好。

原始逻辑可能对 bool mask 直接做：

```text
torch.cumsum(att_masks, dim=1)
```

ONNX/ORT 对 bool `CumSum` 支持不好，所以 `openpi_trt` 使用了 ONNX-friendly 版本：

```python
cumsum = torch.cumsum(att_masks.to(dtype=torch.int64), dim=1)
```

也就是：

```text
bool -> int64 -> cumsum
```

第二，mask shape 和 broadcast 容易出错。

当前 prefix 长度大约是：

```text
prefix_len = 712
```

suffix/action chunk 长度是：

```text
chunk_size = 50
```

后续 denoise_step 还要构造：

```text
suffix query -> prefix + suffix key/value
```

的 full attention mask。只要 prefix/suffix 长度、维度扩展或 concat 方向错了，ONNX 可能能导出，但 runtime 数值会错。

第三，mask dtype 混用复杂。

这里经常同时出现：

```text
bool
int64
float32
```

TensorRT 对某些 bool/int64/shape 组合没有 PyTorch 灵活。

### 5.3 难点总结

```text
prefix attention mask 不是最重模块，
但它是 PyTorch -> ONNX 阶段最容易踩坑的地方之一。
```

## 6. PaliGemma prefix forward

### 6.1 它做什么

prefix_cache 的核心是跑一次 PaliGemma prefix forward：

```text
prefix_embs
prefix attention mask
position ids
  -> PaliGemma prefix forward(use_cache=True)
  -> past_key_values
```

对应 wrapper 中的逻辑是：

```python
_, past_key_values = self.model.paligemma_with_expert.forward(
    attention_mask=prefix_att_2d_masks_4d,
    position_ids=prefix_position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs, None],
    use_cache=True,
)
```

### 6.2 TRT 化难点

第一，Transformer 图很大。

内部包括：

```text
Attention
MLP
Norm
Residual
RoPE / position handling
KV cache generation
```

这会导致：

```text
ONNX external data 很多
TensorRT engine 很大
build 时间长
显存/磁盘压力大
```

第二，attention 实现必须固定。

Transformers 可能根据环境使用：

```text
eager
sdpa
flash attention
其他 fused attention
```

很多 fused attention 路径对 ONNX/TensorRT 不友好。当前 wrapper 里强制：

```python
_attn_implementation = "eager"
```

目的是让导出路径尽可能变成普通 tensor ops。

第三，`use_cache=True` 会产生复杂输出。

它不是单一 tensor 输出，而是产生每层 attention 的 key/value cache。这个输出必须展开成 TensorRT 能表示的多个 tensor。

### 6.3 难点总结

```text
PaliGemma prefix forward 是 prefix_cache TRT 的核心难点：
图大、层多、attention 实现复杂、cache 输出复杂。
```

## 7. past_key_values cache

### 7.1 它做什么

`past_key_values` 是每一层 attention 的 key/value cache，不是每层 transformer 的 hidden output。

当前 prefix_cache 里有：

```text
past_key_values.0.key
past_key_values.0.value
...
past_key_values.17.key
past_key_values.17.value
```

每层 shape 大致是：

```text
key:   [1, 1, 712, 256]
value: [1, 1, 712, 256]
```

### 7.2 TRT 化难点

第一，`DynamicCache` 是 Python 对象。

Transformers 返回的 cache 通常是：

```text
DynamicCache
```

ONNX/TensorRT 不能输出 Python 对象，只能输出 tensor。所以必须：

```text
flatten_past_key_values(...)
```

把它展开成：

```text
past_key_values.0.key
past_key_values.0.value
...
past_key_values.17.key
past_key_values.17.value
```

第二，I/O 数量很多。

18 层，每层 key/value 两个：

```text
18 * 2 = 36
```

再加：

```text
prefix_pad_masks
```

prefix_cache engine 一共有：

```text
37 个输出
```

每个输出都要命名、绑定、验证，并传给 denoise_step engine。

第三，中间 tensor 数值验证更敏感。

历史验证里观察到：

```text
prefix_pad_masks 完全一致
past_key_values 整体 cosine 很高
后层 value cache 有少量 outlier
```

在严格：

```text
rtol=0.02, atol=0.1
```

下，中间 cache 可能不完全 allclose。用：

```text
atol=1.0
```

复验通过。

这些 outlier 没有在最终 action chunk 上明显放大，所以最终判断还要看：

```text
split sample_actions
camera smoke
真实任务成功率
```

### 7.3 难点总结

```text
past_key_values 的难点不是单个算子，
而是 Python 对象展开、I/O 管理、shape/dtype 对齐和中间态数值验证。
```

## 8. 10-step denoise loop

### 8.1 它做什么

PI0.5 不是一步生成 action，而是从 noise 开始迭代：

```text
x_t = noise
for step in range(num_inference_steps):
    time = 1.0 + step * dt
    v_t = denoise_step(...)
    x_t = x_t + dt * v_t
return x_t
```

当前：

```text
num_inference_steps = 10
```

### 8.2 TRT 化难点

第一，Python loop 不适合直接进入一个大 TensorRT 图。

固定 10 次理论上可以展开，但每一步都包含复杂 transformer forward。如果完整展开，会导致：

```text
ONNX 极大
TensorRT build 时间极长
engine 极大
调试困难
```

第二，随机 noise 不能放在图里。

为了 Torch/ONNX/TRT 一致性，noise 必须显式作为输入：

```text
noise: [1, 50, 32]
```

不能让 TensorRT engine 内部随机生成。

第三，误差会跨 step 累积。

单步 `v_t` 的微小误差经过 10 步更新：

```text
x_t = x_t + dt * v_t
```

可能放大到最终 action。因此必须验证完整 `sample_actions`，不能只看单步 denoise。

### 8.3 当前处理方式

当前保留 Python loop：

```text
prefix_cache engine 跑 1 次
denoise_step engine 跑 10 次
Python 负责 x_t 更新和 step 调度
```

这样每个子图小很多，也方便逐层验证。

### 8.4 难点总结

```text
10-step denoise loop 的主要困难是控制流、图膨胀和误差累积。
当前保留 Python loop 是工程可控的选择。
```

## 9. Gemma expert transformer

### 9.1 它做什么

在 denoise_step 里，suffix/action token 会进入 Gemma expert transformer。

对应逻辑：

```python
outputs_embeds, _ = self.model.paligemma_with_expert.forward(
    attention_mask=full_att_2d_masks_4d,
    position_ids=position_ids,
    past_key_values=past_key_values,
    inputs_embeds=[None, suffix_embs],
    use_cache=False,
    adarms_cond=[None, adarms_cond],
)
```

这里 suffix forward 会带上 prefix 的 `past_key_values`，所以不是孤立的 suffix forward。

### 9.2 TRT 化难点

第一，Transformer 层多且算子组合复杂。

内部有：

```text
Q/K/V projection
attention scores
softmax
attention output
MLP
norm
residual
RoPE
AdaRMS
```

第二，suffix 要 attend 到 prefix cache。

denoise_step 输入包含：

```text
prefix_pad_masks
past_key_values.*
x_t
timestep
```

attention 的 key/value 来源包含：

```text
prefix key/value
suffix current key/value
```

mask 和 position ids 必须对齐。

第三，FP16 数值漂移。

Transformer 中的：

```text
LayerNorm / RMSNorm
Softmax
Reduce
Sqrt
Div
```

在 FP16 下容易带来误差。因此 FP16 engine 构建时需要使用 precision constraints，把敏感层保留为 FP32。

### 9.3 难点总结

```text
Gemma expert transformer 是 denoise_step TRT 的核心计算，
也是数值稳定性和导出复杂度最高的部分之一。
```

## 10. RoPE

### 10.1 它做什么

RoPE 是 rotary position embedding，用于 attention 中的位置编码，通常会作用在 Q/K 上。

### 10.2 TRT 化难点

第一，PyTorch 实现形式可能复杂。

RoPE 可能涉及：

```text
sin / cos
reshape / split / concat
复杂 dtype 处理
complex-like 运算
```

已有完整单体 `sample_actions` 导出尝试里，legacy exporter 遇到过：

```text
ComplexDouble / RoPE 相关导出失败
```

第二，RoPE 和 attention 实现路径绑定较深。

如果模型走 fused attention / sdpa / flash attention，exporter 可能捕获不了。当前导出 wrapper 强制：

```text
_attn_implementation = "eager"
```

目的就是让 RoPE/attention 走普通 tensor ops。

### 10.3 难点总结

```text
RoPE 是 PyTorch -> ONNX 阶段的典型困难点，
常见问题是 unsupported dtype/op、图捕获失败或数值差异。
```

## 11. AdaRMS / adarms_cond

### 11.1 它做什么

`adarms_cond` 是 denoise_step 中的 timestep 条件信息。

在 `embed_suffix(...)` 中：

```text
timestep
  -> sinusoidal position embedding
  -> time MLP
  -> adarms_cond
```

然后传入 transformer：

```text
adarms_cond=[None, adarms_cond]
```

用于影响 Gemma expert 里的条件归一化 / AdaRMS。

### 11.2 TRT 化难点

第一，条件 norm 比普通 LayerNorm 更复杂。

普通 norm 只依赖 hidden states，而 AdaRMS 还依赖：

```text
adarms_cond
```

也就是 timestep 条件。

第二，FP16 下 norm 数值敏感。

Norm 类算子常涉及：

```text
Reduce
Sqrt
Div
Add
Mul
```

FP16 下容易出现精度漂移。

第三，ONNX/TensorRT 可能把 norm 分解成多个基础算子。

例如：

```text
ReduceMean
Sub
Pow
ReduceMean
Add
Sqrt
Div
Mul
Add
```

虽然 TensorRT 可以 build，但 FP16 精度需要小心。

### 11.3 难点总结

```text
AdaRMS / adarms_cond 的难点主要是条件归一化和 FP16 数值稳定。
```

## 12. action_out_proj

### 12.1 它做什么

denoise_step 最后会把 suffix hidden state 投影成动作速度：

```text
suffix_out -> action_out_proj -> v_t
```

输出：

```text
v_t: [1, 50, 32]
```

### 12.2 TRT 化难点

这一步相对简单，通常只是：

```text
Linear / MatMul + Bias
```

TensorRT 很擅长处理。

主要注意：

```text
输入 dtype
输出 dtype
suffix_out shape
```

### 12.3 难点总结

```text
action_out_proj 基本不是导出难点，也不是主要性能瓶颈。
```

## 13. 按阶段总结容易出问题的位置

### 13.1 PyTorch -> ONNX 导出阶段

容易出问题：

```text
RoPE
DynamicCache / past_key_values
10-step denoise loop
prefix/suffix full forward 的复杂 Python 路径
attention mask 的 bool cumsum
Transformers fused attention / sdpa 路径
```

原因：

```text
ONNX exporter 需要捕获 tensor graph；
Python 对象、动态结构、复杂控制流、不支持 dtype/op 都会导致失败。
```

### 13.2 ONNX -> TensorRT build 阶段

容易出问题：

```text
prefix_cache engine 图太大
past_key_values 输出太多
int64 / bool shape 处理
某些 ONNX op TensorRT 不支持
FP16 precision constraints
ONNX external data 文件路径
```

原因：

```text
TensorRT 只解析 ONNX；
ONNX 图越大、I/O 越多、op/dtype 越复杂，build 越容易失败或消耗大量资源。
```

### 13.3 TensorRT runtime 阶段

容易出问题：

```text
input name 不匹配
dtype 不匹配
shape 不匹配
past_key_values 顺序不匹配
prefix_cache 输出和 denoise_step 输入对不上
engine 和当前 GPU/TensorRT 版本不匹配
```

原因：

```text
TensorRT runtime 按名字、shape、dtype 绑定 tensor 地址；
任何一项错了都可能失败或输出错误。
```

### 13.4 数值一致性阶段

容易出问题：

```text
prefix cache 中间 value cache outlier
FP16 LayerNorm / Softmax / Reduce
AdaRMS 条件归一化
RoPE 位置编码误差
denoise loop 误差累积
```

原因：

```text
单个子图的小误差可能经过 10-step denoise 累积；
最终需要看 split sample_actions 和真实 camera smoke 是否对齐。
```

## 14. 为什么当前拆成 prefix_cache + denoise_step

完整单体图：

```text
sample_actions:
  prefix + cache + denoise loop + Gemma expert + RoPE + AdaRMS
```

太大、太复杂、太难调。

当前拆成：

```text
prefix_cache TRT:
  负责图像/语言 prefix 编码和 past_key_values 生成

denoise_step TRT:
  负责单步 action denoise transformer forward

Python loop:
  负责 10-step 调度和 x_t 更新
```

这样做的好处：

```text
每个子图边界清楚
每个子图可以单独验证
cache I/O 显式管理
denoise loop 不导致 ONNX 图膨胀 10 倍
最终 sample_actions 仍然能组合验证
```

## 15. 总结表

| 环节 | TRT 难度 | 主要难点 |
| --- | ---: | --- |
| vision encoder | 中 | 图大、输入 shape 固定、dtype、engine 体积 |
| language embedding | 低到中 | tokenizer 不在图里、token 长度固定 |
| prefix attention mask | 中到高 | bool cumsum、mask shape、dtype、broadcast |
| PaliGemma prefix forward | 高 | Transformer 大图、attention 实现、cache 输出 |
| past_key_values cache | 高 | DynamicCache 对象、36 个 KV tensor、I/O 管理 |
| 10-step denoise loop | 高 | Python loop、图膨胀、误差累积 |
| Gemma expert transformer | 高 | attention/MLP/norm/RoPE/AdaRMS，数值敏感 |
| RoPE | 高 | exporter 支持、ComplexDouble、位置编码 reshape/sin/cos |
| AdaRMS / adarms_cond | 中到高 | 条件 norm、FP16 数值稳定、Reduce/Sqrt/Div |
| action_out_proj | 低 | 普通 Linear，基本容易 |

## 16. 最终结论

PI0.5 TensorRT 化最难的不是普通线性层，也不是最终的 `action_out_proj`，而是：

```text
Transformer attention
past_key_values cache
RoPE
AdaRMS / norm
attention mask
denoise loop
```

这些东西叠在一起，会同时带来：

```text
导出难
build 难
runtime I/O 难
数值对齐难
```

所以 `openpi_trt` 当前采用的工程策略是合理的：

```text
不强行导出完整 sample_actions；
而是拆成 prefix_cache TRT 和 denoise_step TRT，
再用 Python loop 组合回原始 PI0.5 推理语义。
```

这条路线牺牲了一点“单 engine 纯粹性”，但换来了：

```text
边界可控
验证可分层
错误可定位
最终 action chunk 一致性可验证
真实部署可接入
```
