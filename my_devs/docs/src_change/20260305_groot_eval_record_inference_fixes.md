# GR00T Eval/Record 推理失败修复报告 (2026-03-05)

## 1. 背景

在 `lerobot_flex` conda 环境下运行评估录制脚本：

```bash
bash my_devs/train/groot/so101/run_groot_eval_record.sh
```

脚本内部实际调用的是 `lerobot-record`，并加载 fine-tune 后的 Groot policy：

- policy path: `outputs/train/.../checkpoints/last/pretrained_model`
- base model path: `assets/modelscope/GR00T-N1.5-3B` (本地)

本次修复针对“之前能推理，现在突然不能”的两类报错：

1. **加载 fine-tuned checkpoint 时 strict 加载失败**：缺少 `embed_tokens.weight` key。
2. **进入 episode 0 后立刻退出**：Eagle fast image processor 在当前 Transformers 版本下 AttributeError。

## 2. 环境信息 (复现时关键版本)

以下版本来自复现时终端输出/本地环境查询：

- conda env: `lerobot_flex`
- Python: 3.10 (conda env 内)
- `safetensors`: 0.7.0
- `transformers`: 4.53.3

## 3. 改动总览 (本次为推理恢复做的代码修改)

本次修复只涉及 `src/lerobot/policies/groot/` 下两个文件：

1. `src/lerobot/policies/groot/modeling_groot.py`
2. `src/lerobot/policies/groot/eagle2_hg_model/image_processing_eagle2_5_vl_fast.py`

说明：

- 当前工作区 `git status` 里还有一些 `my_devs/groot_trt/*` 的改动/新增文件，但它们不属于本次“groot 推理失败”
  的修复范围，也不是本次排查为了解决报错必须修改的部分。本报告仅记录与推理恢复直接相关的两处修复。

## 4. 改动 1：GrootPolicy 加载 checkpoint 时默认不再 strict

### 4.1 现象与错误信息

运行 `lerobot-record` 时，在加载 fine-tuned checkpoint (目录中存在 `model.safetensors`) 触发错误：

- `RuntimeError: Error(s) in loading state_dict for GrootPolicy`
- `Missing key(s) in state_dict: "_groot_model.backbone.eagle_model.language_model.model.embed_tokens.weight"`

### 4.2 根因分析

1. `GrootPolicy.from_pretrained()` 检测到本地目录存在 `model.safetensors` 后，会走 `super().from_pretrained(...)`
   的逻辑，即 `PreTrainedPolicy._load_as_safetensor(...)`。
2. `PreTrainedPolicy.from_pretrained(...)` 的 `strict` 默认是 `False`，但 `GrootPolicy.from_pretrained(...)`
   之前把默认改成了 `strict=True`，导致 safetensors 载入时严格要求**每个参数 key 都必须存在**。
3. 但很多 LLM 会做 weight tying（例如 `embed_tokens.weight` 与 `lm_head.weight` 共享同一份 tensor）。
   在这种情况下，`safetensors.torch.save_model(...)` 为了避免重复存储，可能只保存其中一个 key。

本次实际检查到的 checkpoint 现象：

- `model.safetensors` 中存在：`_groot_model.backbone.eagle_model.language_model.lm_head.weight`
- `model.safetensors` 中不存在：`_groot_model.backbone.eagle_model.language_model.model.embed_tokens.weight`

因此 strict 加载会失败，但这并不意味着 checkpoint 损坏。

### 4.3 修改点

文件：

- `src/lerobot/policies/groot/modeling_groot.py`

修改内容：

- 将 `GrootPolicy.from_pretrained(..., strict: bool = True, ...)` 的默认值改为 `False`
- 在 docstring 中补充说明：tied weights + safetensors 去重可能导致部分 key “合法缺失”

补丁摘要：

```diff
-        strict: bool = True,
+        strict: bool = False,
...
-            strict: Strict state dict loading
+            strict: Strict state dict loading.
+
+                Note: `safetensors.torch.save_model(...)` drops duplicated tensors when weights are tied
+                (e.g. `embed_tokens.weight` tied with `lm_head.weight`). Fine-tuned checkpoints produced by
+                LeRobot may therefore legitimately miss some tied-weight keys. Using `strict=False` keeps
+                the base weights already loaded from `base_model_path` and applies the fine-tuned deltas.
```

### 4.4 预期行为变化与影响

- 现在加载 fine-tuned checkpoint 时不会因为缺少 `embed_tokens.weight` 直接崩溃。
- 仍然会出现 WARNING 级别日志提示缺失 key，这是预期行为（非致命）。
- 如果未来确实需要严格校验 checkpoint 完整性，仍可在调用 `from_pretrained(..., strict=True)` 显式开启。

## 5. 改动 2：Eagle25VLImageProcessorFast 与 Transformers 版本兼容

### 5.1 现象与错误信息

模型权重加载通过后，进入录制循环时在 episode 0 立刻停止，报错：

- `AttributeError: 'Eagle25VLImageProcessorFast' object has no attribute '_prepare_image_like_inputs'`

调用栈显示在 fast image processor 的 `preprocess()` 中触发。

### 5.2 根因分析

vendored 文件 `image_processing_eagle2_5_vl_fast.py` 在 `preprocess()` 中调用了
`self._prepare_image_like_inputs(...)`，但在当前环境 `transformers==4.53.3` 中：

- `BaseImageProcessorFast` **存在**：`_prepare_input_images(...)`
- `BaseImageProcessorFast` **不存在**：`_prepare_image_like_inputs(...)`

因此在 episode 0 首次对图片做预处理时就会直接 AttributeError，导致录制中断。

### 5.3 修改点

文件：

- `src/lerobot/policies/groot/eagle2_hg_model/image_processing_eagle2_5_vl_fast.py`

修改内容：

- 在 `preprocess()` 内部做兼容选择（不同 transformers 版本 helper 命名不同）：
- 优先使用 `_prepare_image_like_inputs`（如果存在）
- 否则回退到 `_prepare_input_images`
- 如果两者都不存在，才抛出明确异常

补丁摘要：

```diff
-        # transformers >= 4.53.0: uses _prepare_image_like_inputs instead of _prepare_input_images
+        # Transformers has changed internal helper names across versions.
+        # - Some versions expose `_prepare_input_images(...)` (BaseImageProcessorFast)
+        # - Some versions expose `_prepare_image_like_inputs(...)`
+        # We support both to keep this vendored processor working across environments.
+        prepare_fn = getattr(self, "_prepare_image_like_inputs", None) or getattr(
+            self, "_prepare_input_images", None
+        )
+        if prepare_fn is None:
+            raise AttributeError(
+                "Neither `_prepare_image_like_inputs` nor `_prepare_input_images` exists on the "
+                "base image processor."
+            )
...
-            images = self._prepare_image_like_inputs(
+            images = prepare_fn(
...
-            videos = self._prepare_image_like_inputs(
+            videos = prepare_fn(
```

### 5.4 预期行为变化与影响

- 解决了 `transformers==4.53.3` 下的 AttributeError，episode 0 不再“刚开始就退出”。
- 该修复对不同 transformers 版本更鲁棒（兼容两种内部 helper 命名）。

## 6. 关于缓存与生效方式 (容易踩坑点)

Eagle 相关文件会在运行时从仓库 vendor 目录复制到 cache：

- vendor 源码目录：`src/lerobot/policies/groot/eagle2_hg_model/`
- cache 目标目录：`~/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5/`

随后 Transformers 的 `trust_remote_code=True` 会将 module 放到动态模块缓存目录：

- `~/.cache/huggingface/modules/transformers_modules/eagle2hg-processor-groot-n1p5/`

一般情况下，重新运行脚本会自动复制 vendor 文件并触发动态模块 hash 变更，从而加载新代码。
如果遇到“明明改了 vendor 文件但仍然报同样错误”，可以手动清理动态模块缓存后重试：

```bash
rm -rf ~/.cache/huggingface/modules/transformers_modules/eagle2hg-processor-groot-n1p5
```

## 7. 验证方式 (建议记录到你自己的复现步骤)

1. 在 `lerobot_flex` 环境下运行：

   ```bash
   bash my_devs/train/groot/so101/run_groot_eval_record.sh
   ```

2. 预期现象：

   - 不再出现 `Missing key(s) ... embed_tokens.weight` 导致的 RuntimeError 崩溃
   - 不再出现 `Eagle25VLImageProcessorFast ... _prepare_image_like_inputs` AttributeError
   - 录制循环能够正常进入并持续执行

## 8. 非本次修复范围的提示

运行日志里还有这类信息：

- `pynput` 报错 / `DISPLAY` 未设置：表示当前是 headless 环境，无法键盘交互/无法显示相机画面。
  这不影响纯自动推理与数据写入，但如果你需要 GUI 交互，需要配置 X server / DISPLAY。
