# Handoff Zip Manifest

## Included Files

```text
START_HERE.md
CURRENT_STATE.md
PROJECT_MAP.md
COMMANDS.md
NEXT_AGENT_PROMPT.md
MANIFEST.md
```

## Zip Purpose

This zip is a compact handoff package for another Agent. It is meant to explain how to continue work in the existing repository, not to snapshot the whole repository.

The actual source tree remains at:

```text
/data/cqy_workspace/flexible_lerobot
```

## Excluded On Purpose

The handoff zip does not include:

- model checkpoints
- `model.safetensors`
- ONNX files
- TensorRT `.engine` / `.plan` files
- runtime outputs
- `__pycache__`
- Git metadata
- third-party reference source trees such as `my_devs/flash_RT_pi/reference_source_code/FlashRT-main/`
- large archives such as `Isaac-GR00T-n1.5-release.zip` or `FlashRT-main.zip`

## Important Existing Context Outside The Zip

Read these from the repository when relevant:

```text
AGENTS.md
my_devs/new_act_trt/README.md
my_devs/groot_trt/README.md
my_devs/openpi_trt/README.md
my_devs/openpi_trt/docs/openpi_trt实现链路报告.md
my_devs/openpi_trt/docs/lerobot_pi05_trt难点分析.md
my_devs/docs/vla_engineering/工作报告.md
my_devs/flash_RT_pi/reference_source_code/FlashRT-main/README.md
my_devs/agilex_web_collection/README.md
my_devs/web_collection/README.md
```

Note: `my_devs/flash_RT_pi/reference_source_code/FlashRT-main/` may be ignored by the current `.gitignore`; it is still useful local context if present on the machine.
