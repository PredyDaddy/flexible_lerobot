from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import flax.nnx as nnx

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import config as openpi_config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders
from openpi import transforms

from openpi_so101 import paths
from openpi_so101 import policy as so101_policy


DATASET_ROOT_ENV = "OPENPI_SO101_DATASET_ROOT"
DATASET_REPO_ID_ENV = "OPENPI_SO101_REPO_ID"
BASE_PARAMS_ENV = "OPENPI_PI05_BASE_PARAMS"
ASSETS_BASE_DIR_ENV = "OPENPI_SO101_ASSETS_BASE_DIR"
CHECKPOINT_BASE_DIR_ENV = "OPENPI_SO101_CHECKPOINT_BASE_DIR"


@dataclasses.dataclass(frozen=True)
class SO101DataConfig(openpi_config.DataConfigFactory):
    repo_id: str = paths.DEFAULT_REPO_ID
    asset_id: str | None = None
    default_prompt: str | None = None
    use_delta_joint_actions: bool = False
    prompt_from_task: bool = False

    def create(self, assets_dirs: Path, model_config: _model.BaseModelConfig) -> openpi_config.DataConfig:
        base_factory = self
        if self.asset_id is not None:
            base_factory = dataclasses.replace(
                self,
                assets=openpi_config.AssetsConfig(
                    assets_dir=self.assets.assets_dir,
                    asset_id=self.asset_id,
                ),
            )
        data_transforms = transforms.Group(
            inputs=[so101_policy.SO101Inputs(model_type=model_config.model_type)],
            outputs=[so101_policy.SO101Outputs()],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = transforms.make_bool_mask(6)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(delta_action_mask)],
                outputs=[transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = openpi_config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            base_factory.create_base_config(assets_dirs, model_config),
            repack_transforms=so101_policy.SO101_REPACK_TRANSFORMS,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            prompt_from_task=self.prompt_from_task,
        )


def _base_params_path() -> str:
    return os.environ.get(BASE_PARAMS_ENV, "gs://openpi-assets/checkpoints/pi05_base/params")


def _assets_base_dir() -> str:
    return os.environ.get(ASSETS_BASE_DIR_ENV, str(paths.DEFAULT_ASSETS_BASE_DIR))


def _checkpoint_base_dir() -> str:
    return os.environ.get(CHECKPOINT_BASE_DIR_ENV, str(paths.DEFAULT_CHECKPOINT_BASE_DIR))


def make_config(
    *,
    name: str = "pi05_so101_eraser_cup_lora",
    exp_name: str = "smoke",
    num_train_steps: int = 10,
    batch_size: int = 1,
    action_horizon: int = 50,
    learning_rate: float = 5e-5,
    save_interval: int = 10,
    log_interval: int = 1,
    wandb_enabled: bool = False,
    overwrite: bool = False,
    resume: bool = False,
    use_delta_joint_actions: bool = False,
    prompt_from_task: bool = False,
    asset_id: str | None = None,
) -> openpi_config.TrainConfig:
    model = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=action_horizon,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    freeze_filter = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=action_horizon,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter()
    repo_id = os.environ.get(DATASET_REPO_ID_ENV, paths.DEFAULT_REPO_ID)

    return openpi_config.TrainConfig(
        name=name,
        project_name="openpi_so101",
        exp_name=exp_name,
        model=model,
        data=SO101DataConfig(
            repo_id=repo_id,
            asset_id=asset_id,
            use_delta_joint_actions=use_delta_joint_actions,
            prompt_from_task=prompt_from_task,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_base_params_path()),
        freeze_filter=freeze_filter,
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=learning_rate,
            decay_steps=max(num_train_steps, 1),
            decay_lr=learning_rate,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        assets_base_dir=_assets_base_dir(),
        checkpoint_base_dir=_checkpoint_base_dir(),
        batch_size=batch_size,
        num_workers=0,
        num_train_steps=num_train_steps,
        log_interval=log_interval,
        save_interval=save_interval,
        keep_period=None,
        overwrite=overwrite,
        resume=resume,
        wandb_enabled=wandb_enabled,
        fsdp_devices=1,
        policy_metadata={
            "robot": "so101_follower",
            "state_dim": 6,
            "action_dim": 6,
            "camera_keys": ["observation.images.top", "observation.images.wrist"],
            "source_dataset_root": os.environ.get(DATASET_ROOT_ENV, str(paths.DEFAULT_SOURCE_DATASET)),
        },
    )


def register_config(config: openpi_config.TrainConfig) -> openpi_config.TrainConfig:
    configs = [cfg for cfg in openpi_config._CONFIGS if cfg.name != config.name]  # noqa: SLF001
    configs.append(config)
    openpi_config._CONFIGS = configs  # noqa: SLF001
    openpi_config._CONFIGS_DICT = {cfg.name: cfg for cfg in configs}  # noqa: SLF001
    return config


def trainable_summary_filter(config: openpi_config.TrainConfig) -> nnx.filterlib.Filter:
    return config.trainable_filter
