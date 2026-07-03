from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import time

import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import optax

from openpi_so101 import config as so101_config
from openpi_so101 import patches
from openpi_so101 import runtime


ROOT = Path("/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train")
DATASET_ROOT = ROOT / "easy_use/data/lerobot_v21/desk_cleanup_v1/eraser_cup_multi_task"
BASE_PARAMS = ROOT / "assets/openpi_cache/openpi-assets/checkpoints/pi05_base/params"
ASSET_ID = "desk_cleanup_v1/eraser_cup_multi_task_v21_full"


def _shape_summary(value):
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, dict):
        return {key: _shape_summary(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_shape_summary(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return str(type(value).__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe a real SO101 OpenPI training step for one batch size.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-frames", type=int, default=512)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--asset-id", default=ASSET_ID)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--base-params", type=Path, default=BASE_PARAMS)
    return parser


def _load_weights_and_validate(loader, params_shape):
    from openpi.shared import array_typing as at

    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    import flax.traverse_util as traverse_util

    return traverse_util.unflatten_dict(
        {key: value for key, value in traverse_util.flatten_dict(loaded_params).items() if not isinstance(value, jax.ShapeDtypeStruct)}
    )


def init_train_state(config, init_rng, mesh):
    from openpi.training import optimizer as _optimizer
    from openpi.training import sharding
    from openpi.training import utils as training_utils
    import openpi.shared.nnx_utils as nnx_utils

    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    return train_state, state_sharding


def train_step(config, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, rng, observation, actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    observation, actions = batch
    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_state = state.replace(
        step=state.step + 1,
        params=nnx.state(model),
        opt_state=new_opt_state,
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
    }
    return new_state, info


def main() -> None:
    runtime.bootstrap()
    args = build_parser().parse_args()
    os.environ["OPENPI_SO101_V21_ROOT"] = str(args.dataset_root.expanduser().resolve())
    os.environ["OPENPI_PI05_BASE_PARAMS"] = str(args.base_params.expanduser().resolve())

    patches.patch_openpi_data_loader(max_frames=args.max_frames, decode_images=True, dataset_format="v21")
    config = so101_config.make_config(
        exp_name=f"probe_bs{args.batch_size}",
        num_train_steps=1,
        batch_size=args.batch_size,
        action_horizon=args.action_horizon,
        learning_rate=args.learning_rate,
        save_interval=999999,
        log_interval=1,
        prompt_from_task=True,
        asset_id=args.asset_id,
        overwrite=True,
    )
    so101_config.register_config(config)

    from openpi.training import data_loader as _data_loader
    from openpi.training import sharding

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    timings = {}
    start = time.monotonic()
    loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True, num_batches=1)
    batch = next(iter(loader))
    jax.block_until_ready(batch)
    timings["batch_seconds"] = round(time.monotonic() - start, 3)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    start = time.monotonic()
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh)
    jax.block_until_ready(train_state)
    timings["init_seconds"] = round(time.monotonic() - start, 3)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    start = time.monotonic()
    train_state, info = ptrain_step(train_rng, train_state, batch)
    jax.block_until_ready((train_state, info))
    timings["step_seconds"] = round(time.monotonic() - start, 3)
    reduced_info = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest([info])))

    print(
        json.dumps(
            {
                "ok": True,
                "batch_size": args.batch_size,
                "max_frames": args.max_frames,
                "device_count": jax.device_count(),
                "devices": [str(device) for device in jax.devices()],
                "batch_shapes": _shape_summary(batch),
                "metrics": {key: float(value) for key, value in reduced_info.items()},
                "timings": timings,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
