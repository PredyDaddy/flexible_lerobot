#!/usr/bin/env python3
"""
LeRobot 数据集左右臂数据交换脚本

由于录制时CAN口接反，导致14维关节数据中：
- 索引 [0:7] 存储的实际是右臂数据（应该是左臂）
- 索引 [7:14] 存储的实际是左臂数据（应该是右臂）

此脚本交换 action 和 observation.state 列的前后7维数据。
# 1. Dry run检查（不修改数据）
python swap_arm_data.py --dataset-path /home/agilex/.cache/huggingface/lerobot/cqy/agilex_both_side_black_cup_new --dry-run

# 2. 执行转换（自动创建备份）
python swap_arm_data.py --dataset-path /home/agilex/.cache/huggingface/lerobot/cqy/agilex_both_side_black_cup_new

# 3. 仅验证数据完整性
python swap_arm_data.py --dataset-path /home/agilex/.cache/huggingface/lerobot/cqy/agilex_both_side_black_cup_new --verify-only
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def swap_array_halves(arr: np.ndarray, split_index: int) -> np.ndarray:
    """
    交换数组的前后两半。

    [0:7, 7:14] -> [7:14, 0:7]

    Args:
        arr: 输入数组
        split_index: 分割索引，默认为7

    Returns:
        交换后的数组
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"swap_array_halves 只支持一维数组，得到 shape={arr.shape}")
    if split_index <= 0 or split_index >= arr.shape[0]:
        raise ValueError(f"split_index 非法: {split_index}, arr.shape={arr.shape}")
    return np.concatenate([arr[split_index:], arr[:split_index]])


def swap_list_halves(lst: list, split_index: int) -> list:
    """
    交换列表的前后两半。

    Args:
        lst: 输入列表
        split_index: 分割索引，默认为7

    Returns:
        交换后的列表
    """
    return lst[split_index:] + lst[:split_index]


def load_info(dataset_path: Path) -> dict:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"未找到 info.json: {info_path}")
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_action_dim_and_split_index(info: dict) -> tuple[int, int]:
    features = info.get("features", {})
    action_spec = features.get("action", {})
    state_spec = features.get("observation.state", {})

    shape = action_spec.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 1 or not isinstance(shape[0], int):
        raise ValueError(f"info.json features.action.shape 非法: {shape}")

    action_dim = int(shape[0])
    if action_dim <= 0:
        raise ValueError(f"action_dim 非法: {action_dim}")

    state_shape = state_spec.get("shape")
    if not isinstance(state_shape, (list, tuple)) or len(state_shape) != 1 or not isinstance(state_shape[0], int):
        raise ValueError(f"info.json features.observation.state.shape 非法: {state_shape}")
    state_dim = int(state_shape[0])
    if state_dim != action_dim:
        raise ValueError(f"action_dim({action_dim}) != observation.state_dim({state_dim})，无法交换")

    if action_dim % 2 != 0:
        raise ValueError(f"action_dim 不是偶数，无法自动均分: {action_dim}")
    split_index = action_dim // 2
    return action_dim, split_index


def swap_fixed_size_list_column(table: pa.Table, column_name: str, split_index: int) -> pa.Table:
    """交换 fixed_size_list 列的前后两半，并保持 schema/metadata 不变。"""
    field_index = table.schema.get_field_index(column_name)
    if field_index == -1:
        return table

    col = table.column(column_name).combine_chunks()
    if not pa.types.is_fixed_size_list(col.type):
        raise ValueError(f"{column_name} 不是 fixed_size_list 类型: {col.type}")

    list_size = col.type.list_size
    if list_size <= split_index:
        raise ValueError(f"{column_name} list_size({list_size}) <= split_index({split_index})")

    values = col.values
    values_np = values.to_numpy(zero_copy_only=False).reshape(-1, list_size)
    swapped_np = np.concatenate([values_np[:, split_index:], values_np[:, :split_index]], axis=1)

    new_values = pa.array(swapped_np.reshape(-1), type=values.type)
    mask = col.is_null() if col.null_count else None
    new_col = pa.FixedSizeListArray.from_arrays(new_values, list_size=list_size, mask=mask)
    return table.set_column(field_index, column_name, new_col)


def create_backup(dataset_path: Path) -> Path:
    """创建数据集备份（只备份会被修改的文件）。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = dataset_path.parent / f"{dataset_path.name}_backup_{timestamp}"

    print(f"创建备份到: {backup_path}")

    # 备份 data/ 目录
    data_dir = dataset_path / "data"
    if data_dir.exists():
        backup_data = backup_path / "data"
        shutil.copytree(data_dir, backup_data)
        print(f"  已备份: data/")

    # 备份 meta/stats.json
    stats_file = dataset_path / "meta" / "stats.json"
    if stats_file.exists():
        backup_meta = backup_path / "meta"
        backup_meta.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stats_file, backup_meta / "stats.json")
        print(f"  已备份: meta/stats.json")

    # 备份 meta/episodes/ 目录
    episodes_dir = dataset_path / "meta" / "episodes"
    if episodes_dir.exists():
        backup_episodes = backup_path / "meta" / "episodes"
        shutil.copytree(episodes_dir, backup_episodes)
        print(f"  已备份: meta/episodes/")

    print(f"备份完成: {backup_path}")
    return backup_path


def process_data_parquet(file_path: Path, output_path: Path, split_index: int) -> dict:
    """处理单个 data parquet 文件，交换 action 和 observation.state 列。"""
    table = pq.read_table(file_path)
    schema_meta = table.schema.metadata

    if "action" in table.column_names:
        table = swap_fixed_size_list_column(table, "action", split_index)
    if "observation.state" in table.column_names:
        table = swap_fixed_size_list_column(table, "observation.state", split_index)

    if schema_meta is not None:
        table = table.replace_schema_metadata(schema_meta)

    pq.write_table(table, output_path)
    return {"rows_processed": table.num_rows, "file": str(file_path)}


def process_stats_json(stats_path: Path, output_path: Path, action_dim: int, split_index: int) -> None:
    """处理 meta/stats.json 文件，交换 action/observation.state 的统计值。"""
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    feature_keys = ["action", "observation.state"]
    stat_types = ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]

    for feature_key in feature_keys:
        if feature_key not in stats:
            continue

        for stat_name in stat_types:
            value = stats[feature_key].get(stat_name)
            if isinstance(value, list) and len(value) == action_dim:
                stats[feature_key][stat_name] = swap_list_halves(value, split_index)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def process_episode_parquet(file_path: Path, output_path: Path, action_dim: int, split_index: int) -> dict:
    """处理单个 episode parquet 文件，交换 stats/action 和 stats/observation.state 的统计列。"""
    table = pq.read_table(file_path)
    feature_prefixes = ["stats/action/", "stats/observation.state/"]
    stat_types = ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]

    for prefix in feature_prefixes:
        for stat_name in stat_types:
            col_name = f"{prefix}{stat_name}"
            field_index = table.schema.get_field_index(col_name)
            if field_index == -1:
                continue

            col = table.column(col_name).combine_chunks()
            swapped = [
                swap_list_halves(x, split_index) if isinstance(x, list) and len(x) == action_dim else x
                for x in col.to_pylist()
            ]
            new_col = pa.array(swapped, type=col.type)
            table = table.set_column(field_index, col_name, new_col)

    pq.write_table(table, output_path)
    return {"episodes_processed": table.num_rows, "file": str(file_path)}


def verify_data_integrity(dataset_root: Path, action_dim: int) -> dict:
    """验证转换后数据的完整性（类型/维度/可读性）。"""
    results: dict[str, object] = {"errors": [], "checks_passed": 0}

    # 检查 data parquet 文件：列存在 + 类型是 fixed_size_list + list_size 正确
    data_dir = dataset_root / "data"
    for pq_file in data_dir.glob("*/*.parquet"):
        try:
            schema = pq.ParquetFile(pq_file).schema_arrow
            for col_name in ["action", "observation.state"]:
                if col_name not in schema.names:
                    continue
                f = schema.field(col_name)
                if not pa.types.is_fixed_size_list(f.type) or f.type.list_size != action_dim:
                    results["errors"].append(f"{pq_file}: {col_name} 类型不匹配: {f.type}")
                else:
                    results["checks_passed"] = int(results["checks_passed"]) + 1
        except Exception as e:
            results["errors"].append(f"{pq_file}: {e}")

    # 检查 stats.json
    stats_path = dataset_root / "meta" / "stats.json"
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        for key in ["action", "observation.state"]:
            if key not in stats:
                continue
            for stat_name in ["min", "max", "mean", "std"]:
                value = stats[key].get(stat_name)
                if isinstance(value, list) and len(value) == action_dim:
                    results["checks_passed"] = int(results["checks_passed"]) + 1
                elif value is not None:
                    results["errors"].append(f"stats.json: {key}/{stat_name} 维度不是 {action_dim}")
    except Exception as e:
        results["errors"].append(f"stats.json: {e}")

    # 检查 episode parquet 文件可读
    episodes_dir = dataset_root / "meta" / "episodes"
    for pq_file in episodes_dir.glob("*/*.parquet"):
        try:
            pq.read_table(pq_file)
            results["checks_passed"] = int(results["checks_passed"]) + 1
        except Exception as e:
            results["errors"].append(f"{pq_file}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="交换 LeRobot 数据集的左右臂数据")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_path)
    if not dataset_root.exists():
        print(f"Error: Dataset not found at {dataset_root}")
        return 1

    info = load_info(dataset_root)
    action_dim, split_index = infer_action_dim_and_split_index(info)

    print(f"action_dim={action_dim}, split_index={split_index}")

    if args.verify_only:
        results = verify_data_integrity(dataset_root, action_dim)
        print(f"Results: {results}")
        return 0 if not results["errors"] else 1

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Data files: {list(dataset_root.glob('data/*/*.parquet'))}")
        print(f"Episode files: {list(dataset_root.glob('meta/episodes/*/*.parquet'))}")
        return 0

    # 创建备份
    backup_path: Path | None = None
    if not args.no_backup:
        backup_path = create_backup(dataset_root)

    try:
        # Step 1: 处理 data parquet
        print("\n=== Processing data parquet ===")
        for pq_file in sorted(dataset_root.glob("data/*/*.parquet")):
            print(f"Processing: {pq_file.name}")
            temp_file = pq_file.with_suffix(".parquet.tmp")
            process_data_parquet(pq_file, temp_file, split_index)
            temp_file.replace(pq_file)

        # Step 2: 处理 stats.json
        print("\n=== Processing stats.json ===")
        stats_path = dataset_root / "meta" / "stats.json"
        temp_stats = stats_path.with_suffix(".json.tmp")
        process_stats_json(stats_path, temp_stats, action_dim, split_index)
        temp_stats.replace(stats_path)

        # Step 3: 处理 episode parquet
        print("\n=== Processing episode parquet ===")
        for pq_file in sorted(dataset_root.glob("meta/episodes/*/*.parquet")):
            print(f"Processing: {pq_file.name}")
            temp_file = pq_file.with_suffix(".parquet.tmp")
            process_episode_parquet(pq_file, temp_file, action_dim, split_index)
            temp_file.replace(pq_file)

        # Step 4: 验证
        print("\n=== Verifying ===")
        results = verify_data_integrity(dataset_root, action_dim)
        if results["errors"]:
            print(f"ERRORS: {results['errors']}")
            return 1

        print(f"Success! {results['checks_passed']} checks passed")
        return 0

    except Exception as e:
        print(f"Error: {e}")
        if backup_path:
            print(f"Backup available at: {backup_path}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
