# LeRobot 数据集左右臂数据交换方案

## 问题描述

**数据集位置**: `/home/agilex/.cache/huggingface/lerobot/cqy/agilex_both_side_black_cup_new`

由于录制时CAN口接反，导致14维关节数据中：
- 索引 [0:7] 存储的实际是**右臂**数据（应该是左臂）
- 索引 [7:14] 存储的实际是**左臂**数据（应该是右臂）

### 关节数据定义（info.json中的names）

```
索引 0-6 (应为左臂):
  [0] left_shoulder_pan
  [1] left_shoulder_lift
  [2] left_shoulder_roll
  [3] left_elbow
  [4] left_wrist_pitch
  [5] left_wrist_roll
  [6] left_gripper

索引 7-13 (应为右臂):
  [7] right_shoulder_pan
  [8] right_shoulder_lift
  [9] right_shoulder_roll
  [10] right_elbow
  [11] right_wrist_pitch
  [12] right_wrist_roll
  [13] right_gripper
```

## 需要修改的文件

| 文件 | 修改内容 |
|------|----------|
| `data/chunk-*/file-*.parquet` | 交换 `action` 和 `observation.state` 列的前后7维 |
| `meta/stats.json` | 交换 `action` 和 `observation.state` 的统计值 |
| `meta/episodes/chunk-*/file-*.parquet` | 交换每个episode的stats列 |

**不需要修改的文件**:
- `meta/info.json` - 特征名称定义正确，只是数据错误
- `meta/tasks.parquet` - 任务描述不变
- `videos/` - 视频内容与关节数据无关

## 脚本架构设计

创建 `swap_arm_data.py` 脚本：

```
swap_arm_data.py
├── load_info()               # 读取 meta/info.json
├── infer_action_dim_and_split_index()  # 从 info.json 推断维度与分割点
├── swap_array_halves()        # 核心：交换14维数组的前后7维
├── swap_fixed_size_list_column()  # pyarrow: 交换 fixed_size_list 列并保持schema/metadata
├── process_data_parquet()     # 处理data parquet文件
├── process_stats_json()       # 处理meta/stats.json
├── process_episode_parquet()  # 处理episode parquet文件
├── create_backup()            # 创建备份
├── verify_data_integrity()    # 验证数据完整性
└── main()                     # 主流程
```

## 核心实现代码

### 1. 核心交换函数

```python
import numpy as np

def swap_array_halves(arr: np.ndarray, split_index: int) -> np.ndarray:
    """
    交换一维数组的前后两半: [0:7, 7:14] -> [7:14, 0:7]

    Args:
        arr: 形状为 (action_dim,) 的一维数组（双臂常见为14）
        split_index: 分割点（通常为 action_dim//2，例如14维时为7）

    Returns:
        交换后的数组
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"Only supports 1D array, got shape={arr.shape}")
    return np.concatenate([arr[split_index:], arr[:split_index]])
```

### 2. 处理 data parquet 文件

```python
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from pathlib import Path

def swap_fixed_size_list_column(table, column_name: str, split_index: int):
    # column: fixed_size_list<float>[14]
    col = table.column(column_name).combine_chunks()
    values = col.values.to_numpy(zero_copy_only=False).reshape(-1, col.type.list_size)
    swapped = np.concatenate([values[:, split_index:], values[:, :split_index]], axis=1)
    new_values = pa.array(swapped.reshape(-1), type=col.values.type)
    new_col = pa.FixedSizeListArray.from_arrays(new_values, list_size=col.type.list_size, mask=col.is_null())
    return table.set_column(table.schema.get_field_index(column_name), column_name, new_col)

def process_data_parquet(file_path: Path, output_path: Path, split_index: int) -> dict:
    """
    处理单个data parquet文件，交换action和observation.state列
    """
    table = pq.read_table(file_path)
    if "action" in table.column_names:
        table = swap_fixed_size_list_column(table, "action", split_index)
    if "observation.state" in table.column_names:
        table = swap_fixed_size_list_column(table, "observation.state", split_index)

    # 保存（保持原始 schema + huggingface metadata）
    pq.write_table(table, output_path)

    return {"rows_processed": table.num_rows, "file": str(file_path)}
```

### 3. 处理 stats.json

```python
import json

def process_stats_json(stats_path: Path, output_path: Path) -> None:
    """处理 meta/stats.json 文件"""
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    keys_to_swap = ["action", "observation.state"]
    stat_types = ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]

    for feature_key in keys_to_swap:
        if feature_key not in stats:
            continue

        for stat_name in stat_types:
            if stat_name in stats[feature_key]:
                value = stats[feature_key][stat_name]
                if isinstance(value, list) and len(value) == 14:
                    # 交换前后7维
                    stats[feature_key][stat_name] = value[7:] + value[:7]

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
```

### 4. 处理 episode parquet 文件

```python
def process_episode_parquet(file_path: Path, output_path: Path) -> dict:
    """
    处理单个episode parquet文件，交换stats列中的数据
    """
    df = pd.read_parquet(file_path)

    # 需要交换的stats列前缀
    prefixes = ["stats/action/", "stats/observation.state/"]
    stat_types = ["min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"]

    for prefix in prefixes:
        for stat_type in stat_types:
            col_name = f"{prefix}{stat_type}"
            if col_name in df.columns:
                df[col_name] = df[col_name].apply(
                    lambda x: swap_array_halves(np.array(x)).tolist()
                    if x is not None and len(x) == 14 else x
                )

    df.to_parquet(output_path, index=False)
    return {"episodes_processed": len(df), "file": str(file_path)}
```

### 5. 备份函数

```python
import shutil
from datetime import datetime

def create_backup(dataset_root: Path) -> Path:
    """创建数据集备份（只备份会被修改的文件）"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = dataset_root.parent / f"{dataset_root.name}_backup_{timestamp}"

    items_to_backup = [
        "data",
        "meta/stats.json",
        "meta/episodes",
    ]

    for item in items_to_backup:
        src = dataset_root / item
        dst = backup_root / item

        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        elif src.is_dir():
            shutil.copytree(src, dst)

    print(f"Backup created at: {backup_root}")
    return backup_root
```

### 6. 验证函数

```python
def verify_data_integrity(dataset_root: Path) -> dict:
    """验证转换后数据的完整性"""
    results = {"errors": [], "checks_passed": 0}

    # 检查data parquet文件
    data_dir = dataset_root / "data"
    for pq_file in data_dir.glob("*/*.parquet"):
        try:
            df = pd.read_parquet(pq_file)
            for col in ["action", "observation.state"]:
                if col in df.columns:
                    sample = df[col].iloc[0]
                    if len(sample) != 14:
                        results["errors"].append(f"{pq_file}: {col} dim={len(sample)}")
                    else:
                        results["checks_passed"] += 1
        except Exception as e:
            results["errors"].append(f"{pq_file}: {e}")

    # 检查stats.json
    stats_path = dataset_root / "meta" / "stats.json"
    try:
        with open(stats_path) as f:
            stats = json.load(f)
        for key in ["action", "observation.state"]:
            if key in stats:
                for stat_name in ["min", "max", "mean", "std"]:
                    if len(stats[key].get(stat_name, [])) == 14:
                        results["checks_passed"] += 1
    except Exception as e:
        results["errors"].append(f"stats.json: {e}")

    return results
```

### 7. 主函数

```python
def main():
    import argparse

    parser = argparse.ArgumentParser(description="交换LeRobot数据集的左右臂数据")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_path)

    if not dataset_root.exists():
        print(f"Error: Dataset not found at {dataset_root}")
        return 1

    if args.verify_only:
        results = verify_data_integrity(dataset_root)
        print(f"Results: {results}")
        return 0 if not results["errors"] else 1

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Data files: {list(dataset_root.glob('data/*/*.parquet'))}")
        print(f"Episode files: {list(dataset_root.glob('meta/episodes/*/*.parquet'))}")
        return 0

    # 创建备份
    backup_path = None
    if not args.no_backup:
        backup_path = create_backup(dataset_root)

    try:
        # Step 1: 处理 data parquet
        print("\n=== Processing data parquet ===")
        for pq_file in sorted(dataset_root.glob("data/*/*.parquet")):
            print(f"Processing: {pq_file.name}")
            temp_file = pq_file.with_suffix(".parquet.tmp")
            process_data_parquet(pq_file, temp_file)
            temp_file.replace(pq_file)

        # Step 2: 处理 stats.json
        print("\n=== Processing stats.json ===")
        stats_path = dataset_root / "meta" / "stats.json"
        temp_stats = stats_path.with_suffix(".json.tmp")
        process_stats_json(stats_path, temp_stats)
        temp_stats.replace(stats_path)

        # Step 3: 处理 episode parquet
        print("\n=== Processing episode parquet ===")
        for pq_file in sorted(dataset_root.glob("meta/episodes/*/*.parquet")):
            print(f"Processing: {pq_file.name}")
            temp_file = pq_file.with_suffix(".parquet.tmp")
            process_episode_parquet(pq_file, temp_file)
            temp_file.replace(pq_file)

        # Step 4: 验证
        print("\n=== Verifying ===")
        results = verify_data_integrity(dataset_root)
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
    exit(main())
```

## 使用方法

```bash
# 1. Dry run检查（不修改数据）
python docs/agilex_datasets_reverse/temp_docs/swap_arm_data.py --dataset-path /home/agilex/.cache/huggingface/lerobot/cqy_vla_dev/agilex_both_side_black_cup_new --dry-run

# 2. 执行转换（自动创建备份）
python docs/agilex_datasets_reverse/temp_docs/swap_arm_data.py --dataset-path /home/agilex/.cache/huggingface/lerobot/cqy_vla_dev/agilex_both_side_yellow_bottle

# 3. 仅验证数据完整性
python docs/agilex_datasets_reverse/temp_docs/swap_arm_data.py --dataset-path /home/agilex/.cache/huggingface/lerobot/cqy_vla_dev/agilex_both_side_black_cup_new --verify-only
```

## 验证方法

转换完成后：
1. 运行 `check_lerobot_v3_dataset_final.py` 验证数据集完整性
2. 使用可视化工具检查关节数据是否符合预期
3. 进行小规模训练测试

## 注意事项

1. **强烈建议保留备份**直到确认训练效果正常
2. 处理 `data/*.parquet` 必须保持 `fixed_size_list<float>[14]`（不要用 pandas 直接 round-trip，否则会变成 `list<double>` 且丢失 huggingface parquet metadata）
3. 使用临时文件+替换的原子操作，避免写入中断导致数据损坏
4. 视频文件不需要修改（内容与关节数据无关）
5. info.json不需要修改（names定义正确，只是数据存储位置错误）
