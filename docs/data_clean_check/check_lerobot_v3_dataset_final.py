#!/usr/bin/env python3
"""
LeRobot v3.0 数据集完整检查脚本

检查项次（按致命程度排序）：
1. 目录结构与必需文件 [致命]
2. info.json 自洽性 [致命]
3. episodes parquet 一致性 [致命]
4. data parquet 帧数据正确性 [致命]
5. videos 视频文件完整性 [致命/如果使用]
6. tasks.parquet 与 task_index 对齐 [重要]
7. stats.json 完整性 [重要]
8. 数据质量检查 [建议]

使用方法:
    python3 check_lerobot_v3_dataset_final.py /path/to/dataset
    python3 check_lerobot_v3_dataset_final.py /home/agilex/.cache/huggingface/lerobot/cqy/agilex_vla_demo_ee_pinocchio_Bowl_on_Plate_Placement_test
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================
# 全局配置
# ============================================================

# LeRobot v3.0 默认路径模板（与 src/lerobot/datasets/utils.py 保持一致）
DEFAULT_DATA_PATH = 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'
DEFAULT_VIDEO_PATH = 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'


def format_data_file_path(root: Path, info: Dict, chunk_index: int, file_index: int) -> Path:
    """根据 info.json 的 data_path 生成 data parquet 文件路径（兼容缺失字段）。"""
    template = info.get('data_path') if isinstance(info, dict) else None
    if not isinstance(template, str) or not template:
        template = DEFAULT_DATA_PATH

    try:
        rel = template.format(chunk_index=int(chunk_index), file_index=int(file_index))
    except Exception:
        rel = DEFAULT_DATA_PATH.format(chunk_index=int(chunk_index), file_index=int(file_index))
    return root / rel


def format_video_file_path(
    root: Path, info: Dict, video_key: str, chunk_index: int, file_index: int
) -> Path:
    """根据 info.json 的 video_path 生成 video 文件路径（兼容缺失字段）。"""
    template = info.get('video_path') if isinstance(info, dict) else None
    if not isinstance(template, str) or not template:
        template = DEFAULT_VIDEO_PATH

    try:
        rel = template.format(
            video_key=video_key, chunk_index=int(chunk_index), file_index=int(file_index)
        )
    except Exception:
        rel = DEFAULT_VIDEO_PATH.format(
            video_key=video_key, chunk_index=int(chunk_index), file_index=int(file_index)
        )
    return root / rel


# 检查结果类型
class CheckResult:
    """检查结果"""
    def __init__(self):
        self.errors: List[str] = []      # 致命错误
        self.warnings: List[str] = []    # 警告
        self.info: List[str] = []        # 信息

    def add_error(self, msg: str):
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_info(self, msg: str):
        self.info.append(msg)

    def is_ok(self) -> bool:
        return len(self.errors) == 0

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


# ============================================================
# 一、目录结构检查 [致命]
# ============================================================
def check_directory_structure(root: Path, info: Dict) -> CheckResult:
    """检查目录结构与必需文件"""
    result = CheckResult()

    # 必需文件
    required_files = [
        'meta/info.json',
        'meta/tasks.parquet',
    ]

    for f in required_files:
        if not (root / f).exists():
            result.add_error(f"缺少必需文件: {f}")

    # 检查 episodes parquet
    episodes_dir = root / 'meta/episodes'
    if not episodes_dir.exists():
        result.add_error("缺少目录: meta/episodes/")
    else:
        ep_files = list(episodes_dir.rglob('*.parquet'))
        if not ep_files:
            result.add_error("meta/episodes/ 下没有 parquet 文件")

    # 检查 data parquet
    data_path_tpl = info.get('data_path', DEFAULT_DATA_PATH) if info else DEFAULT_DATA_PATH
    data_dir = root / Path(str(data_path_tpl)).parts[0] if data_path_tpl else (root / 'data')
    if not data_dir.exists():
        rel = str(data_dir.relative_to(root)) if data_dir.is_relative_to(root) else str(data_dir)
        result.add_error(f"缺少目录: {rel}/")
    else:
        data_files = list(data_dir.rglob('*.parquet'))
        if not data_files:
            rel = str(data_dir.relative_to(root)) if data_dir.is_relative_to(root) else str(data_dir)
            result.add_error(f"{rel}/ 下没有 parquet 文件")

    # 检查 videos 目录（如果 features 有 video 类型）
    if info:
        video_keys = [k for k, v in info.get('features', {}).items()
                      if v.get('dtype') == 'video']
        for vk in video_keys:
            video_dir = format_video_file_path(root, info, vk, 0, 0).parent.parent
            if not video_dir.exists():
                # 兼容自定义 video_path
                rel = str(video_dir.relative_to(root)) if video_dir.is_relative_to(root) else str(video_dir)
                result.add_error(f"缺少视频目录: {rel}/")
            else:
                mp4_files = list(video_dir.rglob('*.mp4'))
                if not mp4_files:
                    rel = str(video_dir.relative_to(root)) if video_dir.is_relative_to(root) else str(video_dir)
                    result.add_error(f"{rel}/ 下没有 mp4 文件")

    return result


# ============================================================
# 二、info.json 检查 [致命]
# ============================================================
def check_info_json(root: Path) -> Tuple[CheckResult, Optional[Dict]]:
    """检查 info.json 自洽性"""
    result = CheckResult()
    info_path = root / 'meta/info.json'

    if not info_path.exists():
        result.add_error("info.json 不存在")
        return result, None

    try:
        with open(info_path) as f:
            info = json.load(f)
    except json.JSONDecodeError as e:
        result.add_error(f"info.json 解析失败: {e}")
        return result, None

    # 检查 codebase_version
    version = info.get('codebase_version', '')
    if version != 'v3.0':
        result.add_warning(f"codebase_version={version}, 期望 v3.0")

    # 检查必需字段（与 `src/lerobot/datasets/utils.py` 的 `create_empty_dataset_info()` 对齐）
    required_fields = [
        'total_episodes',
        'total_frames',
        'total_tasks',
        'fps',
        'features',
        'splits',
        'chunks_size',
        'data_files_size_in_mb',
        'video_files_size_in_mb',
        'data_path',
        'video_path',
    ]
    for field in required_fields:
        if field not in info:
            result.add_error(f"info.json 缺少字段: {field}")

    # 检查数值有效性
    if info.get('total_episodes', 0) <= 0:
        result.add_error(f"total_episodes 非法: {info.get('total_episodes')}")

    if info.get('total_frames', 0) <= 0:
        result.add_error(f"total_frames 非法: {info.get('total_frames')}")

    if info.get('total_tasks', 0) <= 0:
        result.add_error(f"total_tasks 非法: {info.get('total_tasks')}")

    if info.get('fps', 0) <= 0:
        result.add_error(f"fps 非法: {info.get('fps')}")

    if info.get('chunks_size', 0) <= 0:
        result.add_error(f"chunks_size 非法: {info.get('chunks_size')}")

    if info.get('data_files_size_in_mb', 0) <= 0:
        result.add_error(f"data_files_size_in_mb 非法: {info.get('data_files_size_in_mb')}")

    # 如果存在 video 特征，video_files_size_in_mb 应为正数；否则可忽略
    has_video = any(v.get('dtype') == 'video' for v in info.get('features', {}).values())
    if has_video and info.get('video_files_size_in_mb', 0) <= 0:
        result.add_error(f"video_files_size_in_mb 非法: {info.get('video_files_size_in_mb')}")

    # 如果存在 video 特征，video_path 不应为 None
    if has_video and info.get('video_path') in [None, '']:
        result.add_error("存在 video 特征但 video_path 为空")

    # 检查 splits 范围
    splits = info.get('splits', {})
    total_ep = info.get('total_episodes', 0)
    for split_name, split_range in splits.items():
        if ':' in str(split_range):
            parts = str(split_range).split(':')
            if len(parts) != 2:
                result.add_warning(f"splits.{split_name}={split_range} 格式异常，期望 'start:end'")
                continue
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError:
                result.add_warning(f"splits.{split_name}={split_range} 无法解析为整数范围")
                continue
            if start < 0 or end > total_ep:
                result.add_error(f"splits.{split_name}={split_range} 超出范围 [0, {total_ep}]")

    # 检查 features
    features = info.get('features', {})
    bad_feature_names = [k for k in features.keys() if '/' in k]
    if bad_feature_names:
        result.add_error(f"features 中包含非法名称(含'/'): {bad_feature_names[:10]}")

    if 'action' not in features:
        result.add_error("features 缺少 action")
    if 'observation.state' not in features:
        result.add_error("features 缺少 observation.state")

    return result, info


# ============================================================
# 三、episodes parquet 检查 [致命]
# ============================================================
def check_episodes_parquet(root: Path, info: Dict) -> Tuple[CheckResult, Optional[pd.DataFrame]]:
    """检查 episodes parquet 一致性"""
    result = CheckResult()

    # 加载所有 episodes 文件
    episodes_dir = root / 'meta/episodes'
    ep_files = sorted(episodes_dir.rglob('*.parquet'))

    if not ep_files:
        result.add_error("没有找到 episodes parquet 文件")
        return result, None

    try:
        episodes = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
    except Exception as e:
        result.add_error(f"读取 episodes parquet 失败: {e}")
        return result, None

    total_episodes = info.get('total_episodes', 0)
    total_frames = info.get('total_frames', 0)

    # 基本列检查
    required_cols = ['episode_index', 'length', 'dataset_from_index', 'dataset_to_index']
    missing_cols = [c for c in required_cols if c not in episodes.columns]
    if missing_cols:
        result.add_error(f"episodes 缺少列: {missing_cols}")
        return result, episodes

    # 检查行数
    if len(episodes) != total_episodes:
        result.add_error(f"episodes 行数({len(episodes)}) != total_episodes({total_episodes})")

    # episode_index 连续性（不假设 parquet 行已排序）
    ep_indices = episodes['episode_index']
    if ep_indices.duplicated().any():
        dup = ep_indices[ep_indices.duplicated()].unique()
        result.add_error(f"episode_index 有重复: {dup[:10].tolist()}")
    else:
        ep_min = int(ep_indices.min())
        ep_max = int(ep_indices.max())
        ep_nunique = int(ep_indices.nunique())
        if ep_min != 0 or ep_max != total_episodes - 1 or ep_nunique != total_episodes:
            expected_set = set(range(total_episodes))
            actual_set = set(map(int, ep_indices.unique()))
            missing = sorted(list(expected_set - actual_set))[:10]
            extra = sorted(list(actual_set - expected_set))[:10]
            result.add_error(
                f"episode_index 不连续/数量不匹配: min={ep_min}, max={ep_max}, nunique={ep_nunique}, "
                f"missing(sample)={missing}, extra(sample)={extra}"
            )

    # 检查 length 总和
    length_sum = episodes['length'].sum()
    if length_sum != total_frames:
        result.add_error(f"episodes.length 总和({length_sum}) != total_frames({total_frames})")

    # 按 episode_index 排序后，检查 dataset_from/to 严格约束
    if not episodes['episode_index'].is_monotonic_increasing:
        result.add_warning("episodes 未按 episode_index 排序，已自动排序后继续检查")

    episodes_sorted = episodes.sort_values('episode_index').reset_index(drop=True)
    from_idx = episodes_sorted['dataset_from_index'].to_numpy()
    to_idx = episodes_sorted['dataset_to_index'].to_numpy()
    lengths = episodes_sorted['length'].to_numpy()

    if int(from_idx[0]) != 0:
        result.add_error("episode 0 的 dataset_from_index != 0")

    if int(to_idx[-1]) != int(total_frames):
        result.add_error(f"最后一个 episode 的 dataset_to_index({int(to_idx[-1])}) != total_frames({total_frames})")

    diff = to_idx - from_idx
    bad = np.where(diff != lengths)[0]
    if len(bad) > 0:
        i = int(bad[0])
        result.add_error(
            f"Episode {int(episodes_sorted.loc[i,'episode_index'])}: "
            f"to-from({int(diff[i])}) != length({int(lengths[i])})"
        )

    gap = np.where(from_idx[1:] != to_idx[:-1])[0]
    if len(gap) > 0:
        i = int(gap[0])
        result.add_error(
            f"Episode {int(episodes_sorted.loc[i,'episode_index'])} 和 {int(episodes_sorted.loc[i+1,'episode_index'])} 之间不首尾相接: "
            f"{int(to_idx[i])} != {int(from_idx[i+1])}"
        )

    return result, episodes_sorted


def check_episode_file_references(root: Path, info: Dict, episodes: pd.DataFrame) -> CheckResult:
    """检查 episodes 中引用的数据/视频文件是否存在"""
    result = CheckResult()

    # data 引用
    data_cols = ['data/chunk_index', 'data/file_index']
    if not all(c in episodes.columns for c in data_cols):
        result.add_error(f"episodes 缺少 data 引用列: {data_cols}")
        return result

    missing_data = 0
    for chunk, file in episodes[data_cols].drop_duplicates().itertuples(index=False):
        path = format_data_file_path(root, info, int(chunk), int(file))
        if not path.exists():
            result.add_error(f"缺少 data 文件: {path}")
            missing_data += 1
            if missing_data >= 10:
                result.add_error("缺失 data 文件过多，已截断后续报告")
                break

    # videos 引用
    features = info.get('features', {})
    video_keys = [k for k, v in features.items() if v.get('dtype') == 'video']

    for vk in video_keys:
        ccol = f'videos/{vk}/chunk_index'
        fcol = f'videos/{vk}/file_index'
        if ccol not in episodes.columns or fcol not in episodes.columns:
            result.add_error(f"episodes 缺少视频引用列: {ccol}, {fcol}")
            continue

        missing_video = 0
        for chunk, file in episodes[[ccol, fcol]].drop_duplicates().itertuples(index=False):
            path = format_video_file_path(root, info, vk, int(chunk), int(file))
            if not path.exists():
                result.add_error(f"缺少视频文件: {path}")
                missing_video += 1
                if missing_video >= 10:
                    result.add_error(f"{vk} 缺失视频文件过多，已截断后续报告")
                    break

    return result


def check_episodes_timestamps(episodes: pd.DataFrame, info: Dict) -> CheckResult:
    """检查 episodes 时间戳约束"""
    result = CheckResult()
    fps = info.get('fps', 30)
    features = info.get('features', {})

    # 获取视频 keys
    video_keys = [k for k, v in features.items() if v.get('dtype') == 'video']

    for vk in video_keys:
        from_col = f'videos/{vk}/from_timestamp'
        to_col = f'videos/{vk}/to_timestamp'

        if from_col not in episodes.columns or to_col not in episodes.columns:
            continue

        # to > from
        non_positive = episodes[episodes[to_col] <= episodes[from_col]]
        if len(non_positive) > 0:
            ep0 = int(non_positive.iloc[0]['episode_index'])
            result.add_error(f"Episode {ep0} {vk} to_timestamp <= from_timestamp")
            break

        # 检查时间戳约束: (to - from) ≈ length / fps
        for _, row in episodes.iterrows():
            expected_duration = float(row['length']) / float(fps)
            actual_duration = float(row[to_col]) - float(row[from_col])

            # 允许 0.1 秒误差
            if abs(actual_duration - expected_duration) > 0.1:
                result.add_warning(
                    f"Episode {int(row['episode_index'])} {vk} 时间戳不匹配: 实际{actual_duration:.2f}s != 期望{expected_duration:.2f}s"
                )
                break  # 只报告第一个问题

    # 多相机同步（强烈建议）
    if len(video_keys) >= 2:
        global_from_list = []
        global_to_list = []

        for vk in video_keys:
            ccol = f'videos/{vk}/chunk_index'
            fcol = f'videos/{vk}/file_index'
            from_col = f'videos/{vk}/from_timestamp'
            to_col = f'videos/{vk}/to_timestamp'

            required = [ccol, fcol, from_col, to_col]
            if not all(c in episodes.columns for c in required):
                continue

            # 归一化跨文件时间轴：global_ts = file_offset + local_ts
            # file_offset 通过 episodes 中每个 (chunk,file) 的 max(to_timestamp) 近似得到
            key_df = episodes[required].copy()
            key_df[ccol] = key_df[ccol].astype(int)
            key_df[fcol] = key_df[fcol].astype(int)
            key_df[from_col] = key_df[from_col].astype(float)
            key_df[to_col] = key_df[to_col].astype(float)

            file_durations = key_df.groupby([ccol, fcol])[to_col].max().sort_index()
            file_offsets = file_durations.cumsum().shift(fill_value=0.0)
            file_index = pd.MultiIndex.from_frame(key_df[[ccol, fcol]])
            offsets = file_offsets.reindex(file_index).to_numpy(dtype=float)

            global_from_list.append(offsets + key_df[from_col].to_numpy(dtype=float))
            global_to_list.append(offsets + key_df[to_col].to_numpy(dtype=float))

        if len(global_from_list) >= 2 and len(global_to_list) >= 2:
            g_from = np.stack(global_from_list, axis=0)
            g_to = np.stack(global_to_list, axis=0)
            from_diff = (g_from.max(axis=0) - g_from.min(axis=0)).max()
            to_diff = (g_to.max(axis=0) - g_to.min(axis=0)).max()

            if float(from_diff) > 0.05 or float(to_diff) > 0.05:
                result.add_warning(
                    f"多相机同步偏差较大(跨文件归一化): global_from最大差={float(from_diff):.3f}s, "
                    f"global_to最大差={float(to_diff):.3f}s"
                )

    return result


# ============================================================
# 四、data parquet 检查 [致命]
# ============================================================
def check_data_parquet(root: Path, info: Dict, episodes: pd.DataFrame) -> Tuple[CheckResult, Optional[pd.DataFrame]]:
    """检查 data parquet 帧数据正确性"""
    result = CheckResult()

    # 加载所有 data 文件
    data_dir = format_data_file_path(root, info, 0, 0).parent.parent
    data_files = sorted(data_dir.rglob('*.parquet'))

    if not data_files:
        result.add_error("没有找到 data parquet 文件")
        return result, None

    try:
        # 避免将 embedded images 等大列全部读入内存，只读取检查所需字段
        columns = [
            'index',
            'episode_index',
            'frame_index',
            'task_index',
            'timestamp',
            'action',
            'observation.state',
        ]
        data = pd.concat([pd.read_parquet(f, columns=columns) for f in data_files], ignore_index=True)
    except Exception as e:
        result.add_error(f"读取 data parquet 失败: {e}")
        return result, None

    total_frames = info.get('total_frames', 0)
    total_episodes = info.get('total_episodes', 0)
    total_tasks = info.get('total_tasks', 1)

    # 检查总行数
    if len(data) != total_frames:
        result.add_error(f"data 行数({len(data)}) != total_frames({total_frames})")

    # 检查全局 index 连续性（使用 min/max/nunique，不假设行已排序）
    if 'index' in data.columns:
        indices = data['index'].values
        if indices.min() != 0:
            result.add_error(f"data.index 最小值({indices.min()}) != 0")
        if indices.max() != len(data) - 1:
            result.add_error(f"data.index 最大值({indices.max()}) != {len(data)-1}")
        if len(np.unique(indices)) != len(data):
            result.add_error(f"data.index 有重复值 (唯一值数量: {len(np.unique(indices))}, 期望: {len(data)})")

    # 检查 task_index 范围
    if 'task_index' in data.columns:
        task_indices = data['task_index'].values
        if task_indices.min() < 0 or task_indices.max() >= total_tasks:
            result.add_error(f"task_index 超出范围 [0, {total_tasks-1}]")

    # 检查 episode_index 范围/覆盖
    if 'episode_index' not in data.columns:
        result.add_error("data 缺少列: episode_index")
    else:
        ep = data['episode_index']
        if int(ep.min()) != 0 or int(ep.max()) != total_episodes - 1 or int(ep.nunique()) != total_episodes:
            result.add_error(
                f"data.episode_index 不连续/数量不匹配: min={int(ep.min())}, max={int(ep.max())}, nunique={int(ep.nunique())}, "
                f"期望 0..{total_episodes-1}"
            )

    # 每个 episode 的帧数 == episodes.length
    if episodes is not None and 'episode_index' in data.columns:
        expected_lengths = episodes.set_index('episode_index')['length']
        actual_counts = data.groupby('episode_index').size()
        cmp = pd.DataFrame({'expected': expected_lengths, 'actual': actual_counts})

        missing_actual = cmp['actual'].isna()
        if missing_actual.any():
            missing = cmp[missing_actual].index.tolist()[:10]
            result.add_error(f"data 缺少 episode 数据: {missing} (sample)")

        missing_expected = cmp['expected'].isna()
        if missing_expected.any():
            extra = cmp[missing_expected].index.tolist()[:10]
            result.add_error(f"data 存在多余 episode_index: {extra} (sample)")

        bad_len = cmp.dropna()
        bad_len = bad_len[bad_len['expected'] != bad_len['actual']]
        if len(bad_len) > 0:
            ep0 = int(bad_len.index[0])
            result.add_error(f"Episode {ep0} 帧数不匹配: data={int(bad_len.loc[ep0,'actual'])} != meta.length={int(bad_len.loc[ep0,'expected'])}")

    # frame_index 连续性（全量检查，不依赖行顺序）
    if episodes is not None and 'episode_index' in data.columns and 'frame_index' in data.columns:
        lengths = episodes.set_index('episode_index')['length']
        st = data.groupby('episode_index')['frame_index'].agg(['min', 'max', 'nunique', 'count']).join(lengths.rename('length'))
        bad = st[(st['min'] != 0) | (st['max'] != st['length'] - 1) | (st['nunique'] != st['length']) | (st['count'] != st['length'])]
        if len(bad) > 0:
            ep0 = int(bad.index[0])
            row = bad.loc[ep0]
            result.add_error(
                f"Episode {ep0} frame_index 异常: min={int(row['min'])}, max={int(row['max'])}, "
                f"nunique={int(row['nunique'])}, count={int(row['count'])}, length={int(row['length'])}"
            )

    # 对齐 episodes 的 dataset_from/to 与 data.index 范围
    if episodes is not None and 'episode_index' in data.columns and 'index' in data.columns:
        required_meta_cols = {'dataset_from_index', 'dataset_to_index', 'length'}
        if not required_meta_cols.issubset(set(episodes.columns)):
            missing = sorted(list(required_meta_cols - set(episodes.columns)))
            result.add_error(f"episodes 缺少列，无法对齐 data.index 范围: {missing}")
        else:
            meta = episodes.set_index('episode_index')[list(required_meta_cols)]
            st = data.groupby('episode_index')['index'].agg(['min', 'max', 'count']).join(meta, how='left')

            missing_meta = st['dataset_from_index'].isna() | st['dataset_to_index'].isna() | st['length'].isna()
            if missing_meta.any():
                ep0 = int(st[missing_meta].index[0])
                result.add_error(f"Episode {ep0} 在 episodes 中缺失元信息，无法对齐 data.index 范围")
            else:
                exp_from = st['dataset_from_index'].astype(np.int64)
                exp_to = st['dataset_to_index'].astype(np.int64)
                exp_len = st['length'].astype(np.int64)
                act_min = st['min'].astype(np.int64)
                act_max = st['max'].astype(np.int64)
                act_cnt = st['count'].astype(np.int64)

                bad = st[(act_min != exp_from) | (act_max != (exp_to - 1)) | (act_cnt != exp_len)]
                if len(bad) > 0:
                    ep0 = int(bad.index[0])
                    row = bad.loc[ep0]
                    result.add_error(
                        f"Episode {ep0} data.index 范围不匹配: "
                        f"min={int(row['min'])}, max={int(row['max'])}, count={int(row['count'])} "
                        f"!= from={int(row['dataset_from_index'])}, to-1={int(row['dataset_to_index'])-1}, length={int(row['length'])}"
                    )

    return result, data


def check_data_details(data: pd.DataFrame, info: Dict, episodes: pd.DataFrame) -> CheckResult:
    """检查 data 详细内容"""
    result = CheckResult()
    features = info.get('features', {})
    fps = info.get('fps', 30)

    def to_2d_array(series: pd.Series, name: str) -> Optional[np.ndarray]:
        try:
            arr = np.stack(series.to_numpy())
        except Exception:
            arr = np.array(series.tolist(), dtype=object)
        if arr.ndim != 2:
            result.add_error(f"{name} 维度不一致，无法解析为二维数组")
            return None
        return arr

    # 检查 action
    if 'action' in data.columns:
        actions = to_2d_array(data['action'], 'action')
        expected_shape = features.get('action', {}).get('shape', [])

        # 维度检查
        if actions is not None and expected_shape and actions.shape[1] != expected_shape[0]:
            result.add_error(f"action 维度({actions.shape[1]}) != info定义({expected_shape[0]})")

        # NaN/Inf 检查
        if actions is not None:
            if np.isnan(actions).any():
                result.add_error(f"action 包含 NaN ({int(np.isnan(actions).sum())} 个)")
            if np.isinf(actions).any():
                result.add_error(f"action 包含 Inf ({int(np.isinf(actions).sum())} 个)")

    # 检查 observation.state
    if 'observation.state' in data.columns:
        states = to_2d_array(data['observation.state'], 'observation.state')
        expected_shape = features.get('observation.state', {}).get('shape', [])

        # 维度检查
        if states is not None and expected_shape and states.shape[1] != expected_shape[0]:
            result.add_error(f"state 维度({states.shape[1]}) != info定义({expected_shape[0]})")

        # NaN/Inf 检查
        if states is not None:
            if np.isnan(states).any():
                result.add_error("observation.state 包含 NaN")
            if np.isinf(states).any():
                result.add_error("observation.state 包含 Inf")

    # timestamp 单调递增/间隔检查（按 episode+frame_index 排序，不依赖行顺序）
    required_ts_cols = {'episode_index', 'frame_index', 'timestamp'}
    if required_ts_cols.issubset(set(data.columns)):
        ts_df = data[list(required_ts_cols)].sort_values(['episode_index', 'frame_index']).reset_index(drop=True)
        ep = ts_df['episode_index'].to_numpy()
        ts = ts_df['timestamp'].to_numpy()
        dt_all = np.diff(ts)
        same_ep = ep[1:] == ep[:-1]

        dt = dt_all[same_ep]
        if len(dt) > 0 and np.any(dt < -1e-6):
            i = int(np.where(same_ep & (dt_all < -1e-6))[0][0]) + 1
            result.add_error(
                f"timestamp 非单调递增: episode={int(ts_df.loc[i,'episode_index'])}, frame_index={int(ts_df.loc[i,'frame_index'])}"
            )
        else:
            expected = 1.0 / float(fps)
            if len(dt) > 0 and np.any(dt > expected * 2):
                i = int(np.where(same_ep & (dt_all > expected * 2))[0][0]) + 1
                result.add_warning(
                    f"timestamp 间隔异常: episode={int(ts_df.loc[i,'episode_index'])}, "
                    f"frame_index={int(ts_df.loc[i,'frame_index'])}, dt={float(dt_all[i-1]):.4f}s"
                )

    return result


# ============================================================
# 五、视频文件检查 [致命/如果使用]
# ============================================================
def get_video_frame_count(video_path: Path) -> Optional[int]:
    """获取视频帧数"""
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


def get_video_duration(video_path: Path) -> Optional[float]:
    """获取视频时长(秒)"""
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'format=duration', '-of', 'csv=p=0', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def get_video_resolution(video_path: Path) -> Optional[Tuple[int, int]]:
    """获取视频分辨率 (width, height)"""
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None


def check_videos(root: Path, info: Dict, episodes: pd.DataFrame) -> CheckResult:
    """检查视频文件完整性"""
    result = CheckResult()
    features = info.get('features', {})
    total_frames = info.get('total_frames', 0)

    # 获取所有视频 key
    video_keys = [k for k, v in features.items() if v.get('dtype') == 'video']

    if not video_keys:
        result.add_info("没有视频特征，跳过视频检查")
        return result

    for vk in video_keys:
        video_dir = format_video_file_path(root, info, vk, 0, 0).parent.parent
        if not video_dir.exists():
            result.add_error(f"视频目录不存在: {vk}")
            continue

        # 必要列
        ccol = f'videos/{vk}/chunk_index'
        fcol = f'videos/{vk}/file_index'
        from_col = f'videos/{vk}/from_timestamp'
        to_col = f'videos/{vk}/to_timestamp'
        required_cols = [ccol, fcol, from_col, to_col]
        if episodes is None or not all(c in episodes.columns for c in required_cols):
            result.add_error(f"episodes 缺少视频字段，无法检查 {vk}: {required_cols}")
            continue

        # 以 episodes 引用为准，检查文件存在/可读/时长/片段约束
        expected_shape = features.get(vk, {}).get('shape', [])
        unknown_frame_count = 0
        total_video_frames = 0

        refs = episodes[[ccol, fcol]].drop_duplicates()
        for chunk, file in refs.itertuples(index=False):
            mp4 = format_video_file_path(root, info, vk, int(chunk), int(file))
            if not mp4.exists():
                result.add_error(f"缺少视频文件: {mp4}")
                continue

            duration = get_video_duration(mp4)
            if duration is None:
                result.add_error(f"无法读取视频时长: {mp4}")
                continue

            resolution = get_video_resolution(mp4)
            if resolution is None:
                result.add_error(f"无法读取视频分辨率: {mp4}")
            else:
                width, height = resolution
                if expected_shape and len(expected_shape) >= 2:
                    exp_height, exp_width = expected_shape[0], expected_shape[1]
                    if height != exp_height or width != exp_width:
                        result.add_warning(f"{vk} 分辨率({width}x{height}) != info定义({exp_width}x{exp_height})")

            # max(to_timestamp) <= mp4_duration
            seg = episodes[(episodes[ccol] == chunk) & (episodes[fcol] == file)]
            max_to = float(seg[to_col].max())
            if max_to > float(duration) + 1e-3:
                result.add_error(f"{vk} {mp4.name} 时间戳越界: max_to={max_to:.3f}s > duration={float(duration):.3f}s")

            # 片段不重叠/首尾相接（允许极小误差）
            seg = seg.sort_values(from_col)
            prev_to = None
            for _, r in seg.iterrows():
                fr = float(r[from_col])
                to = float(r[to_col])
                if fr < -1e-6 or to < -1e-6:
                    result.add_error(f"{vk} {mp4.name} 存在负时间戳: from={fr:.3f}, to={to:.3f}")
                    break
                if to <= fr:
                    result.add_error(f"{vk} {mp4.name} 存在非法片段: to<=from ({to:.3f}<= {fr:.3f})")
                    break
                if prev_to is not None:
                    if fr < prev_to - 1e-3:
                        result.add_error(f"{vk} {mp4.name} 片段重叠: prev_to={prev_to:.3f} > from={fr:.3f}")
                        break
                    if fr > prev_to + 1e-3:
                        result.add_warning(f"{vk} {mp4.name} 片段有间隙: prev_to={prev_to:.3f} < from={fr:.3f}")
                prev_to = to

            # 帧数统计（可选，部分编码 nb_frames 为空）
            frame_count = get_video_frame_count(mp4)
            if frame_count is None:
                unknown_frame_count += 1
            else:
                total_video_frames += frame_count

        if unknown_frame_count > 0:
            result.add_warning(f"{vk} 有 {unknown_frame_count} 个视频无法读取 nb_frames，已跳过帧数求和校验")
        else:
            if total_video_frames != total_frames:
                result.add_warning(f"{vk} 视频帧数({total_video_frames}) != total_frames({total_frames})")

    return result


# ============================================================
# 六、tasks.parquet 检查 [重要]
# ============================================================
def check_tasks_parquet(root: Path, info: Dict) -> CheckResult:
    """检查 tasks.parquet"""
    result = CheckResult()
    tasks_path = root / 'meta/tasks.parquet'

    if not tasks_path.exists():
        result.add_error("tasks.parquet 不存在")
        return result

    try:
        tasks = pd.read_parquet(tasks_path)
    except Exception as e:
        result.add_error(f"读取 tasks.parquet 失败: {e}")
        return result

    total_tasks = info.get('total_tasks', 1)

    # 检查行数
    if len(tasks) != total_tasks:
        result.add_error(f"tasks 行数({len(tasks)}) != total_tasks({total_tasks})")

    # 必要列：task_index
    if 'task_index' not in tasks.columns:
        result.add_error("tasks.parquet 缺少列: task_index")
        return result

    # 任务名应作为 index（LeRobotDataset.__getitem__ 使用 tasks.iloc[task_idx].name）
    if tasks.index.has_duplicates:
        dup = tasks.index[tasks.index.duplicated()].unique().tolist()[:10]
        result.add_error(f"tasks.parquet index(任务名) 有重复: {dup}")

    if len(tasks) > 0:
        first_name = tasks.index[0]
        if not isinstance(first_name, str):
            result.add_warning("tasks.parquet 的任务名未作为字符串 index 保存，可能导致 task 解析异常")

    # 检查 task_index 连续性 + 行顺序（iloc 依赖）
    try:
        indices = tasks['task_index'].to_numpy().astype(np.int64)
    except Exception:
        result.add_error("tasks.task_index 无法转换为 int")
        return result

    if len(np.unique(indices)) != len(indices):
        result.add_error("tasks.task_index 存在重复值")
        return result

    expected = np.arange(len(tasks), dtype=np.int64)
    if not np.array_equal(np.sort(indices), expected):
        result.add_error("tasks.task_index 不连续或超范围，期望覆盖 0..total_tasks-1")
    elif not np.array_equal(indices, expected):
        result.add_error("tasks.task_index 行顺序与 task_index 不一致（tasks.iloc[task_index] 将解析错误）")

    return result


# ============================================================
# 七、stats.json 检查 [重要]
# ============================================================
def check_stats_json(root: Path, info: Dict) -> CheckResult:
    """检查 stats.json 完整性"""
    result = CheckResult()
    stats_path = root / 'meta/stats.json'

    if not stats_path.exists():
        result.add_error("stats.json 不存在")
        return result

    try:
        with open(stats_path) as f:
            stats = json.load(f)
    except Exception as e:
        result.add_error(f"读取 stats.json 失败: {e}")
        return result

    total_frames = info.get('total_frames', 0)
    features = info.get('features', {})
    required_keys = {'min', 'max', 'mean', 'std', 'count', 'q01', 'q10', 'q50', 'q90', 'q99'}

    # 检查 info.features 中的每个 feature 都有统计
    missing_stats = [feat for feat in features.keys() if feat not in stats]
    if missing_stats:
        result.add_warning(f"stats.json 缺少以下 feature 的统计: {missing_stats[:10]}{'...' if len(missing_stats) > 10 else ''}")

    # 检查每个 feature 的统计
    for feat, feat_stats in stats.items():
        if not isinstance(feat_stats, dict):
            result.add_warning(f"stats.{feat} 格式异常（非 dict），跳过字段检查")
            continue

        missing = required_keys - set(feat_stats.keys())
        if missing:
            result.add_warning(f"stats.{feat} 缺少字段: {missing}")

        # 检查 count
        if 'count' in feat_stats:
            count = feat_stats['count']
            if isinstance(count, list):
                count = count[0]
            # 数值特征的 count 应该等于 total_frames（video/image/string 不强制）
            spec = features.get(feat, {})
            dtype = spec.get('dtype')
            if dtype not in ['video', 'image', 'string'] and count != total_frames:
                result.add_warning(f"stats.{feat}.count({count}) != total_frames({total_frames})")

    return result


# ============================================================
# 八、数据质量检查 [建议]
# ============================================================
def check_data_quality(data: pd.DataFrame, info: Dict) -> CheckResult:
    """检查数据质量"""
    result = CheckResult()

    if 'action' not in data.columns:
        return result

    try:
        actions = np.stack(data['action'].to_numpy())
    except Exception:
        try:
            actions = np.stack(np.array(data['action'].tolist(), dtype=object))
        except Exception:
            result.add_warning("action 无法解析为二维数组，跳过质量检查")
            return result

    # 检查全零帧
    zero_frames = np.all(actions == 0, axis=1).sum()
    if zero_frames > 0:
        result.add_warning(f"发现 {zero_frames} 个全零帧")

    # 检查连续静止帧（默认按 episode 边界计算）
    static_frames = None
    static_ep = None
    static_frame = None
    if {'episode_index', 'frame_index'}.issubset(set(data.columns)):
        df = data[['episode_index', 'frame_index', 'action']].sort_values(['episode_index', 'frame_index'])
        try:
            a_sorted = np.stack(df['action'].to_numpy())
            ep = df['episode_index'].to_numpy(dtype=np.int64)
            frame = df['frame_index'].to_numpy(dtype=np.int64)
            action_diff = np.diff(a_sorted, axis=0)
            same_ep = ep[1:] == ep[:-1]
            static_frames = same_ep & np.all(np.abs(action_diff) < 1e-6, axis=1)
            static_ep = ep
            static_frame = frame
        except Exception:
            static_frames = None

    if static_frames is None:
        action_diff = np.diff(actions, axis=0)
        static_frames = np.all(np.abs(action_diff) < 1e-6, axis=1)

    max_consecutive = 0
    best_end = None
    best_start = None
    current = 0
    for i, is_static in enumerate(static_frames):
        if is_static:
            current += 1
            max_consecutive = max(max_consecutive, current)
            if current == max_consecutive:
                best_end = i
                best_start = i - current + 1
        else:
            current = 0

    max_consecutive_frames = max_consecutive + 1 if max_consecutive > 0 else 0
    if max_consecutive_frames > 60:
        if best_end is not None and best_start is not None and static_ep is not None and static_frame is not None:
            start_row = best_start
            end_row = best_end + 1
            ep0 = int(static_ep[end_row])
            f0 = int(static_frame[start_row])
            f1 = int(static_frame[end_row])
            result.add_warning(
                f"最长连续静止: {max_consecutive_frames} 帧 (>60) (episode={ep0}, frame={f0}->{f1})"
            )
        else:
            result.add_warning(f"最长连续静止: {max_consecutive_frames} 帧 (>60)")

    # 检查关节值范围
    if actions.max() > 3.5 or actions.min() < -3.5:
        result.add_warning(f"关节值可能越界: [{actions.min():.2f}, {actions.max():.2f}]")

    return result


# ============================================================
# 主函数
# ============================================================
def print_result(name: str, result: CheckResult) -> str:
    """打印检查结果"""
    if result.errors:
        status = "✗ 错误"
    elif result.warnings:
        status = "⚠ 警告"
    else:
        status = "✓ 通过"

    print(f"{name} {'.' * (45 - len(name))} {status}")

    for err in result.errors:
        print(f"     [错误] {err}")
    for warn in result.warnings:
        print(f"     [警告] {warn}")

    return status


def run_all_checks(dataset_path: str) -> bool:
    """运行所有检查"""
    root = Path(dataset_path).resolve()

    if not root.exists():
        print(f"错误: 数据集不存在: {root}")
        return False

    print("=" * 60)
    print("LeRobot v3.0 数据集检查报告")
    print("=" * 60)
    print(f"数据集: {root}")
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = {}
    has_fatal_error = False

    # 二、info.json 检查
    print("[二] info.json 检查")
    info_result, info = check_info_json(root)
    results['info'] = print_result("    info.json", info_result)
    if not info_result.is_ok():
        has_fatal_error = True

    if info is None:
        print("\n无法继续检查，info.json 加载失败")
        return False

    # 一、目录结构检查
    print("[一] 目录结构检查")
    dir_result = check_directory_structure(root, info)
    results['dir'] = print_result("    目录结构", dir_result)
    if not dir_result.is_ok():
        has_fatal_error = True

    # 三、episodes 检查
    print("[三] episodes parquet 检查")
    ep_result, episodes = check_episodes_parquet(root, info)
    results['episodes'] = print_result("    episodes", ep_result)
    if not ep_result.is_ok():
        has_fatal_error = True

    # 检查 episodes 引用的 data/video 文件是否存在
    if episodes is not None:
        ref_result = check_episode_file_references(root, info, episodes)
        results['references'] = print_result("    文件引用", ref_result)
        if not ref_result.is_ok():
            has_fatal_error = True

    # 检查时间戳约束
    if episodes is not None:
        ts_result = check_episodes_timestamps(episodes, info)
        results['timestamps'] = print_result("    时间戳约束", ts_result)
        if not ts_result.is_ok():
            has_fatal_error = True

    # 四、data 检查
    print("[四] data parquet 检查")
    data_result, data = check_data_parquet(root, info, episodes)
    results['data'] = print_result("    data", data_result)
    if not data_result.is_ok():
        has_fatal_error = True

    if data is not None:
        detail_result = check_data_details(data, info, episodes)
        results['data_detail'] = print_result("    data详细", detail_result)
        if not detail_result.is_ok():
            has_fatal_error = True

    # 五、视频检查
    print("[五] 视频文件检查")
    video_result = check_videos(root, info, episodes)
    results['videos'] = print_result("    视频", video_result)
    if not video_result.is_ok():
        has_fatal_error = True

    # 六、tasks 检查
    print("[六] tasks.parquet 检查")
    tasks_result = check_tasks_parquet(root, info)
    results['tasks'] = print_result("    tasks", tasks_result)
    if not tasks_result.is_ok():
        has_fatal_error = True

    # 七、stats 检查
    print("[七] stats.json 检查")
    stats_result = check_stats_json(root, info)
    results['stats'] = print_result("    stats", stats_result)
    if not stats_result.is_ok():
        has_fatal_error = True

    # 八、数据质量检查
    print("[八] 数据质量检查")
    if data is not None:
        quality_result = check_data_quality(data, info)
        results['quality'] = print_result("    质量", quality_result)

    # 总结
    print()
    print("=" * 60)
    passed = sum(1 for s in results.values() if '通过' in s)
    warnings = sum(1 for s in results.values() if '警告' in s)
    errors = sum(1 for s in results.values() if '错误' in s)
    print(f"总结: {passed} 通过, {warnings} 警告, {errors} 错误")
    print("=" * 60)

    return not has_fatal_error


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='LeRobot v3.0 数据集完整检查工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python3 check_lerobot_v3_dataset_final.py /path/to/dataset
    python3 check_lerobot_v3_dataset_final.py ~/.cache/huggingface/lerobot/cqy/agilex_left_banana_final
        """
    )
    parser.add_argument('dataset_path', help='数据集完整路径')

    args = parser.parse_args()

    success = run_all_checks(args.dataset_path)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
 
