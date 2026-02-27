from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import h5py
import numpy as np
import yaml

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, JointState


_OPENCV_ENCODING_RE = re.compile(r"^(?P<bits>\\d+)(?P<kind>[USF])C(?P<channels>\\d+)$")


def _encoding_to_dtype_and_channels(encoding: str) -> tuple[np.dtype, int]:
    # Common ROS encodings.
    if encoding in ("rgb8", "bgr8"):
        return np.dtype(np.uint8), 3
    if encoding in ("rgba8", "bgra8"):
        return np.dtype(np.uint8), 4
    if encoding == "mono8":
        return np.dtype(np.uint8), 1
    if encoding == "mono16":
        return np.dtype(np.uint16), 1

    # OpenCV-style encoding like "16UC1", "32FC1".
    m = _OPENCV_ENCODING_RE.match(encoding)
    if not m:
        raise ValueError(f"Unsupported Image encoding: {encoding!r}")

    bits = int(m.group("bits"))
    kind = m.group("kind")
    channels = int(m.group("channels"))

    if kind == "U":
        dtype = {8: np.uint8, 16: np.uint16, 32: np.uint32}.get(bits)
    elif kind == "S":
        dtype = {8: np.int8, 16: np.int16, 32: np.int32}.get(bits)
    elif kind == "F":
        dtype = {32: np.float32, 64: np.float64}.get(bits)
    else:
        dtype = None

    if dtype is None:
        raise ValueError(f"Unsupported Image encoding: {encoding!r}")
    return np.dtype(dtype), channels


def ros_image_to_numpy(msg: Image, *, want_rgb: bool) -> np.ndarray:
    """Convert a ROS Image message to a numpy array without cv_bridge."""
    dtype, channels = _encoding_to_dtype_and_channels(msg.encoding)
    itemsize = dtype.itemsize

    if msg.step % itemsize != 0:
        raise ValueError(f"Invalid Image.step={msg.step} for dtype itemsize={itemsize} (encoding={msg.encoding})")

    row_stride_elems = msg.step // itemsize
    expected_elems = row_stride_elems * msg.height
    arr = np.frombuffer(msg.data, dtype=dtype, count=expected_elems)
    arr = arr.reshape((msg.height, row_stride_elems))

    pixel_elems = msg.width * channels
    if pixel_elems > row_stride_elems:
        raise ValueError(
            f"Invalid Image: width*channels={pixel_elems} > step/itemsize={row_stride_elems} (encoding={msg.encoding})"
        )

    arr = arr[:, :pixel_elems]
    if channels == 1:
        arr = arr.reshape((msg.height, msg.width))
    else:
        arr = arr.reshape((msg.height, msg.width, channels))

    # Handle endianness for multi-byte types.
    if itemsize > 1 and bool(msg.is_bigendian) != (sys.byteorder == "big"):
        arr = arr.byteswap().newbyteorder()

    if want_rgb and msg.encoding in ("bgr8", "bgra8"):
        if channels == 3:
            arr = arr[:, :, ::-1]
        elif channels == 4:
            # BGRA -> RGBA
            arr = arr[:, :, [2, 1, 0, 3]]

    # Ensure we don't hold onto the ROS buffer.
    return np.ascontiguousarray(arr)


@dataclass
class JointBuffer:
    names: list[str] | None = None
    pos: np.ndarray | None = None
    vel: np.ndarray | None = None
    eff: np.ndarray | None = None
    stamp_s: float | None = None

    def update(self, msg: JointState) -> None:
        if self.names is None and msg.name:
            self.names = list(msg.name)

        # If names are present and stable, reorder incoming messages to match the first observed order.
        if self.names and msg.name and list(msg.name) != self.names:
            idx = {n: i for i, n in enumerate(msg.name)}
            order = [idx.get(n, None) for n in self.names]
            if any(i is None for i in order):
                raise ValueError(f"JointState name mismatch. Expected {self.names}, got {list(msg.name)}")
            pos = np.array([msg.position[i] for i in order], dtype=np.float32)
            vel = np.array([msg.velocity[i] for i in order], dtype=np.float32) if msg.velocity else None
            eff = np.array([msg.effort[i] for i in order], dtype=np.float32) if msg.effort else None
        else:
            pos = np.array(msg.position, dtype=np.float32)
            vel = np.array(msg.velocity, dtype=np.float32) if msg.velocity else None
            eff = np.array(msg.effort, dtype=np.float32) if msg.effort else None

        self.pos = pos
        self.vel = vel
        self.eff = eff
        self.stamp_s = msg.header.stamp.to_sec() if msg.header and msg.header.stamp else None


class LatestBuffers:
    def __init__(self):
        self.lock = Lock()
        self.rgb: dict[str, Image] = {}
        self.depth: dict[str, Image] = {}
        self.joints: dict[str, JointBuffer] = {}
        self.odom: Odometry | None = None

    def set_rgb(self, cam: str, msg: Image) -> None:
        with self.lock:
            self.rgb[cam] = msg

    def set_depth(self, cam: str, msg: Image) -> None:
        with self.lock:
            self.depth[cam] = msg

    def set_joint(self, key: str, msg: JointState) -> None:
        with self.lock:
            buf = self.joints.get(key)
            if buf is None:
                buf = JointBuffer()
                self.joints[key] = buf
            buf.update(msg)

    def set_odom(self, msg: Odometry) -> None:
        with self.lock:
            self.odom = msg


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_config_path() -> Path:
    return Path(__file__).parent / "configs" / "default.yaml"


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


class _StopFlag:
    def __init__(self):
        self._stop = False

    def set(self) -> None:
        self._stop = True

    def is_set(self) -> bool:
        return self._stop


def _install_signal_handlers(stop_flag: _StopFlag) -> None:
    def _handler(_sig, _frame):
        stop_flag.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _next_episode_name(task_dir: Path) -> str:
    existing = sorted(task_dir.glob("episode_*.hdf5"))
    max_idx = -1
    for p in existing:
        m = re.match(r"episode_(\\d+)\\.hdf5$", p.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return f"episode_{max_idx + 1:06d}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record ROS topics into an HDF5 episode (LeRobot-friendly schema).")
    p.add_argument("--config", type=str, default=str(_default_config_path()), help="YAML config path.")
    p.add_argument("--dataset_dir", type=str, default="/tmp/web_collection_data", help="Output dataset root.")
    p.add_argument("--task_name", type=str, default=None, help="Override task name (subfolder under dataset_dir).")
    p.add_argument("--episode_name", type=str, default=None, help="Override episode name (without extension).")
    p.add_argument("--num_episodes", type=int, default=1, help="How many episodes to record in a row.")
    p.add_argument("--fps", type=int, default=None, help="Override fps.")
    p.add_argument("--max_frames", type=int, default=None, help="Override max frames (0 means until stopped).")
    p.add_argument(
        "--reset_time_s",
        type=float,
        default=None,
        help="Seconds to wait between episodes for environment reset (only used when recording multiple episodes).",
    )
    p.add_argument("--use_depth", action="store_true", help="Record depth images too.")
    p.add_argument("--instruction", type=str, default=None, help="Override instruction/task text.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)

    fps = int(args.fps if args.fps is not None else cfg.get("fps", 30))
    max_frames = args.max_frames if args.max_frames is not None else int(cfg.get("max_frames", 0))
    use_depth = bool(args.use_depth or cfg.get("use_depth", False))
    num_episodes = int(args.num_episodes)
    if num_episodes < 1:
        raise ValueError("--num_episodes must be >= 1")
    reset_time_s = float(args.reset_time_s if args.reset_time_s is not None else cfg.get("reset_time_s", 0.0))
    if reset_time_s < 0:
        raise ValueError("--reset_time_s must be >= 0")
    if num_episodes > 1 and (not max_frames or max_frames <= 0):
        raise ValueError("When --num_episodes > 1, you must set --max_frames > 0 to auto-split episodes.")

    task_name = args.task_name if args.task_name is not None else cfg.get("task_name", "task")
    instruction_raw = args.instruction if args.instruction is not None else (cfg.get("instruction", "") or "")
    instruction = str(instruction_raw).strip()
    if not instruction:
        raise ValueError(
            "instruction 不能为空。VLA 数据采集必须提供语言指令："
            "请通过 --instruction 传入，或在配置文件中设置非空 instruction。"
        )

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    task_dir = dataset_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # Build episode names list.
    episode_names: list[str] = []
    if args.episode_name:
        if num_episodes == 1:
            episode_names = [args.episode_name]
        else:
            m = re.match(r"episode_(\d+)$", args.episode_name)
            if m:
                start_idx = int(m.group(1))
                width = len(m.group(1))
                episode_names = [f"episode_{start_idx + i:0{width}d}" for i in range(num_episodes)]
            else:
                episode_names = [f"{args.episode_name}_{i:03d}" for i in range(num_episodes)]
    else:
        # Auto increment from existing files.
        first_name = _next_episode_name(task_dir)
        m = re.match(r"episode_(\d+)$", first_name)
        if not m:
            raise RuntimeError(f"Unexpected auto episode name: {first_name}")
        start_idx = int(m.group(1))
        episode_names = [f"episode_{start_idx + i:06d}" for i in range(num_episodes)]

    out_paths = [task_dir / f"{name}.hdf5" for name in episode_names]
    existing = [p for p in out_paths if p.exists()]
    if existing:
        raise FileExistsError(f"Some output episodes already exist, e.g. {existing[0]}")

    topics = cfg.get("topics", {})
    rgb_topics: dict[str, str] = topics.get("rgb_images", {}) or {}
    depth_topics: dict[str, str] = topics.get("depth_images", {}) or {}
    joint_topics: dict[str, str] = topics.get("joints", {}) or {}
    odom_topic: str | None = topics.get("base_odom", None)

    if not rgb_topics:
        raise ValueError("No rgb_images configured.")
    required_joint_keys = ("puppet_left", "puppet_right", "master_left", "master_right")
    missing = [k for k in required_joint_keys if k not in joint_topics]
    if missing:
        raise ValueError(f"Missing joint topics in config: {missing}")

    stop_flag = _StopFlag()
    _install_signal_handlers(stop_flag)

    buffers = LatestBuffers()

    rospy.init_node("web_collection_record", anonymous=True, disable_signals=True)

    for cam, topic in rgb_topics.items():
        rospy.Subscriber(topic, Image, lambda m, c=cam: buffers.set_rgb(c, m), queue_size=10, tcp_nodelay=True)
        print(f"[sub] rgb {cam}: {topic}")

    if use_depth:
        if not depth_topics:
            raise ValueError("use_depth=true but no depth_images configured.")
        for cam, topic in depth_topics.items():
            rospy.Subscriber(topic, Image, lambda m, c=cam: buffers.set_depth(c, m), queue_size=10, tcp_nodelay=True)
            print(f"[sub] depth {cam}: {topic}")

    for key, topic in joint_topics.items():
        rospy.Subscriber(topic, JointState, lambda m, k=key: buffers.set_joint(k, m), queue_size=50, tcp_nodelay=True)
        print(f"[sub] joint {key}: {topic}")

    if odom_topic:
        rospy.Subscriber(odom_topic, Odometry, buffers.set_odom, queue_size=50, tcp_nodelay=True)
        print(f"[sub] odom: {odom_topic}")

    # Wait for first messages.
    print("[wait] waiting for required topics...")
    t_wait0 = time.time()
    while not rospy.is_shutdown() and not stop_flag.is_set():
        with buffers.lock:
            have_rgb = all(cam in buffers.rgb for cam in rgb_topics)
            have_depth = (not use_depth) or all(cam in buffers.depth for cam in depth_topics)
            have_joints = all(k in buffers.joints and buffers.joints[k].pos is not None for k in required_joint_keys)
        if have_rgb and have_depth and have_joints:
            break
        if time.time() - t_wait0 > 30.0:
            raise TimeoutError("Timed out waiting for required ROS topics.")
        time.sleep(0.05)

    # Infer shapes from first images/joints.
    with buffers.lock:
        first_rgb = {cam: buffers.rgb[cam] for cam in rgb_topics}
        first_depth = {cam: buffers.depth[cam] for cam in depth_topics} if use_depth else {}
        jb = buffers.joints
        puppet_left_dim = int(jb["puppet_left"].pos.shape[0])
        puppet_right_dim = int(jb["puppet_right"].pos.shape[0])
        master_left_dim = int(jb["master_left"].pos.shape[0])
        master_right_dim = int(jb["master_right"].pos.shape[0])

    qpos_dim = puppet_left_dim + puppet_right_dim
    action_dim = master_left_dim + master_right_dim

    rgb_shapes: dict[str, tuple[int, int, int]] = {}
    for cam, msg in first_rgb.items():
        img = ros_image_to_numpy(msg, want_rgb=True)
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            raise ValueError(f"RGB image must be HxWx3/4, got shape={img.shape} for cam={cam} enc={msg.encoding}")
        if img.shape[2] == 4:
            img = img[:, :, :3]
        rgb_shapes[cam] = (int(img.shape[0]), int(img.shape[1]), 3)

    depth_dtypes: dict[str, np.dtype] = {}
    depth_shapes: dict[str, tuple[int, int]] = {}
    if use_depth:
        for cam, msg in first_depth.items():
            img = ros_image_to_numpy(msg, want_rgb=False)
            if img.ndim != 2:
                raise ValueError(f"Depth image must be HxW, got shape={img.shape} for cam={cam} enc={msg.encoding}")
            depth_dtypes[cam] = img.dtype
            depth_shapes[cam] = (int(img.shape[0]), int(img.shape[1]))

    for ep_i, (episode_name, out_path) in enumerate(zip(episode_names, out_paths, strict=True), start=1):
        if stop_flag.is_set() or rospy.is_shutdown():
            break

        print(f"[episode] {ep_i}/{num_episodes}: {out_path}")
        with h5py.File(out_path, "w", rdcc_nbytes=1024**2 * 64) as root:
            root.attrs["sim"] = False
            root.attrs["compress"] = False
            root.attrs["fps"] = int(fps)
            root.attrs["created_at"] = _now_ts()
            root.attrs["task_name"] = str(task_name)
            root.attrs["episode_name"] = str(episode_name)
            root.attrs["instruction"] = str(instruction)
            root.attrs["config_path"] = str(cfg_path)
            root.attrs["config_yaml"] = yaml.safe_dump(cfg, sort_keys=False)
            root.attrs["schema"] = "web_collection_hdf5_v1"
            root.attrs["batch_num_episodes"] = int(num_episodes)
            root.attrs["batch_episode_index"] = int(ep_i - 1)

            obs = root.create_group("observations")
            g_img = obs.create_group("images")
            g_depth = obs.create_group("images_depth") if use_depth else None

            # Preallocate when max_frames > 0 for speed; otherwise grow dynamically.
            fixed_T = int(max_frames) if max_frames else 0

            ds_images: dict[str, h5py.Dataset] = {}
            for cam, (h, w, c) in rgb_shapes.items():
                shape0 = fixed_T if fixed_T else 0
                ds_images[cam] = g_img.create_dataset(
                    cam,
                    shape=(shape0, h, w, c),
                    maxshape=(None, h, w, c),
                    dtype="uint8",
                    chunks=(1, h, w, c),
                )

            ds_depth: dict[str, h5py.Dataset] = {}
            if use_depth and g_depth is not None:
                for cam, (h, w) in depth_shapes.items():
                    shape0 = fixed_T if fixed_T else 0
                    ds_depth[cam] = g_depth.create_dataset(
                        cam,
                        shape=(shape0, h, w),
                        maxshape=(None, h, w),
                        dtype=depth_dtypes[cam],
                        chunks=(1, h, w),
                    )

            def _vec_ds(name: str, dim: int, dtype: str) -> h5py.Dataset:
                shape0 = fixed_T if fixed_T else 0
                return obs.create_dataset(name, shape=(shape0, dim), maxshape=(None, dim), dtype=dtype)

            ds_qpos = _vec_ds("qpos", qpos_dim, "float32")
            ds_qvel = _vec_ds("qvel", qpos_dim, "float32")
            ds_eff = _vec_ds("effort", qpos_dim, "float32")

            def _root_vec_ds(name: str, dim: int, dtype: str) -> h5py.Dataset:
                shape0 = fixed_T if fixed_T else 0
                return root.create_dataset(name, shape=(shape0, dim), maxshape=(None, dim), dtype=dtype)

            ds_action = _root_vec_ds("action", action_dim, "float32")
            ds_base = _root_vec_ds("base_action", 2, "float32")
            ds_ts = root.create_dataset("timestamp", shape=(fixed_T if fixed_T else 0,), maxshape=(None,), dtype="float64")

            rate = rospy.Rate(fps)
            n = 0
            t0 = time.time()
            while not rospy.is_shutdown() and not stop_flag.is_set():
                if max_frames and n >= max_frames:
                    break

                with buffers.lock:
                    rgb_msgs = {cam: buffers.rgb.get(cam) for cam in rgb_topics}
                    depth_msgs = {cam: buffers.depth.get(cam) for cam in depth_topics} if use_depth else {}
                    jb = buffers.joints
                    odom = buffers.odom

                if any(m is None for m in rgb_msgs.values()):
                    rate.sleep()
                    continue
                if use_depth and any(m is None for m in depth_msgs.values()):
                    rate.sleep()
                    continue
                if any(k not in jb or jb[k].pos is None for k in required_joint_keys):
                    rate.sleep()
                    continue

                # Grow datasets if needed.
                if fixed_T == 0:
                    for ds in ds_images.values():
                        ds.resize((n + 1,) + ds.shape[1:])
                    for ds in ds_depth.values():
                        ds.resize((n + 1,) + ds.shape[1:])
                    for ds in (ds_qpos, ds_qvel, ds_eff, ds_action, ds_base):
                        ds.resize((n + 1,) + ds.shape[1:])
                    ds_ts.resize((n + 1,))

                # Images: normalize to uint8 RGB (HxWx3).
                for cam, msg in rgb_msgs.items():
                    img = ros_image_to_numpy(msg, want_rgb=True)
                    if img.ndim == 2:
                        img = np.repeat(img[:, :, None], 3, axis=2)
                    if img.shape[2] == 4:
                        img = img[:, :, :3]
                    if img.dtype != np.uint8:
                        raise ValueError(
                            f"RGB image dtype must be uint8, got {img.dtype} (cam={cam}, enc={msg.encoding})"
                        )
                    ds_images[cam][n] = img

                if use_depth:
                    for cam, msg in depth_msgs.items():
                        img = ros_image_to_numpy(msg, want_rgb=False)
                        ds_depth[cam][n] = img

                # Joints.
                puppet_left = jb["puppet_left"].pos
                puppet_right = jb["puppet_right"].pos
                master_left = jb["master_left"].pos
                master_right = jb["master_right"].pos

                qpos = np.concatenate([puppet_left, puppet_right], axis=0).astype(np.float32)
                action = np.concatenate([master_left, master_right], axis=0).astype(np.float32)

                # Optional velocities/effort.
                qvel = np.zeros_like(qpos, dtype=np.float32)
                eff = np.zeros_like(qpos, dtype=np.float32)
                if jb["puppet_left"].vel is not None and jb["puppet_right"].vel is not None:
                    qvel = np.concatenate([jb["puppet_left"].vel, jb["puppet_right"].vel], axis=0).astype(np.float32)
                if jb["puppet_left"].eff is not None and jb["puppet_right"].eff is not None:
                    eff = np.concatenate([jb["puppet_left"].eff, jb["puppet_right"].eff], axis=0).astype(np.float32)

                base = np.zeros((2,), dtype=np.float32)
                if odom is not None:
                    base[0] = float(odom.twist.twist.linear.x)
                    base[1] = float(odom.twist.twist.angular.z)

                ts = float(rospy.Time.now().to_sec())

                ds_qpos[n] = qpos
                ds_qvel[n] = qvel
                ds_eff[n] = eff
                ds_action[n] = action
                ds_base[n] = base
                ds_ts[n] = ts

                n += 1
                if n % max(1, fps) == 0:
                    dt = time.time() - t0
                    print(f"[rec] episode={ep_i}/{num_episodes} frames={n} elapsed={dt:.1f}s ({n/dt:.1f} fps)")
                    root.flush()

                rate.sleep()

            # Shrink to actual recorded size when preallocated.
            if fixed_T and n < fixed_T:
                for ds in ds_images.values():
                    ds.resize((n,) + ds.shape[1:])
                for ds in ds_depth.values():
                    ds.resize((n,) + ds.shape[1:])
                for ds in (ds_qpos, ds_qvel, ds_eff, ds_action, ds_base):
                    ds.resize((n,) + ds.shape[1:])
                ds_ts.resize((n,))

            root.attrs["num_frames"] = int(n)
            root.flush()

        print(f"[done] saved {out_path} frames={n} fps={fps}")
        if ep_i < num_episodes and reset_time_s > 0 and not stop_flag.is_set() and not rospy.is_shutdown():
            print(f"[reset] 等待 {reset_time_s:.1f}s 用于重置环境，然后开始下一条...")
            t0 = time.time()
            while time.time() - t0 < reset_time_s:
                if stop_flag.is_set() or rospy.is_shutdown():
                    break
                time.sleep(0.1)


if __name__ == "__main__":
    main()
