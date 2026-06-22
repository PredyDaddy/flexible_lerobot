# 按照 AgileX 配置新机器人详解

本文基于当前仓库里的 `src/lerobot/robots/agilex/` 实现，说明如何照着 AgileX 的方式添加一个新机器人。

重点回答三个问题：

1. AgileX 这几个文件分别干什么。
2. 里面的类名、字段名、feature key、topic 名是怎么定义的，分别有什么用。
3. 如果从 0 添加一个新机器人，应该怎么仿写。

本文只讲机器人接入本身，不讲模型训练细节。

## 1. AgileX 机器人目录结构

AgileX 的代码目录是：

```text
src/lerobot/robots/agilex/
├── __init__.py
├── config_agilex.py
├── agilex.py
└── agilex_ros_bridge.py
```

这四个文件的分工如下：

```text
config_agilex.py
定义 AgileX 机器人的配置项，比如 ROS topic、相机 key、图像尺寸、控制模式、关节名等。

agilex.py
定义 LeRobot 认识的 AgileXRobot 类，实现 Robot 标准接口，比如 connect、get_observation、send_action、disconnect。

agilex_ros_bridge.py
定义 AgileX 和 ROS 之间的桥接层，负责订阅 JointState、订阅 Image、发布动作、缓存最新状态、图片解码。

__init__.py
导出 AgileXRobot 和 AgileXRobotConfig，让外部可以导入，同时触发 config 注册。
```

可以把这套结构理解成三层：

```text
配置层：config_agilex.py
LeRobot 接口层：agilex.py
硬件通讯层：agilex_ros_bridge.py
```

添加新机器人时，建议也按这个结构拆，不要把所有硬件通讯和 LeRobot 接口都塞进一个文件里。

## 2. AgileX 的名字体系

AgileX 代码里有几类名字，作用不同，不要混在一起。

### 2.1 注册 type 名

在 `config_agilex.py` 里：

```python
@RobotConfig.register_subclass("agilex")
@dataclass(kw_only=True)
class AgileXRobotConfig(RobotConfig):
    ...
```

这里的 `"agilex"` 是 LeRobot 配置系统里的机器人类型名。

命令行里使用的是这个名字：

```bash
--robot.type=agilex
```

所以如果新机器人叫 `my_robot`，就应该写：

```python
@RobotConfig.register_subclass("my_robot")
@dataclass(kw_only=True)
class MyRobotConfig(RobotConfig):
    ...
```

命令行就是：

```bash
--robot.type=my_robot
```

### 2.2 Python 类名

AgileX 有两个主要类名：

```python
class AgileXRobotConfig(RobotConfig):
    ...

class AgileXRobot(Robot):
    ...
```

含义如下：

```text
AgileXRobotConfig
AgileX 的机器人配置类。

AgileXRobot
AgileX 的 LeRobot 机器人实现类。
```

新机器人建议使用类似命名：

```python
class MyRobotConfig(RobotConfig):
    ...

class MyRobot(Robot):
    ...
```

或者：

```python
class CobotMagicRobotConfig(RobotConfig):
    ...

class CobotMagicRobot(Robot):
    ...
```

重点是配置类和机器人类的对应关系要清楚。

### 2.3 Robot.name

在 `agilex.py` 里：

```python
class AgileXRobot(Robot):
    config_class = AgileXRobotConfig
    name = "agilex"
```

`name = "agilex"` 是机器人实例的名字。日志、特殊判断、脚本逻辑都可能用它。

当前仓库里 `lerobot_record.py` 里就有针对 AgileX 的判断：

```python
if robot.name == "agilex":
    ...
```

因此建议新机器人统一这几个名字：

```text
目录名：my_robot
注册名：@RobotConfig.register_subclass("my_robot")
Robot.name：my_robot
CLI：--robot.type=my_robot
```

这样最不容易混乱。

### 2.4 ROS topic 名

AgileX 配置里有：

```python
state_left_topic: str = "/puppet/joint_left"
state_right_topic: str = "/puppet/joint_right"
command_left_topic: str = "/master/joint_left"
command_right_topic: str = "/master/joint_right"
front_camera_topic: str = "/camera_f/color/image_raw"
left_camera_topic: str = "/camera_l/color/image_raw"
right_camera_topic: str = "/camera_r/color/image_raw"
```

这些是 ROS 世界里的名字，用来订阅或发布 ROS 消息。

例如：

```text
/puppet/joint_left
左臂 follower 的 JointState 状态。

/puppet/joint_right
右臂 follower 的 JointState 状态。

/master/joint_left
左臂 master 的 JointState，既可以作为 teleoperator 输入，也可以作为 command topic。

/master/joint_right
右臂 master 的 JointState。

/camera_f/color/image_raw
前方相机图片。

/camera_l/color/image_raw
左腕相机图片。

/camera_r/color/image_raw
右腕相机图片。
```

ROS topic 名可以根据真实机器变化。

### 2.5 LeRobot feature key

AgileX 配置里还有：

```python
front_camera_key: str = "camera_front"
left_camera_key: str = "camera_left"
right_camera_key: str = "camera_right"
```

这些不是 ROS topic，而是 LeRobot observation 和 dataset 里的字段名。

例如：

```text
front_camera_topic = "/camera_f/color/image_raw"
front_camera_key = "camera_front"
```

意思是：

```text
从 ROS topic /camera_f/color/image_raw 读取图片。
在 LeRobot observation 里保存为 observation["camera_front"]。
最终数据集里的图片字段也叫 camera_front。
```

所以要分清：

```text
ROS topic 名
用于真实硬件通讯。

LeRobot feature key
用于 observation、action、dataset、policy 输入输出。
```

## 3. config_agilex.py 详解

文件路径：

```text
src/lerobot/robots/agilex/config_agilex.py
```

核心结构：

```python
from dataclasses import dataclass, field

from ..config import RobotConfig


@RobotConfig.register_subclass("agilex")
@dataclass(kw_only=True)
class AgileXRobotConfig(RobotConfig):
    ...
```

### 3.1 control_mode

```python
control_mode: str = "passive_follow"
```

AgileX 支持两种模式：

```text
passive_follow
只读取机器人状态和图像，不创建 command publisher，不主动下发动作。

command_master
读取机器人状态和图像，同时创建 command publisher，可以主动下发动作。
```

在 `AgileXRobot.__init__()` 里会根据 `control_mode` 决定是否启用 command topic：

```python
command_left_topic=config.command_left_topic if config.control_mode == "command_master" else None
command_right_topic=config.command_right_topic if config.control_mode == "command_master" else None
```

在 `send_action()` 里也会判断：

```python
if self.config.control_mode == "command_master":
    self._bridge.publish_action(sent_action)
```

所以：

```text
passive_follow
适合只采集状态、观察机器人，或者动作由外部系统完成。

command_master
适合 LeRobot 主动控制机器人，比如策略推理和回放。
```

新机器人可以定义自己的模式，比如：

```python
control_mode: str = "position"
```

并在 `__post_init__()` 里校验：

```python
if self.control_mode not in {"position", "passive"}:
    raise ValueError(f"Unsupported control_mode: {self.control_mode}")
```

### 3.2 state topic

```python
state_left_topic: str = "/puppet/joint_left"
state_right_topic: str = "/puppet/joint_right"
```

这两个 topic 用来读取左右臂的当前关节状态。

AgileX 是双臂机器人，所以有 left 和 right 两路。

如果新机器人是单臂，可以改成：

```python
state_topic: str = "/my_robot/joint_states"
```

如果新机器人是底盘加机械臂，可以拆成：

```python
arm_state_topic: str = "/arm/joint_states"
base_state_topic: str = "/base/odom"
```

字段命名原则：

```text
字段名要说明它是哪个硬件模块的状态来源。
```

### 3.3 command topic

```python
command_left_topic: str = "/master/joint_left"
command_right_topic: str = "/master/joint_right"
```

这两个 topic 用来发布动作命令。

AgileX 的动作是双臂 7 轴关节位置，因此发布的是 ROS `sensor_msgs/JointState`。

新机器人如果是单臂，可以写：

```python
command_topic: str = "/my_robot/joint_command"
```

如果动作不是关节位置，而是底盘速度，可能是：

```python
cmd_vel_topic: str = "/cmd_vel"
```

重点是：`send_action()` 最终要把 LeRobot 的 action dict 转成这个 topic 需要的消息。

### 3.4 camera topic 和 camera key

AgileX 有三路相机：

```python
front_camera_topic: str = "/camera_f/color/image_raw"
left_camera_topic: str = "/camera_l/color/image_raw"
right_camera_topic: str = "/camera_r/color/image_raw"

front_camera_key: str = "camera_front"
left_camera_key: str = "camera_left"
right_camera_key: str = "camera_right"
```

每一路相机都由两部分组成：

```text
camera_topic
真实 ROS 图像 topic。

camera_key
LeRobot observation/dataset 里的图片字段名。
```

映射关系如下：

```text
/camera_f/color/image_raw -> camera_front
/camera_l/color/image_raw -> camera_left
/camera_r/color/image_raw -> camera_right
```

新机器人如果只有两路相机，可以写：

```python
front_camera_topic: str = "/my_robot/front_camera/image_raw"
wrist_camera_topic: str = "/my_robot/wrist_camera/image_raw"

front_camera_key: str = "camera_front"
wrist_camera_key: str = "camera_wrist"
```

推荐 camera key 使用稳定、语义清楚的名字：

```text
camera_front
camera_left
camera_right
camera_wrist
camera_top
```

不要把 ROS topic 原样作为 dataset key。topic 可能会变，dataset key 应该尽量稳定。

### 3.5 image_height 和 image_width

```python
image_height: int = 480
image_width: int = 640
```

这两个字段用于声明图片 feature 的 shape。

在 `AgileXRobot.observation_features` 里会用到：

```python
features[key] = (self.config.image_height, self.config.image_width, 3)
```

也就是：

```text
camera_front: (480, 640, 3)
camera_left: (480, 640, 3)
camera_right: (480, 640, 3)
```

顺序是：

```text
height, width, channel
```

如果实际相机是 1280x720，应配置成：

```python
image_height: int = 720
image_width: int = 1280
```

注意：这里声明的 shape 要和 bridge 实际解码出来的图片 shape 一致。

### 3.6 observation_timeout_s

```python
observation_timeout_s: float = 2.0
```

连接机器人时，AgileX 会等待左右臂状态和相机图像到齐。如果超时，就报错。

在 `AgileXRobot.connect()` 里：

```python
self._bridge.wait_for_ready(timeout_s=self.config.observation_timeout_s, require_images=True)
```

这个字段可以帮助快速发现：

```text
ROS topic 没启动。
topic 名配错。
相机没出图。
左右臂 JointState 没发布。
```

如果没有 timeout，程序可能一直卡住。

### 3.7 queue_size

```python
queue_size: int = 1
```

这个字段传给 ROS Subscriber 和 Publisher。

机器人控制通常希望用最新消息，不希望堆积旧消息，所以 AgileX 默认 `queue_size=1`。

如果你的场景允许缓存更多消息，可以改大，但采集和控制一般不建议。

### 3.8 joint_names

```python
joint_names: list[str] = field(default_factory=lambda: [f"joint{i}" for i in range(7)])
```

AgileX 每条臂 7 个关节，所以默认：

```python
["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
```

这个字段用于 ROS `JointState.name`。

在 `publish_action()` 里：

```python
left_message.name = list(self.joint_names)
right_message.name = list(self.joint_names)
```

也就是说：

```text
joint_names 是发给 ROS 下游控制器看的关节名。
```

注意：AgileX 里有两套关节名字：

```text
ROS JointState.name：
joint0, joint1, ..., joint6

LeRobot feature key：
left_joint0.pos, left_joint1.pos, ..., right_joint6.pos
```

这两套名字可以相同，也可以不同，但映射关系必须稳定。

如果真实机器人关节名是语义化的，建议配置成真实名字：

```python
joint_names: list[str] = field(
    default_factory=lambda: [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
        "gripper",
    ]
)
```

### 3.9 camera key alias

AgileX 配置里有：

```python
LEGACY_CAMERA_KEY_ALIASES = {
    "cam_high": "camera_front",
    "cam_left_wrist": "camera_left",
    "cam_right_wrist": "camera_right",
}
```

作用是兼容旧相机 key。

例如旧数据或旧配置里可能叫：

```text
cam_high
cam_left_wrist
cam_right_wrist
```

现在统一成：

```text
camera_front
camera_left
camera_right
```

归一化函数：

```python
def _normalize_camera_key(key: str) -> str:
    return LEGACY_CAMERA_KEY_ALIASES.get(key, key)
```

在 `__post_init__()` 里会执行：

```python
self.front_camera_key = _normalize_camera_key(self.front_camera_key)
self.left_camera_key = _normalize_camera_key(self.left_camera_key)
self.right_camera_key = _normalize_camera_key(self.right_camera_key)
```

新机器人如果没有历史兼容问题，可以不加 alias。

如果已经有旧数据集，建议加 alias，避免新老字段名冲突。

### 3.10 __post_init__

AgileX 的 `__post_init__()` 做三件事：

```python
def __post_init__(self):
    super().__post_init__()
    if self.control_mode not in {"passive_follow", "command_master"}:
        raise ValueError(f"Unsupported control_mode: {self.control_mode}")
    if len(self.joint_names) != 7:
        raise ValueError("AgileX expects exactly 7 joint names per arm")

    self.front_camera_key = _normalize_camera_key(self.front_camera_key)
    self.left_camera_key = _normalize_camera_key(self.left_camera_key)
    self.right_camera_key = _normalize_camera_key(self.right_camera_key)

    camera_keys = (
        self.front_camera_key,
        self.left_camera_key,
        self.right_camera_key,
    )
    if len(set(camera_keys)) != len(camera_keys):
        raise ValueError("AgileX camera keys must be distinct after alias normalization")
```

作用：

```text
校验 control_mode 是否合法。
校验每条臂是否正好 7 个关节。
兼容旧 camera key。
检查 camera key 不能重复。
```

新机器人也应该在 `__post_init__()` 里做类似校验。

## 4. agilex_ros_bridge.py 详解

文件路径：

```text
src/lerobot/robots/agilex/agilex_ros_bridge.py
```

这个文件是 AgileX 的 ROS 通讯层。

它负责：

```text
订阅左右臂 JointState。
订阅三路 ROS Image。
缓存最新关节状态。
缓存最新图片。
把 ROS JointState 转成 LeRobot feature dict。
把 LeRobot action dict 转成 ROS JointState 并发布。
```

### 4.1 feature name 常量

AgileX 定义了：

```python
ACTION_SUFFIX = "pos"
POSITION_SUFFIX = "pos"
VELOCITY_SUFFIX = "vel"
EFFORT_SUFFIX = "effort"
LEFT_PREFIX = "left"
RIGHT_PREFIX = "right"
ARM_PREFIXES = (LEFT_PREFIX, RIGHT_PREFIX)
SUPPORTED_COLOR_ENCODINGS = {"rgb8", "bgr8"}
```

含义：

```text
ACTION_SUFFIX = "pos"
动作是目标关节位置。

POSITION_SUFFIX = "pos"
观测里的关节位置后缀。

VELOCITY_SUFFIX = "vel"
速度后缀。

EFFORT_SUFFIX = "effort"
力或力矩后缀。

LEFT_PREFIX = "left"
左臂前缀。

RIGHT_PREFIX = "right"
右臂前缀。
```

生成 feature 名的函数：

```python
def make_joint_feature_names(suffix: str) -> list[str]:
    return [f"{arm}_joint{i}.{suffix}" for arm in ARM_PREFIXES for i in range(7)]
```

传入 `"pos"` 后生成：

```text
left_joint0.pos
left_joint1.pos
left_joint2.pos
left_joint3.pos
left_joint4.pos
left_joint5.pos
left_joint6.pos
right_joint0.pos
right_joint1.pos
right_joint2.pos
right_joint3.pos
right_joint4.pos
right_joint5.pos
right_joint6.pos
```

然后定义：

```python
ACTION_FEATURE_NAMES = make_joint_feature_names(ACTION_SUFFIX)
POSITION_FEATURE_NAMES = make_joint_feature_names(POSITION_SUFFIX)
VELOCITY_FEATURE_NAMES = make_joint_feature_names(VELOCITY_SUFFIX)
EFFORT_FEATURE_NAMES = make_joint_feature_names(EFFORT_SUFFIX)
```

注意 AgileX 的 `ACTION_SUFFIX` 和 `POSITION_SUFFIX` 都是 `"pos"`，所以动作和观测位置字段名字一样。

概念上它们仍然不同：

```text
POSITION_FEATURE_NAMES
用于 get_observation，表示当前状态。

ACTION_FEATURE_NAMES
用于 send_action 和 teleoperator，表示目标动作。
```

如果新机器人的动作是速度，可以写：

```python
ACTION_SUFFIX = "vel"
POSITION_SUFFIX = "pos"
```

这样 observation 是：

```text
joint0.pos
joint1.pos
```

action 是：

```text
joint0.vel
joint1.vel
```

### 4.2 JointSample

```python
@dataclass(frozen=True)
class JointSample:
    name: tuple[str, ...]
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    effort: tuple[float, ...]
```

这是内部关节状态缓存结构。

ROS `sensor_msgs/JointState` 本身有：

```text
name
position
velocity
effort
```

AgileX 收到 ROS 消息后，不直接到处传 ROS msg，而是转成 `JointSample`。

好处：

```text
内部数据类型稳定。
方便测试。
不让 Robot 层依赖 ROS 消息类型。
提前把所有数值转成 float。
```

`frozen=True` 表示创建后不能修改，避免缓存被意外改掉。

### 4.3 ImageTopicConfig

```python
@dataclass(frozen=True)
class ImageTopicConfig:
    topic: str
    feature_key: str
```

它表示一路相机的映射关系：

```text
ROS topic -> LeRobot feature key
```

例如：

```python
ImageTopicConfig("/camera_f/color/image_raw", "camera_front")
```

表示：

```text
从 /camera_f/color/image_raw 订阅图片。
把图片放到 observation["camera_front"]。
```

### 4.4 BridgeTopics

```python
@dataclass(frozen=True)
class BridgeTopics:
    state_left_topic: str
    state_right_topic: str
    command_left_topic: str | None
    command_right_topic: str | None
    image_topics: tuple[ImageTopicConfig, ...]
```

它把 bridge 需要的 topic 打包在一起。

`AgileXRobot.__init__()` 创建 bridge 时会传：

```python
BridgeTopics(
    state_left_topic=config.state_left_topic,
    state_right_topic=config.state_right_topic,
    command_left_topic=config.command_left_topic if config.control_mode == "command_master" else None,
    command_right_topic=config.command_right_topic if config.control_mode == "command_master" else None,
    image_topics=(
        ImageTopicConfig(config.front_camera_topic, config.front_camera_key),
        ImageTopicConfig(config.left_camera_topic, config.left_camera_key),
        ImageTopicConfig(config.right_camera_topic, config.right_camera_key),
    ),
)
```

这样做的好处是：

```text
RobotConfig 可以比较大。
BridgeTopics 只包含 bridge 真正需要的东西。
Bridge 不直接依赖整个 AgileXRobotConfig。
```

### 4.5 AgileXRosBridge.__init__

```python
class AgileXRosBridge:
    _node_initialized = False

    def __init__(
        self,
        *,
        topics: BridgeTopics,
        joint_names: list[str],
        queue_size: int = 1,
    ):
        ...
```

参数含义：

```text
topics
所有需要订阅和发布的 ROS topic。

joint_names
发布 JointState command 时使用的关节名。

queue_size
ROS subscriber/publisher 队列大小。
```

内部状态包括：

```python
self._latest_state = {
    LEFT_PREFIX: None,
    RIGHT_PREFIX: None,
}

self._latest_images = {
    image_config.feature_key: None for image_config in self.topics.image_topics
}
```

也就是先准备两个缓存：

```text
最新左右臂关节状态缓存。
最新相机图片缓存。
```

### 4.6 connect

```python
def connect(self, *, node_name: str, needs_publishers: bool) -> None:
    ...
```

作用：

```text
导入 rospy 和 ROS message。
初始化 ROS node。
订阅左右臂 JointState。
订阅所有 image topic。
如果需要控制，则创建左右臂 command publisher。
设置 _connected = True。
```

AgileX 把 ROS import 放在函数里面：

```python
import rospy
from sensor_msgs.msg import Image, JointState
```

这样做的好处是：

```text
没有 ROS 环境时，import lerobot.robots.agilex 不会立刻失败。
只有真正 connect 时才要求 rospy 存在。
```

订阅左臂状态：

```python
rospy.Subscriber(
    self.topics.state_left_topic,
    JointState,
    lambda msg: self._store_joint_sample(LEFT_PREFIX, msg),
    queue_size=self.queue_size,
    tcp_nodelay=True,
)
```

意思是：

```text
收到 left topic 的 JointState 后，调用 _store_joint_sample("left", msg)。
```

订阅图片：

```python
rospy.Subscriber(
    image_config.topic,
    Image,
    lambda msg, key=image_config.feature_key: self._store_image(key, msg),
    queue_size=self.queue_size,
    tcp_nodelay=True,
)
```

其中 `key=image_config.feature_key` 很重要，用来绑定当前相机的 feature key。

如果 `needs_publishers=True`，会创建 command publisher：

```python
self._left_command_publisher = rospy.Publisher(
    self.topics.command_left_topic,
    JointState,
    queue_size=self.queue_size,
)
```

### 4.7 _ensure_node_initialized

```python
@classmethod
def _ensure_node_initialized(cls, node_name: str) -> None:
    import rospy

    if cls._node_initialized:
        return
    if not rospy.core.is_initialized():
        rospy.init_node(node_name, anonymous=True, disable_signals=True)
    cls._node_initialized = True
```

作用：

```text
确保 rospy.init_node 只执行一次。
```

ROS 中不能随便重复初始化 node，所以 AgileX 用类变量：

```python
_node_initialized = False
```

第一次连接时初始化，后面就不再初始化。

### 4.8 wait_for_ready

```python
def wait_for_ready(self, *, timeout_s: float, require_images: bool) -> None:
    ...
```

作用：

```text
连接后等待必要数据到齐。
```

Robot 连接时：

```python
self._bridge.wait_for_ready(timeout_s=self.config.observation_timeout_s, require_images=True)
```

表示必须等到：

```text
左臂 JointState
右臂 JointState
三路相机图像
```

Teleoperator 连接时：

```python
self._bridge.wait_for_ready(timeout_s=self.config.observation_timeout_s, require_images=False)
```

表示只需要等到 JointState，不需要图像。

超时时会报缺失 topic：

```text
Timed out waiting for AgileX topics: [...]
```

这对排查 topic 没启动、topic 名错误、相机没出图非常重要。

### 4.9 _store_joint_sample

```python
def _store_joint_sample(self, arm: str, msg: Any) -> None:
    self._latest_state[arm] = JointSample(
        name=tuple(msg.name) if msg.name else self.joint_names,
        position=self._coerce_joint_vector(msg.position, "position"),
        velocity=self._coerce_joint_vector(msg.velocity, "velocity"),
        effort=self._coerce_joint_vector(msg.effort, "effort"),
    )
```

每次收到 ROS JointState，就转成 `JointSample`，然后保存到：

```python
self._latest_state["left"]
```

或者：

```python
self._latest_state["right"]
```

### 4.10 _coerce_joint_vector

```python
def _coerce_joint_vector(self, values: Any, field_name: str) -> tuple[float, ...]:
    if values:
        values_list = [float(value) for value in values]
    else:
        values_list = [0.0] * 7
    if len(values_list) != 7:
        raise ValueError(f"Expected 7 values for JointState.{field_name}, got {len(values_list)}")
    return tuple(values_list)
```

作用：

```text
把 JointState 里的 position/velocity/effort 转成 7 个 float。
如果字段为空，就补 7 个 0。
如果长度不是 7，直接报错。
```

AgileX 写死 7，是因为它每条臂就是 7 轴。

新机器人更通用的写法可以是：

```python
expected = len(self.joint_names)
if len(values_list) != expected:
    raise ValueError(...)
```

### 4.11 _decode_image

```python
SUPPORTED_COLOR_ENCODINGS = {"rgb8", "bgr8"}
```

AgileX 支持 `rgb8` 和 `bgr8` 两种 ROS Image 编码。

`_decode_image()` 做这些事：

```text
检查 encoding 是否支持。
把 msg.data 转成 numpy uint8。
根据 height、width、step reshape 成 H x W x 3。
如果是 bgr8，转成 RGB。
返回连续内存的 numpy array。
```

最后返回：

```python
np.ascontiguousarray(image)
```

LeRobot 数据集和训练流程通常按 RGB 图像处理，所以 BGR 转 RGB 很重要。

### 4.12 get_state_features

```python
def get_state_features(self) -> dict[str, float]:
    left_sample = self._require_joint_sample(LEFT_PREFIX)
    right_sample = self._require_joint_sample(RIGHT_PREFIX)
    state: dict[str, float] = {}
    for arm, sample in ((LEFT_PREFIX, left_sample), (RIGHT_PREFIX, right_sample)):
        for idx in range(7):
            state[f"{arm}_joint{idx}.{POSITION_SUFFIX}"] = sample.position[idx]
    return state
```

作用：

```text
把最新 JointSample 转成 LeRobot observation 里的关节状态 dict。
```

返回结果类似：

```python
{
    "left_joint0.pos": 0.12,
    "left_joint1.pos": -0.44,
    ...
    "right_joint6.pos": 0.55,
}
```

当前 AgileX 只记录关节位置，不记录 velocity 和 effort。

如果以后要记录速度和力矩，需要同时改：

```text
get_state_features()
observation_features
数据集兼容逻辑
训练配置
```

### 4.13 get_images

```python
def get_images(self) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for image_config in self.topics.image_topics:
        image = self._latest_images[image_config.feature_key]
        if image is None:
            raise RuntimeError(f"No image received on topic {image_config.topic}")
        images[image_config.feature_key] = image.copy()
    return images
```

作用：

```text
把最新图片缓存取出来，返回给 Robot.get_observation。
```

返回结果类似：

```python
{
    "camera_front": np.ndarray,
    "camera_left": np.ndarray,
    "camera_right": np.ndarray,
}
```

注意这里返回的是 `image.copy()`，避免外部修改内部缓存。

### 4.14 get_action_features

```python
def get_action_features(self) -> dict[str, float]:
    left_sample = self._require_joint_sample(LEFT_PREFIX)
    right_sample = self._require_joint_sample(RIGHT_PREFIX)
    action: dict[str, float] = {}
    for arm, sample in ((LEFT_PREFIX, left_sample), (RIGHT_PREFIX, right_sample)):
        for idx in range(7):
            action[f"{arm}_joint{idx}.{ACTION_SUFFIX}"] = sample.position[idx]
    return action
```

这个函数主要给 AgileX teleoperator 使用。

它把 master arm 当前姿态转成 LeRobot action dict。

例如：

```python
{
    "left_joint0.pos": 0.1,
    "left_joint1.pos": 0.2,
    ...
    "right_joint6.pos": -0.3,
}
```

### 4.15 publish_action

```python
def publish_action(self, action: dict[str, float]) -> None:
    ...
```

作用：

```text
把 LeRobot action dict 转成 ROS JointState，并发布到左右臂 command topic。
```

输入 action 是：

```python
{
    "left_joint0.pos": 0.1,
    "left_joint1.pos": 0.2,
    ...
    "right_joint6.pos": -0.3,
}
```

发布左臂时：

```python
left_message.name = list(self.joint_names)
left_message.position = [float(action[f"{LEFT_PREFIX}_joint{i}.{ACTION_SUFFIX}"]) for i in range(7)]
```

也就是：

```text
JointState.name = ["joint0", "joint1", ..., "joint6"]
JointState.position = [
    action["left_joint0.pos"],
    action["left_joint1.pos"],
    ...
    action["left_joint6.pos"],
]
```

右臂同理。

如果新机器人不是 ROS，而是 SDK 或串口，那么 `publish_action()` 可以改成：

```python
self.sdk.set_joint_positions([...])
```

或者：

```python
self.serial.write(...)
```

## 5. agilex.py 详解

文件路径：

```text
src/lerobot/robots/agilex/agilex.py
```

这个文件是 LeRobot 正式认识 AgileX 的地方。

### 5.1 类声明

```python
class AgileXRobot(Robot):
    config_class = AgileXRobotConfig
    name = "agilex"
```

含义：

```text
AgileXRobot
LeRobot 里的 AgileX 机器人类。

config_class = AgileXRobotConfig
这个 Robot 类对应的配置类。

name = "agilex"
机器人实例名字。
```

新机器人建议：

```python
class MyRobot(Robot):
    config_class = MyRobotConfig
    name = "my_robot"
```

### 5.2 __init__

```python
def __init__(self, config: AgileXRobotConfig):
    super().__init__(config)
    self.config = config
    self.cameras = {
        config.front_camera_key: None,
        config.left_camera_key: None,
        config.right_camera_key: None,
    }
    self._bridge = AgileXRosBridge(...)
```

作用：

```text
调用 Robot 基类初始化。
保存 config。
声明有哪些相机 key。
创建 AgileXRosBridge。
```

AgileX 的 `self.cameras` 里 value 是 `None`：

```python
self.cameras = {
    "camera_front": None,
    "camera_left": None,
    "camera_right": None,
}
```

原因是 AgileX 不使用 LeRobot 原生 OpenCV/RealSense camera 对象，而是通过 ROS Image topic 读取图片。

这里保留 `self.cameras` 的作用是：

```text
让 observation_features 知道有哪些相机字段。
```

### 5.3 observation_features

```python
@cached_property
def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
    features: dict[str, type | tuple[int, int, int]] = {}
    for key in POSITION_FEATURE_NAMES:
        features[key] = float
    for key in self.cameras:
        features[key] = (self.config.image_height, self.config.image_width, 3)
    return features
```

作用：

```text
声明这个机器人每一帧 observation 有哪些字段，以及字段类型或 shape。
```

AgileX 返回类似：

```python
{
    "left_joint0.pos": float,
    "left_joint1.pos": float,
    ...
    "right_joint6.pos": float,
    "camera_front": (480, 640, 3),
    "camera_left": (480, 640, 3),
    "camera_right": (480, 640, 3),
}
```

这个声明必须和 `get_observation()` 实际返回的字段对齐。

如果这里声明了 `camera_front`，`get_observation()` 就必须返回：

```python
observation["camera_front"] = image
```

### 5.4 action_features

```python
@cached_property
def action_features(self) -> dict[str, type]:
    return {key: float for key in ACTION_FEATURE_NAMES}
```

作用：

```text
声明这个机器人能接收哪些 action 字段。
```

AgileX 返回类似：

```python
{
    "left_joint0.pos": float,
    ...
    "right_joint6.pos": float,
}
```

这个声明必须和 `send_action()` 里实际读取的 action key 对齐。

### 5.5 is_connected

```python
@property
def is_connected(self) -> bool:
    return self._bridge.is_connected
```

作用：

```text
告诉 LeRobot 当前机器人是否已经连接。
```

这个属性会配合装饰器使用：

```python
@check_if_already_connected
@check_if_not_connected
```

### 5.6 connect

```python
@check_if_already_connected
def connect(self, calibrate: bool = True) -> None:
    self._bridge.connect(
        node_name="lerobot_agilex_robot",
        needs_publishers=self.config.control_mode == "command_master",
    )
    self._bridge.wait_for_ready(timeout_s=self.config.observation_timeout_s, require_images=True)
    if calibrate:
        self.calibrate()
    self.configure()
```

作用：

```text
连接 ROS bridge。
根据 control_mode 决定是否创建 publisher。
等待左右臂状态和相机图像到齐。
执行 calibrate。
执行 configure。
```

`@check_if_already_connected` 的作用是防止重复连接。

新机器人可以仿写：

```python
@check_if_already_connected
def connect(self, calibrate: bool = True) -> None:
    self._bridge.connect(
        node_name="lerobot_my_robot",
        needs_publisher=self.config.control_mode == "position",
    )
    self._bridge.wait_for_ready(
        timeout_s=self.config.observation_timeout_s,
        require_images=True,
    )
    if calibrate:
        self.calibrate()
    self.configure()
```

### 5.7 is_calibrated、calibrate、configure

AgileX 写法：

```python
@property
def is_calibrated(self) -> bool:
    return True

def calibrate(self) -> None:
    return None

def configure(self) -> None:
    return None
```

原因：

```text
AgileX 这里不在 LeRobot 内部做电机标定。
底层 ROS 或厂商系统已经处理了硬件标定和使能。
```

如果新机器人也是 ROS bridge，且底层系统已经完成标定，可以先这样写。

如果新机器人直接控制电机，则这里通常要做：

```text
is_calibrated
检查是否有标定文件或标定状态。

calibrate
做零点、方向、关节范围标定。

configure
设置控制模式、速度限制、扭矩、PID 等。
```

### 5.8 get_observation

```python
@check_if_not_connected
def get_observation(self) -> RobotObservation:
    observation = self._bridge.get_state_features()
    observation.update(self._bridge.get_images())
    return observation
```

作用：

```text
每一帧采集 observation。
```

流程：

```text
从 bridge 获取关节状态 dict。
从 bridge 获取图片 dict。
合并后返回。
```

返回类似：

```python
{
    "left_joint0.pos": 0.1,
    ...
    "right_joint6.pos": -0.2,
    "camera_front": np.ndarray,
    "camera_left": np.ndarray,
    "camera_right": np.ndarray,
}
```

`@check_if_not_connected` 的作用是：如果还没连接就调用，会报错。

### 5.9 send_action

```python
@check_if_not_connected
def send_action(self, action: RobotAction) -> RobotAction:
    sent_action = {key: float(action[key]) for key in ACTION_FEATURE_NAMES}
    if self.config.control_mode == "command_master":
        self._bridge.publish_action(sent_action)
    return sent_action
```

作用：

```text
接收 LeRobot action。
取出 AgileX 需要的 action key。
转成 float。
在 command_master 模式下发布给 ROS。
返回实际发送或准备发送的 action。
```

如果新机器人需要安全限幅，通常就在这里加：

```text
检查动作范围。
限制单步最大移动。
检查急停状态。
再发送到底层 bridge。
```

### 5.10 disconnect

```python
@check_if_not_connected
def disconnect(self) -> None:
    self._bridge.disconnect()
```

作用：

```text
释放底层资源。
```

AgileX bridge 会注销 ROS subscriber、publisher，并清空缓存。

如果新机器人是串口、CAN、SDK，要在底层 bridge 的 `disconnect()` 里关闭对应资源。

## 6. __init__.py 详解

AgileX 的 `__init__.py`：

```python
from .agilex import AgileXRobot
from .config_agilex import AgileXRobotConfig

__all__ = ["AgileXRobot", "AgileXRobotConfig"]
```

作用：

```text
让外部可以 from lerobot.robots.agilex import AgileXRobot, AgileXRobotConfig。
触发 config_agilex.py 被导入，从而执行 @RobotConfig.register_subclass("agilex")。
```

新机器人建议：

```python
from .my_robot import MyRobot
from .config_my_robot import MyRobotConfig

__all__ = ["MyRobot", "MyRobotConfig"]
```

## 7. 为什么脚本里要 import agilex

在 `src/lerobot/scripts/lerobot_record.py` 里有：

```python
from lerobot.robots import agilex  # noqa: F401
from lerobot.teleoperators import agilex_teleoperator  # noqa: F401
```

这两行看起来没有直接使用，但作用很重要：

```text
强制导入 agilex 包。
让 AgileXRobotConfig 上的 @RobotConfig.register_subclass("agilex") 执行。
让命令行 --robot.type=agilex 可以被解析。
```

`# noqa: F401` 的意思是告诉 Ruff：

```text
这个 import 是故意的，虽然看起来没有使用，不要报 unused import。
```

如果新机器人写好了，但命令行报找不到 type，首先检查：

```text
对应包有没有在入口脚本里被 import。
```

## 8. 从 0 仿写一个新机器人

假设新机器人叫 `my_robot`，它是单臂 6 轴 ROS 机器人，有两路相机：

```text
关节状态 topic：/my_robot/joint_states
动作命令 topic：/my_robot/joint_command
前方相机 topic：/my_robot/front_camera/image_raw
腕部相机 topic：/my_robot/wrist_camera/image_raw
```

建议目录：

```text
src/lerobot/robots/my_robot/
├── __init__.py
├── config_my_robot.py
├── my_robot.py
└── my_robot_ros_bridge.py
```

### 8.1 config_my_robot.py 模板

```python
from dataclasses import dataclass, field

from ..config import RobotConfig


@RobotConfig.register_subclass("my_robot")
@dataclass(kw_only=True)
class MyRobotConfig(RobotConfig):
    control_mode: str = "position"

    state_topic: str = "/my_robot/joint_states"
    command_topic: str = "/my_robot/joint_command"

    front_camera_topic: str = "/my_robot/front_camera/image_raw"
    wrist_camera_topic: str = "/my_robot/wrist_camera/image_raw"

    front_camera_key: str = "camera_front"
    wrist_camera_key: str = "camera_wrist"

    image_height: int = 480
    image_width: int = 640

    observation_timeout_s: float = 2.0
    queue_size: int = 1

    joint_names: list[str] = field(
        default_factory=lambda: [
            "joint0",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
        ]
    )

    def __post_init__(self):
        super().__post_init__()

        if self.control_mode not in {"position", "passive"}:
            raise ValueError(f"Unsupported control_mode: {self.control_mode}")

        if len(self.joint_names) != 6:
            raise ValueError("MyRobot expects exactly 6 joint names")

        camera_keys = (self.front_camera_key, self.wrist_camera_key)
        if len(set(camera_keys)) != len(camera_keys):
            raise ValueError("MyRobot camera keys must be distinct")
```

### 8.2 my_robot_ros_bridge.py 的 feature name 模板

```python
ACTION_SUFFIX = "pos"
POSITION_SUFFIX = "pos"
JOINT_COUNT = 6
SUPPORTED_COLOR_ENCODINGS = {"rgb8", "bgr8"}


def make_joint_feature_names(suffix: str) -> list[str]:
    return [f"joint{i}.{suffix}" for i in range(JOINT_COUNT)]


ACTION_FEATURE_NAMES = make_joint_feature_names(ACTION_SUFFIX)
POSITION_FEATURE_NAMES = make_joint_feature_names(POSITION_SUFFIX)
```

生成的 action 和 observation 关节字段是：

```text
joint0.pos
joint1.pos
joint2.pos
joint3.pos
joint4.pos
joint5.pos
```

如果是双臂，可以照 AgileX：

```python
ARM_PREFIXES = ("left", "right")


def make_joint_feature_names(suffix: str) -> list[str]:
    return [f"{arm}_joint{i}.{suffix}" for arm in ARM_PREFIXES for i in range(7)]
```

生成：

```text
left_joint0.pos
...
right_joint6.pos
```

### 8.3 my_robot.py 模板

```python
from functools import cached_property

from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_my_robot import MyRobotConfig
from .my_robot_ros_bridge import (
    ACTION_FEATURE_NAMES,
    POSITION_FEATURE_NAMES,
    ImageTopicConfig,
    MyRobotRosBridge,
    MyRobotTopics,
)


class MyRobot(Robot):
    config_class = MyRobotConfig
    name = "my_robot"

    def __init__(self, config: MyRobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = {
            config.front_camera_key: None,
            config.wrist_camera_key: None,
        }
        self._bridge = MyRobotRosBridge(
            topics=MyRobotTopics(
                state_topic=config.state_topic,
                command_topic=config.command_topic if config.control_mode == "position" else None,
                image_topics=(
                    ImageTopicConfig(config.front_camera_topic, config.front_camera_key),
                    ImageTopicConfig(config.wrist_camera_topic, config.wrist_camera_key),
                ),
            ),
            joint_names=config.joint_names,
            queue_size=config.queue_size,
        )

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = {}
        for key in POSITION_FEATURE_NAMES:
            features[key] = float
        for key in self.cameras:
            features[key] = (self.config.image_height, self.config.image_width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in ACTION_FEATURE_NAMES}

    @property
    def is_connected(self) -> bool:
        return self._bridge.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._bridge.connect(
            node_name="lerobot_my_robot",
            needs_publisher=self.config.control_mode == "position",
        )
        self._bridge.wait_for_ready(
            timeout_s=self.config.observation_timeout_s,
            require_images=True,
        )
        if calibrate:
            self.calibrate()
        self.configure()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        observation = self._bridge.get_state_features()
        observation.update(self._bridge.get_images())
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        sent_action = {key: float(action[key]) for key in ACTION_FEATURE_NAMES}
        if self.config.control_mode == "position":
            self._bridge.publish_action(sent_action)
        return sent_action

    @check_if_not_connected
    def disconnect(self) -> None:
        self._bridge.disconnect()
```

### 8.4 __init__.py 模板

```python
from .config_my_robot import MyRobotConfig
from .my_robot import MyRobot

__all__ = ["MyRobot", "MyRobotConfig"]
```

### 8.5 入口脚本 import

如果要让 `lerobot_record.py` 识别新机器人，需要在合适位置 import：

```python
from lerobot.robots import my_robot  # noqa: F401
```

如果还有 teleoperator：

```python
from lerobot.teleoperators import my_robot_teleoperator  # noqa: F401
```

## 9. 新机器人字段命名建议

### 9.1 目录名和注册名

推荐全部小写加下划线：

```text
my_robot
cobot_magic
agilex
unitree_g1
```

统一使用：

```text
目录名：src/lerobot/robots/my_robot/
注册名：@RobotConfig.register_subclass("my_robot")
Robot.name："my_robot"
CLI：--robot.type=my_robot
```

### 9.2 配置类和机器人类

推荐：

```python
class MyRobotConfig(RobotConfig):
    ...

class MyRobot(Robot):
    ...
```

如果名字比较具体：

```python
class CobotMagicRobotConfig(RobotConfig):
    ...

class CobotMagicRobot(Robot):
    ...
```

### 9.3 文件名

推荐：

```text
config_my_robot.py
my_robot.py
my_robot_ros_bridge.py
```

如果不是 ROS，而是 SDK：

```text
my_robot_sdk_bridge.py
```

如果是串口：

```text
my_robot_serial_bridge.py
```

### 9.4 camera key

推荐：

```text
camera_front
camera_left
camera_right
camera_wrist
camera_top
```

不要推荐：

```text
/camera_f/color/image_raw
```

因为这是 ROS topic，不适合作为 dataset key。

### 9.5 joint feature key

单臂可以用：

```text
joint0.pos
joint1.pos
joint2.pos
```

双臂可以用：

```text
left_joint0.pos
left_joint1.pos
right_joint0.pos
right_joint1.pos
```

如果想更语义化，可以用：

```text
shoulder_pan.pos
shoulder_lift.pos
elbow.pos
wrist_1.pos
```

但是 feature key 一旦进入 dataset，后续训练、推理、回放都要对齐，所以不要随便改。

## 10. 新机器人接入检查清单

添加新机器人时，建议按这个顺序检查：

```text
1. 机器人 type 名是否确定，比如 my_robot。
2. 目录是否为 src/lerobot/robots/my_robot/。
3. config_my_robot.py 是否有 @RobotConfig.register_subclass("my_robot")。
4. Robot 类是否设置 config_class = MyRobotConfig。
5. Robot 类是否设置 name = "my_robot"。
6. __init__.py 是否导出 MyRobot 和 MyRobotConfig。
7. observation_features 声明的 key 是否和 get_observation 返回的 key 完全一致。
8. action_features 声明的 key 是否和 send_action 使用的 key 完全一致。
9. camera topic 和 camera key 是否分清。
10. joint_names 和 LeRobot feature key 是否分清。
11. connect 是否会等待必要状态和图像。
12. disconnect 是否释放所有 subscriber、publisher、串口、SDK 或相机资源。
13. record/replay 入口脚本是否 import 了新机器人包。
14. 是否使用 lerobot_flex conda 环境运行测试和脚本。
```

测试导入：

```bash
conda run -n lerobot_flex python -c "from lerobot.robots.my_robot import MyRobot, MyRobotConfig; print(MyRobot.name, MyRobotConfig().type)"
```

如果输出：

```text
my_robot my_robot
```

说明基本注册和导入是通的。

## 11. 一句话总结

AgileX 的接入方式，本质上是在做一张映射表：

```text
ROS 状态 topic -> LeRobot observation key
ROS 图片 topic -> LeRobot camera key
LeRobot action key -> ROS command topic
```

对应代码分工是：

```text
config_agilex.py
定义这张映射表需要哪些配置。

agilex_ros_bridge.py
执行 ROS 数据和 LeRobot dict 之间的转换。

agilex.py
把 bridge 包装成 LeRobot 标准 Robot 接口。

__init__.py
导出类并触发注册。
```

添加新机器人时，先把这张映射表设计清楚，再仿照 AgileX 写 config、bridge、Robot 类，后面的 record、replay、policy 才能稳定对齐。
