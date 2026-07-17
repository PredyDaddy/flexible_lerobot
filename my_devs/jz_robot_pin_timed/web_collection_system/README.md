# JZ Robot Pin Timed 数采系统

本目录提供 timed Robot 的本地 Web 操作界面：

- 启动、停止和打开 VR / IK / Meshcat 可视化；
- 自由指定正整数 episode 数量并录制新数据集；
- 自动发现本机 timed 数据集；
- 选择数据集路径和 episode 进行 armed 回放；
- 查看后台进程状态、日志和最终生成的命令。
- 使用页头的 `English / 中文` 按钮即时切换中英文界面。

语言切换只更新前端显示，不会刷新页面、启动任务或修改录制/回放参数。当前语言保存在浏览器
`localStorage` 的 `jz_web_language` 中；重新打开页面时会沿用上次选择。运行日志和生成的 shell
命令保持原始内容，不做翻译，便于现场排障和复制执行。

Web 后端使用 Python 标准库，不需要安装 FastAPI、Flask 或 Node.js。所有 Python 启动和测试必须使用
lerobot_flex conda 环境。

## 启动

只查看界面、不能执行 armed 录制或回放：

~~~bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

bash my_devs/jz_robot_pin_timed/web_collection_system/start_web.sh
~~~

获得现场授权、确认 Orin executor、机器人工作区和急停后，解锁录制与回放按钮：

~~~bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

JZ_WEB_ARMED_ACTIONS=1 \
HOST=0.0.0.0 \
PORT=8010 \
bash my_devs/jz_robot_pin_timed/web_collection_system/start_web.sh
~~~

浏览器打开：

~~~text
http://10.1.42.3:8010/
~~~

Meshcat 可视化地址：

~~~text
http://10.1.42.3:7000/static/
~~~

## 模拟模式

开发或界面验证必须使用模拟模式。模拟模式不会运行 joystick、record 或 replay 脚本：

~~~bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

JZ_WEB_MOCK_COMMANDS=1 \
HOST=127.0.0.1 \
PORT=8010 \
bash my_devs/jz_robot_pin_timed/web_collection_system/start_web.sh
~~~

## 录制

页面要求填写完整数据集保存路径和本次采集条数。目标目录不能已存在，父目录必须存在；采集条数
可以是 1、5、6、10 等任意正整数，默认值为 10。

默认参数：

~~~text
NUM_EPISODES=10
EPISODE_TIME_S=10
RESET_TIME_S=5
RECORD_FPS=20
VIDEO=true
VIDEO_CRF=18
VIDEO_ENCODING_BATCH_SIZE=<NUM_EPISODES>
RESUME=false
EXECUTION=armed
SEND_ACTION_TRANSPORT=udp
ZMQ_PRESET=jz_three_zmq
RTSP_PRESET=none
TIMING_SIDECAR=true
REQUIRE_STATE_SOURCE_TIMING=true
REQUIRE_STATE_ADVANCE_PER_OBSERVATION=true
STATE_ADVANCE_TIMEOUT_S=0.1
MAX_CAMERA_STATE_RECEIVE_SKEW_MS=200.0
LEFT_GRIPPER_OBSERVATION_SOURCE=measured_opening
RIGHT_GRIPPER_OBSERVATION_SOURCE=commanded_opening
MAX_INITIAL_JOINT_DELTA_RAD=10.0
MAX_JOINT_STEP_RAD=10.0
~~~

`VIDEO_ENCODING_BATCH_SIZE` 由后端自动设置为本次 `NUM_EPISODES`，不允许页面单独填写。例如选择
6 条时，实际命令使用 `NUM_EPISODES=6` 和 `VIDEO_ENCODING_BATCH_SIZE=6`，第 6 条保存后统一编码。
LeRobot 本身不要求必须录满 10 条；如果通过页面正常停止，`VideoEncodingManager` 也会编码尚未达到
完整批量的已保存 episode。录制前页面仍要求可视化进程已经运行，高级参数默认折叠。

`MAX_CAMERA_STATE_RECEIVE_SKEW_MS` 是录制阶段相机帧与机器人 state 本机接收时间的最大允许偏差，
Web 和 `record.sh` 的录制默认值为 `200 ms`，可在页面高级参数中逐次覆盖。它不等于
`STATE_ADVANCE_TIMEOUT_S`；后者只控制等待新 state revision 的时间。机器人类的全局默认仍为
`100 ms`，因此回放和推理不会被录制页面的默认值一起放宽。离线训练质量检查仍可使用更严格的
`100 ms` 门槛，把采集容错与训练准入分开。

录制中断后的 episode 保存边界、数据检查和合并流程见
[`../data_check/DATASET_INTEGRITY_AND_CURATION.md`](../data_check/DATASET_INTEGRITY_AND_CURATION.md)。

## 回放

后端默认扫描：

~~~text
/home/luzhuang/cqy/aaa/flexible_lerobot/tests/outputs
~~~

这个扫描目录只用于生成“已发现的数据集”下拉列表，不是回放路径边界。页面同时提供“自定义数据集
绝对路径”，可以加载 `tests/outputs` 之外的任意目录，例如：

~~~text
/home/luzhuang/cqy/aaa/flexible_lerobot/data/collection_system_vaertify/test1
~~~

点击“加载并校验”后，后端直接读取该绝对路径。无论使用自动发现还是自定义路径，目录都必须包含
`meta/info.json`，其中 `robot_type` 必须是 `jz_robot_pin_timed`，并且至少有一个已保存 episode。
校验通过后，页面根据 `total_episodes` 生成 episode 选项。相对路径会被拒绝。

默认参数：

~~~text
EXECUTION=armed
SEND_ACTION_TRANSPORT=udp
REPLAY_FPS=20
MAX_INITIAL_JOINT_DELTA_RAD=0.5
MAX_JOINT_STEP_RAD=0.05
PLAY_SOUNDS=true
~~~

## 可选配置

修改默认数据扫描目录：

~~~bash
JZ_WEB_DATASET_ROOT=/absolute/dataset/parent
~~~

修改 Meshcat 地址：

~~~bash
JZ_WEB_VISUALIZATION_URL=http://10.1.42.3:7000/static/
~~~

为 POST 控制请求启用可选令牌：

~~~bash
JZ_WEB_CONTROL_TOKEN=replace-with-local-secret
~~~

设置令牌后，浏览器请求必须携带 X-JZ-Control-Token。默认不启用令牌。

## 停止

录制和回放通过页面停止按钮发送 SIGINT，使 LeRobot 有机会保存和关闭当前数据；超时后才升级为
SIGTERM/SIGKILL。

停止 Web 服务使用 Ctrl+C。Web 服务退出时会停止由它启动的录制或回放任务，但不会自动停止 Orin
服务。
