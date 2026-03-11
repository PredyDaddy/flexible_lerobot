## 数据采集
```bash
# 终端1
roscore

# 终端2（若重新插拔操作臂usb，需要重新运行此部分）
conda activate act
cd /home/agilex/cobot_magic//Piper_ros_private-ros-noetic
./can_config.sh
source devel/setup.bash
roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=false

# 终端3
roslaunch astra_camera multi_camera.launch

# 终端4(optiional)
rostopic list
```

打开了上面的终端，我才可以使用my_devs/docs/cobot_magic/collect_data/collect_data.py 这个脚本去录制对应的数据集   


## 模型推理跟回放
```bash
# 终端1
roscore

# 终端2
conda activate act
cd /home/agilex/cobot_magic//Piper_ros_private-ros-noetic
./can_config.sh
source devel/setup.bash
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true

# 终端3
roslaunch astra_camera multi_camera.launch
```

打开了这些脚本，我才可以使用my_devs/docs/cobot_magic/collect_data/replay_data.py 这个脚本去回放数据