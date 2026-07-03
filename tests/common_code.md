 
sudo systemctl stop robot_bringup.service 
sudo systemctl restart robot_bringup.service 



  看头部：

  ros2 topic echo --once /robot1/body/head/joint_states

  看腰部：

  ros2 topic echo --once /robot1/body/waist/joint_states

  看左臂：

  ros2 topic echo --once /robot1/arm_left/joint_states

  看右臂：

  ros2 topic echo --once /robot1/arm_right/joint_states

(base) test@TER30JB3-ubuntu:~/workspace/flexible_lerobot$ ros2 topic echo --once /robot1/body/head/joint_states
header:
  stamp:
    sec: 1783048092
    nanosec: 122483228
  frame_id: head_base_link
name:
- head_joint_1
- head_joint_2
position:
- -0.5378232183231165
- 0.060562925543545934
velocity: []
effort: []
---


(base) test@TER30JB3-ubuntu:~/workspace/flexible_lerobot$ ros2 topic echo --once /robot1/body/waist/joint_states
header:
  stamp:
    sec: 1783045678
    nanosec: 309556254
  frame_id: waist_base_link
name:
- waist_joint_3
- waist_joint_4
- waist_joint_5
position:
- -0.00425860342271984
- -1.102803755546207
- -1.107900113766237
velocity: []
effort: []
---


  ros2 service call /robot1/choreographer/execute \
    multi_robot_choreographer_interfaces/srv/ExecuteChoreography \
    "{choreography_name: 'VR_inital_with_head_body'}"


conda run --no-capture-output -n lerobot python udp_test/local_test/get_pic.py