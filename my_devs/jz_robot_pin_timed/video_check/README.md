# 视频质量检查

本目录不复制编码实现，只包装 `my_devs/jz_robot_pin/video_check` 的只读工具，并把输出隔离到
`my_devs/jz_robot_pin_timed/video_check/outputs/`。

从右腕 RTSP 原流采 10 秒，再生成 CRF 18/20/22/30 对照：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

DURATION_S=10 \
bash my_devs/jz_robot_pin_timed/video_check/capture_rtsp_crf_comparison.sh right
```

可选相机为 `head`、`left`、`right`。该命令只读取 RTSP，不订阅控制输入，也不发送机器人命令。

对已有视频生成对照：

```bash
INPUT_VIDEO=/path/to/file-000.mp4 \
START_S=0 \
DURATION_S=5 \
bash my_devs/jz_robot_pin_timed/video_check/make_crf_comparison.sh
```

输出中的单独 CRF 视频是画质判断依据；2x2 网格为了观看方便还会再编码一次。

