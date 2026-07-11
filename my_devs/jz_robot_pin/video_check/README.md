# JZ Robot Pin 视频质量检查

这个目录用于离线比较 H.264 的不同 CRF，不会连接或控制机器人。

## 生成 CRF 对照视频

默认使用现有 `180651` 数据集的右腕相机，从第 5 秒开始截取 5 秒：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot
bash my_devs/jz_robot_pin/video_check/make_crf_comparison.sh
```

也可以指定其他视频：

```bash
bash my_devs/jz_robot_pin/video_check/make_crf_comparison.sh /absolute/path/to/input.mp4
```

可调整截取范围和输出位置：

```bash
START_S=0 DURATION_S=10 \
OUTPUT_DIR=/tmp/jz_crf_check \
bash my_devs/jz_robot_pin/video_check/make_crf_comparison.sh /absolute/path/to/input.mp4
```

输出包括：

- `reference_ffv1.mkv`：选定片段解码后的无损公共输入。
- `h264_crf*_g2_yuv420p.mp4`：CRF 18、20、22、30 的单独视频。
- `compare_crf_18_20_22_30.mp4`：四档 CRF 的 2x2 并排预览。
- `compare_crf_18_20_22_30.jpg`：并排静态图。
- `report.tsv`：文件大小、实际码率、SSIM 和 PSNR。

判断画质时以四个单独 MP4 为准。并排视频为了方便播放又编码了一次，只适合快速观察。

## 当前对照的限制

LeRobot 在生成现有数据集 MP4 后已经删除了临时 PNG。现有 MP4 本身已经经过
`CRF=30 + yuv420p` 编码，因此这套对照只能显示不同 CRF 带来的额外压缩损失，不能恢复
CRF 30 已经丢失的细节，也不能证明亮区偏青来自 X86 还是 Orin。

正式确认时，应在下一次录制期间同时保存 RTSP 原始码流，或者保留少量编码前 PNG，
然后用同一脚本进行 CRF 对照。

## 从 X86 采集 RTSP 原流后比较

下面的脚本只读取相机 RTSP，不发送机器人状态或控制命令。默认采集右腕相机 10 秒：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot
bash my_devs/jz_robot_pin/video_check/capture_rtsp_crf_comparison.sh right
```

其他相机和时长：

```bash
DURATION_S=10 \
bash my_devs/jz_robot_pin/video_check/capture_rtsp_crf_comparison.sh head
```

脚本先用 `-c copy` 保存 RTSP 原始 H.264，再从完全相同的源片段生成 CRF
18、20、22、30 视频。这个结果比对现有数据集 MP4 再编码更适合决定正式录制参数。
