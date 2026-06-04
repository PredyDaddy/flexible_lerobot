 cd /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering

 conda run --no-capture-output -n lerobot_flex python -u -m vlash_iner.server.run_pi05_async_erver \
    --host 127.0.0.1 \
    --port 8008 \
    --endpoint /infer \
    --policy-path /data/cqy_workspace/flexible_lerobot/outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --task "Put the eraser into the small box" \
    --robot-type so101_follower \
    --compile-model true \
    --compile-mode reduce-overhead \
    --warmup-steps 3 \
    --img-width 640 \
    --img-height 480 \
    --state-dim 6



 cd /data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering

conda run --no-capture-output -n lerobot_flex python -u -m vlash_iner.server.run_pi05_async_client \
    --server-url http://127.0.0.1:8008 \
    --endpoint /infer \
    --task "Put the eraser into the small box" \
    --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
    --top-cam /dev/video4 \
    --wrist-cam /dev/video6 \
    --run-time-s 120 \
    --fps 30 \
    --control-fps 45 \
    --reuse-observation-within-chunk true \
    --inference-overlap-steps 8 \
    --background-inference true \
    --future-state-aware false \
    --chunk-blend-steps 2 \
    --log-interval 10 \
    --action-quant-ratio 1 \
    --connect-retries 3 \
    --connect-retry-s 1.0