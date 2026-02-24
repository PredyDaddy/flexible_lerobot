#!/usr/bin/env python3
"""
Position alignment tool for LeRobot datasets (Web version).

This script overlays the first frame from a dataset onto the live camera feed,
helping users align object positions for consistent data collection.
Access via web browser for real-time visualization.

Usage:
    python docs/move_position/align_position.py \
      --data-path /home/agilex/.cache/huggingface/lerobot/cqy/agilex_left_box \
      --alpha 0.4 \
      --port 5001

    Then visit: http://10.1.26.29:5001
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify

# Add src to path for local development
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from lerobot.cameras.ros_camera import RosCamera
from lerobot.cameras.ros_camera.configuration_ros_camera import RosCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

# Default camera topic
DEFAULT_CAMERA_TOPIC = "/camera_f/color/image_raw"
DEFAULT_CAMERA_KEY = "observation.images.camera_front"

# HTML template for web page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Position Alignment Tool</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            color: #00d4ff;
            margin-bottom: 20px;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }
        .video-container {
            border: 3px solid #00d4ff;
            border-radius: 8px;
            overflow: hidden;
        }
        .video-container img {
            display: block;
            width: 1280px;
            height: auto;
        }
        .controls {
            background-color: #16213e;
            padding: 20px 40px;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
        }
        .slider-container {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .slider-container label {
            font-size: 16px;
            min-width: 120px;
        }
        .slider-container input[type="range"] {
            width: 300px;
            height: 8px;
            cursor: pointer;
        }
        .alpha-value {
            font-size: 24px;
            font-weight: bold;
            color: #00d4ff;
            min-width: 60px;
            text-align: center;
        }
        .info {
            color: #888;
            font-size: 14px;
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>Position Alignment Tool</h1>
    <div class="container">
        <div class="video-container">
            <img src="/video_feed" alt="Video Stream">
        </div>
        <div class="controls">
            <div class="slider-container">
                <label>Reference Alpha:</label>
                <input type="range" id="alphaSlider" min="0" max="100" value="{{ alpha_percent }}">
                <span class="alpha-value" id="alphaValue">{{ alpha_percent }}%</span>
            </div>
            <p class="info">Adjust slider to change reference image transparency</p>
        </div>
    </div>
    <script>
        const slider = document.getElementById('alphaSlider');
        const alphaValue = document.getElementById('alphaValue');

        slider.addEventListener('input', function() {
            alphaValue.textContent = this.value + '%';
        });

        slider.addEventListener('change', function() {
            fetch('/set_alpha/' + (this.value / 100))
                .then(response => response.json())
                .then(data => {
                    console.log('Alpha set to:', data.alpha);
                });
        });
    </script>
</body>
</html>
"""


class GracefulExit:
    """Handle graceful exit on Ctrl+C."""

    def __init__(self):
        self.should_exit = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        self.should_exit = True


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Position alignment tool - overlay dataset first frame on live camera"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the dataset root directory",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Overlay transparency (0.0-1.0), default: 0.4",
    )
    parser.add_argument(
        "--camera-topic",
        type=str,
        default=DEFAULT_CAMERA_TOPIC,
        help=f"ROS camera topic, default: {DEFAULT_CAMERA_TOPIC}",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default=DEFAULT_CAMERA_KEY,
        help=f"Dataset camera key, default: {DEFAULT_CAMERA_KEY}",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera width, default: 640",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera height, default: 480",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Web server port, default: 5001",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Web server host, default: 0.0.0.0",
    )
    return parser.parse_args()


def load_reference_image(data_path: Path, camera_key: str) -> np.ndarray:
    """Load the first frame from the dataset.

    Args:
        data_path: Path to the dataset root directory.
        camera_key: Key for the camera image in the dataset.

    Returns:
        Reference image as numpy array (H, W, 3) in BGR format for OpenCV.
    """
    logger.info(f"Loading dataset from {data_path}")

    # Load dataset with first episode
    dataset = LeRobotDataset(repo_id="local", root=data_path, episodes=[0])

    # Get available camera keys
    available_keys = dataset.meta.camera_keys
    logger.info(f"Available camera keys: {available_keys}")

    if camera_key not in available_keys:
        raise ValueError(
            f"Camera key '{camera_key}' not found in dataset. "
            f"Available keys: {available_keys}"
        )

    # Get first frame image
    first_frame = dataset[0]
    image_tensor = first_frame[camera_key]

    # Convert from torch.Tensor (C, H, W) to numpy (H, W, C)
    if hasattr(image_tensor, "numpy"):
        image = image_tensor.numpy()
    else:
        image = np.array(image_tensor)

    # Handle different tensor formats
    if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
        # (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0))

    # Convert to uint8 if normalized
    if image.dtype in [np.float32, np.float64]:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    logger.info(f"Loaded reference image with shape: {image.shape}")
    return image


def create_camera(topic: str, width: int, height: int) -> RosCamera:
    """Create and configure the ROS camera.

    Args:
        topic: ROS topic name for the camera.
        width: Camera image width.
        height: Camera image height.

    Returns:
        Configured RosCamera instance.
    """
    config = RosCameraConfig(
        topic_name=topic,
        width=width,
        height=height,
        fps=30,
        mock=False,
    )
    return RosCamera(config)


class AlignmentWebApp:
    """Flask-based web application for position alignment."""

    def __init__(
        self,
        camera: RosCamera,
        reference_image: np.ndarray,
        alpha: float,
    ):
        self.camera = camera
        self.reference_image = reference_image
        self.alpha = alpha
        self._lock = threading.Lock()
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route("/")
        def index():
            alpha_percent = int(self.alpha * 100)
            return render_template_string(HTML_TEMPLATE, alpha_percent=alpha_percent)

        @self.app.route("/video_feed")
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.route("/set_alpha/<value>")
        def set_alpha(value):
            try:
                new_alpha = float(value)
                if 0.0 <= new_alpha <= 1.0:
                    with self._lock:
                        self.alpha = new_alpha
                    logger.info(f"Alpha set to {new_alpha:.2f}")
                    return jsonify({"status": "ok", "alpha": new_alpha})
                else:
                    return jsonify({"status": "error", "message": "Alpha must be 0.0-1.0"}), 400
            except ValueError:
                return jsonify({"status": "error", "message": "Invalid alpha value"}), 400

    def _generate_frames(self):
        """Generate MJPEG frames for video streaming."""
        while True:
            try:
                live_frame = self.camera.read()
            except Exception as e:
                logger.warning(f"Failed to read camera frame: {e}")
                continue

            # Convert RGB to BGR for OpenCV
            live_bgr = cv2.cvtColor(live_frame, cv2.COLOR_RGB2BGR)

            # Resize reference image to match live frame if needed
            if self.reference_image.shape[:2] != live_bgr.shape[:2]:
                ref_resized = cv2.resize(
                    self.reference_image,
                    (live_bgr.shape[1], live_bgr.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                ref_resized = self.reference_image

            # Get current alpha value (thread-safe)
            with self._lock:
                alpha = self.alpha

            # Convert reference to BGR-swapped for better color contrast
            ref_color_swapped = ref_resized[:, :, ::-1].copy()

            # Create overlay: live * (1-alpha) + reference * alpha
            overlay = cv2.addWeighted(live_bgr, 1 - alpha, ref_color_swapped, alpha, 0)

            # Encode frame as JPEG
            _, jpeg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Yield frame in MJPEG format
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )

    def run(self, host: str, port: int):
        """Run the Flask web server."""
        logger.info(f"Starting web server at http://{host}:{port}")
        logger.info(f"Access from browser: http://localhost:{port}")
        self.app.run(host=host, port=port, threaded=True, debug=False)


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    # Validate alpha
    if not 0.0 <= args.alpha <= 1.0:
        logger.error("Alpha must be between 0.0 and 1.0")
        sys.exit(1)

    # Validate data path
    if not args.data_path.exists():
        logger.error(f"Dataset path does not exist: {args.data_path}")
        sys.exit(1)

    # Setup graceful exit handler
    exit_handler = GracefulExit()

    # Load reference image from dataset
    try:
        reference_image = load_reference_image(args.data_path, args.camera_key)
    except Exception as e:
        logger.error(f"Failed to load reference image: {e}")
        sys.exit(1)

    # Initialize ROS node
    try:
        import rospy
        rospy.init_node("align_position", anonymous=True, disable_signals=True)
        logger.info("ROS node initialized")
    except Exception as e:
        logger.error(f"Failed to initialize ROS node: {e}")
        sys.exit(1)

    # Create and connect camera
    camera = create_camera(args.camera_topic, args.width, args.height)

    try:
        logger.info(f"Connecting to camera topic: {args.camera_topic}")
        camera.connect(warmup=True)
        logger.info("Camera connected successfully")

        # Create and run web application
        web_app = AlignmentWebApp(camera, reference_image, args.alpha)
        web_app.run(args.host, args.port)

    except Exception as e:
        logger.error(f"Error during alignment: {e}")
        sys.exit(1)
    finally:
        if camera.is_connected:
            camera.disconnect()
            logger.info("Camera disconnected")


if __name__ == "__main__":
    main()
