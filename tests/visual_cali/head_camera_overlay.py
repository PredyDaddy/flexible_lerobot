#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import yaml
except ImportError:  # pragma: no cover - only used when PyYAML is unavailable in a runtime env.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml"
DEFAULT_CAMERA_NAME = "camera_head"
DEFAULT_RTSP_URL = "rtsp://192.168.1.81:8554/robot_camera/camera_head"
LOW_LATENCY_TCP_OPTIONS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
MJPEG_BOUNDARY = "frame"


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Head Camera Overlay</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101418;
      --panel: #171d22;
      --panel-2: #20272e;
      --line: #38434d;
      --text: #edf2f4;
      --muted: #a7b2bb;
      --accent: #2fbf9f;
      --warn: #f2b84b;
      --danger: #ee6c4d;
    }

    * {
      box-sizing: border-box;
    }

    html,
    body {
      width: 100%;
      min-height: 100%;
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family:
        Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    body {
      display: flex;
      flex-direction: column;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 58px;
      padding: 10px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }

    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }

    .status {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
      min-width: 170px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    .status-dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--warn);
      box-shadow: 0 0 0 3px rgba(242, 184, 75, 0.16);
    }

    .status-dot.live {
      background: var(--accent);
      box-shadow: 0 0 0 3px rgba(47, 191, 159, 0.16);
    }

    .status-dot.error {
      background: var(--danger);
      box-shadow: 0 0 0 3px rgba(238, 108, 77, 0.16);
    }

    main {
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      width: 100%;
      min-height: calc(100vh - 58px);
    }

    .stage-wrap {
      display: grid;
      min-height: 0;
      padding: 14px;
    }

    .stage {
      position: relative;
      place-self: center;
      width: min(100%, calc((100vh - 178px) * 16 / 9));
      max-width: 1280px;
      aspect-ratio: 16 / 9;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #050607;
    }

    .video,
    .reference {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-position: center;
      user-select: none;
    }

    .video {
      z-index: 1;
      object-fit: fill;
      background: #050607;
    }

    .reference {
      z-index: 2;
      display: none;
      object-fit: fill;
      opacity: 0.45;
      pointer-events: none;
    }

    .reference.visible {
      display: block;
    }

    .empty-reference {
      position: absolute;
      right: 12px;
      bottom: 12px;
      z-index: 3;
      padding: 6px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(16, 20, 24, 0.76);
      color: var(--muted);
      font-size: 12px;
    }

    .empty-reference.hidden {
      display: none;
    }

    .toolbar {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 160px 140px 120px 92px;
      gap: 10px;
      align-items: center;
      padding: 12px 16px 16px;
      border-top: 1px solid var(--line);
      background: var(--panel);
    }

    .field {
      display: grid;
      gap: 6px;
      min-width: 0;
    }

    label {
      color: var(--muted);
      font-size: 12px;
      line-height: 1;
    }

    input[type="file"],
    select,
    button {
      width: 100%;
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      color: var(--text);
      font: inherit;
      font-size: 14px;
    }

    input[type="file"] {
      padding: 6px;
    }

    select {
      padding: 0 8px;
    }

    input[type="range"] {
      width: 100%;
      margin: 0;
      accent-color: var(--accent);
    }

    button {
      cursor: pointer;
    }

    button:hover,
    select:hover,
    input[type="file"]:hover {
      border-color: #5b6975;
    }

    .range-row {
      display: grid;
      grid-template-columns: 1fr 44px;
      gap: 8px;
      align-items: center;
    }

    .value {
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      font-size: 13px;
      text-align: right;
    }

    @media (max-width: 760px) {
      header {
        align-items: flex-start;
        flex-direction: column;
      }

      .status {
        justify-content: flex-start;
        min-width: 0;
      }

      main {
        min-height: calc(100vh - 92px);
      }

      .stage-wrap {
        padding: 10px;
      }

      .stage {
        width: 100%;
      }

      .toolbar {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Head Camera Overlay</h1>
    <div class="status">
      <span id="statusDot" class="status-dot"></span>
      <span id="statusText">connecting</span>
    </div>
  </header>
  <main>
    <section class="stage-wrap">
      <div class="stage">
        <img id="video" class="video" alt="Head camera stream" src="/video.mjpg">
        <img id="reference" class="reference" alt="Reference overlay">
        <div id="emptyReference" class="empty-reference">No reference image</div>
      </div>
    </section>
    <section class="toolbar">
      <div class="field">
        <label for="referenceInput">Reference image</label>
        <input id="referenceInput" type="file" accept="image/*">
      </div>
      <div class="field">
        <label for="opacity">Opacity</label>
        <div class="range-row">
          <input id="opacity" type="range" min="0" max="100" value="45">
          <span id="opacityValue" class="value">45%</span>
        </div>
      </div>
      <div class="field">
        <label for="fitMode">Fit</label>
        <select id="fitMode">
          <option value="fill" selected>fill</option>
          <option value="contain">contain</option>
          <option value="cover">cover</option>
        </select>
      </div>
      <div class="field">
        <label for="videoFitMode">Video fit</label>
        <select id="videoFitMode">
          <option value="fill" selected>fill</option>
          <option value="contain">contain</option>
          <option value="cover">cover</option>
        </select>
      </div>
      <div class="field">
        <label>&nbsp;</label>
        <button id="clearReference" type="button">Clear</button>
      </div>
    </section>
  </main>
  <script>
    const hasInitialReference = __HAS_INITIAL_REFERENCE__;
    const reference = document.getElementById("reference");
    const referenceInput = document.getElementById("referenceInput");
    const emptyReference = document.getElementById("emptyReference");
    const opacity = document.getElementById("opacity");
    const opacityValue = document.getElementById("opacityValue");
    const fitMode = document.getElementById("fitMode");
    const videoFitMode = document.getElementById("videoFitMode");
    const video = document.getElementById("video");
    const clearReference = document.getElementById("clearReference");
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");
    let objectUrl = null;

    function setReference(src, managedObjectUrl = null) {
      if (objectUrl && objectUrl !== managedObjectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
      objectUrl = managedObjectUrl;
      reference.src = src;
      reference.classList.add("visible");
      emptyReference.classList.add("hidden");
    }

    function clearOverlay() {
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
        objectUrl = null;
      }
      reference.removeAttribute("src");
      reference.classList.remove("visible");
      emptyReference.classList.remove("hidden");
      referenceInput.value = "";
    }

    referenceInput.addEventListener("change", () => {
      const file = referenceInput.files && referenceInput.files[0];
      if (!file) {
        return;
      }
      const nextObjectUrl = URL.createObjectURL(file);
      setReference(nextObjectUrl, nextObjectUrl);
    });

    opacity.addEventListener("input", () => {
      const value = Number(opacity.value);
      reference.style.opacity = String(value / 100);
      opacityValue.textContent = `${value}%`;
    });

    fitMode.addEventListener("change", () => {
      reference.style.objectFit = fitMode.value;
    });

    videoFitMode.addEventListener("change", () => {
      video.style.objectFit = videoFitMode.value;
    });

    clearReference.addEventListener("click", clearOverlay);

    async function refreshStatus() {
      try {
        const response = await fetch("/status", { cache: "no-store" });
        const status = await response.json();
        statusDot.classList.remove("live", "error");
        if (status.connected && status.last_frame_age_s !== null && status.last_frame_age_s < 2.0) {
          statusDot.classList.add("live");
          statusText.textContent = `live - ${status.frames_read} frames`;
        } else if (status.last_error) {
          statusDot.classList.add("error");
          statusText.textContent = status.last_error;
        } else {
          statusText.textContent = "waiting for camera";
        }
      } catch (error) {
        statusDot.classList.remove("live");
        statusDot.classList.add("error");
        statusText.textContent = "status unavailable";
      }
    }

    if (hasInitialReference) {
      setReference(`/reference-image?ts=${Date.now()}`);
    }
    refreshStatus();
    window.setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""


@dataclass(frozen=True)
class CameraSettings:
    name: str
    rtsp_url: str
    width: int | None
    height: int | None
    fps: int
    transport: str


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int
    camera: CameraSettings
    reference_image: Path | None
    jpeg_quality: int
    reconnect_sleep_s: float


class HeadCameraStream:
    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._jpeg: bytes | None = None
        self._frame_id = 0
        self._frames_read = 0
        self._last_frame_monotonic_s: float | None = None
        self._connected = False
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="head-camera-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def wait_for_jpeg(self, last_seen_frame_id: int, timeout_s: float) -> tuple[int, bytes] | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._frame_id > last_seen_frame_id or self._stop.is_set(),
                timeout=timeout_s,
            )
            if self._jpeg is None or self._frame_id <= last_seen_frame_id:
                return None
            return self._frame_id, self._jpeg

    def status(self) -> dict[str, Any]:
        with self._condition:
            last_frame_age_s = None
            if self._last_frame_monotonic_s is not None:
                last_frame_age_s = max(0.0, time.monotonic() - self._last_frame_monotonic_s)
            return {
                "camera_name": self.settings.camera.name,
                "rtsp_url": self.settings.camera.rtsp_url,
                "connected": self._connected,
                "frames_read": self._frames_read,
                "last_frame_age_s": last_frame_age_s,
                "last_error": self._last_error,
            }

    def _set_status(self, *, connected: bool, last_error: str | None = None) -> None:
        with self._condition:
            self._connected = connected
            self._last_error = last_error
            self._condition.notify_all()

    def _store_frame(self, jpeg: bytes) -> None:
        with self._condition:
            self._jpeg = jpeg
            self._frame_id += 1
            self._frames_read += 1
            self._last_frame_monotonic_s = time.monotonic()
            self._connected = True
            self._last_error = None
            self._condition.notify_all()

    def _run(self) -> None:
        import cv2

        camera = self.settings.camera
        if camera.transport == "tcp":
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = LOW_LATENCY_TCP_OPTIONS

        while not self._stop.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(camera.rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                if camera.width is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(camera.width))
                if camera.height is not None:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(camera.height))
                if camera.fps > 0:
                    cap.set(cv2.CAP_PROP_FPS, float(camera.fps))
                if not cap.isOpened():
                    self._set_status(connected=False, last_error="camera open failed")
                    self._sleep_before_reconnect()
                    continue

                self._set_status(connected=True)
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        self._set_status(connected=False, last_error="camera read failed")
                        break
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(self.settings.jpeg_quality)],
                    )
                    if not ok:
                        self._set_status(connected=True, last_error="jpeg encode failed")
                        continue
                    self._store_frame(encoded.tobytes())
            except Exception as exc:  # pragma: no cover - depends on camera/network runtime behavior.
                self._set_status(connected=False, last_error=str(exc))
            finally:
                if cap is not None:
                    cap.release()

            self._sleep_before_reconnect()

    def _sleep_before_reconnect(self) -> None:
        self._stop.wait(timeout=self.settings.reconnect_sleep_s)


class OverlayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], settings: ServerSettings, stream: HeadCameraStream):
        super().__init__(server_address, OverlayRequestHandler)
        self.settings = settings
        self.stream = stream


class OverlayRequestHandler(BaseHTTPRequestHandler):
    server: OverlayHTTPServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_index()
        elif parsed.path == "/video.mjpg":
            self._send_mjpeg_stream()
        elif parsed.path == "/status":
            self._send_json(self.server.stream.status())
        elif parsed.path == "/reference-image":
            self._send_reference_image()
        elif parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), format % args))

    def _send_index(self) -> None:
        html = INDEX_HTML.replace(
            "__HAS_INITIAL_REFERENCE__",
            json.dumps(self.server.settings.reference_image is not None),
        )
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=True).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_reference_image(self) -> None:
        reference_image = self.server.settings.reference_image
        if reference_image is None or not reference_image.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Reference image not configured")
            return

        content_type = mimetypes.guess_type(reference_image.name)[0] or "application/octet-stream"
        body = reference_image.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_mjpeg_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}")
        self.end_headers()

        last_seen_frame_id = 0
        while True:
            frame = self.server.stream.wait_for_jpeg(last_seen_frame_id, timeout_s=10.0)
            if frame is None:
                continue
            last_seen_frame_id, jpeg = frame
            try:
                self.wfile.write(f"--{MJPEG_BOUNDARY}\r\n".encode("ascii"))
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return


def main() -> int:
    args = parse_args()
    settings = build_settings(args)
    validate_settings(settings)

    stream = HeadCameraStream(settings)
    server = OverlayHTTPServer((settings.host, settings.port), settings, stream)

    def handle_signal(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, name="overlay-server-shutdown", daemon=True).start()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    stream.start()
    url_host = "127.0.0.1" if settings.host in ("", "0.0.0.0") else settings.host
    print(f"Head camera overlay: http://{url_host}:{settings.port}", flush=True)
    print(f"Camera: {settings.camera.name} -> {settings.camera.rtsp_url}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        stream.stop()
        server.server_close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the JZ robot head camera with a transparent image overlay."
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host to bind.")
    parser.add_argument("--port", type=int, default=8090, help="HTTP port to bind.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Robot camera YAML config.")
    parser.add_argument(
        "--camera-name",
        default=DEFAULT_CAMERA_NAME,
        help="Camera key inside the YAML config.",
    )
    parser.add_argument("--rtsp-url", help="Override the camera RTSP URL.")
    parser.add_argument("--width", type=int, help="Override capture width.")
    parser.add_argument("--height", type=int, help="Override capture height.")
    parser.add_argument("--fps", type=int, help="Override capture FPS.")
    parser.add_argument("--transport", choices=("tcp", "udp"), help="Override RTSP transport.")
    parser.add_argument("--reference-image", type=Path, help="Optional initial reference image path.")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="MJPEG JPEG quality, 1-100.")
    parser.add_argument(
        "--reconnect-sleep-s",
        type=float,
        default=1.0,
        help="Delay before reconnect attempts.",
    )
    return parser.parse_args()


def build_settings(args: argparse.Namespace) -> ServerSettings:
    camera = load_camera_settings(args.config, args.camera_name)
    camera = CameraSettings(
        name=args.camera_name,
        rtsp_url=args.rtsp_url or camera.rtsp_url,
        width=args.width if args.width is not None else camera.width,
        height=args.height if args.height is not None else camera.height,
        fps=args.fps if args.fps is not None else camera.fps,
        transport=args.transport or camera.transport,
    )
    reference_image = args.reference_image.expanduser().resolve() if args.reference_image else None
    return ServerSettings(
        host=args.host,
        port=args.port,
        camera=camera,
        reference_image=reference_image,
        jpeg_quality=args.jpeg_quality,
        reconnect_sleep_s=args.reconnect_sleep_s,
    )


def load_camera_settings(config_path: Path, camera_name: str) -> CameraSettings:
    fallback = CameraSettings(
        name=camera_name,
        rtsp_url=DEFAULT_RTSP_URL,
        width=1280,
        height=720,
        fps=30,
        transport="tcp",
    )
    if not config_path.is_file():
        return fallback
    if yaml is None:
        return fallback

    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    if "cameras" in raw:
        camera = (raw.get("cameras") or {}).get(camera_name)
        if camera is None:
            raise ValueError(f"Camera {camera_name!r} not found in {config_path}")
        return CameraSettings(
            name=camera_name,
            rtsp_url=str(camera["rtsp_url"]),
            width=_optional_int(camera.get("width")),
            height=_optional_int(camera.get("height")),
            fps=int(camera.get("fps", 30)),
            transport=str(camera.get("transport", "tcp")),
        )

    if "rtsp_cameras" in raw:
        camera = (raw.get("rtsp_cameras") or {}).get(camera_name)
        if camera is None:
            raise ValueError(f"Camera {camera_name!r} not found in {config_path}")
        return CameraSettings(
            name=camera_name,
            rtsp_url=str(camera["url"]),
            width=_optional_int(camera.get("width")),
            height=_optional_int(camera.get("height")),
            fps=int(camera.get("fps", 30)),
            transport=str(camera.get("transport", "tcp")),
        )

    return fallback


def validate_settings(settings: ServerSettings) -> None:
    if not settings.camera.rtsp_url.startswith("rtsp://"):
        raise ValueError(f"RTSP URL must start with rtsp://, got {settings.camera.rtsp_url}")
    if settings.camera.transport not in ("tcp", "udp"):
        raise ValueError("RTSP transport must be 'tcp' or 'udp'")
    if settings.camera.fps <= 0:
        raise ValueError("FPS must be positive")
    if settings.jpeg_quality < 1 or settings.jpeg_quality > 100:
        raise ValueError("--jpeg-quality must be in 1..100")
    if settings.reconnect_sleep_s < 0:
        raise ValueError("--reconnect-sleep-s must be non-negative")
    if settings.reference_image is not None and not settings.reference_image.is_file():
        raise FileNotFoundError(f"Reference image not found: {settings.reference_image}")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
