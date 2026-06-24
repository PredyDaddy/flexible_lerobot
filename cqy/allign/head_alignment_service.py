from __future__ import annotations

import argparse
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, AsyncIterator, Callable, Iterator

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8012
DEFAULT_CAPTURE_INTERVAL_S = 1.0
DEFAULT_CAPTURE_TIMEOUT_S = 8.0
DEFAULT_MJPEG_POLL_INTERVAL_S = 0.2
DEFAULT_JPEG_QUALITY = 90
DEFAULT_CAPTURE_SCRIPT = REPO_ROOT / "cqy/capture_head_once.sh"
DEFAULT_CAPTURE_OUTPUT_PATH = Path("/home/test/cam_shots/head_0001.png")
DEFAULT_REFERENCE_IMAGE = Path(__file__).with_name("allign.png")


@dataclass(frozen=True)
class CaptureResult:
    frame_bgr: np.ndarray
    captured_at: float
    source_path: str


class ConfigUpdateRequest(BaseModel):
    capture_interval_s: float | None = Field(default=None, ge=0.2, le=10.0)


def _encode_image(image_bgr: np.ndarray, suffix: str) -> bytes:
    encode_params: list[int] = []
    if suffix in {".jpg", ".jpeg"}:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), DEFAULT_JPEG_QUALITY]

    ok, buffer = cv2.imencode(suffix, image_bgr, encode_params)
    if not ok:
        raise RuntimeError(f"Could not encode image as {suffix}.")
    return bytes(buffer)


def _resize_to_match(frame_bgr: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    target_height, target_width = target_shape[:2]
    if frame_bgr.shape[:2] == (target_height, target_width):
        return frame_bgr.copy()
    return cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def prepare_display_frame(
    frame_bgr: np.ndarray,
    *,
    channel_mode: str = "rgb",
    contrast: float = 1.0,
) -> np.ndarray:
    normalized_channel_mode = channel_mode.strip().lower()
    if normalized_channel_mode == "rgb":
        display_bgr = frame_bgr.copy()
    elif normalized_channel_mode == "bgr":
        display_bgr = frame_bgr[:, :, ::-1].copy()
    else:
        raise ValueError(f"Unsupported channel mode: {channel_mode}")

    if contrast <= 0:
        raise ValueError("contrast must be > 0")
    if contrast == 1.0:
        return display_bgr

    adjusted = np.clip(display_bgr.astype(np.float32) * contrast, 0, 255)
    return adjusted.astype(np.uint8)


class HeadAlignmentService:
    def __init__(
        self,
        *,
        reference_image_path: Path = DEFAULT_REFERENCE_IMAGE,
        capture_script_path: Path = DEFAULT_CAPTURE_SCRIPT,
        capture_output_path: Path = DEFAULT_CAPTURE_OUTPUT_PATH,
        capture_interval_s: float = DEFAULT_CAPTURE_INTERVAL_S,
        capture_timeout_s: float = DEFAULT_CAPTURE_TIMEOUT_S,
        capture_frame_fn: Callable[[], CaptureResult] | None = None,
        autostart: bool = False,
    ) -> None:
        self.reference_image_path = Path(reference_image_path)
        self.capture_script_path = Path(capture_script_path)
        self.capture_output_path = Path(capture_output_path)
        self.capture_timeout_s = capture_timeout_s
        self.capture_frame_fn = capture_frame_fn or self._capture_with_script

        self.reference_image_bgr = cv2.imread(str(self.reference_image_path), cv2.IMREAD_COLOR)
        if self.reference_image_bgr is None:
            raise FileNotFoundError(f"Could not load reference image: {self.reference_image_path}")

        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None

        self._capture_interval_s = capture_interval_s
        self._capture_count = 0
        self._last_capture_at: float | None = None
        self._last_source_path: str | None = None
        self._last_error: str | None = None
        self._latest_frame_bgr: np.ndarray | None = None

        if autostart:
            self.start()

    @property
    def capture_interval_s(self) -> float:
        with self._lock:
            return self._capture_interval_s

    def update_config(self, *, capture_interval_s: float | None = None) -> dict[str, Any]:
        with self._lock:
            if capture_interval_s is not None:
                self._capture_interval_s = capture_interval_s
            return {"ok": True, "capture_interval_s": self._capture_interval_s}

    def _capture_with_script(self) -> CaptureResult:
        completed = subprocess.run(
            ["bash", str(self.capture_script_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=self.capture_timeout_s,
            check=False,
        )
        stdout_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        stderr_text = completed.stderr.strip()
        capture_output = stdout_lines[-1] if stdout_lines else str(self.capture_output_path)

        if completed.returncode != 0:
            raise RuntimeError(stderr_text or capture_output or "capture script failed")
        if capture_output == "timeout":
            raise RuntimeError("capture script timed out")

        image_path = Path(capture_output)
        if not image_path.is_file():
            image_path = self.capture_output_path
        if not image_path.is_file():
            raise RuntimeError(f"capture output image not found: {image_path}")

        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"could not read capture output: {image_path}")

        return CaptureResult(
            frame_bgr=frame_bgr,
            captured_at=time.time(),
            source_path=str(image_path),
        )

    def set_latest_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        source_path: str,
        captured_at: float | None = None,
    ) -> None:
        with self._lock:
            self._latest_frame_bgr = frame_bgr.copy()
            self._capture_count += 1
            self._last_capture_at = time.time() if captured_at is None else captured_at
            self._last_source_path = source_path
            self._last_error = None

    def capture_once(self) -> dict[str, Any]:
        capture_result = self.capture_frame_fn()
        self.set_latest_frame(
            capture_result.frame_bgr,
            source_path=capture_result.source_path,
            captured_at=capture_result.captured_at,
        )
        return self.status()

    def _record_error(self, exc: Exception) -> None:
        with self._lock:
            self._last_error = str(exc)

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            try:
                self.capture_once()
            except Exception as exc:
                self._record_error(exc)

            elapsed_s = time.perf_counter() - loop_start
            sleep_s = max(self.capture_interval_s - elapsed_s, 0.0)
            if self._stop_event.wait(sleep_s):
                return

    def start(self) -> None:
        if self.is_running():
            return
        self._stop_event = Event()
        self._thread = Thread(target=self._capture_loop, name="head-alignment-capture", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        if not self.is_running():
            return
        self._stop_event.set()
        assert self._thread is not None
        self._thread.join(timeout=timeout_s)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _get_latest_frame(self) -> np.ndarray:
        with self._lock:
            if self._latest_frame_bgr is None:
                raise RuntimeError("No frame captured yet.")
            return self._latest_frame_bgr.copy()

    def render_live_view(self, *, channel_mode: str = "rgb", contrast: float = 1.0) -> np.ndarray:
        latest_frame_bgr = self._get_latest_frame()
        return prepare_display_frame(latest_frame_bgr, channel_mode=channel_mode, contrast=contrast)

    def render_alignment_view(
        self,
        *,
        mode: str = "difference",
        channel_mode: str = "bgr",
        alpha: float = 0.5,
        contrast: float = 1.0,
        diff_gain: float = 4.0,
    ) -> np.ndarray:
        latest_frame_bgr = self._get_latest_frame()
        resized_live_bgr = _resize_to_match(latest_frame_bgr, self.reference_image_bgr.shape)
        prepared_live_bgr = prepare_display_frame(
            resized_live_bgr,
            channel_mode=channel_mode,
            contrast=contrast,
        )

        normalized_mode = mode.strip().lower()
        if normalized_mode == "blend":
            normalized_alpha = min(max(alpha, 0.0), 1.0)
            return cv2.addWeighted(
                self.reference_image_bgr,
                normalized_alpha,
                prepared_live_bgr,
                1.0 - normalized_alpha,
                0.0,
            )
        if normalized_mode == "difference":
            if diff_gain <= 0:
                raise ValueError("diff_gain must be > 0")
            difference = cv2.absdiff(self.reference_image_bgr, prepared_live_bgr)
            boosted = np.clip(difference.astype(np.float32) * diff_gain, 0, 255)
            return boosted.astype(np.uint8)
        raise ValueError(f"Unsupported alignment mode: {mode}")

    def render_reference_png(self) -> bytes:
        return _encode_image(self.reference_image_bgr, ".png")

    def render_live_jpg(self, *, channel_mode: str = "rgb", contrast: float = 1.0) -> bytes:
        return _encode_image(self.render_live_view(channel_mode=channel_mode, contrast=contrast), ".jpg")

    def render_alignment_jpg(
        self,
        *,
        mode: str = "difference",
        channel_mode: str = "bgr",
        alpha: float = 0.5,
        contrast: float = 1.0,
        diff_gain: float = 4.0,
    ) -> bytes:
        return _encode_image(
            self.render_alignment_view(
                mode=mode,
                channel_mode=channel_mode,
                alpha=alpha,
                contrast=contrast,
                diff_gain=diff_gain,
            ),
            ".jpg",
        )

    def iter_live_mjpeg(
        self,
        *,
        channel_mode: str = "rgb",
        contrast: float = 1.0,
        max_frames: int | None = None,
        poll_interval_s: float = DEFAULT_MJPEG_POLL_INTERVAL_S,
    ) -> Iterator[bytes]:
        yielded_frames = 0
        while max_frames is None or yielded_frames < max_frames:
            try:
                jpeg_bytes = self.render_live_jpg(channel_mode=channel_mode, contrast=contrast)
            except RuntimeError:
                time.sleep(poll_interval_s)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode()
                + jpeg_bytes
                + b"\r\n"
            )
            yielded_frames += 1
            time.sleep(poll_interval_s)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "service": "head_alignment_service",
                "capture": {
                    "running": self.is_running(),
                    "capture_interval_s": self._capture_interval_s,
                    "capture_count": self._capture_count,
                    "has_frame": self._latest_frame_bgr is not None,
                    "last_capture_at": self._last_capture_at,
                    "last_source_path": self._last_source_path,
                    "last_error": self._last_error,
                },
                "reference": {
                    "path": str(self.reference_image_path),
                    "shape": list(self.reference_image_bgr.shape),
                },
            }


def _render_index_html(service: HeadAlignmentService) -> str:
    default_interval = service.capture_interval_s
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Head Camera Alignment</title>
  <style>
    :root {{
      --bg: #f4efe5;
      --panel: rgba(255, 252, 247, 0.92);
      --ink: #1f2421;
      --muted: #5b635b;
      --line: #c8bda9;
      --accent: #0f766e;
      --accent-soft: #d6efe9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 30%),
        radial-gradient(circle at bottom right, rgba(210, 127, 70, 0.16), transparent 25%),
        linear-gradient(135deg, #f6f1e7 0%, #ece3d4 100%);
      min-height: 100vh;
    }}
    .shell {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero, .controls, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 18px 60px rgba(55, 42, 22, 0.08);
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      line-height: 1.05;
    }}
    .sub {{
      margin: 0;
      color: var(--muted);
      max-width: 900px;
      line-height: 1.55;
    }}
    .status {{
      margin-top: 16px;
      padding: 12px 14px;
      background: #fbf8f1;
      border-radius: 14px;
      border: 1px solid #ddd1bb;
      font-family: "SFMono-Regular", Consolas, monospace;
      white-space: pre-wrap;
      min-height: 74px;
    }}
    .controls {{
      padding: 18px 20px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }}
    label {{
      display: grid;
      gap: 8px;
      font-size: 14px;
      color: var(--muted);
    }}
    input, select, button {{
      width: 100%;
      border: 1px solid #c9baa1;
      border-radius: 12px;
      padding: 10px 12px;
      font: inherit;
      background: white;
      color: var(--ink);
    }}
    button {{
      background: var(--accent);
      color: white;
      cursor: pointer;
      font-weight: 600;
      border: none;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    .panel {{
      padding: 14px;
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    .panel p {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 14px;
    }}
    img {{
      display: block;
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      background:
        linear-gradient(45deg, #f1ebdf 25%, #f7f2e8 25%, #f7f2e8 50%, #f1ebdf 50%, #f1ebdf 75%, #f7f2e8 75%);
      background-size: 24px 24px;
      border-radius: 16px;
      border: 1px solid #ddd1bb;
    }}
    .meta {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .slider-row {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .slider-row output {{
      min-width: 48px;
      text-align: right;
      color: var(--ink);
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>Head Camera Alignment</h1>
      <p class="sub">This page keeps sampling the robot head camera at a low rate and compares it with the saved lab reference image. Use the live stream, blend mode, and RGB/BGR color mismatch to manually align the desk and dual-arm placement.</p>
      <div class="status" id="status">Loading status...</div>
    </section>

    <section class="controls">
      <label>
        Capture Interval (s)
        <input id="capture-interval" type="number" min="0.2" max="10" step="0.1" value="{default_interval}">
      </label>
      <label>
        Compare Mode
        <select id="mode">
          <option value="difference" selected>difference</option>
          <option value="blend">blend</option>
        </select>
      </label>
      <label>
        Live Channel Mode
        <select id="channel-mode">
          <option value="bgr" selected>bgr swapped</option>
          <option value="rgb">rgb normal</option>
        </select>
      </label>
      <label>
        Blend Alpha
        <div class="slider-row">
          <input id="alpha" type="range" min="0" max="1" step="0.05" value="0.5">
          <output id="alpha-value">0.50</output>
        </div>
      </label>
      <label>
        Contrast
        <div class="slider-row">
          <input id="contrast" type="range" min="0.5" max="3" step="0.1" value="1.4">
          <output id="contrast-value">1.40</output>
        </div>
      </label>
      <label>
        Difference Gain
        <div class="slider-row">
          <input id="diff-gain" type="range" min="1" max="8" step="0.5" value="4">
          <output id="diff-gain-value">4.00</output>
        </div>
      </label>
      <label>
        Apply Interval
        <button id="save-config" type="button">Update Capture Rate</button>
      </label>
    </section>

    <section class="grid">
      <article class="panel">
        <h2>Reference Image</h2>
        <p>Saved lab layout from <code>cqy/allign/allign.png</code>.</p>
        <img id="reference-image" src="/api/frame/reference.png" alt="Reference image">
        <div class="meta">Keep this scene fixed and move the physical desk setup until it matches.</div>
      </article>

      <article class="panel">
        <h2>Live Head Camera</h2>
        <p>Low-rate MJPEG stream backed by repeated calls to <code>cqy/capture_head_once.sh</code>.</p>
        <img id="live-image" src="/api/stream/live.mjpg" alt="Live head camera stream">
        <div class="meta" id="live-meta">Stream ready.</div>
      </article>

      <article class="panel">
        <h2>Alignment View</h2>
        <p>Difference mode highlights mismatched regions. Blend mode helps with direct overlay alignment.</p>
        <img id="align-image" src="/api/frame/align.jpg" alt="Alignment comparison image">
        <div class="meta">Use the BGR-swapped mode when you want stronger chromatic mismatch cues.</div>
      </article>
    </section>
  </main>

  <script>
    const statusBox = document.getElementById("status");
    const liveImage = document.getElementById("live-image");
    const alignImage = document.getElementById("align-image");
    const liveMeta = document.getElementById("live-meta");
    const captureInterval = document.getElementById("capture-interval");
    const mode = document.getElementById("mode");
    const channelMode = document.getElementById("channel-mode");
    const alpha = document.getElementById("alpha");
    const contrast = document.getElementById("contrast");
    const diffGain = document.getElementById("diff-gain");
    const alphaValue = document.getElementById("alpha-value");
    const contrastValue = document.getElementById("contrast-value");
    const diffGainValue = document.getElementById("diff-gain-value");
    const saveConfig = document.getElementById("save-config");

    function updateOutputs() {{
      alphaValue.textContent = Number(alpha.value).toFixed(2);
      contrastValue.textContent = Number(contrast.value).toFixed(2);
      diffGainValue.textContent = Number(diffGain.value).toFixed(2);
    }}

    function buildImageQuery() {{
      const params = new URLSearchParams();
      params.set("mode", mode.value);
      params.set("channel_mode", channelMode.value);
      params.set("alpha", alpha.value);
      params.set("contrast", contrast.value);
      params.set("diff_gain", diffGain.value);
      params.set("ts", Date.now().toString());
      return params.toString();
    }}

    function refreshImages() {{
      const params = buildImageQuery();
      liveImage.src = `/api/stream/live.mjpg?channel_mode=${{encodeURIComponent(channelMode.value)}}&contrast=${{encodeURIComponent(contrast.value)}}&ts=${{Date.now()}}`;
      alignImage.src = `/api/frame/align.jpg?${{params}}`;
      liveMeta.textContent = `channel=${{channelMode.value}}, contrast=${{Number(contrast.value).toFixed(2)}}`;
    }}

    async function refreshStatus() {{
      try {{
        const response = await fetch("/api/status");
        const payload = await response.json();
        const capture = payload.capture;
        const lines = [
          `running: ${{capture.running}}`,
          `capture_count: ${{capture.capture_count}}`,
          `capture_interval_s: ${{capture.capture_interval_s}}`,
          `last_capture_at: ${{capture.last_capture_at}}`,
          `last_source_path: ${{capture.last_source_path}}`,
          `last_error: ${{capture.last_error}}`,
          `reference: ${{payload.reference.path}}`,
        ];
        statusBox.textContent = lines.join("\\n");
      }} catch (error) {{
        statusBox.textContent = `status fetch failed: ${{error}}`;
      }}
    }}

    async function saveCaptureInterval() {{
      const payload = {{ capture_interval_s: Number(captureInterval.value) }};
      const response = await fetch("/api/config", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload),
      }});
      const result = await response.json();
      captureInterval.value = result.capture_interval_s;
      await refreshStatus();
    }}

    [mode, channelMode, alpha, contrast, diffGain].forEach((element) => {{
      element.addEventListener("input", () => {{
        updateOutputs();
        refreshImages();
      }});
      element.addEventListener("change", () => {{
        updateOutputs();
        refreshImages();
      }});
    }});
    saveConfig.addEventListener("click", saveCaptureInterval);

    updateOutputs();
    refreshImages();
    refreshStatus();
    setInterval(refreshStatus, 1000);
    setInterval(() => {{
      alignImage.src = `/api/frame/align.jpg?${{buildImageQuery()}}`;
    }}, 700);
  </script>
</body>
</html>"""


def create_app(service: HeadAlignmentService | None = None, *, autostart: bool = True) -> FastAPI:
    alignment_service = service or HeadAlignmentService()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if autostart:
            alignment_service.start()
        try:
            yield
        finally:
            alignment_service.stop()

    app = FastAPI(title="head_alignment_service", version="0.1.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _render_index_html(alignment_service)

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return alignment_service.status()

    @app.post("/api/config")
    def update_config(request: ConfigUpdateRequest) -> dict[str, Any]:
        return alignment_service.update_config(capture_interval_s=request.capture_interval_s)

    @app.get("/api/frame/reference.png")
    def reference_frame() -> Response:
        return Response(content=alignment_service.render_reference_png(), media_type="image/png")

    @app.get("/api/frame/live.jpg")
    def live_frame(channel_mode: str = "rgb", contrast: float = 1.0) -> Response:
        try:
            return Response(
                content=alignment_service.render_live_jpg(channel_mode=channel_mode, contrast=contrast),
                media_type="image/jpeg",
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/frame/align.jpg")
    def alignment_frame(
        mode: str = "difference",
        channel_mode: str = "bgr",
        alpha: float = 0.5,
        contrast: float = 1.0,
        diff_gain: float = 4.0,
    ) -> Response:
        try:
            return Response(
                content=alignment_service.render_alignment_jpg(
                    mode=mode,
                    channel_mode=channel_mode,
                    alpha=alpha,
                    contrast=contrast,
                    diff_gain=diff_gain,
                ),
                media_type="image/jpeg",
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/stream/live.mjpg")
    def live_stream(
        channel_mode: str = "rgb",
        contrast: float = 1.0,
        max_frames: int | None = None,
    ) -> StreamingResponse:
        return StreamingResponse(
            alignment_service.iter_live_mjpeg(
                channel_mode=channel_mode,
                contrast=contrast,
                max_frames=max_frames,
            ),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a browser UI for head-camera scene alignment.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--capture-interval-s", type=float, default=DEFAULT_CAPTURE_INTERVAL_S)
    parser.add_argument("--capture-timeout-s", type=float, default=DEFAULT_CAPTURE_TIMEOUT_S)
    parser.add_argument("--capture-script", default=str(DEFAULT_CAPTURE_SCRIPT))
    parser.add_argument("--reference-image", default=str(DEFAULT_REFERENCE_IMAGE))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = HeadAlignmentService(
        reference_image_path=Path(args.reference_image),
        capture_script_path=Path(args.capture_script),
        capture_interval_s=args.capture_interval_s,
        capture_timeout_s=args.capture_timeout_s,
    )

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("uvicorn is required to run cqy.allign.head_alignment_service") from exc

    uvicorn.run(create_app(service=service, autostart=True), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
