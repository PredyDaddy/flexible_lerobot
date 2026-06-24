# Head Camera Alignment Service Design

## Goal

Build a small browser-accessible service under `cqy/allign` that repeatedly captures the robot head camera at a low frame rate and exposes:

- a live view
- a reference view
- a color-difference alignment view
- a simple browser UI for manual scene alignment

The service must only read from the head camera capture script and must not move the robot.

## User Workflow

1. Start the service on a known host and port.
2. Open the URL in a browser.
3. See the latest head-camera frame update automatically.
4. Compare the live frame with a reference image from `cqy/allign/allign.png`.
5. Use blend, contrast, and RGB/BGR channel swapping to make misalignment visible through color fringes or difference heat.
6. Adjust the table and robot placement until the live scene matches the lab reference scene.

## Architecture

### Capture loop

- The backend runs a background thread.
- Each loop iteration calls `cqy/capture_head_once.sh`.
- The script writes a single image to disk and returns the captured path.
- The backend reads that image, stores it in memory, records status metadata, and sleeps for a configurable interval.

### Rendering

- Keep the latest frame in memory as a BGR OpenCV image.
- Load the reference image from `cqy/allign/allign.png`.
- Resize the live frame to match the reference frame when generating comparison views.
- Support two display channel modes:
  - `rgb`: normal color display
  - `bgr`: intentionally swapped display to amplify visual color mismatch
- Support two comparison modes:
  - `blend`: alpha blend of reference and live frame
  - `difference`: absolute pixel difference multiplied by a gain factor

### HTTP API

- `GET /`: returns the HTML page.
- `GET /api/status`: returns capture status, timestamps, frame counters, and current configuration.
- `GET /api/frame/reference.png`: returns the reference image.
- `GET /api/frame/live.jpg`: returns the latest captured frame with optional channel and contrast adjustments.
- `GET /api/frame/align.jpg`: returns a rendered comparison image.
- `GET /api/stream/live.mjpg`: returns an MJPEG stream for the live frame.
- `POST /api/config`: updates capture interval and optional rendering defaults.

### Frontend

- Minimal static HTML served by FastAPI.
- Auto-refresh the status, live image, and alignment image.
- Controls:
  - capture interval
  - comparison mode
  - channel mode
  - blend alpha
  - contrast
  - difference gain

## Testing Strategy

- Test pure rendering helpers with synthetic images.
- Test the FastAPI app using a fake capture source so tests do not touch ROS or the robot.
- Test MJPEG response shape and status/config endpoints.
- Keep end-to-end runtime-free by injecting frames directly or through a fake capture callable.

## File Plan

- Create `cqy/allign/head_alignment_service.py` for the backend and embedded page.
- Create `tests/cqy/test_head_alignment_service.py` for API and rendering tests.
- Optionally add a tiny shell runner in `cqy/allign` if the service needs a one-command launch path.

## Risks And Mitigations

- Capture script timeout: report the error in `/api/status` and keep serving the last good frame.
- Missing reference file: fail clearly at startup.
- Slow capture script: use low default FPS and a background thread so the browser stays responsive.
