# Head Alignment Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-based alignment service that repeatedly captures the robot head camera, serves a live stream and comparison images, and helps manually align a physical scene to a saved lab reference image.

**Architecture:** A FastAPI app hosts a background capture loop that repeatedly invokes `cqy/capture_head_once.sh`, keeps the newest OpenCV frame in memory, and renders live plus reference comparison views on demand. A single HTML page polls status and image endpoints so the browser can be used as the alignment console.

**Tech Stack:** Python 3.10, FastAPI, Uvicorn, OpenCV, NumPy, pytest

---

### Task 1: Add the failing tests

**Files:**
- Create: `tests/cqy/test_head_alignment_service.py`
- Test: `tests/cqy/test_head_alignment_service.py`

- [ ] **Step 1: Write the failing test**

```python
def test_create_app_serves_html_and_status():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n lerobot pytest -q tests/cqy/test_head_alignment_service.py`
Expected: FAIL because `cqy.allign.head_alignment_service` does not exist yet

- [ ] **Step 3: Commit**

```bash
git add tests/cqy/test_head_alignment_service.py
git commit -m "test(allign): add alignment service coverage"
```

### Task 2: Implement the service

**Files:**
- Create: `cqy/allign/head_alignment_service.py`
- Modify: `tests/cqy/test_head_alignment_service.py`

- [ ] **Step 1: Write minimal implementation**

```python
def create_app(...):
    ...
```

- [ ] **Step 2: Run targeted tests**

Run: `conda run -n lerobot pytest -q tests/cqy/test_head_alignment_service.py`
Expected: PASS

- [ ] **Step 3: Refine the rendering and stream helpers**

```python
def render_alignment_view(...):
    ...
```

- [ ] **Step 4: Re-run targeted tests**

Run: `conda run -n lerobot pytest -q tests/cqy/test_head_alignment_service.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cqy/allign/head_alignment_service.py tests/cqy/test_head_alignment_service.py
git commit -m "feat(allign): add head camera alignment service"
```

### Task 3: Verify startup path

**Files:**
- Modify: `cqy/allign/head_alignment_service.py`

- [ ] **Step 1: Run a startup smoke check**

Run: `conda run -n lerobot python -m cqy.allign.head_alignment_service --help`
Expected: PASS and prints CLI usage

- [ ] **Step 2: Run lint-style import check**

Run: `conda run -n lerobot python -m py_compile cqy/allign/head_alignment_service.py tests/cqy/test_head_alignment_service.py`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add cqy/allign/head_alignment_service.py
git commit -m "chore(allign): verify alignment service startup"
```
