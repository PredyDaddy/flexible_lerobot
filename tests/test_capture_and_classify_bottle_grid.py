#!/usr/bin/env python

import importlib.util
from pathlib import Path


def load_capture_and_classify_module():
    module_path = Path(__file__).resolve().parents[1] / "cqy" / "capture_and_classify_bottle_grid.py"
    assert module_path.is_file(), f"Expected helper script to exist at {module_path}"

    spec = importlib.util.spec_from_file_location("cqy.capture_and_classify_bottle_grid", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_and_classify_bottle_grid_copies_image_and_returns_grid_result(tmp_path, monkeypatch):
    module = load_capture_and_classify_module()
    source_image = tmp_path / "head_0001.png"
    source_image.write_bytes(b"fake-image-bytes")

    captured = {}

    monkeypatch.setattr(module, "run_capture_once", lambda capture_script: source_image)

    def fake_classify_image(*, image_path, output_path=None, prompt=None, model=None, max_tokens=64, temperature=0.0):
        captured["image_path"] = image_path
        captured["output_path"] = output_path
        return {
            "image_path": str(image_path),
            "label": "B3",
            "position_text": "第二排3列",
            "raw_response": "<answer>B3</answer>",
            "output_path": str(image_path.with_suffix(".grid.json")),
            "model": model or "fake-model",
        }

    monkeypatch.setattr(module, "classify_image", fake_classify_image)

    result = module.capture_and_classify_bottle_grid(output_dir=tmp_path / "captures")

    copied_image = Path(result["image_path"])
    assert copied_image.is_file()
    assert copied_image.parent == tmp_path / "captures"
    assert copied_image.read_bytes() == b"fake-image-bytes"
    assert captured["image_path"] == copied_image
    assert result["label"] == "B3"
    assert result["position_text"] == "第二排3列"
