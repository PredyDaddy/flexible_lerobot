#!/usr/bin/env python

import importlib.util
import json
from pathlib import Path


def load_classify_bottle_grid_module():
    module_path = Path(__file__).resolve().parents[1] / "cqy" / "classify_bottle_grid.py"
    assert module_path.is_file(), f"Expected helper script to exist at {module_path}"

    spec = importlib.util.spec_from_file_location("cqy.classify_bottle_grid", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_grid_label_prefers_answer_tag():
    module = load_classify_bottle_grid_module()

    raw_response = """
    用户要求识别白色药瓶所在格子。
    先观察图像，再判断位置。
    <answer>B2</answer>
    """

    assert module.extract_grid_label(raw_response) == "B2"


def test_geometry_to_label_uses_body_center_for_top_row_and_foot_for_bottom_row():
    module = load_classify_bottle_grid_module()

    top_row = {
        "body_x": 171.6,
        "body_y": 321.4,
        "foot_y": 428.6,
        "v1_x": 234.3,
        "v2_x": 392.9,
        "h1_y": 407.1,
        "h2_y": 592.9,
    }
    bottom_row = {
        "body_x": 458.0,
        "body_y": 605.0,
        "foot_y": 680.0,
        "v1_x": 230.0,
        "v2_x": 340.0,
        "h1_y": 430.0,
        "h2_y": 590.0,
    }

    assert module.geometry_to_label(top_row) == "A1"
    assert module.geometry_to_label(bottom_row) == "C3"


def test_geometry_to_label_allows_small_tolerance_near_right_middle_boundary():
    module = load_classify_bottle_grid_module()

    near_right_boundary = {
        "body_x": 337.0,
        "body_y": 650.0,
        "foot_y": 740.0,
        "v1_x": 220.0,
        "v2_x": 330.0,
        "h1_y": 430.0,
        "h2_y": 590.0,
    }

    assert module.geometry_to_label(near_right_boundary) == "C2"


def test_classify_bottle_grid_writes_result_json(tmp_path, monkeypatch):
    module = load_classify_bottle_grid_module()
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"fake-image-bytes")

    monkeypatch.setattr(module, "build_client", lambda model=None: ("fake-client", model or "fake-model"))
    monkeypatch.setattr(
        module,
        "request_grid_geometry",
        lambda **kwargs: {
            "body_x": 10.0,
            "body_y": 20.0,
            "foot_y": 20.0,
            "v1_x": 40.0,
            "v2_x": 80.0,
            "h1_y": 15.0,
            "h2_y": 25.0,
            "raw_response": '{"body_x": 10.0, "body_y": 20.0, "foot_y": 20.0, "v1_x": 40.0, "v2_x": 80.0, "h1_y": 15.0, "h2_y": 25.0}',
        },
    )

    result = module.classify_bottle_grid(image_path=image_path)

    output_path = Path(result["output_path"])
    assert result["label"] == "B1"
    assert result["position_text"] == "第二排1列"
    assert output_path.is_file()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["label"] == "B1"
    assert payload["position_text"] == "第二排1列"
    assert payload["raw_response"] == '{"body_x": 10.0, "body_y": 20.0, "foot_y": 20.0, "v1_x": 40.0, "v2_x": 80.0, "h1_y": 15.0, "h2_y": 25.0}'
    assert payload["geometry"]["body_x"] == 10.0
