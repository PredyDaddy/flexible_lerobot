#!/usr/bin/env python

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def load_capture_and_describe_module():
    module_path = Path(__file__).resolve().parents[1] / "cqy" / "capture_and_describe.py"
    assert module_path.is_file(), f"Expected helper script to exist at {module_path}"

    spec = importlib.util.spec_from_file_location("cqy.capture_and_describe", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_and_describe_copies_image_and_writes_json(tmp_path, monkeypatch):
    module = load_capture_and_describe_module()

    source_image = tmp_path / "head_0001.png"
    source_image.write_bytes(b"fake-image-bytes")

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout=f"{source_image}\n", returncode=0, stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "build_client", lambda model=None: ("fake-client", model or "fake-model"))
    monkeypatch.setattr(module, "describe_captured_image", lambda **kwargs: "桌上有一个杯子。")

    result = module.capture_and_describe(output_dir=tmp_path / "captures")

    copied_image = Path(result["image_path"])
    metadata_path = Path(result["metadata_path"])

    assert copied_image.is_file()
    assert copied_image.parent == tmp_path / "captures"
    assert copied_image.read_bytes() == b"fake-image-bytes"
    assert metadata_path.is_file()

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["description"] == "桌上有一个杯子。"
    assert payload["image_path"] == str(copied_image)
    assert payload["source_image_path"] == str(source_image.resolve())
    assert payload["model"] == "fake-model"
