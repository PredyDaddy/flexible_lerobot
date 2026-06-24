#!/usr/bin/env python

import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_qwen_max_request_module():
    module_path = Path(__file__).resolve().parents[1] / "cqy" / "qwen" / "qwen_max_request.py"
    assert module_path.is_file(), f"Expected helper script to exist at {module_path}"

    spec = importlib.util.spec_from_file_location("cqy.qwen.qwen_max_request", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_chunk(*, reasoning: str | None = None, content: str | None = None):
    delta = SimpleNamespace(reasoning_content=reasoning, content=content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def test_resolve_api_key_reads_common_env_names(monkeypatch):
    module = load_qwen_max_request_module()

    for name in ("QWEN_API_KEY", "DASHSCOPE_API_KEY", "API_KEY", "ACCESS_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError):
        module.resolve_api_key()

    monkeypatch.setenv("API_KEY", "test-api-key")
    assert module.resolve_api_key() == "test-api-key"


def test_render_stream_response_prints_reasoning_then_answer():
    module = load_qwen_max_request_module()

    stream = [
        make_chunk(reasoning="先分析一下。"),
        make_chunk(reasoning="再补充一步。"),
        make_chunk(content="我是一个助手。"),
        make_chunk(content="很高兴为你服务。"),
    ]
    buffer = io.StringIO()

    module.render_stream_response(stream, output=buffer)

    assert (
        buffer.getvalue()
        == "\n====================思考过程====================\n"
        "先分析一下。再补充一步。\n"
        "====================完整回复====================\n"
        "我是一个助手。很高兴为你服务。\n"
    )


def test_render_response_supports_sse_string_payload():
    module = load_qwen_max_request_module()

    response = (
        'data:{"choices":[{"delta":{"role":"assistant","content":"第一段回复。"}}]}\n\n'
        'data:{"choices":[{"delta":{"content":"第二段回复。"}}]}\n\n'
        "data:[DONE]\n"
    )
    buffer = io.StringIO()

    module.render_response(response, output=buffer)

    assert buffer.getvalue() == "====================完整回复====================\n第一段回复。\n第二段回复。\n"
