"""AgileX web collection backend package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_entry_module() -> ModuleType:
    module_name = f"{__name__}._entrypoint"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    module_path = Path(__file__).with_name("app.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load AgileX web app entrypoint from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_entry_module = _load_entry_module()
app = _entry_module.app
create_app = _entry_module.create_app

__all__ = ["app", "create_app"]

