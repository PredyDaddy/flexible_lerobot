"""Opt-in bootstrap for the JZ Pin extensions used by ``lerobot-train``.

Python imports ``sitecustomize`` during interpreter startup.  The training
shell script exposes this directory on ``PYTHONPATH`` and sets an explicit
enable flag.  Other Python commands are therefore unaffected.
"""

from __future__ import annotations

import os


if os.environ.get("JZ_PIN_ENABLE_LEROBOT_TRAIN_HOOK") == "1":
    from jz_lerobot_train_hook import install

    install()
