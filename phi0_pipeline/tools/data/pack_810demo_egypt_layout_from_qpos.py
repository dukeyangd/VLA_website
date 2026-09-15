#!/usr/bin/env python3
"""Deprecated shim → pack_teleop_qpos_unified_lerobot."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("pack_teleop_qpos_unified_lerobot.py")),
        run_name="__main__",
    )
