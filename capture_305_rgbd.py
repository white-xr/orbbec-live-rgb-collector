#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gemini 305 RGB-D capture entry.

This is a thin launcher around orbbec_live_capture.py with config_305_rgbd.yaml.
It opens the normal RGB-D preview window and does not save immediately:

- SPACE/S: start saving
- SPACE/E: stop saving
- Q/ESC: quit safely
"""

from __future__ import annotations

import sys
from pathlib import Path

import orbbec_live_capture


ROOT = Path(__file__).resolve().parent
CONFIG_305_RGBD = ROOT / "config_305_rgbd.yaml"


def main() -> int:
    if not CONFIG_305_RGBD.exists():
        raise SystemExit(f"Config not found: {CONFIG_305_RGBD}")
    # Keep user-provided args after the default config so explicit CLI values can override it.
    sys.argv = [sys.argv[0], "--config", str(CONFIG_305_RGBD), *sys.argv[1:]]
    return orbbec_live_capture.main()


if __name__ == "__main__":
    raise SystemExit(main())
