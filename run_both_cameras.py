#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
One-command controller for two Orbbec cameras.

335L: RGB-D, config.yaml, SN CP2N1630005C
305 : Dual RGB, config_dual_rgb.yaml, SN CV2L36000024

This is software-level synchronized start/stop. It is not hardware trigger
sync, so exposure time is not guaranteed to be exactly identical.
"""

from __future__ import annotations

import argparse
import msvcrt
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CAPTURE_SCRIPT = ROOT / "orbbec_live_capture.py"
CONFIG_RGBD_335L = ROOT / "config.yaml"
CONFIG_DUAL_RGB_305 = ROOT / "config_dual_rgb.yaml"
SN_335L = "CP2N1630005C"
SN_305 = "CV2L36000024"


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(f'"{x}"' if " " in x else x for x in cmd)


def launch(
    name: str,
    config: Path,
    serial: str,
    tag: str,
    auto_start_at: float,
    stop_file: Path,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(CAPTURE_SCRIPT),
        "--config",
        str(config),
        "--serial",
        serial,
        "--tag",
        tag,
        "--auto-start",
        "--auto-start-at",
        f"{auto_start_at:.6f}",
        "--stop-file",
        str(stop_file),
    ]
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Start {name}:")
    print("  " + quote_cmd(cmd))
    return subprocess.Popen(cmd, cwd=str(ROOT), creationflags=subprocess.CREATE_NEW_CONSOLE)


def wait_for_control_key(stop_file: Path, procs: list[subprocess.Popen]) -> None:
    print()
    print("============================================================")
    print("Controller")
    print("  E: stop saving BOTH cameras")
    print("  Q: stop saving BOTH cameras and close this controller")
    print("  This controller window must stay open while recording.")
    print("============================================================")
    while True:
        alive = any(p.poll() is None for p in procs)
        if not alive:
            print("Both capture processes have exited.")
            return

        if msvcrt.kbhit():
            key = msvcrt.getwch().lower()
            if key in ("e", "q"):
                stop_file.write_text(f"stop requested at {datetime.now():%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Stop signal written: {stop_file}")
                if key == "q":
                    return
            else:
                print(f"Unknown key: {key}. Use E or Q.")
        time.sleep(0.05)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start 335L RGB-D and 305 Dual RGB with one controller.")
    parser.add_argument("--delay", type=float, default=8.0, help="Seconds before both processes auto-start saving")
    parser.add_argument("--tag", default="", help="Session tag. Empty means both_YYYYmmdd_HHMMSS")
    args = parser.parse_args()

    if not CAPTURE_SCRIPT.exists():
        raise SystemExit(f"Capture script not found: {CAPTURE_SCRIPT}")
    if not CONFIG_RGBD_335L.exists():
        raise SystemExit(f"335L RGB-D config not found: {CONFIG_RGBD_335L}")
    if not CONFIG_DUAL_RGB_305.exists():
        raise SystemExit(f"305 Dual RGB config not found: {CONFIG_DUAL_RGB_305}")

    tag = args.tag.strip() or f"both_{datetime.now():%Y%m%d_%H%M%S}"
    auto_start_at = time.time() + max(1.0, float(args.delay))
    stop_file = ROOT / "captures" / f"{tag}_STOP.flag"
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    if stop_file.exists():
        stop_file.unlink()

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Shared software start time: {auto_start_at:.6f}")
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Session tag: {tag}")
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Shared stop file: {stop_file}")
    print("Note: this is software synchronized start/stop, not hardware exposure sync.")

    procs = [
        launch("335L RGB-D", CONFIG_RGBD_335L, SN_335L, f"{tag}_335L_rgbd", auto_start_at, stop_file),
        launch("305 Dual RGB", CONFIG_DUAL_RGB_305, SN_305, f"{tag}_305_dual_rgb", auto_start_at, stop_file),
    ]

    wait_for_control_key(stop_file, procs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
