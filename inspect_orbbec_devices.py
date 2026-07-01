#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enumerate Orbbec devices, SDK information, USB connection type, presets, and stream profiles.

Usage:
python inspect_orbbec_devices.py
python inspect_orbbec_devices.py --sdk-bin D:\OrbbecSDK_v2\bin
"""

from __future__ import annotations

import argparse
import ctypes
import os
import string
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
DEFAULT_SDK_BIN = Path(r"D:\OrbbecSDK_v2\bin")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import orbbec_live_capture as cap  # noqa: E402


SENSOR_ITEMS = [
    ("COLOR", cap.OB_SENSOR_COLOR),
    ("COLOR_LEFT", cap.OB_SENSOR_COLOR_LEFT),
    ("COLOR_RIGHT", cap.OB_SENSOR_COLOR_RIGHT),
    ("DEPTH", cap.OB_SENSOR_DEPTH),
    ("IR_LEFT", cap.OB_SENSOR_IR_LEFT),
    ("IR_RIGHT", cap.OB_SENSOR_IR_RIGHT),
    ("IR", cap.OB_SENSOR_IR),
]


def read_c_string(value: bytes | None) -> str:
    return value.decode("utf-8", errors="ignore") if value else ""


def printable_text(value: str) -> str:
    if not value:
        return ""
    allowed = set(string.printable)
    cleaned = "".join(ch if ch in allowed and ch not in "\r\n\t\x0b\x0c" else "" for ch in value)
    return cleaned.strip()


def optional_device_info_value(sdk: cap.SDK, info, fn_name: str, restype, hex_value: bool = False) -> str:
    fn = getattr(sdk.lib, fn_name, None)
    if fn is None:
        return "SDK未导出"
    try:
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        fn.restype = restype
        err = ctypes.c_void_p()
        value = fn(info, ctypes.byref(err))
        if err:
            message = sdk._err_msg(err)
            return f"读取失败: {message}"
        if restype is ctypes.c_char_p:
            return printable_text(read_c_string(value))
        if hex_value:
            return f"0x{int(value):04X}"
        return str(int(value))
    except Exception as ex:
        return f"读取失败: {ex}"


def get_extended_device_info(sdk: cap.SDK, dev) -> dict[str, str]:
    err = ctypes.c_void_p()
    info = sdk.lib.ob_device_get_device_info(dev, ctypes.byref(err))
    sdk._check(err, "ob_device_get_device_info")
    try:
        return {
            "connection_type": optional_device_info_value(
                sdk,
                info,
                "ob_device_info_get_connection_type",
                ctypes.c_char_p,
            ),
            "vid": optional_device_info_value(sdk, info, "ob_device_info_get_vid", ctypes.c_int, hex_value=True),
            "pid": optional_device_info_value(sdk, info, "ob_device_info_get_pid", ctypes.c_int, hex_value=True),
            "uid": optional_device_info_value(sdk, info, "ob_device_info_get_uid", ctypes.c_char_p),
        }
    finally:
        err2 = ctypes.c_void_p()
        sdk.lib.ob_delete_device_info(info, ctypes.byref(err2))
        if err2:
            sdk._err_msg(err2)


def classify_usb(connection_type: str) -> str:
    text = (connection_type or "").upper()
    if "USB3.2" in text:
        return "USB3.2"
    if "USB3.1" in text:
        return "USB3.1"
    if "USB3.0" in text:
        return "USB3.0"
    if "USB2.1" in text:
        return "USB2.1"
    if "USB2.0" in text:
        return "USB2.0"
    if "USB" in text:
        return connection_type or "USB"
    return connection_type or "unknown"


def profile_lines(sdk: cap.SDK, pipe, sensor_label: str, sensor_type: int, show_full: bool) -> list[str]:
    try:
        profiles = sdk.list_video_stream_profiles(pipe, sensor_type)
    except Exception as ex:
        return [f"  {sensor_label}: unavailable ({ex})"]

    if not profiles:
        return [f"  {sensor_label}: none"]

    lines = [f"  {sensor_label}: {cap.summarize_profiles(profiles, limit=128)}"]
    if show_full:
        for prof in sorted(
            profiles,
            key=lambda p: (
                -int(p["width"]) * int(p["height"]),
                -int(p["width"]),
                -int(p["height"]),
                -int(p["fps"]),
                cap.format_name(int(p["format"])),
            ),
        ):
            lines.append(
                "    - "
                f'{int(prof["width"])}x{int(prof["height"])}@{int(prof["fps"])} '
                f'{cap.format_name(int(prof["format"]))}'
            )
    return lines


def print_device_report(sdk: cap.SDK, dev, index: int, show_full_profiles: bool) -> None:
    basic = sdk.get_device_info_detail(dev)
    extra = get_extended_device_info(sdk, dev)
    name = basic.get("name", "UNKNOWN_DEVICE")
    sn = basic.get("serial_number", "UNKNOWN_SN")
    connection = extra.get("connection_type", "")

    print("")
    print(f"Device[{index}]")
    print(f"  Name          : {name}")
    print(f"  SN            : {sn}")
    print(f"  USB/Connection: {classify_usb(connection)}")
    print(f"  Raw connection: {connection or 'unknown'}")
    print(f"  VID/PID       : {extra.get('vid', '')} / {extra.get('pid', '')}")
    print(f"  UID           : {extra.get('uid', '') or 'unknown'}")
    print(f"  Firmware      : {basic.get('firmware_version', '') or 'unknown'}")
    print(f"  Hardware      : {basic.get('hardware_version', '') or 'unknown'}")
    print(f"  Min SDK       : {basic.get('supported_min_sdk_version', '') or 'unknown'}")

    try:
        current_preset = sdk.get_current_preset_name(dev) or "unknown"
        presets = sdk.get_available_presets(dev)
        print(f"  Current mode  : {current_preset}")
        print(f"  All modes     : {', '.join(presets) if presets else 'none'}")
    except Exception as ex:
        print(f"  Preset modes  : unavailable ({ex})")

    pipe = 0
    print("  Stream profiles:")
    try:
        pipe = sdk.create_pipeline(dev)
        for sensor_label, sensor_type in SENSOR_ITEMS:
            for line in profile_lines(sdk, pipe, sensor_label, sensor_type, show_full_profiles):
                print(line)
    except Exception as ex:
        print(f"  Stream profiles unavailable: {ex}")
    finally:
        if pipe:
            sdk.delete_pipeline(pipe)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Orbbec SDK/device/mode information.")
    parser.add_argument("--sdk-bin", default=str(DEFAULT_SDK_BIN), help="Folder containing OrbbecSDK.dll")
    parser.add_argument(
        "--full-profiles",
        action="store_true",
        help="Print every profile row instead of only grouped summaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sdk_bin = Path(args.sdk_bin).expanduser().resolve()
    dll_path = sdk_bin / "OrbbecSDK.dll"

    print("Orbbec device inspector")
    print(f"Working dir   : {ROOT}")
    print(f"Python        : {sys.executable}")
    print(f"SDK bin       : {sdk_bin}")
    print(f"SDK dll       : {dll_path}")

    sdk = cap.SDK(sdk_bin)
    print(f"SDK version   : {sdk.get_sdk_version_text()}")
    print(f"Process cwd   : {Path.cwd()}")
    print(f"PATH head     : {os.environ.get('PATH', '').split(os.pathsep)[0]}")

    ctx = 0
    dl = 0
    devices: list[Any] = []
    try:
        ctx = sdk.create_context()
        dl = sdk.query_device_list(ctx)
        count = sdk.device_count(dl)
        print(f"Device count  : {count}")
        if count <= 0:
            print("No Orbbec device found.")
            return 1

        for idx in range(count):
            dev = sdk.get_device(dl, idx)
            devices.append(dev)
            print_device_report(sdk, dev, idx, bool(args.full_profiles))
        return 0
    finally:
        for dev in devices:
            sdk.delete_device(dev)
        if dl:
            sdk.delete_device_list(dl)
        if ctx:
            sdk.delete_context(ctx)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
    except Exception as ex:
        print(f"[ERROR] {ex}")
        raise SystemExit(1)
