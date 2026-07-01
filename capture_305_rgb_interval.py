#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Orbbec Gemini 305 单路 RGB 间隔采集工具。

功能：
- 打开 Gemini 305 普通 COLOR 流，默认 1280x800@30fps。
- 按 S 保存当前单张 RGB 图片。
- 按空格/R 或点击左下角按钮开始/停止间隔保存。
- 支持每 N 帧保存一张，或每 N 秒保存一张。

示例：
python D:\OrbbecLiveCollector\capture_305_rgb_interval.py
python D:\OrbbecLiveCollector\capture_305_rgb_interval.py --save-every-seconds 2 --auto
python D:\OrbbecLiveCollector\capture_305_rgb_interval.py --save-every-frames 30 --auto
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

import orbbec_live_capture as cap


ROOT = Path(__file__).resolve().parent
DEFAULT_SDK_BIN = Path(r"D:\OrbbecSDK_v2\bin")
DEFAULT_MODEL_HINT = "305"
WINDOW_NAME = "Orbbec 305 RGB interval capture"
CONTROL_BUTTON_RECT = (24, 704, 230, 780)


def make_session_dir(output_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_root / f"capture_305_rgb_{ts}"
    if not base.exists():
        return base
    i = 1
    while True:
        candidate = output_root / f"capture_305_rgb_{ts}_{i:02d}"
        if not candidate.exists():
            return candidate
        i += 1


def select_color_profile(sdk: cap.SDK, pipe, width: int, height: int, fps: int, fmt_names: list[str]) -> dict[str, Any]:
    profiles = sdk.list_video_stream_profiles(pipe, cap.OB_SENSOR_COLOR)
    fmt_ids = cap.format_candidates_from_config(fmt_names, [
        cap.OB_FORMAT_BGR,
        cap.OB_FORMAT_RGB,
        cap.OB_FORMAT_MJPG,
        cap.OB_FORMAT_YUYV,
        cap.OB_FORMAT_BGRA,
        cap.OB_FORMAT_RGBA,
        cap.OB_FORMAT_UYVY,
    ])
    cfg = {"width": width, "height": height, "fps": fps, "formats": fmt_names}
    chosen = cap.choose_profile_from_config(profiles, cfg, fmt_ids)
    if chosen is None:
        print(f"[{cap.now_str()}] Available COLOR profiles: {cap.summarize_profiles(profiles, limit=64)}")
        raise RuntimeError(f"305 没有找到 COLOR {width}x{height}@{fps} profile")
    return chosen


def on_mouse(event, x, y, flags, state):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    x1, y1, x2, y2 = CONTROL_BUTTON_RECT
    if x1 <= x <= x2 and y1 <= y <= y2:
        state["toggle"] = True


def draw_overlay(frame, saved_count: int, interval_on: bool, fps_value: float, save_every_frames: int, save_every_seconds: float, out_dir: Path):
    view = frame.copy()
    h, w = view.shape[:2]
    panel_h = 118
    cv2.rectangle(view, (0, 0), (w, panel_h), (0, 0, 0), -1)
    save_rule = "manual"
    if save_every_seconds > 0:
        save_rule = f"every {save_every_seconds:g}s"
    elif save_every_frames > 0:
        save_rule = f"every {save_every_frames} frames"
    lines = [
        f"305 RGB 1280x800@30 | FPS: {fps_value:.1f}",
        f"Status: {'SAVING' if interval_on else 'IDLE'} | Saved: {saved_count} | Rule: {save_rule}",
        "SPACE/click=START/STOP interval | S=save one | Q/ESC=quit",
        f"Session: {out_dir.name}",
    ]
    y = 24
    for line in lines:
        cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1, cv2.LINE_AA)
        y += 24

    x1, y1, x2, y2 = CONTROL_BUTTON_RECT
    button_color = (0, 80, 220) if interval_on else (0, 150, 0)
    border_color = (40, 40, 255) if interval_on else (80, 255, 80)
    label = "STOP SAVE" if interval_on else "START SAVE"
    cv2.rectangle(view, (x1, y1), (x2, y2), button_color, -1)
    cv2.rectangle(view, (x1, y1), (x2, y2), border_color, 2)
    cv2.putText(view, label, (x1 + 18, y1 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return view


def save_frame(images_dir: Path, timestamps_file, frame, frame_fd: cap.FrameData, saved_count: int, first_ts_us: int | None) -> tuple[int, int | None]:
    if first_ts_us is None:
        first_ts_us = int(frame_fd.dev_ts)
    saved_count += 1
    name = f"{saved_count:06d}.png"
    path = images_dir / name
    ok = cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise RuntimeError(f"保存失败: {path}")
    ts = max(0.0, (int(frame_fd.dev_ts) - int(first_ts_us)) / 1_000_000.0)
    frame_index = int(getattr(frame_fd, "frame_index", getattr(frame_fd, "fi", getattr(frame_fd, "index", -1))))
    timestamps_file.write(f"{name},{ts:.6f},{frame_index},{int(frame_fd.dev_ts)},{int(frame_fd.sys_ts)}\n")
    timestamps_file.flush()
    return saved_count, first_ts_us


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemini 305 单路 RGB 单张/间隔采集")
    parser.add_argument("--sdk-bin", default=str(DEFAULT_SDK_BIN), help="Orbbec SDK bin 路径")
    parser.add_argument("--serial", default="", help="305 序列号；留空则按型号自动选择")
    parser.add_argument("--model-hint", default=DEFAULT_MODEL_HINT, help="留空未指定序列号时，按设备型号名匹配")
    parser.add_argument("--output-root", default=str(ROOT / "captures"), help="保存根目录")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--formats", nargs="+", default=["BGR", "RGB", "MJPG", "YUYV"], help="优先尝试的 COLOR 格式")
    parser.add_argument("--single", action="store_true", help="启动后保存第一张 RGB 图并退出")
    parser.add_argument("--save-every-frames", type=int, default=0, help="每 N 帧保存一张；0 表示不用帧间隔自动保存")
    parser.add_argument("--save-every-seconds", type=float, default=0.0, help="每 N 秒保存一张；0 表示不用时间间隔自动保存")
    parser.add_argument("--auto", action="store_true", help="启动后自动开启间隔保存")
    parser.add_argument("--max-saves", type=int, default=0, help="最多保存多少张，0 表示不限")
    parser.add_argument("--no-preview", action="store_true", help="不显示预览窗口")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    session_dir = make_session_dir(output_root)
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    timestamps_path = session_dir / "timestamps.csv"
    metadata_path = session_dir / "metadata.json"

    sdk = cap.SDK(Path(args.sdk_bin))
    ctx = dl = dev = pipe = cfg = 0
    timestamps_file = None
    saved_count = 0
    first_ts_us: int | None = None
    frame_count = 0
    last_save_perf = 0.0
    interval_on = bool(args.auto or args.single)
    fps_count = 0
    fps_t0 = time.perf_counter()
    fps_value = 0.0

    try:
        print(f"[{cap.now_str()}] Orbbec SDK version: {sdk.get_sdk_version_text()}")
        ctx = sdk.create_context()
        dl = sdk.query_device_list(ctx)
        dev, sn, dev_name = cap.select_device(sdk, dl, str(args.serial), None, str(args.model_hint))
        print(f"[{cap.now_str()}] Device: {dev_name}, SN: {sn}")

        settings = {
            "device_preset": {"enabled": True, "name": "Default", "required": True, "settle_ms": 800},
            "streams": {"color": True, "depth": False, "color_left": False, "color_right": False},
            "color": {"auto_exposure": True, "auto_white_balance": True},
        }
        try:
            cap.switch_device_preset_if_configured(sdk, dev, dev_name, settings)
        except Exception as ex:
            print(f"[{cap.now_str()}] WARN preset switch failed, continue: {ex}")

        pipe = sdk.create_pipeline(dev)
        profile = select_color_profile(sdk, pipe, int(args.width), int(args.height), int(args.fps), list(args.formats))
        cfg = sdk.create_config()
        sdk.enable_video_stream(cfg, cap.OB_STREAM_COLOR, int(profile["width"]), int(profile["height"]), int(profile["fps"]), int(profile["format"]))
        sdk.set_aggregate_all_type(cfg)
        sdk.start_pipeline(pipe, cfg)
        print(
            f"[{cap.now_str()}] Started COLOR: {profile['width']}x{profile['height']}@{profile['fps']} "
            f"{cap.format_name(profile['format'])}"
        )

        metadata = {
            "device_name": dev_name,
            "serial_number": sn,
            "stream": "COLOR",
            "width": int(profile["width"]),
            "height": int(profile["height"]),
            "fps": int(profile["fps"]),
            "format": cap.format_name(profile["format"]),
            "save_every_frames": int(args.save_every_frames),
            "save_every_seconds": float(args.save_every_seconds),
            "manual_single_save": True,
            "session_dir": str(session_dir),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        timestamps_file = timestamps_path.open("w", encoding="utf-8", newline="")
        timestamps_file.write("file,timestamp_s,device_frame_index,device_timestamp_us,system_timestamp_us\n")

        if not args.no_preview:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 1280, 800)
            mouse_state = {"toggle": False}
            cv2.setMouseCallback(WINDOW_NAME, on_mouse, mouse_state)

        print(f"[{cap.now_str()}] Save dir: {session_dir}")
        print("Keys: SPACE/click=START/STOP interval | S=save one | Q/ESC=quit")
        if interval_on:
            print(f"[{cap.now_str()}] Interval save ON")

        while True:
            fs = sdk.wait_frameset(pipe, 200)
            if not fs:
                continue
            frame = 0
            fd = None
            img = None
            try:
                frame = sdk.get_optional_frame(fs, cap.OB_FRAME_COLOR)
                if not frame:
                    continue
                fd = sdk.extract(frame)
                img = cap.decode_color(fd)
            finally:
                if frame:
                    sdk.delete_frame(frame)
                sdk.delete_frame(fs)

            if img is None:
                print(f"[{cap.now_str()}] WARN decode color failed")
                continue

            frame_count += 1
            fps_count += 1
            now_perf = time.perf_counter()
            if now_perf - fps_t0 >= 1.0:
                fps_value = fps_count / (now_perf - fps_t0)
                fps_count = 0
                fps_t0 = now_perf

            save_req = False
            key = 255
            if not args.no_preview:
                preview = draw_overlay(img, saved_count, interval_on, fps_value, int(args.save_every_frames), float(args.save_every_seconds), session_dir)
                cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF
                mouse_toggle = bool(mouse_state.get("toggle", False))
                mouse_state["toggle"] = False
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (32, ord("r"), ord("R")) or mouse_toggle:
                    interval_on = not interval_on
                    print(f"[{cap.now_str()}] Interval save {'ON' if interval_on else 'OFF'}")
                    if interval_on and int(args.save_every_frames) <= 0 and float(args.save_every_seconds) <= 0:
                        print(f"[{cap.now_str()}] WARN interval is ON, but no --save-every-seconds/--save-every-frames was set")
                if key in (ord("s"), ord("S")):
                    save_req = True

            if args.single and saved_count == 0:
                save_req = True

            if interval_on:
                if int(args.save_every_frames) > 0 and frame_count % int(args.save_every_frames) == 0:
                    save_req = True
                if float(args.save_every_seconds) > 0 and (now_perf - last_save_perf) >= float(args.save_every_seconds):
                    save_req = True

            if save_req:
                saved_count, first_ts_us = save_frame(images_dir, timestamps_file, img, fd, saved_count, first_ts_us)
                last_save_perf = now_perf
                print(f"[{cap.now_str()}] saved {saved_count:06d}.png")
                if args.single:
                    break
                if int(args.max_saves) > 0 and saved_count >= int(args.max_saves):
                    print(f"[{cap.now_str()}] Reached max saves: {saved_count}")
                    break

            if args.no_preview and not interval_on:
                break

        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as ex:
        print(f"[ERROR] {ex}")
        return 1
    finally:
        if timestamps_file is not None:
            timestamps_file.close()
        if pipe:
            try:
                sdk.stop_pipeline(pipe)
            except Exception:
                pass
            try:
                sdk.delete_pipeline(pipe)
            except Exception:
                pass
        if cfg:
            try:
                sdk.delete_config(cfg)
            except Exception:
                pass
        if dev:
            try:
                sdk.delete_device(dev)
            except Exception:
                pass
        if dl:
            try:
                sdk.delete_device_list(dl)
            except Exception:
                pass
        if ctx:
            try:
                sdk.delete_context(ctx)
            except Exception:
                pass
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
