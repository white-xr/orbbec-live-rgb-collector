#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Generic Orbbec RGB dataset collector for YOLO detection/segmentation images.

Examples:
python D:\OrbbecLiveCollector\scripts\collect_orbbec_rgb_dataset.py --camera 335L --task coarse
python D:\OrbbecLiveCollector\scripts\collect_orbbec_rgb_dataset.py --camera 305 --task precise
python D:\OrbbecLiveCollector\scripts\collect_orbbec_rgb_dataset.py --camera 305 --task precise --width 1280 --height 720 --fps 30
"""

from __future__ import annotations

import argparse
import base64
import csv
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

import orbbec_live_capture as cap


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_SDK_BIN = Path(r"D:\OrbbecSDK_v2\bin")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "captures" / "rgb_dataset"
WINDOW_NAME = "Orbbec RGB dataset collector"

CAMERA_DEFAULTS = {
    "335L": {
        "task": "coarse",
        "name_hint": "335",
    },
    "305": {
        "task": "precise",
        "name_hint": "305",
    },
}

DEFAULT_COLOR_FORMATS = ["BGR", "RGB", "MJPG", "YUYV", "BGRA", "RGBA", "UYVY"]
DEFAULT_COLOR_FORMAT_IDS = [
    cap.OB_FORMAT_BGR,
    cap.OB_FORMAT_RGB,
    cap.OB_FORMAT_MJPG,
    cap.OB_FORMAT_YUYV,
    cap.OB_FORMAT_BGRA,
    cap.OB_FORMAT_RGBA,
    cap.OB_FORMAT_UYVY,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Orbbec RGB PNG images for YOLO datasets.")
    parser.add_argument("--camera", required=True, choices=sorted(CAMERA_DEFAULTS), help="Camera model: 335L or 305.")
    parser.add_argument("--task", required=True, choices=["coarse", "precise"], help="Dataset task: coarse or precise.")
    parser.add_argument("--width", type=int, default=1280, help="Requested RGB width.")
    parser.add_argument("--height", type=int, default=800, help="Requested RGB height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested RGB FPS.")
    parser.add_argument("--auto-interval", type=float, default=1.0, help="Seconds between auto saves.")
    parser.add_argument("--sdk-bin", default=str(DEFAULT_SDK_BIN), help="Folder containing OrbbecSDK.dll.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root folder for dataset sessions.")
    parser.add_argument("--session", default="", help="Session folder token; default is current time.")
    parser.add_argument(
        "--serial",
        default="",
        help="Override camera serial. Empty/auto selects by camera model name; any selects the first device.",
    )
    parser.add_argument("--device-index", type=int, default=None, help="Select by Orbbec device index instead of serial.")
    parser.add_argument("--preset", default="Default", help="Device preset to load before starting COLOR stream.")
    parser.add_argument("--formats", nargs="+", default=DEFAULT_COLOR_FORMATS, help="Preferred COLOR formats.")
    parser.add_argument("--png-compression", type=int, default=3, help="PNG compression 0..9.")
    parser.add_argument("--start-auto", action="store_true", help="Use auto save mode; press Space/S to start or stop.")
    parser.add_argument("--no-preview", action="store_true", help="Run without preview; useful for non-interactive tests only.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    expected_task = CAMERA_DEFAULTS[args.camera]["task"]
    if args.task != expected_task:
        raise ValueError(
            f"Invalid camera/task combination: {args.camera} + {args.task}. "
            f"Use {args.camera} + {expected_task}."
        )
    if int(args.width) <= 0 or int(args.height) <= 0 or int(args.fps) <= 0:
        raise ValueError("--width, --height and --fps must be positive.")
    if float(args.auto_interval) <= 0:
        raise ValueError("--auto-interval must be > 0.")
    if str(args.preset or "").strip() == cap.DUAL_COLOR_PRESET_NAME:
        raise ValueError(
            "Dual Color Streams exposes COLOR_LEFT/COLOR_RIGHT only. "
            "Use Default/High Accuracy for YOLO RGB dataset collection, "
            "or switch to the 305 dual RGB collector."
        )


def sanitize_token(value: str, fallback: str) -> str:
    text = (value or "").strip() or fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._-") or fallback


def make_session_dir(output_root: Path, camera: str, task: str, session: str) -> Path:
    base = output_root / f"{camera}_{task}_{session}"
    if not base.exists():
        return base
    index = 1
    while True:
        candidate = output_root / f"{camera}_{task}_{session}_{index:02d}"
        if not candidate.exists():
            return candidate
        index += 1


def color_profile_suggestions(profiles: list[dict[str, Any]], max_items: int = 8) -> str:
    seen: set[tuple[int, int, int]] = set()
    suggestions: list[str] = []
    for prof in sorted(
        profiles,
        key=lambda p: (-int(p["width"]) * int(p["height"]), -int(p["fps"]), int(p["width"])),
    ):
        key = (int(prof["width"]), int(prof["height"]), int(prof["fps"]))
        if key in seen:
            continue
        seen.add(key)
        suggestions.append(f"--width {key[0]} --height {key[1]} --fps {key[2]}")
        if len(suggestions) >= max_items:
            break
    return "; ".join(suggestions) if suggestions else "none"


def select_color_profile(
    sdk: cap.SDK,
    pipe,
    width: int,
    height: int,
    fps: int,
    fmt_names: list[str],
) -> dict[str, Any]:
    profiles = sdk.list_video_stream_profiles(pipe, cap.OB_SENSOR_COLOR)
    fmt_ids = cap.format_candidates_from_config(fmt_names, DEFAULT_COLOR_FORMAT_IDS)
    profile = cap.choose_profile_from_config(
        profiles,
        {"width": width, "height": height, "fps": fps, "formats": fmt_names},
        fmt_ids,
    )
    if profile is not None:
        return profile

    print(f"[{cap.now_str()}] Requested COLOR profile is not available: {width}x{height}@{fps}")
    print(f"[{cap.now_str()}] Available COLOR profiles: {cap.summarize_profiles(profiles, limit=64)}")
    print(f"[{cap.now_str()}] Try one of these arguments: {color_profile_suggestions(profiles)}")
    raise RuntimeError(f"No COLOR {width}x{height}@{fps} profile found.")


def maybe_switch_device_preset(sdk: cap.SDK, dev, device_name: str, preset_name: str) -> None:
    target = str(preset_name or "").strip()
    if not target:
        print(f"[{cap.now_str()}] Device preset switch skipped.")
        return
    settings = {
        "device_preset": {"enabled": True, "name": target, "required": False, "settle_ms": 800},
        "streams": {"color": True, "depth": False, "color_left": False, "color_right": False},
    }
    try:
        cap.switch_device_preset_if_configured(sdk, dev, device_name, settings)
    except Exception as ex:
        print(f"[{cap.now_str()}] WARN preset switch to {target} failed; continue with current preset: {ex}")


def select_device_for_camera(
    sdk: cap.SDK,
    dl,
    camera: str,
    serial_arg: str,
    device_index: int | None,
):
    serial = str(serial_arg or "").strip()
    serial_mode = serial.lower()

    if serial and serial_mode not in ("auto", "none", "any"):
        return cap.select_device(sdk, dl, serial, device_index)

    if serial_mode == "any" or device_index is not None:
        return cap.select_device(sdk, dl, "", device_index)

    count = sdk.device_count(dl)
    if count < 1:
        raise RuntimeError("No Orbbec device found.")

    hint = str(CAMERA_DEFAULTS[camera]["name_hint"]).lower()
    print(f"[{cap.now_str()}] Found {count} Orbbec device(s).")
    selected_dev = 0
    selected_sn = ""
    selected_name = ""
    for idx in range(count):
        dev = 0
        try:
            dev = sdk.get_device(dl, idx)
            sn, name = sdk.get_device_info(dev)
            print(f"[{cap.now_str()}] Device[{idx}]: {name}, SN: {sn}")
            if hint in name.lower():
                selected_dev = dev
                selected_sn = sn
                selected_name = name
                dev = 0
                break
        finally:
            if dev:
                sdk.delete_device(dev)

    if not selected_dev:
        raise RuntimeError(
            f"No Orbbec device name matched camera {camera}. "
            f"Use --serial with the printed SN, or use --serial any to select the first device."
        )

    print(f"[{cap.now_str()}] Auto-selected {camera} by model name. SN: {selected_sn}")
    return selected_dev, selected_sn, selected_name


def draw_overlay(
    frame,
    camera: str,
    task: str,
    width: int,
    height: int,
    fps: int,
    measured_fps: float,
    auto_mode: bool,
    auto_running: bool,
    auto_interval: float,
    saved_count: int,
):
    view = frame.copy()
    lines = [
        f"camera: {camera}",
        f"task: {task}",
        f"resolution: {width}x{height}",
        f"fps: {fps} target / {measured_fps:.1f} live",
        f"save mode: {'auto' if auto_mode else 'manual'}",
        f"auto status: {'running' if auto_running else 'idle'}",
        f"auto interval: {auto_interval:g}s",
        f"saved count: {saved_count}",
    ]

    panel_w = min(view.shape[1], 430)
    panel_h = 24 + len(lines) * 24
    overlay = view.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, view, 0.38, 0, view)
    y = 28
    for line in lines:
        cv2.putText(view, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return view


class PreviewWindow:
    """Preview abstraction with an OpenCV window first and a Tkinter fallback."""

    def __init__(self, title: str, width: int, height: int) -> None:
        self.title = title
        self.mode = "opencv"
        self.key = 255
        self.closed = False
        self.tk = None
        self.root = None
        self.label = None
        self.photo = None

        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, min(width, 1280), min(height, 800))
            print(f"[{cap.now_str()}] Preview backend: OpenCV HighGUI")
        except Exception as ex:
            print(f"[{cap.now_str()}] WARN OpenCV preview unavailable: {ex}")
            print(f"[{cap.now_str()}] Preview backend: Tkinter fallback")
            self._init_tkinter(title)

    def _init_tkinter(self, title: str) -> None:
        import tkinter as tk

        self.mode = "tkinter"
        self.tk = tk
        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Key>", self._on_key)
        self.label = tk.Label(self.root, bd=0)
        self.label.pack()

    def _on_key(self, event) -> None:
        if event.keysym == "Escape":
            self.key = 27
            return
        if event.keysym == "space":
            self.key = 32
            return
        if event.char:
            self.key = ord(event.char[0])

    def _on_close(self) -> None:
        self.closed = True
        self.key = ord("q")

    def show(self, frame) -> int:
        if self.closed:
            return ord("q")

        if self.mode == "opencv":
            try:
                cv2.imshow(self.title, frame)
                return cv2.waitKey(1) & 0xFF
            except Exception as ex:
                print(f"[{cap.now_str()}] WARN OpenCV preview failed while showing frame: {ex}")
                print(f"[{cap.now_str()}] Preview backend: Tkinter fallback")
                self._init_tkinter(self.title)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        ppm = f"P6\n{w} {h}\n255\n".encode("ascii") + rgb.tobytes()
        payload = base64.b64encode(ppm).decode("ascii")
        self.photo = self.tk.PhotoImage(data=payload, format="PPM")
        self.label.configure(image=self.photo)
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self.closed = True
            return ord("q")

        key = self.key
        self.key = 255
        return key

    def close(self) -> None:
        if self.mode == "opencv":
            try:
                cv2.destroyWindow(self.title)
            except Exception:
                pass
            return

        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass


def open_metadata_csv(session_dir: Path):
    path = session_dir / "metadata.csv"
    file_obj = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        file_obj,
        fieldnames=["filename", "timestamp", "camera", "task", "width", "height", "fps", "save_mode"],
    )
    writer.writeheader()
    file_obj.flush()
    return file_obj, writer


def save_rgb_png(
    session_dir: Path,
    metadata_writer: csv.DictWriter,
    metadata_file,
    image,
    camera: str,
    task: str,
    index: int,
    width: int,
    height: int,
    fps: int,
    save_mode: str,
) -> str:
    filename = f"{index:06d}.png"
    path = session_dir / filename
    cap.write_png_file(path, image)
    metadata_writer.writerow(
        {
            "filename": filename,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "camera": camera,
            "task": task,
            "width": int(width),
            "height": int(height),
            "fps": int(fps),
            "save_mode": save_mode,
        }
    )
    metadata_file.flush()
    return filename


def cleanup_sdk(sdk: cap.SDK | None, ctx, dl, dev, pipe, cfg) -> None:
    if sdk is None:
        return
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


def main() -> int:
    args = parse_args()
    metadata_file = None
    sdk: cap.SDK | None = None
    ctx = dl = dev = pipe = cfg = 0
    preview_window: PreviewWindow | None = None

    try:
        validate_args(args)
        cap.PNG_COMPRESSION = int(max(0, min(9, int(args.png_compression))))

        session = sanitize_token(args.session, datetime.now().strftime("%Y%m%d_%H%M%S"))
        output_root = Path(args.output_root).expanduser().resolve()
        session_dir = make_session_dir(output_root, args.camera, args.task, session)
        session_dir.mkdir(parents=True, exist_ok=True)
        metadata_file, metadata_writer = open_metadata_csv(session_dir)

        sdk = cap.SDK(Path(args.sdk_bin))
        print(f"[{cap.now_str()}] Orbbec SDK version: {sdk.get_sdk_version_text()}")
        ctx = sdk.create_context()
        dl = sdk.query_device_list(ctx)
        dev, sn, device_name = select_device_for_camera(sdk, dl, args.camera, args.serial, args.device_index)
        print(f"[{cap.now_str()}] Device: {device_name}, SN: {sn}")

        name_hint = str(CAMERA_DEFAULTS[args.camera]["name_hint"]).lower()
        if name_hint not in device_name.lower():
            print(f"[{cap.now_str()}] WARN selected device name does not look like Gemini {args.camera}: {device_name}")

        maybe_switch_device_preset(sdk, dev, device_name, args.preset)

        pipe = sdk.create_pipeline(dev)
        profile = select_color_profile(
            sdk,
            pipe,
            int(args.width),
            int(args.height),
            int(args.fps),
            [str(item).upper() for item in args.formats],
        )
        cfg = sdk.create_config()
        sdk.enable_video_stream(
            cfg,
            cap.OB_STREAM_COLOR,
            int(profile["width"]),
            int(profile["height"]),
            int(profile["fps"]),
            int(profile["format"]),
        )
        sdk.set_aggregate_all_type(cfg)
        sdk.start_pipeline(pipe, cfg)

        actual_width = int(profile["width"])
        actual_height = int(profile["height"])
        actual_fps = int(profile["fps"])
        print(
            f"[{cap.now_str()}] Started COLOR: {actual_width}x{actual_height}@{actual_fps} "
            f"{cap.format_name(profile['format'])}"
        )
        print(f"[{cap.now_str()}] Save dir: {session_dir}")
        print("Keys: A=manual/auto mode | S/SPACE=save once or start/stop auto | Q/ESC=quit")

        if not args.no_preview:
            preview_window = PreviewWindow(WINDOW_NAME, actual_width, actual_height)

        auto_mode = bool(args.start_auto)
        auto_running = bool(args.no_preview and args.start_auto)
        saved_count = 0
        last_auto_save = time.perf_counter() - float(args.auto_interval)
        measured_fps = 0.0
        fps_frames = 0
        fps_t0 = time.perf_counter()

        while True:
            fs = sdk.wait_frameset(pipe, 200)
            if not fs:
                continue

            frame = 0
            fd = None
            image = None
            try:
                frame = sdk.get_optional_frame(fs, cap.OB_FRAME_COLOR)
                if not frame:
                    continue
                fd = sdk.extract(frame)
                image = cap.decode_color(fd)
            finally:
                if frame:
                    sdk.delete_frame(frame)
                sdk.delete_frame(fs)

            if image is None:
                print(f"[{cap.now_str()}] WARN failed to decode COLOR frame")
                continue

            now_perf = time.perf_counter()
            fps_frames += 1
            if now_perf - fps_t0 >= 1.0:
                measured_fps = fps_frames / (now_perf - fps_t0)
                fps_frames = 0
                fps_t0 = now_perf

            manual_save = False
            if not args.no_preview:
                preview = draw_overlay(
                    image,
                    args.camera,
                    args.task,
                    int(fd.width) if fd else actual_width,
                    int(fd.height) if fd else actual_height,
                    actual_fps,
                    measured_fps,
                    auto_mode,
                    auto_running,
                    float(args.auto_interval),
                    saved_count,
                )
                key = preview_window.show(preview)
            else:
                key = 255

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("a"), ord("A")):
                auto_mode = not auto_mode
                auto_running = False
                last_auto_save = now_perf - float(args.auto_interval)
                print(f"[{cap.now_str()}] save mode = {'auto' if auto_mode else 'manual'}")
            if key in (ord("s"), ord("S"), 32):
                if auto_mode:
                    auto_running = not auto_running
                    last_auto_save = now_perf - float(args.auto_interval)
                    print(f"[{cap.now_str()}] auto save {'START' if auto_running else 'STOP'}")
                else:
                    manual_save = True

            auto_due = auto_mode and auto_running and (now_perf - last_auto_save) >= float(args.auto_interval)
            save_mode = ""
            if manual_save:
                save_mode = "manual"
            elif auto_due:
                save_mode = "auto"

            if save_mode:
                saved_count += 1
                filename = save_rgb_png(
                    session_dir,
                    metadata_writer,
                    metadata_file,
                    image,
                    args.camera,
                    args.task,
                    saved_count,
                    int(fd.width) if fd else actual_width,
                    int(fd.height) if fd else actual_height,
                    actual_fps,
                    save_mode,
                )
                if auto_mode:
                    last_auto_save = now_perf
                print(f"[{cap.now_str()}] saved {filename} ({save_mode})")

            if args.no_preview and not auto_running:
                print(f"[{cap.now_str()}] --no-preview is idle; exiting.")
                break

        print(f"[{cap.now_str()}] Exit. Saved count: {saved_count}")
        return 0
    except KeyboardInterrupt:
        print(f"[{cap.now_str()}] Interrupted by user.")
        return 0
    except Exception as ex:
        print(f"[ERROR] {ex}")
        return 1
    finally:
        if metadata_file is not None:
            metadata_file.close()
        cleanup_sdk(sdk, ctx, dl, dev, pipe, cfg)
        if preview_window is not None:
            preview_window.close()


if __name__ == "__main__":
    raise SystemExit(main())
