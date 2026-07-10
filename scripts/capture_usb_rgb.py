#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""USB RGB camera preview and dataset capture.

This script is for normal UVC cameras, not Orbbec SDK devices.
Default profile is tuned for DECXIN CAMERA (1bcf:2cd1): 1920x1080@60 MJPG.
"""

from __future__ import annotations

import argparse
import csv
import queue
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


BACKENDS = {
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "any": cv2.CAP_ANY,
}

STOP_REQUESTED = False


def request_stop(_signum=None, _frame=None) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def sanitize_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    bad = '<>:"/\\|?*'
    for ch in bad:
        text = text.replace(ch, "_")
    return "_".join(text.split()).strip(" ._")


def unique_session_dir(root: Path, tag: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base_name = sanitize_name(tag) or f"usb_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base = root / base_name
    if not base.exists():
        return base
    idx = 1
    while True:
        candidate = root / f"{base.name}_{idx:02d}"
        if not candidate.exists():
            return candidate
        idx += 1


def fourcc_text(value: float) -> str:
    code = int(value)
    chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
    return "".join(ch if 32 <= ord(ch) <= 126 else "." for ch in chars)


def frame_stats(frame: np.ndarray) -> str:
    return (
        f"shape={frame.shape}, mean={float(np.mean(frame)):.1f}, "
        f"min={int(np.min(frame))}, max={int(np.max(frame))}"
    )


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    backend = BACKENDS[str(args.backend)]
    cap = cv2.VideoCapture(int(args.index), backend)
    if str(args.fourcc).strip():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(args.fourcc).strip().upper()))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    cap.set(cv2.CAP_PROP_FPS, int(args.fps))

    # Keep camera-side auto behavior enabled by default. DirectShow drivers may ignore these.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(args.auto_exposure))
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    if args.apply_controls:
        cap.set(cv2.CAP_PROP_EXPOSURE, float(args.exposure))
        cap.set(cv2.CAP_PROP_GAIN, float(args.gain))
        cap.set(cv2.CAP_PROP_BRIGHTNESS, float(args.brightness))
        cap.set(cv2.CAP_PROP_CONTRAST, float(args.contrast))
    if args.settings:
        cap.set(cv2.CAP_PROP_SETTINGS, 1)
    return cap


def auto_brightness(frame: np.ndarray, target_mean: float) -> np.ndarray:
    mean = float(np.mean(frame))
    if mean <= 1.0:
        return frame
    gain = float(np.clip(float(target_mean) / mean, 1.0, 4.0))
    return cv2.convertScaleAbs(frame, alpha=gain, beta=0)


def prepare_preview(frame: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    shown = frame if args.no_enhance else auto_brightness(frame, float(args.enhance_target))
    scale = float(args.preview_scale)
    if 0 < scale < 1.0:
        shown = cv2.resize(shown, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return shown


def prepare_save_frame(frame: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.save_enhanced:
        return auto_brightness(frame, float(args.enhance_target))
    return frame


def draw_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    width = out.shape[1]
    height = min(out.shape[0], 30 + 26 * len(lines))
    cv2.rectangle(out, (0, 0), (width, height), (0, 0, 0), -1)
    y = 26
    for line in lines:
        cv2.putText(out, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        y += 26
    return out


def write_image(path: Path, frame: np.ndarray, image_format: str, jpg_quality: int) -> None:
    image_format = image_format.lower().lstrip(".")
    if image_format in {"jpg", "jpeg"}:
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)]
    elif image_format == "png":
        ext = ".png"
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    elif image_format == "bmp":
        ext = ".bmp"
        params = []
    else:
        raise ValueError(f"Unsupported image format: {image_format}")
    ok, encoded = cv2.imencode(ext, frame, params)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    path.write_bytes(encoded.tobytes())


class AsyncImageWriter:
    def __init__(self, color_dir: Path, args: argparse.Namespace) -> None:
        self.color_dir = color_dir
        self.args = args
        self.queue: queue.Queue | None = queue.Queue(maxsize=max(1, int(args.write_queue_size)))
        self.threads: list[threading.Thread] = []
        self.deferred_items: list[tuple[str, np.ndarray]] = []
        self.error: Exception | None = None
        self.error_lock = threading.Lock()
        if not args.defer_write_until_stop:
            for idx in range(max(1, int(args.writer_threads))):
                thread = threading.Thread(target=self._worker, name=f"UsbImageWriter-{idx + 1}", daemon=True)
                thread.start()
                self.threads.append(thread)

    def _worker(self) -> None:
        assert self.queue is not None
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                filename, frame = item
                write_image(self.color_dir / filename, frame, self.args.image_format, self.args.jpg_quality)
            except Exception as ex:
                with self.error_lock:
                    if self.error is None:
                        self.error = ex
            finally:
                self.queue.task_done()

    def raise_if_failed(self) -> None:
        with self.error_lock:
            if self.error is not None:
                raise RuntimeError(f"Background image writer failed: {self.error}")

    def enqueue(self, filename: str, frame: np.ndarray) -> bool:
        self.raise_if_failed()
        if self.args.defer_write_until_stop:
            if self.args.drop_when_full and len(self.deferred_items) >= int(self.args.write_queue_size):
                return False
            self.deferred_items.append((filename, frame.copy()))
            return True
        assert self.queue is not None
        item = (filename, frame.copy())
        if self.args.drop_when_full:
            try:
                self.queue.put_nowait(item)
                return True
            except queue.Full:
                return False
        self.queue.put(item)
        return True

    def close(self) -> None:
        if self.args.defer_write_until_stop:
            for filename, frame in self.deferred_items:
                write_image(self.color_dir / filename, frame, self.args.image_format, self.args.jpg_quality)
            self.deferred_items.clear()
            self.raise_if_failed()
            return
        if self.queue is None:
            return
        self.queue.join()
        for _ in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join(timeout=5.0)
        self.raise_if_failed()
        self.queue = None

    def pending(self) -> int:
        if self.args.defer_write_until_stop:
            return len(self.deferred_items)
        return 0 if self.queue is None else int(self.queue.qsize())


def start_session(args: argparse.Namespace) -> tuple[Path, Path, object, csv.writer, AsyncImageWriter]:
    session_dir = unique_session_dir(Path(args.output_root), str(args.tag or ""))
    color_dir = session_dir / "color"
    color_dir.mkdir(parents=True, exist_ok=True)
    ts_file = (session_dir / "timestamps.csv").open("w", encoding="utf-8", newline="")
    writer = csv.writer(ts_file)
    writer.writerow(["frame_id", "timestamp_s", "system_time_s", "filename"])
    image_writer = AsyncImageWriter(color_dir, args)
    print(f"[INFO] session started: {session_dir}")
    if args.defer_write_until_stop:
        print("[INFO] writer mode: defer image encoding/writing until stop")
    else:
        print(f"[INFO] background writer: threads={args.writer_threads}, queue_size={args.write_queue_size}")
    return session_dir, color_dir, ts_file, writer, image_writer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USB RGB camera capture.")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="msmf")
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--allow-profile-fallback", action="store_true", help="Allow capture to continue if the actual camera profile differs from the requested one.")
    parser.add_argument("--output-root", default=str(Path("captures") / "usb_rgb"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--image-format", choices=["jpg", "png", "bmp"], default="jpg")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--writer-threads", type=int, default=4)
    parser.add_argument("--write-queue-size", type=int, default=256)
    parser.add_argument("--defer-write-until-stop", action="store_true")
    parser.add_argument("--drop-when-full", action="store_true")
    parser.add_argument("--preview-scale", type=float, default=0.5)
    parser.add_argument("--preview-fps", type=float, default=30.0)
    parser.add_argument("--no-enhance", action="store_true", help="Disable preview brightness lift.")
    parser.add_argument("--save-enhanced", action="store_true", help="Save the preview-enhanced frame instead of raw frame.")
    parser.add_argument("--enhance-target", type=float, default=95.0)
    parser.add_argument("--start-auto", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--settings", action="store_true", help="Open DirectShow camera settings dialog when using dshow.")
    parser.add_argument("--apply-controls", action="store_true")
    parser.add_argument("--auto-exposure", type=float, default=0.75)
    parser.add_argument("--exposure", type=float, default=-4)
    parser.add_argument("--gain", type=float, default=64)
    parser.add_argument("--brightness", type=float, default=128)
    parser.add_argument("--contrast", type=float, default=32)
    return parser.parse_args()


def profile_matches_request(actual: dict[str, float | int | str], args: argparse.Namespace) -> bool:
    actual_fps = float(actual["fps"])
    return (
        int(actual["width"]) == int(args.width)
        and int(actual["height"]) == int(args.height)
        and actual_fps >= max(1.0, float(args.fps) - 2.0)
    )


def main() -> int:
    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            signal.signal(sig, request_stop)

    args = parse_args()
    if args.settings and args.backend != "dshow":
        print("[INFO] --settings requires DirectShow; switching backend to dshow.")
        args.backend = "dshow"
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        print("[ERROR] width/height/fps must be positive.")
        return 1

    cap = open_camera(args)
    if not cap.isOpened():
        print("[ERROR] Failed to open USB camera.")
        return 1

    actual = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "fourcc": fourcc_text(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    print(f"[INFO] opened USB camera index={args.index}, backend={args.backend}, requested={args.width}x{args.height}@{args.fps} {args.fourcc}")
    print(f"[INFO] actual={actual}")
    if not args.allow_profile_fallback and not profile_matches_request(actual, args):
        cap.release()
        print(
            "[ERROR] USB camera did not open the required profile. "
            f"Required {args.width}x{args.height}@{args.fps}, actual "
            f"{actual['width']}x{actual['height']}@{actual['fps']:.1f}. "
            "Try another USB Index/backend, close other camera apps, or use the probe script."
        )
        return 1
    print("Keys: SPACE/S=start/stop saving | P=save one | Q/ESC=quit")

    capturing = bool(args.start_auto or args.no_preview)
    session_dir = color_dir = None
    ts_file = None
    ts_writer = None
    image_writer = None
    first_t = None
    saved = 0
    dropped = 0
    read_count = 0
    live_fps = 0.0
    fps_t0 = time.perf_counter()
    preview_interval = 1.0 / max(1.0, float(args.preview_fps or 30.0))
    next_preview_t = 0.0
    ts_flush_count = 0

    if capturing:
        session_dir, color_dir, ts_file, ts_writer, image_writer = start_session(args)

    try:
        while not STOP_REQUESTED:
            ok, frame = cap.read()
            now = time.perf_counter()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            read_count += 1
            if now - fps_t0 >= 1.0:
                live_fps = read_count / (now - fps_t0)
                read_count = 0
                fps_t0 = now

            save_one = False
            key = -1
            if not args.no_preview and now >= next_preview_t:
                next_preview_t = now + preview_interval
                shown = prepare_preview(frame, args)
                lines = [
                    f"USB RGB {actual['width']}x{actual['height']}@{actual['fps']:.1f} {args.fourcc}",
                    f"read fps: {live_fps:.1f} | save: {'ON' if capturing else 'OFF'} | saved: {saved} | dropped: {dropped}",
                    f"writer queue: {image_writer.pending() if image_writer else 0}",
                    f"{'enhanced preview' if not args.no_enhance else 'raw preview'} | SPACE/S=start-stop | P=one | Q=quit",
                    frame_stats(frame),
                ]
                cv2.imshow("USB RGB Capture", draw_overlay(shown, lines))
                key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (32, ord("s"), ord("S")):
                if capturing:
                    capturing = False
                    print("[INFO] saving stopped")
                    if image_writer is not None:
                        print(f"[INFO] waiting for image writer, pending={image_writer.pending()}")
                        image_writer.close()
                        image_writer = None
                    if ts_file is not None:
                        ts_file.flush()
                        ts_file.close()
                        ts_file = None
                    print(f"[INFO] session finalized: saved={saved}, dropped={dropped}, session={session_dir}")
                    session_dir = color_dir = None
                    ts_writer = None
                    first_t = None
                    saved = 0
                    dropped = 0
                    ts_flush_count = 0
                else:
                    capturing = True
                    if ts_file is None:
                        session_dir, color_dir, ts_file, ts_writer, image_writer = start_session(args)
                    print("[INFO] saving started")
            if key in (ord("p"), ord("P")):
                save_one = True

            if capturing or save_one:
                if ts_file is None:
                    session_dir, color_dir, ts_file, ts_writer, image_writer = start_session(args)
                if first_t is None:
                    first_t = now
                next_id = saved + 1
                ext = "jpg" if args.image_format == "jpg" else args.image_format
                filename = f"{next_id:06d}.{ext}"
                out_frame = prepare_save_frame(frame, args)
                if image_writer.enqueue(filename, out_frame):
                    saved = next_id
                    ts_writer.writerow([f"{saved:06d}", f"{now - first_t:.6f}", f"{time.time():.6f}", filename])
                    ts_flush_count += 1
                    if ts_flush_count >= 30:
                        ts_file.flush()
                        ts_flush_count = 0
                else:
                    dropped += 1

            if args.no_preview and not capturing:
                break
    finally:
        if image_writer is not None:
            print(f"[INFO] waiting for background writer, pending={image_writer.pending()}")
            image_writer.close()
        if ts_file is not None:
            ts_file.flush()
            ts_file.close()
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
        print(f"[INFO] saved={saved}, dropped={dropped}, session={session_dir if session_dir else '--'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
