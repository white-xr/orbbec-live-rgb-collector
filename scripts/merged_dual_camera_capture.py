#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-window dual Orbbec capture.

One process opens either:
- Gemini 335L RGB-D + Gemini 305 Dual RGB
- Gemini 335L RGB-D + Gemini 305 RGB-D

Keys:
- S: start saving both cameras
- E: stop saving both cameras
- Q/ESC: stop and exit
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import orbbec_live_capture as cap


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_335L_RGBD = CONFIG_DIR / "config.yaml"
CONFIG_305_DUAL_RGB = CONFIG_DIR / "config_dual_rgb.yaml"
CONFIG_305_RGBD = CONFIG_DIR / "config_305_rgbd.yaml"
MODEL_335L = "335L"
MODEL_305 = "305"
MODE_RGBD_DUAL_RGB = "rgbd-dual-rgb"
MODE_RGBD_RGBD = "rgbd-rgbd"
CONTROL_BUTTON_RECT = (24, 724, 220, 790)
ROLE_FOLDERS = {
    "335L_rgbd": "eye_to_hand_335L",
    "305_dual_rgb": "eye_in_hand_305",
    "305_rgbd": "eye_in_hand_305",
}
ROLE_TITLES = {
    "335L_rgbd": "335L RGB-D",
    "305_dual_rgb": "305 Dual RGB",
    "305_rgbd": "305 RGB-D",
}

# 启动并打开两台相机后，等待多少秒自动开始保存。0 表示手动按 S/空格/按钮。
AUTO_START_DELAY_SECONDS = 0.0
# 合并窗口只负责预览，刷新率独立于保存帧率。
PREVIEW_TARGET_FPS = 15.0
# 同步保存线程轮询两台相机最新帧的间隔；快相机多出来的帧会被最新帧覆盖。
SYNC_SAVE_POLL_SECONDS = 0.002
# 合并窗口中的 depth 伪彩色预览；RGB-D + RGB-D 模式会默认打开。
SHOW_DEPTH_PREVIEW = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-window dual Orbbec capture.")
    parser.add_argument(
        "--capture-mode",
        choices=[MODE_RGBD_DUAL_RGB, MODE_RGBD_RGBD],
        default=MODE_RGBD_DUAL_RGB,
        help="Dual capture mode.",
    )
    parser.add_argument("--sdk-bin", default=r"D:\OrbbecSDK_v2\bin", help="Folder containing OrbbecSDK.dll")
    parser.add_argument("--config-335l", default=str(CONFIG_335L_RGBD), help="335L RGB-D config YAML")
    parser.add_argument("--config-305", default="", help="305 config YAML; default depends on --capture-mode")
    parser.add_argument("--output-root", default="", help="Override capture output root for both cameras")
    parser.add_argument("--width", type=int, default=0, help="Override color/depth width for both cameras")
    parser.add_argument("--height", type=int, default=0, help="Override color/depth height for both cameras")
    parser.add_argument("--fps", type=int, default=0, help="Override color/depth FPS for both cameras")
    parser.add_argument("--preset-335l", default="", help="Override 335L device preset")
    parser.add_argument("--preset-305", default="", help="Override 305 device preset")
    parser.add_argument("--preview-fps", type=float, default=15.0, help="Merged window refresh FPS")
    parser.add_argument("--preview-every-n", type=int, default=1, help="Update RGB preview every N camera frames")
    parser.add_argument("--depth-preview-every-n", type=int, default=5, help="Update depth pseudo-color preview every N camera frames")
    parser.add_argument(
        "--show-depth-preview",
        action="store_true",
        help="Force depth pseudo-color panels; RGB-D + RGB-D mode enables them by default",
    )
    return parser.parse_args()


def apply_runtime_overrides(settings: dict[str, Any], args: argparse.Namespace) -> None:
    output_root = str(args.output_root or "").strip()
    if output_root:
        settings["output_root"] = output_root
        settings.setdefault("output", {})["base_dir"] = output_root

    width = int(args.width or 0)
    height = int(args.height or 0)
    fps = int(args.fps or 0)
    if width <= 0 and height <= 0 and fps <= 0:
        return

    stream_profile = settings.setdefault("stream_profile", {})
    for key in ("color", "depth", "dual_color"):
        profile = stream_profile.setdefault(key, {})
        if width > 0:
            profile["width"] = width
        if height > 0:
            profile["height"] = height
        if fps > 0:
            profile["fps"] = fps


def apply_preset_override(settings: dict[str, Any], preset_name: str) -> None:
    preset = str(preset_name or "").strip()
    if not preset:
        return
    preset_cfg = settings.setdefault("device_preset", {})
    preset_cfg["enabled"] = True
    preset_cfg["name"] = preset
    preset_cfg["required"] = True


def resize_panel(img: np.ndarray | None, width: int, height: int, label: str) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if img is not None and img.size > 0:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        scale = min(width / max(1, w), height / max(1, h))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        x = (width - nw) // 2
        y = (height - nh) // 2
        canvas[y : y + nh, x : x + nw] = resized
    cv2.rectangle(canvas, (0, 0), (width - 1, 28), (20, 20, 20), -1)
    cv2.putText(canvas, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def state_preview_panels(state: dict[str, Any], panel_w: int, panel_h: int) -> tuple[np.ndarray, np.ndarray]:
    role_title = ROLE_TITLES.get(str(state.get("role", "")), str(state.get("role", "")))
    streams = state.get("streams", {}) or {}
    if streams.get("color_left", False) and streams.get("color_right", False) and not streams.get("depth", False):
        left = resize_panel(state.get("left"), panel_w, panel_h, f"{role_title} Left RGB")
        right = resize_panel(state.get("right"), panel_w, panel_h, f"{role_title} Right RGB")
        return left, right

    color = resize_panel(state.get("color"), panel_w, panel_h, f"{role_title} RGB")
    depth_label = f"{role_title} Depth" if SHOW_DEPTH_PREVIEW else f"{role_title} Depth preview off"
    depth = resize_panel(state.get("depth_vis") if SHOW_DEPTH_PREVIEW else None, panel_w, panel_h, depth_label)
    return color, depth


def make_merged_preview(state_335: dict[str, Any], state_305: dict[str, Any], capturing: bool) -> np.ndarray:
    panel_w = 640
    panel_h = 400
    p1, p2 = state_preview_panels(state_335, panel_w, panel_h)
    p3, p4 = state_preview_panels(state_305, panel_w, panel_h)
    top = np.hstack([p1, p2])
    bottom = np.hstack([p3, p4])
    preview = np.vstack([top, bottom])

    sections = [
        (
            ROLE_TITLES.get(str(state_335.get("role", "")), "335L"),
            [
                f"SN: {state_335.get('sn', '--')}",
                f"FPS: {state_335.get('fps', 0.0):.1f}",
                f"Saved: {state_335.get('writer').pair_index if state_335.get('writer') else 0}",
            ],
        ),
        (
            ROLE_TITLES.get(str(state_305.get("role", "")), "305"),
            [
                f"SN: {state_305.get('sn', '--')}",
                f"FPS: {state_305.get('fps', 0.0):.1f}",
                f"Saved: {state_305.get('writer').pair_index if state_305.get('writer') else 0}",
            ],
        ),
        (
            "Capture",
            [
                f"Status: {'RUNNING' if capturing else 'IDLE'}",
                "Keys: S=start, E=stop, Q=quit",
            ],
        ),
    ]
    return cap.draw_status_sections(preview, sections)


def draw_capture_button(img: np.ndarray, capturing: bool) -> np.ndarray:
    x1, y1, x2, y2 = CONTROL_BUTTON_RECT
    y2 = min(y2, img.shape[0] - 18)
    y1 = max(16, min(y1, y2 - 66))
    bg = (0, 150, 60) if not capturing else (0, 115, 220)
    border = (210, 255, 230) if not capturing else (230, 240, 255)
    text = "START" if not capturing else "STOP"
    hint = "SPACE / Click"

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
    cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 2)
    cv2.putText(img, text, (x1 + 22, y1 + 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, hint, (x1 + 22, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    return img


def on_mouse(event: int, x: int, y: int, flags: int, param: dict[str, bool]) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    x1, y1, x2, y2 = CONTROL_BUTTON_RECT
    if x1 <= x <= x2 and y1 <= y <= y2:
        param["toggle"] = True


def make_shared_session_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = root / time.strftime("capture_%Y%m%d_%H%M%S")
    if not base.exists():
        return base
    idx = 1
    while True:
        candidate = root / f"{base.name}_{idx:02d}"
        if not candidate.exists():
            return candidate
        idx += 1


def write_shared_manifest(shared_dir: Path, states: list[dict[str, Any]]) -> None:
    lines = [
        "# merged dual camera capture",
        f"created_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "roles:",
    ]
    for state in states:
        role = state.get("role", "")
        folder = ROLE_FOLDERS.get(role, role)
        lines.extend([
            f"  - role: {role}",
            f"    folder: {folder}",
            f"    device: {state.get('name', '')}",
            f"    serial: {state.get('sn', '')}",
        ])
    (shared_dir / "session_manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def setup_camera(
    sdk: cap.SDK,
    dl,
    config_path: Path,
    model_hint: str,
    role: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    settings = cap.load_capture_config(config_path)
    apply_runtime_overrides(settings, args)
    if model_hint == MODEL_305:
        apply_preset_override(settings, str(args.preset_305 or ""))
    elif model_hint == MODEL_335L:
        apply_preset_override(settings, str(args.preset_335l or ""))
    device_cfg = settings.setdefault("device", {})
    device_cfg.setdefault("model_hint", model_hint)
    output_cfg = settings.get("output", {}) or {}
    streams_cfg = settings.get("streams", {}) or {}
    pointcloud_cfg = settings.get("pointcloud", {}) or {}

    serial = str(device_cfg.get("serial", "") or "")
    device_index = device_cfg.get("index", None)
    device_index = None if device_index is None else int(device_index)
    dev, sn, dev_name = cap.select_device(sdk, dl, serial, device_index, str(device_cfg.get("model_hint", model_hint) or ""))
    print(f"[{cap.now_str()}] {role}: Device {dev_name}, SN: {sn}")
    try:
        print(f"[{cap.now_str()}] {role}: Current preset: {sdk.get_current_preset_name(dev) or 'unknown'}")
    except Exception as ex:
        print(f"[{cap.now_str()}] WARN {role}: query preset failed: {ex}")

    preset_report = cap.switch_device_preset_if_configured(sdk, dev, dev_name, settings)
    cap.print_all_stream_profiles(sdk, dev, dev_name, f"{role} stream profiles after preset selection")
    property_report = cap.apply_camera_properties(sdk, dev, settings)
    pipe, cfg, align_name = cap.start_rgbd_pipeline(sdk, dev, dev_name, settings)

    try:
        cam_params = cap.camera_param_to_dict(sdk.get_camera_param(pipe), align_name)
    except Exception as ex:
        print(f"[{cap.now_str()}] WARN {role}: get camera params failed: {ex}")
        cam_params = {"align_mode": align_name, "camera_param_error": str(ex)}
    cam_params["device_name"] = dev_name
    cam_params["serial_number"] = sn
    cam_params["capture_config_path"] = str(config_path)
    cam_params["device_preset_report"] = preset_report
    cam_params["camera_property_apply_report"] = property_report
    cam_params["resolved_capture_config"] = settings
    cam_params = cap.apply_intrinsics_reference_fallback(
        cam_params,
        settings,
        bool(streams_cfg.get("color_left", False) and streams_cfg.get("color_right", False) and not streams_cfg.get("depth", False)),
    )

    save_color = bool(streams_cfg.get("color", True))
    save_color_left = bool(streams_cfg.get("color_left", False))
    save_color_right = bool(streams_cfg.get("color_right", False))
    save_depth = bool(streams_cfg.get("depth", True))
    save_depth_vis = bool(output_cfg.get("save_depth_vis", True))
    minimal_dual = bool(output_cfg.get("minimal_dual_rgb_layout", False) and save_color_left and save_color_right and not save_depth)

    writer = cap.SessionWriter(
        Path(settings.get("output_root", PROJECT_ROOT / "captures")),
        sn,
        role,
        target_width=0,
        target_height=0,
        align_mode_name=align_name,
        role=role,
        require_aligned_depth_to_color=bool(pointcloud_cfg.get("require_aligned_depth_to_color", True)),
        save_color=save_color,
        save_color_left=save_color_left,
        save_color_right=save_color_right,
        save_depth=save_depth,
        save_depth_vis=save_depth_vis,
        save_ir_left=False,
        save_ir_right=False,
        minimal_dual_rgb_layout=minimal_dual,
        color_format=str(output_cfg.get("color_format", "png") or "png"),
        depth_raw_format=str(output_cfg.get("depth_raw_format", "png") or "png"),
        writer_thread_count=int(output_cfg.get("writer_threads", 1) or 1),
        write_queue_maxsize=int(output_cfg.get("write_queue_maxsize", 256) or 256),
    )

    return {
        "role": role,
        "settings": settings,
        "streams": streams_cfg,
        "output": output_cfg,
        "dev": dev,
        "pipe": pipe,
        "cfg": cfg,
        "sn": sn,
        "name": dev_name,
        "align": align_name,
        "cam_params": cam_params,
        "writer": writer,
        "fps": 0.0,
        "fps_t0": time.perf_counter(),
        "fps_count": 0,
        "color": None,
        "depth_vis": None,
        "left": None,
        "right": None,
        "sync_lock": threading.Lock(),
        "latest_payload": None,
        "latest_payload_id": 0,
        "latest_saved_payload_id": 0,
        "preview_every_n": max(1, int(args.preview_every_n or 1)),
        "depth_preview_every_n": max(1, int(args.depth_preview_every_n or 5)),
        "preview_count": 0,
    }


def update_fps(state: dict[str, Any]) -> None:
    state["fps_count"] += 1
    now = time.perf_counter()
    dt = now - state["fps_t0"]
    if dt >= 1.0:
        state["fps"] = state["fps_count"] / dt
        state["fps_count"] = 0
        state["fps_t0"] = now


def reset_sync_payloads(states: list[dict[str, Any]]) -> None:
    for state in states:
        with state["sync_lock"]:
            state["latest_payload"] = None
            state["latest_payload_id"] = 0
            state["latest_saved_payload_id"] = 0


def set_latest_payload(state: dict[str, Any], payload: dict[str, Any]) -> None:
    with state["sync_lock"]:
        payload_id = int(state["latest_payload_id"]) + 1
        payload["payload_id"] = payload_id
        state["latest_payload"] = payload
        state["latest_payload_id"] = payload_id


def get_unsaved_payload(state: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    with state["sync_lock"]:
        payload = state.get("latest_payload")
        payload_id = int(state.get("latest_payload_id", 0) or 0)
        saved_payload_id = int(state.get("latest_saved_payload_id", 0) or 0)
        if payload is None or payload_id <= saved_payload_id:
            return None
        return payload_id, payload


def mark_payload_saved(state: dict[str, Any], payload_id: int) -> None:
    with state["sync_lock"]:
        state["latest_saved_payload_id"] = max(int(state.get("latest_saved_payload_id", 0) or 0), int(payload_id))


def make_save_payload(
    streams: dict[str, Any],
    color_fd,
    color_img: np.ndarray | None,
    color_left_fd,
    left_img: np.ndarray | None,
    color_right_fd,
    right_img: np.ndarray | None,
    depth_fd,
    depth_raw: np.ndarray | None,
) -> dict[str, Any] | None:
    if streams.get("color_left", False) and streams.get("color_right", False) and not streams.get("depth", False):
        if color_left_fd and color_right_fd and left_img is not None and right_img is not None:
            return {
                "kind": "dual_rgb",
                "left_fd": color_left_fd,
                "left_img": left_img,
                "right_fd": color_right_fd,
                "right_img": right_img,
            }
        return None

    if streams.get("color", False) and streams.get("depth", False):
        if color_fd and depth_fd and color_img is not None and depth_raw is not None:
            return {
                "kind": "rgbd",
                "color_fd": color_fd,
                "color_img": color_img,
                "depth_fd": depth_fd,
                "depth_raw": depth_raw,
            }
        return None

    return None


def save_payload(writer: cap.SessionWriter, payload: dict[str, Any]) -> None:
    kind = str(payload.get("kind", ""))
    if kind == "dual_rgb":
        writer.save_dual_color_pair(
            payload["left_fd"],
            payload["left_img"],
            payload["right_fd"],
            payload["right_img"],
        )
    elif kind == "rgbd":
        writer.save_pair(
            payload["color_fd"],
            payload["color_img"],
            payload["depth_fd"],
            payload["depth_raw"],
        )
    else:
        writer.mark_skip("format", f"unknown payload kind: {kind}")


def sync_save_worker(states: list[dict[str, Any]], capture_event: threading.Event, stop_event: threading.Event) -> None:
    """Save one sample per camera only after every camera has a fresh payload."""
    while not stop_event.is_set():
        if not capture_event.is_set():
            time.sleep(0.01)
            continue

        ready: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
        for state in states:
            item = get_unsaved_payload(state)
            if item is None:
                break
            payload_id, payload = item
            ready.append((state, payload_id, payload))

        if len(ready) != len(states):
            time.sleep(SYNC_SAVE_POLL_SECONDS)
            continue

        try:
            for state, _payload_id, payload in ready:
                writer: cap.SessionWriter = state["writer"]
                if writer.active:
                    save_payload(writer, payload)
            for state, payload_id, _payload in ready:
                mark_payload_saved(state, payload_id)
        except Exception as ex:
            for state, _payload_id, _payload in ready:
                state["error"] = str(ex)
            print(f"[{cap.now_str()}] WARN synchronized saver stopped: {ex}")
            stop_event.set()
            break


def poll_camera(sdk: cap.SDK, state: dict[str, Any], capturing: bool) -> None:
    streams = state["streams"]
    writer: cap.SessionWriter = state["writer"]
    fs = sdk.wait_frameset(state["pipe"], int((state["settings"].get("pipeline", {}) or {}).get("frame_timeout_ms", 200) or 200))
    if not fs:
        return

    ptrs = []
    color_fd = color_left_fd = color_right_fd = depth_fd = None
    try:
        if streams.get("color", False):
            f = sdk.get_optional_frame(fs, cap.OB_FRAME_COLOR)
            if f:
                ptrs.append(f)
                color_fd = sdk.extract(f)
        if streams.get("color_left", False):
            f = sdk.get_optional_frame(fs, cap.OB_FRAME_COLOR_LEFT)
            if f:
                ptrs.append(f)
                color_left_fd = sdk.extract(f)
        if streams.get("color_right", False):
            f = sdk.get_optional_frame(fs, cap.OB_FRAME_COLOR_RIGHT)
            if f:
                ptrs.append(f)
                color_right_fd = sdk.extract(f)
        if streams.get("depth", False):
            f = sdk.get_optional_frame(fs, cap.OB_FRAME_DEPTH)
            if f:
                ptrs.append(f)
                depth_fd = sdk.extract(f)
    finally:
        for p in ptrs:
            sdk.delete_frame(p)
        sdk.delete_frame(fs)

    color_img = cap.decode_color(color_fd) if color_fd else None
    left_img = cap.decode_color(color_left_fd) if color_left_fd else None
    right_img = cap.decode_color(color_right_fd) if color_right_fd else None
    depth_raw = cap.decode_depth(depth_fd) if depth_fd else None

    if color_img is not None or left_img is not None or right_img is not None:
        update_fps(state)
        state["preview_count"] = int(state.get("preview_count", 0)) + 1

    preview_every_n = max(1, int(state.get("preview_every_n", 1) or 1))
    depth_preview_every_n = max(1, int(state.get("depth_preview_every_n", 5) or 5))
    preview_count = int(state.get("preview_count", 0))
    preview_now = preview_count <= 1 or preview_count % preview_every_n == 0
    depth_preview_now = preview_count <= 1 or preview_count % depth_preview_every_n == 0
    if preview_now:
        if color_img is not None:
            state["color"] = color_img
        if left_img is not None:
            state["left"] = left_img
        if right_img is not None:
            state["right"] = right_img
    if depth_raw is not None and SHOW_DEPTH_PREVIEW and depth_preview_now:
        state["depth_vis"] = cap.depth_to_vis(depth_raw)
    if not capturing or not writer.active:
        return

    payload = make_save_payload(
        streams,
        color_fd,
        color_img,
        color_left_fd,
        left_img,
        color_right_fd,
        right_img,
        depth_fd,
        depth_raw,
    )
    if payload is not None:
        set_latest_payload(state, payload)
    else:
        writer.mark_skip("missing")


def camera_worker(sdk: cap.SDK, state: dict[str, Any], capture_event: threading.Event, stop_event: threading.Event) -> None:
    """后台读相机和保存；GUI 线程只负责显示，不阻塞采集。"""
    while not stop_event.is_set():
        try:
            poll_camera(sdk, state, capture_event.is_set())
        except Exception as ex:
            state["error"] = str(ex)
            print(f"[{cap.now_str()}] WARN {state.get('role', '')} worker stopped: {ex}")
            break


def main() -> int:
    global PREVIEW_TARGET_FPS, SHOW_DEPTH_PREVIEW
    args = parse_args()
    PREVIEW_TARGET_FPS = max(1.0, float(args.preview_fps or 15.0))
    cap.PNG_COMPRESSION = 0
    sdk = cap.SDK(Path(args.sdk_bin))
    ctx = dl = 0
    states: list[dict[str, Any]] = []
    capturing = False
    capture_event = threading.Event()
    stop_event = threading.Event()
    worker_threads: list[threading.Thread] = []
    sync_thread: threading.Thread | None = None
    try:
        mode = str(args.capture_mode)
        SHOW_DEPTH_PREVIEW = bool(args.show_depth_preview or mode == MODE_RGBD_RGBD)
        config_335l = Path(args.config_335l)
        if args.config_305:
            config_305 = Path(args.config_305)
        else:
            config_305 = CONFIG_305_RGBD if mode == MODE_RGBD_RGBD else CONFIG_305_DUAL_RGB
        role_305 = "305_rgbd" if mode == MODE_RGBD_RGBD else "305_dual_rgb"
        window_name = "Orbbec 335L RGB-D + 305 RGB-D" if mode == MODE_RGBD_RGBD else "Orbbec 335L RGB-D + 305 Dual RGB"

        print(f"[{cap.now_str()}] Orbbec SDK version: {sdk.get_sdk_version_text()}")
        print(f"[{cap.now_str()}] Capture mode: {mode}")
        ctx = sdk.create_context()
        dl = sdk.query_device_list(ctx)
        states.append(setup_camera(sdk, dl, config_335l, MODEL_335L, "335L_rgbd", args))
        states.append(setup_camera(sdk, dl, config_305, MODEL_305, role_305, args))
        for state in states:
            thread = threading.Thread(target=camera_worker, args=(sdk, state, capture_event, stop_event), daemon=True, name=f"CameraWorker-{state['role']}")
            thread.start()
            worker_threads.append(thread)
        sync_thread = threading.Thread(
            target=sync_save_worker,
            args=(states, capture_event, stop_event),
            daemon=True,
            name="SynchronizedSaveWorker",
        )
        sync_thread.start()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 850)
        mouse_state = {"toggle": False}
        cv2.setMouseCallback(window_name, on_mouse, mouse_state)

        print("Keys: S=start both | E=stop both | Q/ESC=quit")
        auto_start_at = time.perf_counter() + float(AUTO_START_DELAY_SECONDS) if AUTO_START_DELAY_SECONDS > 0 else None
        auto_started = False
        if auto_start_at is not None:
            print(f"[{cap.now_str()}] Auto start after {AUTO_START_DELAY_SECONDS:.1f}s")
        next_preview_at = 0.0
        while True:
            now_perf = time.perf_counter()
            if now_perf < next_preview_at:
                time.sleep(min(0.005, next_preview_at - now_perf))
            next_preview_at = time.perf_counter() + (1.0 / max(1.0, PREVIEW_TARGET_FPS))

            preview = make_merged_preview(states[0], states[1], capturing)
            preview = draw_capture_button(preview, capturing)
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            mouse_toggle = bool(mouse_state.get("toggle"))
            mouse_state["toggle"] = False

            toggle_req = key == 32 or mouse_toggle
            start_req = key in (ord("s"), ord("S")) or (toggle_req and not capturing)
            stop_req = key in (ord("e"), ord("E")) or (toggle_req and capturing)
            if auto_start_at is not None and not auto_started and not capturing and time.perf_counter() >= auto_start_at:
                start_req = True
                auto_started = True

            if start_req and not capturing:
                output_root = Path(states[0]["settings"].get("output_root", PROJECT_ROOT / "captures"))
                shared_dir = make_shared_session_dir(output_root)
                shared_dir.mkdir(parents=True, exist_ok=True)
                write_shared_manifest(shared_dir, states)
                print(f"[{cap.now_str()}] Shared session started: {shared_dir}")
                reset_sync_payloads(states)
                for state in states:
                    writer: cap.SessionWriter = state["writer"]
                    role_folder = ROLE_FOLDERS.get(state["role"], state["role"])
                    session_dir = writer.start(state["cam_params"], session_dir=shared_dir / role_folder)
                    if not writer.minimal_dual_rgb_layout:
                        cap.write_capture_config_snapshot(session_dir, state["settings"], Path(state["cam_params"]["capture_config_path"]))
                    print(f"[{cap.now_str()}] {state['role']} session started: {session_dir}")
                capturing = True
                capture_event.set()
            elif stop_req and capturing:
                capture_event.clear()
                time.sleep(0.25)
                for state in states:
                    writer = state["writer"]
                    if writer.active:
                        writer.stop()
                        print(f"[{cap.now_str()}] {state['role']} session ended: {writer.session_dir}")
                        print(f"[{cap.now_str()}] {state['role']} summary: {writer.skip_summary()}")
                capturing = False
            elif key in (ord("q"), ord("Q"), 27):
                capture_event.clear()
                if capturing:
                    time.sleep(0.25)
                    for state in states:
                        writer = state["writer"]
                        if writer.active:
                            writer.stop()
                break

        return 0
    except Exception as ex:
        print(f"[ERROR] {ex}")
        return 1
    finally:
        stop_event.set()
        capture_event.clear()
        for thread in worker_threads:
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass
        if sync_thread is not None:
            try:
                sync_thread.join(timeout=1.0)
            except Exception:
                pass
        for state in states:
            try:
                writer = state.get("writer")
                if writer and writer.active:
                    writer.stop()
            except Exception:
                pass
            try:
                if state.get("pipe"):
                    sdk.stop_pipeline(state["pipe"])
            except Exception:
                pass
            try:
                if state.get("dev"):
                    sdk.delete_device(state["dev"])
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())










