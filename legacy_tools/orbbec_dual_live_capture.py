#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from orbbec_live_capture import (
    ALIGN_D2C_HW_MODE,
    ALIGN_D2C_SW_MODE,
    OB_FORMAT_Y16,
    OB_FRAME_COLOR,
    OB_FRAME_DEPTH,
    OB_STREAM_COLOR,
    OB_STREAM_DEPTH,
    SDK,
    SessionWriter,
    camera_param_to_dict,
    decode_color,
    decode_depth,
    depth_to_vis,
    now_str,
    fixed_1280x800_30_model,
    select_fixed_1280x800_30_profile,
)


ROLE_EYE_TO_HAND = 'eye_to_hand'
ROLE_EYE_IN_HAND = 'eye_in_hand'


@dataclass
class DeviceRecord:
    index: int
    sn: str
    name: str
    dev: object


def unique_scan_dir(root: Path) -> Path:
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base = root / f'scan_data_{stamp}'
    if not base.exists():
        return base
    i = 1
    while True:
        candidate = root / f'scan_data_{stamp}_{i:02d}'
        if not candidate.exists():
            return candidate
        i += 1


def choose_output_root(current: Path) -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog

        app = tk.Tk()
        app.withdraw()
        app.attributes('-topmost', True)
        selected = filedialog.askdirectory(initialdir=str(current), title='Select capture output folder')
        app.destroy()
        return Path(selected).resolve() if selected else current
    except Exception as ex:
        print(f'[{now_str()}] WARN folder dialog failed: {ex}')
        return current


def build_capture_config(sdk: SDK, align_mode: int, name: str, stream_profile: Optional[dict] = None):
    cfg = sdk.create_config()
    try:
        if stream_profile:
            color_width = stream_profile['color_width'] if 'color_width' in stream_profile else stream_profile['width']
            color_height = stream_profile['color_height'] if 'color_height' in stream_profile else stream_profile['height']
            color_fps = stream_profile['color_fps'] if 'color_fps' in stream_profile else stream_profile['fps']
            depth_width = stream_profile['depth_width'] if 'depth_width' in stream_profile else stream_profile['width']
            depth_height = stream_profile['depth_height'] if 'depth_height' in stream_profile else stream_profile['height']
            depth_fps = stream_profile['depth_fps'] if 'depth_fps' in stream_profile else stream_profile['fps']
            sdk.enable_video_stream(
                cfg,
                OB_STREAM_COLOR,
                color_width,
                color_height,
                color_fps,
                stream_profile['color_format'],
            )
            sdk.enable_video_stream(
                cfg,
                OB_STREAM_DEPTH,
                depth_width,
                depth_height,
                depth_fps,
                stream_profile['depth_format'],
            )
        else:
            sdk.enable_stream(cfg, OB_STREAM_COLOR)
            sdk.enable_stream(cfg, OB_STREAM_DEPTH)
        if not sdk.set_align_mode_try(cfg, align_mode):
            raise RuntimeError(f'Unable to enable {name}')
        sdk.set_depth_scale_after_align(cfg, True)
        sdk.set_aggregate_all_type(cfg)
        return cfg
    except Exception:
        sdk.delete_config(cfg)
        raise


def start_rgbd_pipeline(sdk: SDK, dev, role: str, device_name: str = ''):
    stream_profile = None
    fixed_required = fixed_1280x800_30_model(device_name)
    if fixed_required:
        probe_pipe = 0
        try:
            probe_pipe = sdk.create_pipeline(dev)
            stream_profile = select_fixed_1280x800_30_profile(sdk, probe_pipe, device_name)
        except Exception as ex:
            print(f'[{now_str()}] WARN {role}: failed to query SDK stream profiles for {device_name}: {ex}')
            stream_profile = None
        finally:
            if probe_pipe:
                sdk.delete_pipeline(probe_pipe)
        if stream_profile is None:
            raise RuntimeError(f'{role}: {device_name} does not provide required RGB-D 1280x800@30fps profile in this SDK/device mode.')
    attempts: list[Optional[dict]] = [stream_profile] if stream_profile else [None]

    for stream_profile in attempts:
        for mode, name in (
            (ALIGN_D2C_HW_MODE, 'ALIGN_D2C_HW_MODE'),
            (ALIGN_D2C_SW_MODE, 'ALIGN_D2C_SW_MODE'),
        ):
            pipe = 0
            cfg = 0
            try:
                pipe = sdk.create_pipeline(dev)
                cfg = build_capture_config(sdk, mode, name, stream_profile)
                sdk.start_pipeline(pipe, cfg)
                if stream_profile:
                    print(f'[{now_str()}] {role}: started with profile {stream_profile["label"]}')
                return pipe, cfg, name
            except Exception as ex:
                profile_name = stream_profile['label'] if stream_profile else 'SDK default'
                print(f'[{now_str()}] WARN {role} start with {name}, profile={profile_name} failed: {ex}')
                if pipe:
                    try:
                        sdk.stop_pipeline(pipe)
                    except Exception:
                        pass
                    sdk.delete_pipeline(pipe)
                if cfg:
                    sdk.delete_config(cfg)
    raise RuntimeError(f'Unable to start {role} RGB-D pipeline with HW or SW D2C alignment.')


class CameraRuntime:
    def __init__(
        self,
        sdk: SDK,
        role: str,
        dev,
        sn: str,
        name: str,
        pipe,
        cfg,
        align_mode_name: str,
        camera_params: dict,
        max_sync_diff_ms: float,
    ):
        self.sdk = sdk
        self.role = role
        self.dev = dev
        self.sn = sn
        self.name = name
        self.pipe = pipe
        self.cfg = cfg
        self.align_mode_name = align_mode_name
        self.camera_params = camera_params
        self.max_sync_diff_ms = float(max_sync_diff_ms)

        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.state_lock = threading.Lock()
        self.save_lock = threading.Lock()

        self.writer: Optional[SessionWriter] = None
        self.saving_enabled = False
        self.session_dir: Optional[Path] = None

        self.last_color: Optional[np.ndarray] = None
        self.last_depth_vis: Optional[np.ndarray] = None
        self.last_error = ''
        self.last_skip = ''
        self.color_size = ''
        self.depth_size = ''
        self.frame_count = 0
        self.fps = 0.0
        self._fps_window_start = time.perf_counter()
        self._fps_window_frames = 0
        self.last_frame_time = ''

    def start_thread(self) -> None:
        self.thread = threading.Thread(target=self._loop, name=f'{self.role}CaptureThread', daemon=True)
        self.thread.start()

    def stop_thread(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
            self.thread = None

    def start_saving(
        self,
        session_root: Path,
        tag: str,
        target_width: int,
        target_height: int,
        enable_immediately: bool = True,
    ) -> Path:
        role_dir = session_root / self.role
        writer = SessionWriter(
            session_root,
            self.sn,
            tag,
            target_width=target_width,
            target_height=target_height,
            align_mode_name=self.align_mode_name,
            role=self.role,
        )
        writer.start(self.camera_params, session_dir=role_dir)
        with self.save_lock:
            self.writer = writer
            self.session_dir = role_dir
            self.saving_enabled = enable_immediately
        return role_dir

    def enable_saving(self) -> None:
        with self.save_lock:
            if self.writer is None or not self.writer.active:
                raise RuntimeError(f'{self.role} writer is not ready')
            self.saving_enabled = True

    def stop_saving(self) -> str:
        with self.save_lock:
            self.saving_enabled = False
            writer = self.writer
            if writer is None:
                return 'not started'
            writer.stop()
            summary = writer.skip_summary()
            if writer.pair_index == 0:
                detail = f' ({writer.last_skip_detail})' if writer.last_skip_detail else ''
                summary += f'; no pairs saved, last skip={writer.last_skip_reason or "none"}{detail}'
            return summary

    def release(self) -> None:
        self.stop_thread()
        try:
            if self.saving_enabled or (self.writer and self.writer.active):
                self.stop_saving()
        except Exception as ex:
            print(f'[{now_str()}] WARN {self.role} stop saving failed: {ex}')
        if self.pipe:
            try:
                self.sdk.stop_pipeline(self.pipe)
            except Exception:
                pass
        if self.cfg:
            self.sdk.delete_config(self.cfg)
            self.cfg = 0
        if self.pipe:
            self.sdk.delete_pipeline(self.pipe)
            self.pipe = 0
        if self.dev:
            self.sdk.delete_device(self.dev)
            self.dev = 0

    def snapshot(self) -> dict:
        with self.state_lock:
            return {
                'last_color': None if self.last_color is None else self.last_color.copy(),
                'last_depth_vis': None if self.last_depth_vis is None else self.last_depth_vis.copy(),
                'last_error': self.last_error,
                'last_skip': self.last_skip,
                'color_size': self.color_size,
                'depth_size': self.depth_size,
                'frame_count': self.frame_count,
                'fps': self.fps,
                'last_frame_time': self.last_frame_time,
            }

    def writer_summary(self) -> str:
        writer = self.writer
        return writer.skip_summary() if writer else 'saved=0'

    def saved_count(self) -> int:
        writer = self.writer
        return int(writer.pair_index) if writer else 0

    def _set_error(self, message: str) -> None:
        with self.state_lock:
            self.last_error = message

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            color_fd = None
            depth_fd = None
            color_img = None
            depth_raw = None

            try:
                fs = self.sdk.wait_frameset(self.pipe, 100)
                if fs:
                    ptrs = []
                    try:
                        f_color = self.sdk.get_optional_frame(fs, OB_FRAME_COLOR)
                        f_depth = self.sdk.get_optional_frame(fs, OB_FRAME_DEPTH)
                        if f_color:
                            ptrs.append(f_color)
                            color_fd = self.sdk.extract(f_color)
                        if f_depth:
                            ptrs.append(f_depth)
                            depth_fd = self.sdk.extract(f_depth)
                    finally:
                        for fp in ptrs:
                            self.sdk.delete_frame(fp)
                        self.sdk.delete_frame(fs)

                color_img = decode_color(color_fd) if color_fd else None
                depth_raw = decode_depth(depth_fd) if depth_fd else None
                depth_vis = None
                if depth_raw is not None:
                    try:
                        depth_vis = depth_to_vis(depth_raw)
                    except Exception as ex:
                        self._set_error(f'depth preview failed: {ex}')

                with self.state_lock:
                    if color_img is not None:
                        self.last_color = color_img
                    if depth_vis is not None:
                        self.last_depth_vis = depth_vis
                    if color_fd is not None:
                        self.color_size = f'{color_fd.width}x{color_fd.height}'
                    if depth_fd is not None:
                        self.depth_size = f'{depth_fd.width}x{depth_fd.height}'
                    if color_fd is not None or depth_fd is not None:
                        self.frame_count += 1
                        self._fps_window_frames += 1
                        now_perf = time.perf_counter()
                        elapsed = now_perf - self._fps_window_start
                        if elapsed >= 1.0:
                            self.fps = self._fps_window_frames / elapsed
                            self._fps_window_frames = 0
                            self._fps_window_start = now_perf
                        self.last_frame_time = datetime.now().strftime('%H:%M:%S')
                    self.last_error = ''

                self._maybe_save(color_fd, color_img, depth_fd, depth_raw)
            except Exception as ex:
                self._set_error(str(ex))
                time.sleep(0.05)

    def _maybe_save(self, color_fd, color_img, depth_fd, depth_raw) -> None:
        if not self.saving_enabled:
            return
        with self.save_lock:
            writer = self.writer
            if not self.saving_enabled or writer is None or not writer.active:
                return

            if color_fd is None or depth_fd is None:
                detail = f'color={"yes" if color_fd is not None else "no"}, depth={"yes" if depth_fd is not None else "no"}'
                writer.mark_skip('missing', detail)
            elif depth_fd.fmt != OB_FORMAT_Y16:
                writer.mark_skip('format', f'depth_fmt={depth_fd.fmt}, expected={OB_FORMAT_Y16}')
            elif color_img is None or depth_raw is None:
                detail = f'color_img={"yes" if color_img is not None else "no"}, depth_raw={"yes" if depth_raw is not None else "no"}'
                writer.mark_skip('decode', detail)
            elif abs(int(color_fd.dev_ts) - int(depth_fd.dev_ts)) > int(self.max_sync_diff_ms * 1000.0):
                diff_ms = abs(int(color_fd.dev_ts) - int(depth_fd.dev_ts)) / 1000.0
                writer.mark_skip('sync', f'diff_ms={diff_ms:.3f}, max_ms={self.max_sync_diff_ms:.3f}')
            else:
                writer.save_pair(color_fd, color_img, depth_fd, depth_raw)

            if writer.last_skip_reason:
                with self.state_lock:
                    self.last_skip = f'{writer.last_skip_reason} {writer.last_skip_detail}'.strip()


def enumerate_devices(sdk: SDK, dl) -> list[DeviceRecord]:
    records = []
    count = sdk.device_count(dl)
    print(f'[{now_str()}] Found {count} Orbbec device(s).')
    for idx in range(count):
        dev = sdk.get_device(dl, idx)
        try:
            sn, name = sdk.get_device_info(dev)
            print(f'[{now_str()}] Device[{idx}]: {name}, SN: {sn}')
            records.append(DeviceRecord(idx, sn, name, dev))
            dev = 0
        finally:
            if dev:
                sdk.delete_device(dev)
    return records


def find_by_serial(records: list[DeviceRecord], serial: str) -> Optional[DeviceRecord]:
    wanted = serial.strip().lower()
    if not wanted:
        return None
    for rec in records:
        if rec.sn.strip().lower() == wanted:
            return rec
    return None


def pop_record(records: list[DeviceRecord], rec: DeviceRecord) -> DeviceRecord:
    records.remove(rec)
    return rec


def find_by_model(records: list[DeviceRecord], model_text: str) -> Optional[DeviceRecord]:
    wanted = model_text.strip().lower()
    for rec in records:
        if wanted in rec.name.lower():
            return rec
    return None


def select_role_devices(args, records: list[DeviceRecord]) -> tuple[DeviceRecord, DeviceRecord]:
    if len(records) < 2:
        raise RuntimeError('Dual capture requires at least two Orbbec RGB-D cameras.')

    available = records[:]

    def select_one(role: str, serial: str, index: Optional[int], default_index: int) -> DeviceRecord:
        if serial.strip():
            rec = find_by_serial(available, serial)
            if rec is None:
                raise RuntimeError(f'{role} serial not found: {serial}')
            return pop_record(available, rec)

        if index is not None:
            for rec in available:
                if rec.index == index:
                    return pop_record(available, rec)
            raise RuntimeError(f'{role} device index not available: {index}')

        for rec in available:
            if rec.index == default_index:
                return pop_record(available, rec)
        return pop_record(available, available[0])

    if (
        not args.eye_to_hand_serial.strip()
        and args.eye_to_hand_index is None
        and not args.eye_in_hand_serial.strip()
        and args.eye_in_hand_index is None
    ):
        eye_to_auto = find_by_model(available, 'Gemini 335L')
        eye_in_auto = find_by_model(available, 'Gemini 305')
        if eye_to_auto is not None and eye_in_auto is not None and eye_to_auto is not eye_in_auto:
            eye_to_hand = pop_record(available, eye_to_auto)
            eye_in_hand = pop_record(available, eye_in_auto)
            print(f'[{now_str()}] Fixed role binding: Gemini 335L -> {ROLE_EYE_TO_HAND}, Gemini 305 -> {ROLE_EYE_IN_HAND}')
            return eye_to_hand, eye_in_hand
        print(f'[{now_str()}] WARN fixed role binding not found; falling back to index/argument selection.')

    eye_to_hand = select_one(ROLE_EYE_TO_HAND, args.eye_to_hand_serial, args.eye_to_hand_index, 0)
    eye_in_hand = select_one(ROLE_EYE_IN_HAND, args.eye_in_hand_serial, args.eye_in_hand_index, 1)
    return eye_to_hand, eye_in_hand


def prompt_camera_count(args, device_count: int) -> int:
    if args.camera_count in (1, 2):
        return int(args.camera_count)

    default_count = 2 if device_count >= 2 else 1
    while True:
        raw = input(f'Start camera count: 1=single, 2=dual (Enter={default_count}): ').strip()
        if not raw:
            return default_count
        if raw in ('1', '2'):
            return int(raw)
        print('Please enter 1 or 2.')


def prompt_single_role(args) -> str:
    role = str(args.single_role or '').strip()
    if role in (ROLE_EYE_TO_HAND, ROLE_EYE_IN_HAND):
        return role

    if args.eye_in_hand_serial.strip() or args.eye_in_hand_index is not None:
        return ROLE_EYE_IN_HAND
    if args.eye_to_hand_serial.strip() or args.eye_to_hand_index is not None:
        return ROLE_EYE_TO_HAND

    while True:
        raw = input('Single camera role: 1=eye_to_hand, 2=eye_in_hand (Enter=1): ').strip()
        if not raw or raw == '1':
            return ROLE_EYE_TO_HAND
        if raw == '2':
            return ROLE_EYE_IN_HAND
        if raw in (ROLE_EYE_TO_HAND, ROLE_EYE_IN_HAND):
            return raw
        print('Please enter 1 or 2.')


def select_single_role_device(args, records: list[DeviceRecord]) -> tuple[str, DeviceRecord]:
    if not records:
        raise RuntimeError('No Orbbec RGB-D camera found.')

    role = str(args.single_role or '').strip()
    serial = str(args.single_serial or '').strip()
    index = args.single_index
    if not serial and index is None:
        if role == ROLE_EYE_TO_HAND:
            serial = args.eye_to_hand_serial.strip()
            index = args.eye_to_hand_index
        elif role == ROLE_EYE_IN_HAND:
            serial = args.eye_in_hand_serial.strip()
            index = args.eye_in_hand_index

    selected = None
    if serial:
        selected = find_by_serial(records, serial)
        if selected is None:
            raise RuntimeError(f'Single camera serial not found: {serial}')
    elif index is not None:
        for rec in records:
            if rec.index == index:
                selected = rec
                break
        if selected is None:
            raise RuntimeError(f'Single camera device index not available: {index}')
    elif len(records) == 1:
        selected = records[0]
    else:
        default_index = records[0].index
        while True:
            raw = input(f'Single camera device index (Enter={default_index}): ').strip()
            if not raw:
                selected = records[0]
                break
            try:
                wanted = int(raw)
            except ValueError:
                print('Please enter a device index number.')
                continue
            for rec in records:
                if rec.index == wanted:
                    selected = rec
                    break
            if selected is not None:
                break
            print(f'Device index not available: {wanted}')

    if role not in (ROLE_EYE_TO_HAND, ROLE_EYE_IN_HAND):
        name = selected.name.lower()
        if 'gemini 305' in name:
            role = ROLE_EYE_IN_HAND
        elif 'gemini 335l' in name:
            role = ROLE_EYE_TO_HAND
        else:
            role = prompt_single_role(args)
    print(f'[{now_str()}] Fixed role binding: {selected.name} -> {role}')
    return role, selected


def release_unselected(sdk: SDK, all_records: list[DeviceRecord], selected: list[DeviceRecord]) -> None:
    selected_devs = {id(rec.dev) for rec in selected}
    for rec in all_records:
        if id(rec.dev) not in selected_devs and rec.dev:
            sdk.delete_device(rec.dev)
            rec.dev = 0


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    nw = max(1, int(w * (target_h / max(1, h))))
    return cv2.resize(img, (nw, target_h), interpolation=cv2.INTER_AREA)


def make_camera_panel(cam: CameraRuntime, capturing: bool, target_h: int = 360) -> np.ndarray:
    snap = cam.snapshot()
    panels = []
    if snap['last_color'] is not None:
        panels.append(resize_to_height(snap['last_color'], target_h))
    if snap['last_depth_vis'] is not None:
        panels.append(resize_to_height(snap['last_depth_vis'], target_h))
    if panels:
        panel = np.hstack(panels)
    else:
        panel = np.zeros((target_h, 960, 3), dtype=np.uint8)

    lines = [
        f'{cam.role}: {cam.name} | SN: {cam.sn}',
        f'Align: {cam.align_mode_name} | Status: {"RUNNING" if capturing else "IDLE"}',
        f'RGB: {snap["color_size"] or "-"} | Depth: {snap["depth_size"] or "-"} | FPS: {snap["fps"]:.1f} | Frames: {snap["frame_count"]}',
        f'Save: {cam.writer_summary()}',
    ]
    if snap['last_skip'] and cam.saved_count() == 0:
        lines.append(f'Last skip: {snap["last_skip"]}')
    if snap['last_error']:
        lines.append(f'ERROR: {snap["last_error"][:96]}')

    y = 28
    for text in lines:
        cv2.putText(panel, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
        y += 28
    return panel


def pad_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w >= width:
        return img
    pad = np.zeros((h, width - w, 3), dtype=np.uint8)
    return np.hstack([img, pad])


def draw_global_controls(img: np.ndarray, capturing: bool, output_root: Path, session_root: Optional[Path]) -> tuple[np.ndarray, dict]:
    footer_h = 86
    h, w = img.shape[:2]
    footer = np.zeros((footer_h, w, 3), dtype=np.uint8)
    lines = [
        f'Output: {output_root}',
        f'Session: {session_root if session_root else "-"}',
        'Keys: S=start | E=end | SPACE=toggle | O=path | Q/ESC=quit',
    ]
    y = 22
    for text in lines:
        cv2.putText(footer, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        y += 22

    labels = [
        ('toggle', 'STOP (E/SPACE)' if capturing else 'START (S/SPACE)'),
        ('path', 'PATH (O)'),
        ('quit', 'QUIT (Q/ESC)'),
    ]
    buttons = {}
    x = 12
    y1 = footer_h - 46
    y2 = footer_h - 8
    for name, text in labels:
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
        bw = tw + 24
        x2 = min(w - 12, x + bw)
        color = (0, 70, 230) if name == 'quit' else ((0, 140, 255) if name == 'toggle' and capturing else (0, 160, 0))
        if name == 'path':
            color = (120, 95, 0)
        cv2.rectangle(footer, (x, y1), (x2, y2), color, thickness=-1)
        cv2.rectangle(footer, (x, y1), (x2, y2), (255, 255, 255), thickness=1)
        cv2.putText(footer, text, (x + 10, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        buttons[name] = (x, h + y1, x2, h + y2)
        x = x2 + 12

    return np.vstack([img, footer]), buttons


def on_mouse(event, x, y, flags, state):
    if state is None or event != cv2.EVENT_LBUTTONUP:
        return
    for name, (x1, y1, x2, y2) in state.get('buttons', {}).items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            state['action'] = name
            return


def write_session_info(session_root: Path, cameras: list[CameraRuntime], started_at: str, ended_at: str = '') -> None:
    lines = [
        f'session_dir: "{session_root}"',
        f'started_at: "{started_at}"',
        f'ended_at: "{ended_at}"',
        'roles:',
    ]
    for cam in cameras:
        lines.extend([
            f'  {cam.role}:',
            f'    name: "{cam.name}"',
            f'    serial_number: "{cam.sn}"',
            f'    align_mode: "{cam.align_mode_name}"',
            f'    folder: "{cam.role}"',
        ])
    (session_root / 'session_info.yaml').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args():
    p = argparse.ArgumentParser(description='Dual Orbbec RGB-D collector for eye_to_hand and eye_in_hand cameras')
    p.add_argument('--sdk-bin', default=r'D:\OrbbecSDK_v2\bin', help='Folder containing OrbbecSDK.dll')
    p.add_argument('--output-root', default=r'D:\OrbbecLiveCollector\captures', help='Root output folder')
    p.add_argument('--tag', default='', help='Optional tag recorded in camera_info.yaml')
    p.add_argument('--camera-count', type=int, choices=(1, 2), default=0, help='Number of cameras to start; default asks at startup')
    p.add_argument('--single-role', default=None, choices=(ROLE_EYE_TO_HAND, ROLE_EYE_IN_HAND), help='Role name when starting one camera')
    p.add_argument('--single-serial', default='', help='Serial number when starting one camera')
    p.add_argument('--single-index', type=int, default=None, help='Device index when starting one camera')
    p.add_argument('--width', type=int, default=0, help='Output width for rgb/depth PNG, 0 means keep native RGB width')
    p.add_argument('--height', type=int, default=0, help='Output height for rgb/depth PNG, 0 means keep native RGB height')
    p.add_argument('--max-sync-diff-ms', type=float, default=15.0, help='Max allowed per-camera |rgb_ts-depth_ts| in ms')
    p.add_argument('--eye-to-hand-serial', default='', help='Serial number for the external eye_to_hand camera')
    p.add_argument('--eye-in-hand-serial', default='', help='Serial number for the hand-mounted eye_in_hand camera')
    p.add_argument('--eye-to-hand-index', type=int, default=None, help='Device index for eye_to_hand when serial is not provided')
    p.add_argument('--eye-in-hand-index', type=int, default=None, help='Device index for eye_in_hand when serial is not provided')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    sdk_bin = Path(args.sdk_bin).resolve()

    sdk = SDK(sdk_bin)
    ctx = dl = 0
    cameras: list[CameraRuntime] = []
    all_records: list[DeviceRecord] = []
    capturing = False
    session_root: Optional[Path] = None
    session_started_at = ''

    try:
        ctx = sdk.create_context()
        dl = sdk.query_device_list(ctx)
        all_records = enumerate_devices(sdk, dl)
        camera_count = prompt_camera_count(args, len(all_records))
        if camera_count == 1:
            role, rec = select_single_role_device(args, all_records)
            selected = [(role, rec)]
        else:
            eye_to_rec, eye_in_rec = select_role_devices(args, all_records)
            selected = [(ROLE_EYE_TO_HAND, eye_to_rec), (ROLE_EYE_IN_HAND, eye_in_rec)]
        release_unselected(sdk, all_records, [rec for _, rec in selected])
        print(f'[{now_str()}] Capture mode: {len(selected)} camera(s)')

        for role, rec in selected:
            pipe, cfg, align_name = start_rgbd_pipeline(sdk, rec.dev, role, rec.name)
            params = camera_param_to_dict(sdk.get_camera_param(pipe), align_name)
            cam = CameraRuntime(
                sdk=sdk,
                role=role,
                dev=rec.dev,
                sn=rec.sn,
                name=rec.name,
                pipe=pipe,
                cfg=cfg,
                align_mode_name=align_name,
                camera_params=params,
                max_sync_diff_ms=args.max_sync_diff_ms,
            )
            rec.dev = 0
            cam.start_thread()
            cameras.append(cam)
            print(f'[{now_str()}] {role}: {rec.name}, SN: {rec.sn}, align={align_name}')

        def start_capture():
            nonlocal capturing, session_root, session_started_at
            if capturing:
                return
            output_root.mkdir(parents=True, exist_ok=True)
            session_root = unique_scan_dir(output_root)
            session_root.mkdir(parents=True, exist_ok=False)
            session_started_at = now_str()
            write_session_info(session_root, cameras, session_started_at)
            prepared = []
            try:
                for cam in cameras:
                    cam.start_saving(session_root, args.tag, args.width, args.height, enable_immediately=False)
                    prepared.append(cam)
                for cam in cameras:
                    cam.enable_saving()
                capturing = True
                print(f'[{now_str()}] Capture session started: {session_root}')
            except Exception:
                for cam in prepared:
                    try:
                        cam.stop_saving()
                    except Exception:
                        pass
                raise

        def stop_capture():
            nonlocal capturing
            if not capturing:
                return
            summaries = []
            for cam in cameras:
                summaries.append((cam.role, cam.stop_saving()))
            ended_at = now_str()
            if session_root is not None:
                write_session_info(session_root, cameras, session_started_at, ended_at)
            capturing = False
            print(f'[{now_str()}] Capture session ended: {session_root}')
            for role, summary in summaries:
                print(f'[{now_str()}] {role} summary: {summary}')

        window_name = 'Orbbec Dual Live Capture' if len(cameras) > 1 else 'Orbbec Live Capture'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1600, 900)
        cv2.moveWindow(window_name, 60, 40)
        ui_state = {'buttons': {}, 'action': None}
        cv2.setMouseCallback(window_name, on_mouse, ui_state)

        print('Keys in preview window: S=start | E=end | SPACE=toggle | O=path | Q/ESC=quit')
        print('Mouse in preview window: click START/STOP/PATH/QUIT buttons')

        while True:
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    if capturing:
                        stop_capture()
                    break
            except Exception:
                pass

            panels = [make_camera_panel(cam, capturing) for cam in cameras]
            width = max(panel.shape[1] for panel in panels)
            body = np.vstack([pad_width(panel, width) for panel in panels])
            preview, buttons = draw_global_controls(body, capturing, output_root, session_root)
            ui_state['buttons'] = buttons
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            click_action = ui_state.get('action')
            ui_state['action'] = None

            quit_req = (key in (ord('q'), ord('Q'), 27)) or click_action == 'quit'
            start_req = (key in (ord('s'), ord('S'))) or (click_action == 'toggle' and not capturing)
            stop_req = (key in (ord('e'), ord('E'))) or (click_action == 'toggle' and capturing)
            path_req = (key in (ord('o'), ord('O'))) or click_action == 'path'
            toggle_req = key in (32,)

            if toggle_req:
                if capturing:
                    stop_req = True
                else:
                    start_req = True

            if path_req and not capturing:
                output_root = choose_output_root(output_root)
                print(f'[{now_str()}] Output root: {output_root}')
            elif path_req and capturing:
                print(f'[{now_str()}] WARN stop capture before changing output path.')

            if quit_req:
                if capturing:
                    stop_capture()
                break

            if start_req and not capturing:
                try:
                    start_capture()
                except Exception as ex:
                    print(f'[{now_str()}] WARN capture start failed: {ex}')

            if stop_req and capturing:
                try:
                    stop_capture()
                except Exception as ex:
                    print(f'[{now_str()}] WARN capture stop failed: {ex}')

        cv2.destroyAllWindows()
        print(f'[{now_str()}] Exit.')
        return 0
    except Exception as ex:
        print(f'[ERROR] {ex}')
        return 1
    finally:
        try:
            if capturing:
                for cam in cameras:
                    if cam.saving_enabled or (cam.writer and cam.writer.active):
                        cam.stop_saving()
        except Exception:
            pass
        for cam in cameras:
            cam.release()
        for rec in all_records:
            if rec.dev:
                sdk.delete_device(rec.dev)
                rec.dev = 0
        if dl:
            sdk.delete_device_list(dl)
        if ctx:
            sdk.delete_context(ctx)


if __name__ == '__main__':
    raise SystemExit(main())
