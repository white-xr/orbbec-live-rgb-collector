#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


ROLES = ('eye_to_hand', 'eye_in_hand')
DEFAULT_FPS = 30.0


def read_timestamp_map(role_dir: Path) -> dict[int, float]:
    path = role_dir / 'timestamps.txt'
    if not path.exists():
        return {}

    rows = {}
    for raw in path.read_text(encoding='utf-8').splitlines():
        parts = raw.split()
        if len(parts) < 2:
            continue
        try:
            rows[int(parts[0])] = float(parts[1])
        except ValueError:
            continue
    return rows


def read_existing_rgb_rows(role_dir: Path) -> list[tuple[int, float]]:
    timestamp_map = read_timestamp_map(role_dir)
    rows = []
    for path in sorted((role_dir / 'rgb').glob('*.png'), key=lambda p: p.name):
        try:
            frame_id = int(path.stem)
        except ValueError:
            continue
        if frame_id in timestamp_map:
            rows.append((frame_id, timestamp_map[frame_id]))
    return rows


def nearest_index(times: list[float], target: float) -> int:
    pos = bisect.bisect_left(times, target)
    if pos <= 0:
        return 0
    if pos >= len(times):
        return len(times) - 1
    before = pos - 1
    after = pos
    if abs(times[after] - target) < abs(times[before] - target):
        return after
    return before


class FrameReader:
    def __init__(self, role_dir: Path, rows: list[tuple[int, float]]):
        self.role_dir = role_dir
        self.rows = rows
        self.times = [row[1] for row in rows]
        self.last_path: Optional[Path] = None
        self.last_frame: Optional[np.ndarray] = None
        self.bad_reads = 0
        self.reused_reads = 0
        self.max_diff_ms = 0.0
        self.sum_diff_ms = 0.0

    def first_frame(self) -> np.ndarray:
        for frame_id, _ in self.rows:
            path = self.role_dir / 'rgb' / f'{frame_id:06d}.png'
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
        raise RuntimeError(f'No readable RGB frames in {self.role_dir / "rgb"}')

    def read_nearest(self, target: float) -> np.ndarray:
        idx = nearest_index(self.times, target)
        frame_id, ts = self.rows[idx]
        diff_ms = abs(ts - target) * 1000.0
        self.max_diff_ms = max(self.max_diff_ms, diff_ms)
        self.sum_diff_ms += diff_ms

        path = self.role_dir / 'rgb' / f'{frame_id:06d}.png'
        if self.last_path == path and self.last_frame is not None:
            self.reused_reads += 1
            return self.last_frame

        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            self.bad_reads += 1
            if self.last_frame is not None:
                return self.last_frame
            raise RuntimeError(f'Failed to read RGB frame: {path}')

        self.last_path = path
        self.last_frame = frame
        return frame


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f'Failed to open VideoWriter: {path}')
    return writer


def resize_to(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == height:
        return frame
    width = max(1, int(w * (height / max(1, h))))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 520), 38), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def make_synced_session(session_dir: Path, fps: float, make_preview: bool = True) -> dict:
    role_rows = {role: read_existing_rgb_rows(session_dir / role) for role in ROLES}
    missing = [role for role, rows in role_rows.items() if not rows]
    if missing:
        return {'session': str(session_dir), 'status': 'skipped_missing_timestamps', 'missing': missing}

    start = max(role_rows[role][0][1] for role in ROLES)
    end = min(role_rows[role][-1][1] for role in ROLES)
    if end <= start:
        return {'session': str(session_dir), 'status': 'skipped_no_overlap'}

    frame_count = int((end - start) * fps) + 1
    readers = {role: FrameReader(session_dir / role, role_rows[role]) for role in ROLES}
    first_frames = {role: readers[role].first_frame() for role in ROLES}
    sizes = {role: (first_frames[role].shape[1], first_frames[role].shape[0]) for role in ROLES}

    out_dir = session_dir / 'synced_videos'
    writers = {
        role: open_writer(out_dir / f'{role}_sync.mp4', fps, sizes[role])
        for role in ROLES
    }

    preview_writer = None
    preview_size = None
    if make_preview:
        preview_h = max(first_frames[role].shape[0] for role in ROLES)
        preview_frames = [resize_to_height(first_frames[role], preview_h) for role in ROLES]
        preview_size = (sum(frame.shape[1] for frame in preview_frames), preview_h)
        preview_writer = open_writer(out_dir / 'synced_preview.mp4', fps, preview_size)

    try:
        for i in range(frame_count):
            target = start + (i / fps)
            frames = {}
            for role in ROLES:
                frame = readers[role].read_nearest(target)
                frame = resize_to(frame, sizes[role])
                writers[role].write(frame)
                frames[role] = frame

            if preview_writer is not None:
                preview_parts = []
                for role in ROLES:
                    part = resize_to_height(frames[role], preview_size[1])
                    part = draw_label(part, f'{role}  t={target - start:.3f}s')
                    preview_parts.append(part)
                preview_writer.write(np.hstack(preview_parts))
    finally:
        for writer in writers.values():
            writer.release()
        if preview_writer is not None:
            preview_writer.release()

    stats = {
        'session': str(session_dir),
        'status': 'ok',
        'fps': fps,
        'frames': frame_count,
        'start': start,
        'end': end,
        'duration': (frame_count - 1) / fps if frame_count > 1 else 0.0,
        'outputs': {role: str(out_dir / f'{role}_sync.mp4') for role in ROLES},
    }
    if make_preview:
        stats['preview'] = str(out_dir / 'synced_preview.mp4')
    for role in ROLES:
        reader = readers[role]
        stats[f'{role}_max_diff_ms'] = reader.max_diff_ms
        stats[f'{role}_mean_diff_ms'] = reader.sum_diff_ms / max(1, frame_count)
        stats[f'{role}_bad_reads'] = reader.bad_reads
        stats[f'{role}_reused_reads'] = reader.reused_reads
        stats[f'{role}_source_frames'] = len(role_rows[role])
    return stats


def find_sessions(root: Path) -> list[Path]:
    sessions = []
    for path in sorted(root.glob('scan_data_*')):
        if path.is_dir() and all((path / role / 'rgb').is_dir() for role in ROLES):
            sessions.append(path)
    return sessions


def parse_args():
    parser = argparse.ArgumentParser(description='Create timestamp-synchronized videos for dual Orbbec RGB captures')
    parser.add_argument('--root', default='captures', help='Root folder containing scan_data_* sessions')
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS, help='Output FPS for synchronized videos')
    parser.add_argument('--no-preview', action='store_true', help='Do not create side-by-side synced_preview.mp4')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f'Root not found: {root}')
    if args.fps <= 0 or args.fps > 240:
        raise SystemExit(f'Invalid --fps: {args.fps}')

    sessions = find_sessions(root)
    if not sessions:
        print(f'No dual-camera sessions found under {root}')
        return 0

    print(f'Found {len(sessions)} dual-camera sessions under {root}')
    for session in sessions:
        stats = make_synced_session(session, args.fps, make_preview=not args.no_preview)
        if stats['status'] != 'ok':
            print(f'[SKIP] {stats["session"]}: {stats["status"]}')
            continue
        print(
            f'[OK] {Path(stats["session"]).name} | frames={stats["frames"]} | fps={stats["fps"]:.3f} | '
            f'duration={stats["duration"]:.3f}s | '
            f'eye_to_hand diff max/mean={stats["eye_to_hand_max_diff_ms"]:.2f}/{stats["eye_to_hand_mean_diff_ms"]:.2f}ms | '
            f'eye_in_hand diff max/mean={stats["eye_in_hand_max_diff_ms"]:.2f}/{stats["eye_in_hand_mean_diff_ms"]:.2f}ms'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
