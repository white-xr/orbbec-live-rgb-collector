#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


DEFAULT_FPS = 30.0
OUTPUT_NAME = 'rgb_video.mp4'


def read_fps(timestamps_path: Path, fallback: float = DEFAULT_FPS) -> float:
    if not timestamps_path.exists():
        return fallback

    times = []
    for raw in timestamps_path.read_text(encoding='utf-8').splitlines():
        parts = raw.split()
        if len(parts) < 2:
            continue
        try:
            times.append(float(parts[1]))
        except ValueError:
            continue

    if len(times) < 2:
        return fallback

    duration = times[-1] - times[0]
    if duration <= 0:
        return fallback

    fps = (len(times) - 1) / duration
    if fps <= 0 or fps > 240:
        return fallback
    return fps


def list_images(rgb_dir: Path) -> list[Path]:
    return sorted([p for p in rgb_dir.glob('*.png') if p.is_file()], key=lambda p: p.name)


def open_writer(output_path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    if output_path.exists():
        output_path.unlink()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f'Failed to open VideoWriter: {output_path}')
    return writer


def make_video(rgb_dir: Path) -> dict:
    images = list_images(rgb_dir)
    output_path = rgb_dir.parent / OUTPUT_NAME
    timestamps_path = rgb_dir.parent / 'timestamps.txt'

    if not images:
        return {
            'rgb_dir': str(rgb_dir),
            'output': str(output_path),
            'status': 'skipped_empty',
            'image_count': 0,
        }

    first = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f'Failed to read first image: {images[0]}')

    height, width = first.shape[:2]
    fps = read_fps(timestamps_path)
    writer = open_writer(output_path, fps, (width, height))

    written = 0
    skipped = 0
    resized = 0
    try:
        for path in images:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                skipped += 1
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                resized += 1
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    return {
        'rgb_dir': str(rgb_dir),
        'output': str(output_path),
        'status': 'ok',
        'image_count': len(images),
        'written': written,
        'skipped': skipped,
        'resized': resized,
        'fps': fps,
        'width': width,
        'height': height,
    }


def find_rgb_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob('rgb') if p.is_dir()])


def parse_args():
    parser = argparse.ArgumentParser(description='Convert every captures/**/rgb PNG sequence into rgb_video.mp4')
    parser.add_argument('--root', default='captures', help='Root folder to scan for rgb directories')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f'Root not found: {root}')

    rgb_dirs = find_rgb_dirs(root)
    if not rgb_dirs:
        print(f'No rgb folders found under {root}')
        return 0

    print(f'Found {len(rgb_dirs)} rgb folders under {root}')
    for rgb_dir in rgb_dirs:
        result = make_video(rgb_dir)
        if result['status'] == 'skipped_empty':
            print(f'[SKIP] {result["rgb_dir"]}: no PNG files')
            continue
        print(
            f'[OK] {result["output"]} | '
            f'{result["written"]}/{result["image_count"]} frames | '
            f'{result["width"]}x{result["height"]} | '
            f'fps={result["fps"]:.3f} | '
            f'skipped={result["skipped"]} | resized={result["resized"]}'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
