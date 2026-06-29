#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算采集图片文件夹的帧率。

优先使用采集目录里的 timestamps.csv / timestamps.txt，因为它记录的是采集程序保存帧
时写入的设备时间戳或会话时间戳；如果没有时间戳文件，才退回用图片文件修改时间估算。
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".raw", ".npy"}
TIMESTAMP_NAMES = ("timestamps.csv", "timestamps.txt")


def find_images(path: Path) -> list[Path]:
    """只统计当前文件夹第一层图片，不递归，避免把 left/right 混在一起算。"""
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def find_timestamp_file(path: Path) -> Path | None:
    """在当前目录和父目录中寻找 timestamps 文件。"""
    candidates: list[Path] = []
    for base in (path, path.parent):
        for name in TIMESTAMP_NAMES:
            candidates.append(base / name)
    for item in candidates:
        if item.exists() and item.is_file():
            return item
    return None


def read_timestamps(ts_path: Path, image_count: int | None = None) -> list[tuple[str, float]]:
    """读取 timestamps.csv 或 timestamps.txt，返回 [(frame_id, timestamp_s), ...]。"""
    rows: list[tuple[str, float]] = []
    text = ts_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    if not text:
        return rows

    if ts_path.suffix.lower() == ".csv":
        reader = csv.reader(text)
        for row in reader:
            if not row or len(row) < 2:
                continue
            if row[0].strip().lower() in {"frame_id", "id", "frame"}:
                continue
            try:
                rows.append((row[0].strip(), float(row[1])))
            except ValueError:
                continue
    else:
        for line in text:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            if parts[0].lower() in {"frame_id", "id", "frame"}:
                continue
            try:
                rows.append((parts[0], float(parts[1])))
            except ValueError:
                continue

    if image_count is not None and len(rows) > image_count:
        rows = rows[:image_count]
    return rows


def calc_stats(times: list[float]) -> dict[str, float]:
    """根据时间戳列表计算平均 FPS、间隔分布和掉帧数量。"""
    if len(times) < 2:
        return {}

    intervals = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not intervals:
        return {}

    duration = times[-1] - times[0]
    avg_fps = (len(times) - 1) / duration if duration > 0 else 0.0
    median_dt = statistics.median(intervals)
    median_fps = 1.0 / median_dt if median_dt > 0 else 0.0
    max_dt = max(intervals)
    min_dt = min(intervals)
    over_40ms = sum(1 for x in intervals if x > 0.040)
    over_60ms = sum(1 for x in intervals if x > 0.060)

    return {
        "duration": duration,
        "avg_fps": avg_fps,
        "median_fps": median_fps,
        "avg_dt": sum(intervals) / len(intervals),
        "median_dt": median_dt,
        "min_dt": min_dt,
        "max_dt": max_dt,
        "over_40ms": float(over_40ms),
        "over_60ms": float(over_60ms),
    }


def calc_from_timestamps(path: Path, images: list[Path]) -> tuple[Path | None, list[tuple[str, float]], dict[str, float]]:
    ts_path = find_timestamp_file(path)
    if ts_path is None:
        return None, [], {}
    rows = read_timestamps(ts_path, image_count=len(images) if images else None)
    stats = calc_stats([t for _, t in rows])
    return ts_path, rows, stats


def calc_from_file_mtime(images: list[Path]) -> dict[str, float]:
    times = [p.stat().st_mtime for p in images]
    return calc_stats(times)


def print_report(path: Path) -> None:
    images = find_images(path)
    ts_path, rows, ts_stats = calc_from_timestamps(path, images)
    mtime_stats = calc_from_file_mtime(images) if len(images) >= 2 else {}

    print("=" * 72)
    print(f"检查目录: {path}")
    print(f"图片数量: {len(images)}")
    if images:
        print(f"第一张: {images[0].name}")
        print(f"最后一张: {images[-1].name}")

    if ts_path is not None:
        print(f"时间戳文件: {ts_path}")
        print(f"时间戳行数: {len(rows)}")
        if images and len(rows) != len(images):
            print(f"警告: 图片数量({len(images)}) 和时间戳行数({len(rows)}) 不一致")
        if ts_stats:
            print("\n基于 timestamps 的结果，推荐看这个:")
            print(f"  总时长: {ts_stats['duration']:.3f} s")
            print(f"  平均 FPS: {ts_stats['avg_fps']:.3f}")
            print(f"  中位 FPS: {ts_stats['median_fps']:.3f}")
            print(f"  平均帧间隔: {ts_stats['avg_dt'] * 1000:.3f} ms")
            print(f"  中位帧间隔: {ts_stats['median_dt'] * 1000:.3f} ms")
            print(f"  最小/最大帧间隔: {ts_stats['min_dt'] * 1000:.3f} / {ts_stats['max_dt'] * 1000:.3f} ms")
            print(f"  >40ms 间隔数: {int(ts_stats['over_40ms'])}")
            print(f"  >60ms 间隔数: {int(ts_stats['over_60ms'])}")
        else:
            print("警告: 找到 timestamps，但可用时间戳不足，无法计算 FPS")
    else:
        print("时间戳文件: 未找到，将使用图片文件修改时间粗略估算")

    if mtime_stats:
        print("\n基于文件修改时间的粗略结果:")
        print(f"  总时长: {mtime_stats['duration']:.3f} s")
        print(f"  平均 FPS: {mtime_stats['avg_fps']:.3f}")
        print(f"  中位 FPS: {mtime_stats['median_fps']:.3f}")
        print(f"  最大帧间隔: {mtime_stats['max_dt'] * 1000:.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="计算图片文件夹帧率，优先读取 timestamps.csv/txt。")
    parser.add_argument("path", nargs="?", help="图片文件夹路径，例如 D:\\OrbbecLiveCollector\\captures\\xxx\\left_rgb")
    args = parser.parse_args()

    path_text = args.path or input("请输入图片文件夹路径: ").strip().strip('"')
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"路径不存在: {path}")
    if not path.is_dir():
        raise SystemExit(f"不是文件夹: {path}")

    print_report(path)


if __name__ == "__main__":
    main()

