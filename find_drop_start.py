#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
查找采集序列从哪里开始掉帧。

用法 1：检查单个图片目录，例如 335L color：
python D:\OrbbecLiveCollector\find_drop_start.py D:\OrbbecLiveCollector\captures\capture_xxx\eye_to_hand_335L\color

用法 2：检查整个合并采集会话目录：
python D:\OrbbecLiveCollector\find_drop_start.py D:\OrbbecLiveCollector\captures\capture_xxx

默认认为 30fps 下帧间隔应约 33.33ms；超过 40ms 就算一次掉帧。
脚本只分析，不删除、不移动任何图片。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median

TIMESTAMP_NAMES = ("timestamps.csv", "timestamps.txt")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".raw", ".npy"}


def read_timestamps(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if not path.exists():
        return rows
    text = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    if path.suffix.lower() == ".csv":
        reader = csv.reader(text)
        for row in reader:
            if len(row) < 2:
                continue
            if row[0].strip().lower() in {"frame_id", "id", "frame"}:
                continue
            try:
                rows.append((Path(row[0].strip()).stem, float(row[1])))
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
                rows.append((Path(parts[0]).stem, float(parts[1])))
            except ValueError:
                continue
    return rows


def find_timestamp_file(path: Path) -> Path | None:
    candidates: list[Path] = []
    for base in (path, path.parent):
        for name in TIMESTAMP_NAMES:
            candidates.append(base / name)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def count_images(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def frame_num(frame_id: str) -> int | None:
    try:
        return int(Path(frame_id).stem)
    except ValueError:
        return None


def analyze_rows(name: str, rows: list[tuple[str, float]], threshold_ms: float, consecutive: int) -> dict[str, object]:
    result: dict[str, object] = {
        "name": name,
        "row_count": len(rows),
        "first_bad_index": None,
        "safe_frame_id": None,
        "safe_frame_num": None,
        "bad_from_frame": None,
        "bad_to_frame": None,
        "bad_dt_ms": None,
        "bad_count": 0,
        "max_dt_ms": None,
        "avg_fps_before_drop": None,
    }
    if len(rows) < 2:
        return result

    intervals = []
    bad_flags = []
    for i in range(len(rows) - 1):
        dt_ms = (rows[i + 1][1] - rows[i][1]) * 1000.0
        intervals.append(dt_ms)
        bad_flags.append(dt_ms > threshold_ms)

    result["bad_count"] = sum(1 for x in bad_flags if x)
    result["max_dt_ms"] = max(intervals) if intervals else None

    first_bad_i = None
    run = 0
    for i, is_bad in enumerate(bad_flags):
        if is_bad:
            run += 1
            if run >= consecutive:
                first_bad_i = i - consecutive + 1
                break
        else:
            run = 0

    if first_bad_i is None:
        first_id = rows[0][0]
        last_id = rows[-1][0]
        result["safe_frame_id"] = last_id
        result["safe_frame_num"] = frame_num(last_id)
        duration = rows[-1][1] - rows[0][1]
        result["avg_fps_before_drop"] = (len(rows) - 1) / duration if duration > 0 else None
        return result

    safe_i = max(0, first_bad_i)
    safe_id = rows[safe_i][0]
    result["first_bad_index"] = first_bad_i + 1
    result["safe_frame_id"] = safe_id
    result["safe_frame_num"] = frame_num(safe_id)
    result["bad_from_frame"] = rows[first_bad_i][0]
    result["bad_to_frame"] = rows[first_bad_i + 1][0]
    result["bad_dt_ms"] = intervals[first_bad_i]

    if safe_i >= 1:
        duration = rows[safe_i][1] - rows[0][1]
        result["avg_fps_before_drop"] = safe_i / duration if duration > 0 else None
    return result


def print_analysis(result: dict[str, object], threshold_ms: float) -> None:
    print(f"\n[{result['name']}]")
    print(f"  timestamps: {result['row_count']}")
    print(f"  掉帧间隔数(>{threshold_ms:.1f}ms): {result['bad_count']}")
    max_dt = result.get("max_dt_ms")
    if isinstance(max_dt, float):
        print(f"  最大帧间隔: {max_dt:.3f} ms")
    if result.get("first_bad_index") is None:
        print("  未发现超过阈值的掉帧点。")
        print(f"  建议保留到最后一帧: {result.get('safe_frame_id')}")
    else:
        print(f"  第一处掉帧: {result.get('bad_from_frame')} -> {result.get('bad_to_frame')}")
        print(f"  该间隔: {float(result.get('bad_dt_ms')):.3f} ms")
        print(f"  建议保留到掉帧前一帧: {result.get('safe_frame_id')}")
    fps = result.get("avg_fps_before_drop")
    if isinstance(fps, float):
        print(f"  掉帧前平均 FPS: {fps:.3f}")


def analyze_single_path(path: Path, threshold_ms: float, consecutive: int) -> list[dict[str, object]]:
    ts_path = find_timestamp_file(path)
    if ts_path is None:
        raise SystemExit(f"找不到 timestamps.csv/timestamps.txt: {path}")
    rows = read_timestamps(ts_path)
    img_count = count_images(path)
    name = f"{path.name} ({ts_path.name}, images={img_count})"
    return [analyze_rows(name, rows, threshold_ms, consecutive)]


def analyze_session(session: Path, threshold_ms: float, consecutive: int) -> list[dict[str, object]]:
    targets = [
        ("305 Dual RGB", session / "eye_in_hand_305" / "timestamps.csv"),
        ("335L RGB-D", session / "eye_to_hand_335L" / "timestamps.txt"),
    ]
    results: list[dict[str, object]] = []
    for name, ts_path in targets:
        if ts_path.exists():
            results.append(analyze_rows(name, read_timestamps(ts_path), threshold_ms, consecutive))
    if not results:
        return analyze_single_path(session, threshold_ms, consecutive)
    return results


def write_keep_summary(session: Path, results: list[dict[str, object]], threshold_ms: float) -> None:
    safe_nums = [r.get("safe_frame_num") for r in results if isinstance(r.get("safe_frame_num"), int)]
    if not safe_nums:
        return
    keep_until = min(safe_nums)
    out = session / "drop_start_report.txt"
    lines = [
        f"threshold_ms={threshold_ms:.3f}",
        f"recommended_keep_until_frame={keep_until:06d}",
        "",
    ]
    for r in results:
        lines.append(f"[{r.get('name')}]")
        lines.append(f"timestamps={r.get('row_count')}")
        lines.append(f"bad_count={r.get('bad_count')}")
        lines.append(f"safe_frame_id={r.get('safe_frame_id')}")
        lines.append(f"first_bad={r.get('bad_from_frame')}->{r.get('bad_to_frame')}")
        lines.append(f"bad_dt_ms={r.get('bad_dt_ms')}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已写出报告: {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="查找采集序列第一处掉帧位置")
    parser.add_argument("path", nargs="?", help="图片目录或 capture_xxx 会话目录")
    parser.add_argument("--threshold-ms", type=float, default=40.0, help="超过多少 ms 算掉帧，默认 40ms")
    parser.add_argument("--consecutive", type=int, default=1, help="连续几次超过阈值才算开始掉帧，默认 1")
    args = parser.parse_args()

    path_text = args.path or input("请输入图片目录或 capture 会话目录: ").strip().strip('"')
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"路径不存在: {path}")

    print("=" * 72)
    print(f"检查路径: {path}")
    print(f"掉帧阈值: >{args.threshold_ms:.1f} ms")
    print(f"连续掉帧判定: {max(1, args.consecutive)} 次")

    if (path / "eye_in_hand_305").exists() or (path / "eye_to_hand_335L").exists():
        results = analyze_session(path, args.threshold_ms, max(1, args.consecutive))
        for r in results:
            print_analysis(r, args.threshold_ms)
        safe_nums = [r.get("safe_frame_num") for r in results if isinstance(r.get("safe_frame_num"), int)]
        if safe_nums:
            keep_until = min(safe_nums)
            print("\n[共同建议]")
            print(f"  两台相机保守共同保留到: {keep_until:06d}")
            print(f"  也就是后处理只用 <= {keep_until:06d} 的同步帧。")
            write_keep_summary(path, results, args.threshold_ms)
    else:
        results = analyze_single_path(path, args.threshold_ms, max(1, args.consecutive))
        for r in results:
            print_analysis(r, args.threshold_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())