# -*- coding: utf-8 -*-
r"""
检查一次合并采集的跨相机同步情况。

用法：
python D:\OrbbecLiveCollector\check_capture_sync.py D:\OrbbecLiveCollector\captures\capture_YYYYMMDD_HHMMSS

说明：
- 不修改、不移动图片。
- 305 内部：检查 left_rgb/right_rgb 数量和编号是否一致。
- 335L 内部：检查 color/depth_raw 数量和编号是否一致。
- 跨相机：用 timestamps 做最近邻匹配，不用“第 N 张 = 第 N 张”的假设。
- 输出 sync_pairs.csv，记录每一对匹配帧的编号、时间戳和时间差。
r"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".raw", ".npy"}


def list_frame_ids(folder: Path) -> list[str]:
    if not folder.exists():
        return []
    ids = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            ids.append(p.stem)
    return sorted(ids)


def read_timestamps(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            parts = [x.strip() for x in line.split(",")]
        else:
            parts = line.split()
        if len(parts) < 2:
            continue
        try:
            rows.append((Path(parts[0]).stem, float(parts[1])))
        except ValueError:
            continue
    return rows


def compare_ids(name_a: str, ids_a: list[str], name_b: str, ids_b: list[str]) -> None:
    set_a = set(ids_a)
    set_b = set(ids_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    print(f"\n[{name_a} vs {name_b}]")
    print(f"  {name_a}: {len(ids_a)}")
    print(f"  {name_b}: {len(ids_b)}")
    print(f"  matched ids: {len(set_a & set_b)}")
    print(f"  only {name_a}: {len(only_a)}")
    print(f"  only {name_b}: {len(only_b)}")
    if only_a[:5]:
        print(f"  only {name_a} examples: {', '.join(only_a[:5])}")
    if only_b[:5]:
        print(f"  only {name_b} examples: {', '.join(only_b[:5])}")


def nearest_pairs(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> list[tuple[str, float, str, float, float]]:
    pairs: list[tuple[str, float, str, float, float]] = []
    if not a or not b:
        return pairs
    j = 0
    for aid, ats in a:
        while j + 1 < len(b) and abs(b[j + 1][1] - ats) <= abs(b[j][1] - ats):
            j += 1
        bid, bts = b[j]
        pairs.append((aid, ats, bid, bts, abs(ats - bts)))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 305 Dual RGB 和 335L RGB-D 的帧同步")
    parser.add_argument("session", help="capture_xxx 会话目录")
    parser.add_argument("--max-delta-ms", type=float, default=50.0, help="认为可接受的跨相机时间差，默认 50ms")
    args = parser.parse_args()

    session = Path(args.session).expanduser().resolve()
    cam305 = session / "eye_in_hand_305"
    cam335 = session / "eye_to_hand_335L"

    left_ids = list_frame_ids(cam305 / "left_rgb")
    right_ids = list_frame_ids(cam305 / "right_rgb")
    color_ids = list_frame_ids(cam335 / "color")
    depth_ids = list_frame_ids(cam335 / "depth_raw")

    print("=" * 72)
    print(f"Session: {session}")
    compare_ids("305 left_rgb", left_ids, "305 right_rgb", right_ids)
    compare_ids("335L color", color_ids, "335L depth_raw", depth_ids)

    ts305 = read_timestamps(cam305 / "timestamps.csv")
    ts335 = read_timestamps(cam335 / "timestamps.txt")
    print("\n[Timestamps]")
    print(f"  305 timestamps: {len(ts305)}")
    print(f"  335L timestamps: {len(ts335)}")

    pairs = nearest_pairs(ts305, ts335)
    if not pairs:
        print("  ERROR: 无法生成跨相机同步匹配，请检查 timestamps 文件。")
        return 2

    deltas_ms = [p[4] * 1000.0 for p in pairs]
    bad = [d for d in deltas_ms if d > args.max_delta_ms]
    print("\n[Cross-camera nearest timestamp match]")
    print(f"  matched pairs: {len(pairs)}")
    print(f"  mean delta: {mean(deltas_ms):.3f} ms")
    print(f"  median delta: {median(deltas_ms):.3f} ms")
    print(f"  max delta: {max(deltas_ms):.3f} ms")
    print(f"  >{args.max_delta_ms:.1f}ms pairs: {len(bad)}")

    out_csv = session / "sync_pairs.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["pair_index", "frame_305", "timestamp_305_s", "frame_335L", "timestamp_335L_s", "delta_ms", "ok"])
        for idx, (id305, t305, id335, t335, delta_s) in enumerate(pairs, start=1):
            delta_ms = delta_s * 1000.0
            w.writerow([idx, id305, f"{t305:.6f}", id335, f"{t335:.6f}", f"{delta_ms:.3f}", delta_ms <= args.max_delta_ms])

    print(f"\n已写出同步索引: {out_csv}")
    if bad:
        print("结论: 存在跨相机时间差过大的帧，后处理时应剔除这些 pair。")
    else:
        print("结论: 跨相机时间差在阈值内，可以按 sync_pairs.csv 做同步读取。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())