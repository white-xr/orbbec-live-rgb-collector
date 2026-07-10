#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Probe USB camera indices for a requested OpenCV profile."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe USB camera indices/backends.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--max-index", type=int, default=6)
    parser.add_argument("--backends", nargs="+", default=["msmf", "dshow"])
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=8.0)
    return parser.parse_args()


def probe_one(index: int, backend: str, args: argparse.Namespace) -> dict[str, object]:
    code = r"""
import cv2, json, time, sys
index = int(sys.argv[1])
backend_name = sys.argv[2]
width = int(sys.argv[3])
height = int(sys.argv[4])
fps = int(sys.argv[5])
fourcc = sys.argv[6]
seconds = float(sys.argv[7])
backends = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}
cap = cv2.VideoCapture(index, backends[backend_name])
if fourcc:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc.upper()))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)
opened = cap.isOpened()
count = 0
failed = 0
shape = None
t0 = time.perf_counter()
while opened and time.perf_counter() - t0 < seconds:
    ok, frame = cap.read()
    if ok and frame is not None:
        count += 1
        shape = list(frame.shape)
    else:
        failed += 1
        time.sleep(0.01)
elapsed = max(0.000001, time.perf_counter() - t0)
actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)) if opened else 0
chars = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) if 32 <= ((actual_fourcc >> (8 * i)) & 0xFF) <= 126 else "." for i in range(4))
result = {
    "index": index,
    "backend": backend_name,
    "opened": opened,
    "actual_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0,
    "actual_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0,
    "actual_fps_prop": float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0,
    "actual_fourcc": chars,
    "read_fps": count / elapsed,
    "frames": count,
    "failed": failed,
    "shape": shape,
}
cap.release()
print(json.dumps(result, ensure_ascii=False))
"""
    cmd = [
        sys.executable,
        "-c",
        code,
        str(index),
        backend,
        str(args.width),
        str(args.height),
        str(args.fps),
        str(args.fourcc),
        str(args.seconds),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(args.timeout),
        )
    except subprocess.TimeoutExpired:
        return {"index": index, "backend": backend, "opened": False, "timeout": True}

    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    return {
        "index": index,
        "backend": backend,
        "opened": False,
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip()[-300:],
    }


def main() -> int:
    args = parse_args()
    print(f"Probe target: {args.width}x{args.height}@{args.fps} {args.fourcc}")
    print("Close browser camera tests, Windows Camera, meeting apps, and other capture scripts first.\n")
    matches: list[dict[str, object]] = []
    for backend in args.backends:
        for index in range(max(0, int(args.max_index))):
            result = probe_one(index, backend, args)
            opened = bool(result.get("opened"))
            actual = f"{result.get('actual_width', 0)}x{result.get('actual_height', 0)}@{float(result.get('actual_fps_prop', 0.0)):.1f}"
            read_fps = float(result.get("read_fps", 0.0))
            ok = (
                opened
                and int(result.get("actual_width", 0)) == int(args.width)
                and int(result.get("actual_height", 0)) == int(args.height)
                and read_fps >= float(args.fps) * 0.9
            )
            mark = "OK" if ok else "--"
            if result.get("timeout"):
                print(f"{mark} backend={backend:5s} index={index}: timeout")
                continue
            print(
                f"{mark} backend={backend:5s} index={index}: opened={opened} "
                f"actual={actual} read_fps={read_fps:.1f} frames={result.get('frames', 0)} "
                f"fourcc={result.get('actual_fourcc', '')} shape={result.get('shape')}"
            )
            if ok:
                matches.append(result)

    if matches:
        print("\nRecommended:")
        best = max(matches, key=lambda item: float(item.get("read_fps", 0.0)))
        print(
            f"  USB Index={best['index']}, backend={best['backend']}, "
            f"{args.width}x{args.height}@{args.fps} {args.fourcc}"
        )
        return 0
    print("\nNo 1080p60 match found. Try unplugging other cameras or changing USB port.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
