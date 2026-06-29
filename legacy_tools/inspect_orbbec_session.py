#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Inspect an Orbbec RGB-D scan_data session')
    p.add_argument('session_dir', help='Path to scan_data_xxx folder')
    p.add_argument('--max-samples', type=int, default=120, help='Max RGB/depth pairs to sample for image-quality metrics')
    return p.parse_args()


def parse_simple_yaml(path: Path) -> dict:
    data = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line in ('%YAML:1.0', '---') or ':' not in line:
            continue
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            data[key] = value[1:-1]
            continue
        try:
            if any(ch in value for ch in '.eE'):
                data[key] = float(value)
            else:
                data[key] = int(value)
        except ValueError:
            data[key] = value
    return data


def list_pngs(folder: Path) -> list[Path]:
    return sorted([p for p in folder.glob('*.png') if p.is_file()], key=lambda p: p.name)


def expected_names(count: int) -> list[str]:
    return [f'{i:06d}.png' for i in range(1, count + 1)]


def sample_indices(count: int, max_samples: int) -> list[int]:
    if count <= 0:
        return []
    if count <= max_samples:
        return list(range(count))
    picks = np.linspace(0, count - 1, num=max_samples, dtype=int)
    return sorted(set(int(x) for x in picks))


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def read_timestamps(path: Path) -> list[tuple[int, float]]:
    rows = []
    if not path.exists():
        return rows
    for idx, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f'timestamps.txt line {idx} should have at least 2 columns, got: {line}')
        rows.append((int(parts[0]), float(parts[1])))
    return rows


def classify(metric_name: str, value: float) -> str:
    rules = {
        'blur_mean': (40.0, 80.0),
        'depth_valid_mean': (0.15, 0.35),
        'bright_clip_mean': (0.20, 0.10),
        'dark_clip_mean': (0.35, 0.20),
        'timestamp_std_ms': (8.0, 4.0),
        'gap_ratio_max': (2.0, 1.5),
        'fps_ratio': (0.50, 0.75),
    }
    fail_th, warn_th = rules[metric_name]
    if metric_name in ('blur_mean', 'depth_valid_mean', 'fps_ratio'):
        if value < fail_th:
            return 'FAIL'
        if value < warn_th:
            return 'WARN'
        return 'PASS'
    if value > fail_th:
        return 'FAIL'
    if value > warn_th:
        return 'WARN'
    return 'PASS'


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir).resolve()
    rgb_dir = session_dir / 'rgb'
    depth_dir = session_dir / 'depth'
    timestamps_path = session_dir / 'timestamps.txt'
    yaml_path = session_dir / 'camera_info.yaml'
    pose_note_path = session_dir / 'pose_note.txt'

    if not session_dir.exists():
        raise SystemExit(f'Session not found: {session_dir}')

    rgb_files = list_pngs(rgb_dir)
    depth_files = list_pngs(depth_dir)
    yaml_data = parse_simple_yaml(yaml_path)
    timestamp_rows = read_timestamps(timestamps_path)

    findings = []
    summary = {
        'session_dir': str(session_dir),
        'rgb_count': len(rgb_files),
        'depth_count': len(depth_files),
        'timestamps_count': len(timestamp_rows),
        'camera_info_exists': yaml_path.exists(),
        'pose_note_exists': pose_note_path.exists(),
    }

    if not rgb_files:
        findings.append({'level': 'FAIL', 'message': 'rgb folder has no PNG files'})
    if not depth_files:
        findings.append({'level': 'FAIL', 'message': 'depth folder has no PNG files'})

    rgb_names = [p.name for p in rgb_files]
    depth_names = [p.name for p in depth_files]
    exp_rgb = expected_names(len(rgb_names))
    exp_depth = expected_names(len(depth_names))
    if rgb_names != exp_rgb:
        findings.append({'level': 'FAIL', 'message': 'rgb filenames are not contiguous 000001.png ...'})
    if depth_names != exp_depth:
        findings.append({'level': 'FAIL', 'message': 'depth filenames are not contiguous 000001.png ...'})
    if rgb_names != depth_names:
        findings.append({'level': 'FAIL', 'message': 'rgb/depth filenames are not exactly paired'})
    if len(rgb_names) != len(timestamp_rows):
        findings.append({'level': 'FAIL', 'message': 'timestamps.txt line count does not match image pair count'})

    timestamp_id_errors = 0
    for expected_id, row in enumerate(timestamp_rows, start=1):
        frame_id = row[0]
        if frame_id != expected_id:
            timestamp_id_errors += 1
    if timestamp_id_errors:
        findings.append({'level': 'FAIL', 'message': f'timestamps.txt has {timestamp_id_errors} frame-id mismatches'})

    sampled = sample_indices(min(len(rgb_files), len(depth_files)), args.max_samples)

    rgb_shapes = set()
    depth_shapes = set()
    blur_vals = []
    bright_clip_vals = []
    dark_clip_vals = []
    mean_gray_vals = []
    depth_valid_vals = []
    depth_p50_vals = []
    depth_p95_vals = []
    rgb_type_errors = 0
    depth_type_errors = 0

    for idx in sampled:
        rgb = cv2.imread(str(rgb_files[idx]), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(str(depth_files[idx]), cv2.IMREAD_UNCHANGED)

        if rgb is None:
            findings.append({'level': 'FAIL', 'message': f'failed to read rgb/{rgb_files[idx].name}'})
            continue
        if depth is None:
            findings.append({'level': 'FAIL', 'message': f'failed to read depth/{depth_files[idx].name}'})
            continue

        rgb_shapes.add((int(rgb.shape[1]), int(rgb.shape[0])))
        depth_shapes.add((int(depth.shape[1]), int(depth.shape[0])))

        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            rgb_type_errors += 1
        if depth.dtype != np.uint16 or depth.ndim != 2:
            depth_type_errors += 1

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        blur_vals.append(variance_of_laplacian(gray))
        mean_gray_vals.append(float(gray.mean()))
        bright_clip_vals.append(float(np.mean(gray >= 250)))
        dark_clip_vals.append(float(np.mean(gray <= 5)))

        valid = depth > 0
        depth_valid_vals.append(float(np.mean(valid)))
        if np.any(valid):
            vals = depth[valid].astype(np.float32)
            depth_p50_vals.append(float(np.percentile(vals, 50)))
            depth_p95_vals.append(float(np.percentile(vals, 95)))

    if rgb_type_errors:
        findings.append({'level': 'FAIL', 'message': f'{rgb_type_errors} sampled RGB files are not 8-bit 3-channel PNG'})
    if depth_type_errors:
        findings.append({'level': 'FAIL', 'message': f'{depth_type_errors} sampled depth files are not uint16 single-channel PNG'})
    if len(rgb_shapes) > 1:
        findings.append({'level': 'FAIL', 'message': f'RGB resolution is inconsistent across samples: {sorted(rgb_shapes)}'})
    if len(depth_shapes) > 1:
        findings.append({'level': 'FAIL', 'message': f'Depth resolution is inconsistent across samples: {sorted(depth_shapes)}'})
    if len(rgb_shapes) == 1 and len(depth_shapes) == 1 and next(iter(rgb_shapes)) != next(iter(depth_shapes)):
        findings.append({'level': 'FAIL', 'message': 'RGB and depth resolutions differ'})

    actual_w = actual_h = None
    if len(rgb_shapes) == 1:
        actual_w, actual_h = next(iter(rgb_shapes))
        summary['actual_width'] = actual_w
        summary['actual_height'] = actual_h

    yaml_w = int(yaml_data.get('image_width', -1)) if 'image_width' in yaml_data else None
    yaml_h = int(yaml_data.get('image_height', -1)) if 'image_height' in yaml_data else None
    summary['camera_info_width'] = yaml_w
    summary['camera_info_height'] = yaml_h
    if actual_w is not None and actual_h is not None:
        if yaml_w != actual_w or yaml_h != actual_h:
            findings.append({
                'level': 'FAIL',
                'message': f'camera_info.yaml resolution ({yaml_w}x{yaml_h}) does not match images ({actual_w}x{actual_h})',
            })

    required_yaml_keys = [
        'image_width', 'image_height',
        'color_fx', 'color_fy', 'color_cx', 'color_cy',
        'color_k1', 'color_k2', 'color_p1', 'color_p2', 'color_k3',
        'depth_scale_from_raw',
    ]
    missing_yaml = [k for k in required_yaml_keys if k not in yaml_data]
    if missing_yaml:
        findings.append({'level': 'FAIL', 'message': f'camera_info.yaml is missing keys: {missing_yaml}'})

    deltas = []
    gap_ratio_max = None
    timestamp_std_ms = None
    if len(timestamp_rows) >= 2:
        ts = np.array([row[1] for row in timestamp_rows], dtype=np.float64)
        deltas = np.diff(ts)
        if np.any(deltas <= 0):
            findings.append({'level': 'FAIL', 'message': 'timestamps are not strictly increasing'})
        if deltas.size > 0:
            delta_mean = float(np.mean(deltas))
            delta_std = float(np.std(deltas))
            median_delta = float(np.median(deltas))
            gap_ratio_max = float(np.max(deltas) / median_delta) if median_delta > 0 else math.inf
            timestamp_std_ms = delta_std * 1000.0
            fps_from_timestamps = (1.0 / delta_mean) if delta_mean > 0 else 0.0
            summary['timestamp_mean_delta_s'] = delta_mean
            summary['timestamp_std_ms'] = timestamp_std_ms
            summary['timestamp_gap_ratio_max'] = gap_ratio_max
            summary['fps_from_timestamps'] = fps_from_timestamps
            for metric_name, value in (
                ('timestamp_std_ms', timestamp_std_ms),
                ('gap_ratio_max', gap_ratio_max),
            ):
                status = classify(metric_name, value)
                if status != 'PASS':
                    findings.append({'level': status, 'message': f'{metric_name}={value:.3f}'})

    metric_blocks = []
    if blur_vals:
        blur_mean = float(np.mean(blur_vals))
        blur_min = float(np.min(blur_vals))
        summary['blur_mean'] = blur_mean
        summary['blur_min'] = blur_min
        status = classify('blur_mean', blur_mean)
        metric_blocks.append({'name': 'blur_mean', 'value': blur_mean, 'status': status})
        if status != 'PASS':
            findings.append({'level': status, 'message': f'blur_mean={blur_mean:.2f}, image motion blur/weak texture may hurt tracking'})

    if bright_clip_vals:
        bright_clip_mean = float(np.mean(bright_clip_vals))
        dark_clip_mean = float(np.mean(dark_clip_vals))
        mean_gray = float(np.mean(mean_gray_vals))
        summary['bright_clip_mean'] = bright_clip_mean
        summary['dark_clip_mean'] = dark_clip_mean
        summary['mean_gray'] = mean_gray
        for metric_name, value in (
            ('bright_clip_mean', bright_clip_mean),
            ('dark_clip_mean', dark_clip_mean),
        ):
            status = classify(metric_name, value)
            metric_blocks.append({'name': metric_name, 'value': value, 'status': status})
            if status != 'PASS':
                findings.append({'level': status, 'message': f'{metric_name}={value:.3f}, exposure may be hurting tracking'})

    if depth_valid_vals:
        depth_valid_mean = float(np.mean(depth_valid_vals))
        summary['depth_valid_mean'] = depth_valid_mean
        summary['depth_p50_mm'] = float(np.mean(depth_p50_vals)) if depth_p50_vals else None
        summary['depth_p95_mm'] = float(np.mean(depth_p95_vals)) if depth_p95_vals else None
        status = classify('depth_valid_mean', depth_valid_mean)
        metric_blocks.append({'name': 'depth_valid_mean', 'value': depth_valid_mean, 'status': status})
        if status != 'PASS':
            findings.append({'level': status, 'message': f'depth_valid_mean={depth_valid_mean:.3f}, valid depth coverage is low'})

    health = 'PASS'
    if any(f['level'] == 'FAIL' for f in findings):
        health = 'FAIL'
    elif any(f['level'] == 'WARN' for f in findings):
        health = 'WARN'

    report = {
        'health': health,
        'summary': summary,
        'metrics': metric_blocks,
        'findings': findings,
        'notes': [
            'ORB-SLAM3 Current Frame being grayscale is normal.',
            'Tracking usually fails on low-texture, backlit, reflective, or motion-blurred data.',
        ],
    }

    report_json = session_dir / 'health_report.json'
    report_txt = session_dir / 'health_report.txt'
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = [
        f'Health: {health}',
        f'Session: {session_dir}',
        f'RGB pairs: {len(rgb_files)}',
        f'Depth pairs: {len(depth_files)}',
        f'Timestamps: {len(timestamp_rows)}',
    ]
    if 'fps_from_timestamps' in summary:
        lines.append(f'FPS from timestamps: {summary["fps_from_timestamps"]:.3f}')
    if actual_w is not None and actual_h is not None:
        lines.append(f'Image size: {actual_w}x{actual_h}')
    lines.append('')
    lines.append('Metrics:')
    for item in metric_blocks:
        lines.append(f'- {item["name"]}: {item["value"]:.6f} [{item["status"]}]')
    lines.append('')
    lines.append('Findings:')
    if findings:
        for finding in findings:
            lines.append(f'- {finding["level"]}: {finding["message"]}')
    else:
        lines.append('- PASS: no obvious structural/data-quality issues found in sampled files')
    lines.append('')
    lines.append('Notes:')
    for note in report['notes']:
        lines.append(f'- {note}')
    report_txt.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('\n'.join(lines))
    print(f'\nSaved: {report_txt}')
    print(f'Saved: {report_json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
