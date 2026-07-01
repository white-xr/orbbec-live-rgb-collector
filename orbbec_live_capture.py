#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import ctypes
import importlib.util
import json
import os
import queue
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# Stream type
OB_STREAM_IR = 1
OB_STREAM_COLOR = 2
OB_STREAM_DEPTH = 3
OB_STREAM_IR_LEFT = 6
OB_STREAM_IR_RIGHT = 7
OB_STREAM_COLOR_LEFT = 11
OB_STREAM_COLOR_RIGHT = 12

# Sensor type
OB_SENSOR_IR = 1
OB_SENSOR_COLOR = 2
OB_SENSOR_DEPTH = 3
OB_SENSOR_IR_LEFT = 6
OB_SENSOR_IR_RIGHT = 7
OB_SENSOR_COLOR_LEFT = 11
OB_SENSOR_COLOR_RIGHT = 12

# Frame type
OB_FRAME_IR = 1
OB_FRAME_COLOR = 2
OB_FRAME_DEPTH = 3
OB_FRAME_IR_LEFT = 8
OB_FRAME_IR_RIGHT = 9
OB_FRAME_COLOR_LEFT = 13
OB_FRAME_COLOR_RIGHT = 14

# Frame formats
OB_FORMAT_YUYV = 0
OB_FORMAT_UYVY = 2
OB_FORMAT_MJPG = 5
OB_FORMAT_Y16 = 8
OB_FORMAT_Y8 = 9
OB_FORMAT_GRAY = 13
OB_FORMAT_RGB = 22
OB_FORMAT_BGR = 23
OB_FORMAT_BGRA = 25
OB_FORMAT_RGBA = 31

FORMAT_NAMES = {
    OB_FORMAT_YUYV: 'YUYV',
    OB_FORMAT_UYVY: 'UYVY',
    OB_FORMAT_MJPG: 'MJPG',
    OB_FORMAT_Y16: 'Y16',
    OB_FORMAT_Y8: 'Y8',
    OB_FORMAT_GRAY: 'GRAY',
    OB_FORMAT_RGB: 'RGB',
    OB_FORMAT_BGR: 'BGR',
    OB_FORMAT_BGRA: 'BGRA',
    OB_FORMAT_RGBA: 'RGBA',
}

FORMAT_IDS_BY_NAME = {name: fmt for fmt, name in FORMAT_NAMES.items()}
DUAL_COLOR_PRESET_NAME = 'Dual Color Streams'
PNG_COMPRESSION = 3

# Orbbec SDK property IDs. These are applied to the real camera before the RGB-D
# pipeline starts when they are enabled in config.yaml.
OB_PROP_LASER_BOOL = 3
OB_PROP_LASER_CURRENT_FLOAT = 5
OB_PROP_FLOOD_BOOL = 6
OB_PROP_FLOOD_LEVEL_INT = 7
OB_PROP_DEPTH_MIRROR_BOOL = 14
OB_PROP_DEPTH_FLIP_BOOL = 15
OB_PROP_DEPTH_POSTFILTER_BOOL = 16
OB_PROP_DEPTH_HOLEFILTER_BOOL = 17
OB_PROP_MIN_DEPTH_INT = 22
OB_PROP_MAX_DEPTH_INT = 23
OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_BOOL = 24
OB_PROP_DEPTH_PRECISION_LEVEL_INT = 75
OB_PROP_COLOR_MIRROR_BOOL = 81
OB_PROP_COLOR_FLIP_BOOL = 82
OB_PROP_LASER_POWER_LEVEL_CONTROL_INT = 99
OB_PROP_COLOR_AUTO_EXPOSURE_BOOL = 2000
OB_PROP_COLOR_EXPOSURE_INT = 2001
OB_PROP_COLOR_GAIN_INT = 2002
OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL = 2003
OB_PROP_COLOR_WHITE_BALANCE_INT = 2004
OB_PROP_COLOR_BRIGHTNESS_INT = 2005
OB_PROP_COLOR_SHARPNESS_INT = 2006
OB_PROP_COLOR_SATURATION_INT = 2008
OB_PROP_COLOR_CONTRAST_INT = 2009
OB_PROP_COLOR_GAMMA_INT = 2010
OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT = 2015
OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL = 2016
OB_PROP_DEPTH_EXPOSURE_INT = 2017
OB_PROP_DEPTH_GAIN_INT = 2018
OB_PROP_IR_AUTO_EXPOSURE_BOOL = 2025
OB_PROP_IR_EXPOSURE_INT = 2026
OB_PROP_IR_GAIN_INT = 2027
OB_PROP_IR_RECTIFY_BOOL = 2040
OB_PROP_DEPTH_AUTO_EXPOSURE_PRIORITY_INT = 2052

PROPERTY_SPECS = {
    'OB_PROP_LASER_BOOL': (OB_PROP_LASER_BOOL, 'bool'),
    'OB_PROP_LASER_CURRENT_FLOAT': (OB_PROP_LASER_CURRENT_FLOAT, 'float'),
    'OB_PROP_FLOOD_BOOL': (OB_PROP_FLOOD_BOOL, 'bool'),
    'OB_PROP_FLOOD_LEVEL_INT': (OB_PROP_FLOOD_LEVEL_INT, 'int'),
    'OB_PROP_DEPTH_MIRROR_BOOL': (OB_PROP_DEPTH_MIRROR_BOOL, 'bool'),
    'OB_PROP_DEPTH_FLIP_BOOL': (OB_PROP_DEPTH_FLIP_BOOL, 'bool'),
    'OB_PROP_DEPTH_POSTFILTER_BOOL': (OB_PROP_DEPTH_POSTFILTER_BOOL, 'bool'),
    'OB_PROP_DEPTH_HOLEFILTER_BOOL': (OB_PROP_DEPTH_HOLEFILTER_BOOL, 'bool'),
    'OB_PROP_MIN_DEPTH_INT': (OB_PROP_MIN_DEPTH_INT, 'int'),
    'OB_PROP_MAX_DEPTH_INT': (OB_PROP_MAX_DEPTH_INT, 'int'),
    'OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_BOOL': (OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_BOOL, 'bool'),
    'OB_PROP_DEPTH_PRECISION_LEVEL_INT': (OB_PROP_DEPTH_PRECISION_LEVEL_INT, 'int'),
    'OB_PROP_COLOR_MIRROR_BOOL': (OB_PROP_COLOR_MIRROR_BOOL, 'bool'),
    'OB_PROP_COLOR_FLIP_BOOL': (OB_PROP_COLOR_FLIP_BOOL, 'bool'),
    'OB_PROP_LASER_POWER_LEVEL_CONTROL_INT': (OB_PROP_LASER_POWER_LEVEL_CONTROL_INT, 'int'),
    'OB_PROP_COLOR_AUTO_EXPOSURE_BOOL': (OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, 'bool'),
    'OB_PROP_COLOR_EXPOSURE_INT': (OB_PROP_COLOR_EXPOSURE_INT, 'int'),
    'OB_PROP_COLOR_GAIN_INT': (OB_PROP_COLOR_GAIN_INT, 'int'),
    'OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL': (OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, 'bool'),
    'OB_PROP_COLOR_WHITE_BALANCE_INT': (OB_PROP_COLOR_WHITE_BALANCE_INT, 'int'),
    'OB_PROP_COLOR_BRIGHTNESS_INT': (OB_PROP_COLOR_BRIGHTNESS_INT, 'int'),
    'OB_PROP_COLOR_SHARPNESS_INT': (OB_PROP_COLOR_SHARPNESS_INT, 'int'),
    'OB_PROP_COLOR_SATURATION_INT': (OB_PROP_COLOR_SATURATION_INT, 'int'),
    'OB_PROP_COLOR_CONTRAST_INT': (OB_PROP_COLOR_CONTRAST_INT, 'int'),
    'OB_PROP_COLOR_GAMMA_INT': (OB_PROP_COLOR_GAMMA_INT, 'int'),
    'OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT': (OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT, 'int'),
    'OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL': (OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, 'bool'),
    'OB_PROP_DEPTH_EXPOSURE_INT': (OB_PROP_DEPTH_EXPOSURE_INT, 'int'),
    'OB_PROP_DEPTH_GAIN_INT': (OB_PROP_DEPTH_GAIN_INT, 'int'),
    'OB_PROP_IR_AUTO_EXPOSURE_BOOL': (OB_PROP_IR_AUTO_EXPOSURE_BOOL, 'bool'),
    'OB_PROP_IR_EXPOSURE_INT': (OB_PROP_IR_EXPOSURE_INT, 'int'),
    'OB_PROP_IR_GAIN_INT': (OB_PROP_IR_GAIN_INT, 'int'),
    'OB_PROP_IR_RECTIFY_BOOL': (OB_PROP_IR_RECTIFY_BOOL, 'bool'),
    'OB_PROP_DEPTH_AUTO_EXPOSURE_PRIORITY_INT': (OB_PROP_DEPTH_AUTO_EXPOSURE_PRIORITY_INT, 'int'),
}

# Config enum
OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE = 0
ALIGN_DISABLE = 0
ALIGN_D2C_HW_MODE = 1
ALIGN_D2C_SW_MODE = 2

ALIGN_MODES_BY_NAME = {
    'hardware': (ALIGN_D2C_HW_MODE, 'ALIGN_D2C_HW_MODE'),
    'hw': (ALIGN_D2C_HW_MODE, 'ALIGN_D2C_HW_MODE'),
    'software': (ALIGN_D2C_SW_MODE, 'ALIGN_D2C_SW_MODE'),
    'sw': (ALIGN_D2C_SW_MODE, 'ALIGN_D2C_SW_MODE'),
    'disable': (ALIGN_DISABLE, 'ALIGN_DISABLE'),
    'none': (ALIGN_DISABLE, 'ALIGN_DISABLE'),
}

DEFAULT_CONFIG_PATH = Path(__file__).with_name('config.yaml')

DEFAULT_CAPTURE_CONFIG: dict[str, Any] = {
    'sdk_bin': r'D:\OrbbecSDK_v2\bin',
    'output_root': r'D:\OrbbecLiveCollector\captures',
    'output': {
        'base_dir': r'D:\OrbbecLiveCollector\captures',
        'color_format': 'png',
        'depth_raw_format': 'png',
        'depth_vis_format': 'png',
        'ir_format': 'png',
        'save_depth_vis': True,
        'png_compression': 3,
        'jpg_quality': 95,
        'minimal_dual_rgb_layout': False,
        'writer_threads': 1,
        'write_queue_maxsize': 256,
    },
    'streams': {
        'color': True,
        'color_left': False,
        'color_right': False,
        'depth': True,
        'ir_left': False,
        'ir_right': False,
        'imu': False,
    },
    'device': {
        'index': None,
        'serial': '',
    },
    'device_preset': {
        'enabled': False,
        'name': '',
        'required': True,
        'settle_ms': 800,
    },
    'intrinsics_reference': {},
    'session': {
        'tag': '',
        'output_width': 0,
        'output_height': 0,
        'max_sync_diff_ms': 15.0,
    },
    'stream_profile': {
        'enabled': True,
        'use_fixed_profile_for_305_335l': True,
        'fallback_to_sdk_default': False,
        'color': {
            'width': 1280,
            'height': 800,
            'fps': 30,
            'formats': ['BGR', 'RGB', 'YUYV', 'MJPG', 'BGRA', 'RGBA', 'UYVY'],
        },
        'depth': {
            'width': 1280,
            'height': 800,
            'fps': 30,
            'formats': ['Y16'],
        },
        'dual_color': {
            'width': 1280,
            'height': 800,
            'fps': 30,
            'formats': ['BGR', 'RGB', 'YUYV', 'MJPG', 'BGRA', 'RGBA', 'UYVY'],
        },
        'ir': {
            'width': 1280,
            'height': 800,
            'fps': 30,
            'formats': ['Y16', 'Y8', 'GRAY'],
        },
    },
    'align': {
        'mode_order': ['hardware', 'software'],
        'depth_scale_after_align': True,
    },
    'pipeline': {
        'frame_timeout_ms': 200,
        'frame_aggregate_mode': 'FULL_FRAME_REQUIRE',
        'align_mode': 'ALIGN_D2C_HW_MODE',
    },
    'pointcloud': {
        'enabled': True,
        'require_aligned_depth_to_color': True,
        'depth_unit': 'mm',
        'preferred_alignment': 'depth_to_color',
    },
    'rectification': {
        # Fast-FoundationStereo needs rectified stereo pairs. The SDK property
        # can be requested, but this collector still marks output as unverified
        # unless assume_rectified is explicitly set by the user.
        'try_sdk_rectify': False,
        'assume_rectified': False,
    },
    'stereo': {
        'baseline_m': 0.0,
        'left_right_order_verified_by_user': False,
        'write_identity_poses_csv': False,
        'identity_pose_assumes_static_scene': False,
    },
    'fusion': {
        'enabled': False,
    },
    'pose_source': {
        'mode': 'none',
    },
    'preview': {
        'enabled': True,
        'window_width': 1400,
        'window_height': 800,
    },
    'camera_properties': {
        'enabled': True,
        'strict': False,
        'values': {},
    },
}


def now_str() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def normalize_repo_style_config(user_config: dict[str, Any]) -> dict[str, Any]:
    if 'stream_profile' in user_config or 'camera_properties' in user_config:
        return deep_merge(DEFAULT_CAPTURE_CONFIG, user_config)

    settings = copy.deepcopy(DEFAULT_CAPTURE_CONFIG)
    output_cfg = user_config.get('output', {}) or {}
    streams_cfg = user_config.get('streams', {}) or {}
    color_cfg = user_config.get('color', {}) or {}
    dual_color_cfg = user_config.get('dual_color', {}) or {}
    depth_cfg = user_config.get('depth', {}) or {}
    ir_cfg = user_config.get('ir', {}) or {}
    device_cfg = user_config.get('device', {}) or {}
    session_cfg = user_config.get('session', {}) or {}
    device_preset_cfg = user_config.get('device_preset', {}) or {}
    intrinsics_reference_cfg = user_config.get('intrinsics_reference', {}) or {}
    pipeline_cfg = user_config.get('pipeline', {}) or {}
    preview_cfg = user_config.get('preview', {}) or {}
    pointcloud_cfg = user_config.get('pointcloud', {}) or {}
    rectification_cfg = user_config.get('rectification', {}) or {}
    stereo_cfg = user_config.get('stereo', {}) or {}
    fusion_cfg = user_config.get('fusion', {}) or {}
    pose_source_cfg = user_config.get('pose_source', {}) or {}

    if output_cfg:
        settings['output'] = deep_merge(settings['output'], output_cfg)
        settings['output_root'] = str(output_cfg.get('base_dir', settings['output_root']))
    if streams_cfg:
        settings['streams'] = deep_merge(settings['streams'], streams_cfg)
    if device_cfg:
        settings['device'] = deep_merge(settings.get('device', {}), device_cfg)
    if session_cfg:
        settings['session'] = deep_merge(settings.get('session', {}), session_cfg)
    if preview_cfg:
        settings['preview'] = deep_merge(settings['preview'], preview_cfg)
    if pipeline_cfg:
        settings['pipeline'] = deep_merge(settings['pipeline'], pipeline_cfg)
    if pointcloud_cfg:
        settings['pointcloud'] = deep_merge(settings['pointcloud'], pointcloud_cfg)
    if rectification_cfg:
        settings['rectification'] = deep_merge(settings.get('rectification', {}), rectification_cfg)
    if stereo_cfg:
        settings['stereo'] = deep_merge(settings.get('stereo', {}), stereo_cfg)
    if fusion_cfg:
        settings['fusion'] = deep_merge(settings.get('fusion', {}), fusion_cfg)
    if pose_source_cfg:
        settings['pose_source'] = deep_merge(settings.get('pose_source', {}), pose_source_cfg)
    if device_preset_cfg:
        settings['device_preset'] = deep_merge(settings.get('device_preset', {}), device_preset_cfg)
    if intrinsics_reference_cfg:
        settings['intrinsics_reference'] = copy.deepcopy(intrinsics_reference_cfg)

    settings['stream_profile']['enabled'] = bool(streams_cfg.get('color', True) and streams_cfg.get('depth', True))
    settings['stream_profile']['fallback_to_sdk_default'] = bool(pipeline_cfg.get('fallback_to_sdk_default', False))

    if color_cfg:
        settings['stream_profile']['color']['width'] = int(color_cfg.get('width', settings['stream_profile']['color']['width']))
        settings['stream_profile']['color']['height'] = int(color_cfg.get('height', settings['stream_profile']['color']['height']))
        settings['stream_profile']['color']['fps'] = int(color_cfg.get('fps', settings['stream_profile']['color']['fps']))
        if color_cfg.get('format'):
            settings['stream_profile']['color']['formats'] = [str(color_cfg['format']).upper()]
    if dual_color_cfg:
        settings['stream_profile']['dual_color']['width'] = int(dual_color_cfg.get('width', settings['stream_profile']['dual_color']['width']))
        settings['stream_profile']['dual_color']['height'] = int(dual_color_cfg.get('height', settings['stream_profile']['dual_color']['height']))
        settings['stream_profile']['dual_color']['fps'] = int(dual_color_cfg.get('fps', settings['stream_profile']['dual_color']['fps']))
        if dual_color_cfg.get('formats'):
            settings['stream_profile']['dual_color']['formats'] = [str(v).upper() for v in dual_color_cfg['formats']]
        elif dual_color_cfg.get('format'):
            settings['stream_profile']['dual_color']['formats'] = [str(dual_color_cfg['format']).upper()]

    if depth_cfg:
        settings['stream_profile']['depth']['width'] = int(depth_cfg.get('width', settings['stream_profile']['depth']['width']))
        settings['stream_profile']['depth']['height'] = int(depth_cfg.get('height', settings['stream_profile']['depth']['height']))
        settings['stream_profile']['depth']['fps'] = int(depth_cfg.get('fps', settings['stream_profile']['depth']['fps']))
        if depth_cfg.get('format'):
            settings['stream_profile']['depth']['formats'] = [str(depth_cfg['format']).upper()]
    if ir_cfg:
        settings['stream_profile']['ir']['width'] = int(ir_cfg.get('width', settings['stream_profile']['ir']['width']))
        settings['stream_profile']['ir']['height'] = int(ir_cfg.get('height', settings['stream_profile']['ir']['height']))
        settings['stream_profile']['ir']['fps'] = int(ir_cfg.get('fps', settings['stream_profile']['ir']['fps']))
        if ir_cfg.get('formats'):
            settings['stream_profile']['ir']['formats'] = [str(v).upper() for v in ir_cfg['formats']]
        elif ir_cfg.get('format'):
            settings['stream_profile']['ir']['formats'] = [str(ir_cfg['format']).upper()]

    align_mode = str(pipeline_cfg.get('align_mode', settings['pipeline']['align_mode'])).strip().upper()
    if align_mode == 'ALIGN_D2C_HW_MODE':
        settings['align']['mode_order'] = ['hardware']
    elif align_mode == 'ALIGN_D2C_SW_MODE':
        settings['align']['mode_order'] = ['software']
    elif align_mode == 'DISABLE':
        settings['align']['mode_order'] = ['disable']

    values = settings['camera_properties']['values']
    if 'auto_exposure' in color_cfg:
        values['OB_PROP_COLOR_AUTO_EXPOSURE_BOOL'] = bool(color_cfg.get('auto_exposure'))
    if color_cfg.get('auto_exposure') is False:
        values['OB_PROP_COLOR_EXPOSURE_INT'] = color_cfg.get('exposure')
        values['OB_PROP_COLOR_GAIN_INT'] = color_cfg.get('gain')
    if 'auto_white_balance' in color_cfg:
        values['OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL'] = bool(color_cfg.get('auto_white_balance'))
    if color_cfg.get('auto_white_balance') is False:
        values['OB_PROP_COLOR_WHITE_BALANCE_INT'] = color_cfg.get('white_balance')
    if dual_color_cfg:
        if 'auto_exposure' in dual_color_cfg:
            values['OB_PROP_COLOR_AUTO_EXPOSURE_BOOL'] = bool(dual_color_cfg.get('auto_exposure'))
        if dual_color_cfg.get('auto_exposure') is False:
            values['OB_PROP_COLOR_EXPOSURE_INT'] = dual_color_cfg.get('exposure')
            values['OB_PROP_COLOR_GAIN_INT'] = dual_color_cfg.get('gain')
        if 'auto_white_balance' in dual_color_cfg:
            values['OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL'] = bool(dual_color_cfg.get('auto_white_balance'))
        if dual_color_cfg.get('auto_white_balance') is False:
            values['OB_PROP_COLOR_WHITE_BALANCE_INT'] = dual_color_cfg.get('white_balance')

    if streams_cfg.get('depth', True):
        if 'auto_exposure' in depth_cfg:
            values['OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL'] = bool(depth_cfg.get('auto_exposure'))
        if depth_cfg.get('auto_exposure') is False:
            values['OB_PROP_DEPTH_EXPOSURE_INT'] = depth_cfg.get('exposure')
            values['OB_PROP_DEPTH_GAIN_INT'] = depth_cfg.get('gain')
        if depth_cfg.get('min_depth_mm') is not None:
            values['OB_PROP_MIN_DEPTH_INT'] = int(depth_cfg['min_depth_mm'])
        if depth_cfg.get('max_depth_mm') is not None:
            values['OB_PROP_MAX_DEPTH_INT'] = int(depth_cfg['max_depth_mm'])

    if streams_cfg.get('ir_left', False) or streams_cfg.get('ir_right', False):
        if 'auto_exposure' in ir_cfg:
            values['OB_PROP_IR_AUTO_EXPOSURE_BOOL'] = bool(ir_cfg.get('auto_exposure'))
        if ir_cfg.get('auto_exposure') is False:
            values['OB_PROP_IR_EXPOSURE_INT'] = ir_cfg.get('exposure')
            values['OB_PROP_IR_GAIN_INT'] = ir_cfg.get('gain')
        if 'rectify' in ir_cfg:
            rectify = bool(ir_cfg.get('rectify'))
            values['OB_PROP_IR_RECTIFY_BOOL'] = rectify
            settings.setdefault('rectification', {})['try_sdk_rectify'] = rectify

    return settings


def load_capture_config(config_path: Path) -> dict[str, Any]:
    settings = copy.deepcopy(DEFAULT_CAPTURE_CONFIG)
    settings['_config_path'] = str(config_path)
    if not config_path.exists():
        print(f'[{now_str()}] WARN config file not found, using built-in defaults: {config_path}')
        return settings

    suffix = config_path.suffix.lower()
    if suffix in ('.yaml', '.yml'):
        try:
            import yaml
        except ImportError as ex:
            raise RuntimeError('缺少 PyYAML，请先运行：pip install pyyaml') from ex
        with config_path.open('r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f) or {}
    elif suffix == '.py':
        spec = importlib.util.spec_from_file_location('orbbec_capture_config_runtime', str(config_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Unable to load config file: {config_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        user_config = getattr(module, 'CONFIG', None)
    else:
        raise RuntimeError(f'Unsupported config file type: {config_path}')

    if not isinstance(user_config, dict):
        raise RuntimeError(f'{config_path} must contain a dict config.')

    settings = normalize_repo_style_config(user_config)
    settings['_config_path'] = str(config_path)
    return settings


def cli_option_present(*names: str) -> bool:
    for arg in sys.argv[1:]:
        for name in names:
            if arg == name or arg.startswith(name + '='):
                return True
    return False


def apply_config_defaults_to_args(args, settings: dict[str, Any]) -> None:
    device_cfg = settings.get('device', {}) or {}
    session_cfg = settings.get('session', {}) or {}

    if not cli_option_present('--sdk-bin'):
        args.sdk_bin = str(settings.get('sdk_bin', args.sdk_bin))
    if not cli_option_present('--output-root'):
        args.output_root = str(settings.get('output_root', args.output_root))
    if not cli_option_present('--tag'):
        args.tag = str(session_cfg.get('tag', args.tag) or '')
    if not cli_option_present('--width'):
        args.width = int(session_cfg.get('output_width', args.width) or 0)
    if not cli_option_present('--height'):
        args.height = int(session_cfg.get('output_height', args.height) or 0)
    if not cli_option_present('--max-sync-diff-ms'):
        args.max_sync_diff_ms = float(session_cfg.get('max_sync_diff_ms', args.max_sync_diff_ms))
    if not cli_option_present('--device-index'):
        index = device_cfg.get('index', args.device_index)
        args.device_index = None if index is None else int(index)
    if not cli_option_present('--serial'):
        args.serial = str(device_cfg.get('serial', args.serial) or '')
    if not cli_option_present('--model-hint'):
        args.model_hint = str(device_cfg.get('model_hint', args.model_hint) or '')


def validate_capture_settings(settings: dict[str, Any]) -> None:
    streams_cfg = settings.get('streams', {}) or {}
    stereo_cfg = settings.get('stereo', {}) or {}
    fusion_cfg = settings.get('fusion', {}) or {}
    pose_source_cfg = settings.get('pose_source', {}) or {}

    dual_rgb_mode = bool(
        streams_cfg.get('color_left', False)
        and streams_cfg.get('color_right', False)
        and not streams_cfg.get('depth', False)
    )
    fusion_enabled = bool(fusion_cfg.get('enabled', False))
    pose_mode = str(pose_source_cfg.get('mode', 'none') or 'none').strip().lower()
    writes_identity = bool(stereo_cfg.get('write_identity_poses_csv', False))

    if dual_rgb_mode and fusion_enabled and pose_mode in ('', 'none') and not writes_identity:
        raise RuntimeError(
            'fusion.enabled=true, but no real pose source is configured. '
            'Set pose_source.mode to a real provider such as robot/apriltag/slam, '
            'or keep fusion.enabled=false during raw capture.'
        )


def format_id_from_value(value: Any) -> int:
    if isinstance(value, int):
        return int(value)
    key = str(value).strip().upper()
    if key not in FORMAT_IDS_BY_NAME:
        raise ValueError(f'Unknown stream format: {value}')
    return FORMAT_IDS_BY_NAME[key]


def format_candidates_from_config(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return [format_id_from_value(item) for item in value]
    return [format_id_from_value(value)]


def bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ('1', 'true', 'yes', 'on'):
        return True
    if text in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'Invalid bool value: {value}')


def align_modes_from_settings(settings: dict[str, Any]) -> list[tuple[int, str]]:
    align_cfg = settings.get('align', {}) or {}
    order = align_cfg.get('mode_order') or ['hardware', 'software']
    modes: list[tuple[int, str]] = []
    for item in order:
        key = str(item).strip().lower()
        mode = ALIGN_MODES_BY_NAME.get(key)
        if not mode:
            print(f'[{now_str()}] WARN unknown align mode in config: {item}')
            continue
        if mode not in modes:
            modes.append(mode)
    return modes or [(ALIGN_D2C_HW_MODE, 'ALIGN_D2C_HW_MODE'), (ALIGN_D2C_SW_MODE, 'ALIGN_D2C_SW_MODE')]


def write_capture_config_snapshot(session_dir: Path, settings: dict[str, Any], config_path: Path) -> None:
    try:
        (session_dir / 'capture_config_resolved.json').write_text(
            json.dumps(settings, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        if config_path.exists():
            shutil.copy2(config_path, session_dir / config_path.name)
    except Exception as ex:
        print(f'[{now_str()}] WARN failed to write config snapshot: {ex}')


def write_latest_camera_intrinsics(output_root: Path, cam_params: dict[str, Any]) -> None:
    """相机 pipeline 启动后立即导出当前内参/外参，方便不录制也能查看。"""
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / 'last_camera_params.json').write_text(
            json.dumps(cam_params, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

        color = cam_params.get('color_intrinsic', {}) or {}
        depth = cam_params.get('depth_intrinsic', {}) or {}
        color_dist = cam_params.get('color_distortion', {}) or {}
        depth_dist = cam_params.get('depth_distortion', {}) or {}
        extrinsic = cam_params.get('depth_to_color_extrinsic', {}) or {}

        def val(data: dict[str, Any], key: str, default: Any = 0) -> Any:
            return data.get(key, default)

        lines = [
            '# 当前相机启动后由 Orbbec SDK 读取的内参/外参',
            '# 这个文件会在每次打开相机时刷新；每次正式采集目录里也会保存 camera_info.yaml。',
            f'created_at: "{now_str()}"',
            f'device_name: "{cam_params.get("device_name", "")}"',
            f'serial_number: "{cam_params.get("serial_number", "")}"',
            f'align_mode: "{cam_params.get("align_mode", "")}"',
            '',
            'color_intrinsic:',
            f'  width: {int(val(color, "width"))}',
            f'  height: {int(val(color, "height"))}',
            f'  fx: {float(val(color, "fx", 0.0)):.8f}',
            f'  fy: {float(val(color, "fy", 0.0)):.8f}',
            f'  cx: {float(val(color, "cx", 0.0)):.8f}',
            f'  cy: {float(val(color, "cy", 0.0)):.8f}',
            '',
            'depth_intrinsic:',
            f'  width: {int(val(depth, "width"))}',
            f'  height: {int(val(depth, "height"))}',
            f'  fx: {float(val(depth, "fx", 0.0)):.8f}',
            f'  fy: {float(val(depth, "fy", 0.0)):.8f}',
            f'  cx: {float(val(depth, "cx", 0.0)):.8f}',
            f'  cy: {float(val(depth, "cy", 0.0)):.8f}',
            '',
            'color_distortion:',
            f'  model: {int(val(color_dist, "model"))}',
            f'  k1: {float(val(color_dist, "k1", 0.0)):.8f}',
            f'  k2: {float(val(color_dist, "k2", 0.0)):.8f}',
            f'  k3: {float(val(color_dist, "k3", 0.0)):.8f}',
            f'  p1: {float(val(color_dist, "p1", 0.0)):.8f}',
            f'  p2: {float(val(color_dist, "p2", 0.0)):.8f}',
            '',
            'depth_distortion:',
            f'  model: {int(val(depth_dist, "model"))}',
            f'  k1: {float(val(depth_dist, "k1", 0.0)):.8f}',
            f'  k2: {float(val(depth_dist, "k2", 0.0)):.8f}',
            f'  k3: {float(val(depth_dist, "k3", 0.0)):.8f}',
            f'  p1: {float(val(depth_dist, "p1", 0.0)):.8f}',
            f'  p2: {float(val(depth_dist, "p2", 0.0)):.8f}',
            '',
            'depth_to_color_extrinsic:',
            '  rotation_3x3:',
        ]
        for row in extrinsic.get('rotation_3x3', []):
            lines.append('    - [' + ', '.join(f'{float(v):.8f}' for v in row) + ']')
        lines.extend([
            '  translation_mm: [' + ', '.join(f'{float(v):.8f}' for v in extrinsic.get('translation_mm', [])) + ']',
            '',
            'pointcloud_note: "彩色点云通常使用 color_intrinsic + depth_raw(D2C aligned)；未对齐时不要直接套 RGB 颜色。"',
        ])
        (output_root / 'last_camera_intrinsics.yaml').write_text('\n'.join(lines) + '\n', encoding='utf-8')

        print(
            f'[{now_str()}] Color intrinsics: '
            f'{int(val(color, "width"))}x{int(val(color, "height"))}, '
            f'fx={float(val(color, "fx", 0.0)):.3f}, fy={float(val(color, "fy", 0.0)):.3f}, '
            f'cx={float(val(color, "cx", 0.0)):.3f}, cy={float(val(color, "cy", 0.0)):.3f}'
        )
        print(
            f'[{now_str()}] Depth intrinsics: '
            f'{int(val(depth, "width"))}x{int(val(depth, "height"))}, '
            f'fx={float(val(depth, "fx", 0.0)):.3f}, fy={float(val(depth, "fy", 0.0)):.3f}, '
            f'cx={float(val(depth, "cx", 0.0)):.3f}, cy={float(val(depth, "cy", 0.0)):.3f}'
        )
        print(f'[{now_str()}] Camera intrinsics exported: {output_root / "last_camera_intrinsics.yaml"}')
    except Exception as ex:
        print(f'[{now_str()}] WARN failed to export latest camera intrinsics: {ex}')


@dataclass
class FrameData:
    frame_type: int
    fmt: int
    width: int
    height: int
    frame_index: int
    dev_ts: int
    sys_ts: int
    data: bytes
    depth_scale: Optional[float] = None


class OBCameraIntrinsic(ctypes.Structure):
    _fields_ = [
        ('fx', ctypes.c_float),
        ('fy', ctypes.c_float),
        ('cx', ctypes.c_float),
        ('cy', ctypes.c_float),
        ('width', ctypes.c_int16),
        ('height', ctypes.c_int16),
    ]


class OBCameraDistortion(ctypes.Structure):
    _fields_ = [
        ('k1', ctypes.c_float),
        ('k2', ctypes.c_float),
        ('k3', ctypes.c_float),
        ('k4', ctypes.c_float),
        ('k5', ctypes.c_float),
        ('k6', ctypes.c_float),
        ('p1', ctypes.c_float),
        ('p2', ctypes.c_float),
        ('model', ctypes.c_int),
    ]


class OBD2CTransform(ctypes.Structure):
    _fields_ = [
        ('rot', ctypes.c_float * 9),
        ('trans', ctypes.c_float * 3),
    ]


class OBCameraParam(ctypes.Structure):
    _fields_ = [
        ('depthIntrinsic', OBCameraIntrinsic),
        ('rgbIntrinsic', OBCameraIntrinsic),
        ('depthDistortion', OBCameraDistortion),
        ('rgbDistortion', OBCameraDistortion),
        ('transform', OBD2CTransform),
        ('isMirrored', ctypes.c_bool),
    ]


class SDK:
    def __init__(self, sdk_bin: Path):
        self.dll = sdk_bin / 'OrbbecSDK.dll'
        if not self.dll.exists():
            raise FileNotFoundError(f'OrbbecSDK.dll not found: {self.dll}')

        os.environ['PATH'] = f'{sdk_bin}{os.pathsep}' + os.environ.get('PATH', '')
        os.chdir(sdk_bin)
        self.lib = ctypes.CDLL(str(self.dll))
        self._bind()

    def _bind(self) -> None:
        L = self.lib
        p = ctypes.c_void_p
        u32 = ctypes.c_uint32
        u64 = ctypes.c_uint64

        L.ob_error_get_message.argtypes = [p]
        L.ob_error_get_message.restype = ctypes.c_char_p
        L.ob_delete_error.argtypes = [p]

        L.ob_create_context.argtypes = [ctypes.POINTER(p)]
        L.ob_create_context.restype = p
        L.ob_delete_context.argtypes = [p, ctypes.POINTER(p)]
        L.ob_query_device_list.argtypes = [p, ctypes.POINTER(p)]
        L.ob_query_device_list.restype = p
        L.ob_delete_device_list.argtypes = [p, ctypes.POINTER(p)]
        L.ob_device_list_get_count.argtypes = [p, ctypes.POINTER(p)]
        L.ob_device_list_get_count.restype = u32
        L.ob_device_list_get_device.argtypes = [p, u32, ctypes.POINTER(p)]
        L.ob_device_list_get_device.restype = p
        L.ob_delete_device.argtypes = [p, ctypes.POINTER(p)]

        L.ob_device_get_device_info.argtypes = [p, ctypes.POINTER(p)]
        L.ob_device_get_device_info.restype = p
        L.ob_device_info_get_serial_number.argtypes = [p, ctypes.POINTER(p)]
        L.ob_device_info_get_serial_number.restype = ctypes.c_char_p
        L.ob_device_info_get_name.argtypes = [p, ctypes.POINTER(p)]
        L.ob_device_info_get_name.restype = ctypes.c_char_p
        L.ob_device_info_get_firmware_version.argtypes = [p, ctypes.POINTER(p)]
        L.ob_device_info_get_firmware_version.restype = ctypes.c_char_p
        L.ob_device_info_get_hardware_version.argtypes = [p, ctypes.POINTER(p)]
        L.ob_device_info_get_hardware_version.restype = ctypes.c_char_p
        L.ob_device_info_get_supported_min_sdk_version.argtypes = [p, ctypes.POINTER(p)]
        L.ob_device_info_get_supported_min_sdk_version.restype = ctypes.c_char_p
        L.ob_delete_device_info.argtypes = [p, ctypes.POINTER(p)]
        L.ob_device_set_bool_property.argtypes = [p, ctypes.c_int, ctypes.c_bool, ctypes.POINTER(p)]
        L.ob_device_set_bool_property.restype = None
        L.ob_device_set_int_property.argtypes = [p, ctypes.c_int, ctypes.c_int32, ctypes.POINTER(p)]
        L.ob_device_set_int_property.restype = None
        L.ob_device_set_float_property.argtypes = [p, ctypes.c_int, ctypes.c_float, ctypes.POINTER(p)]
        L.ob_device_set_float_property.restype = None

        L.ob_create_pipeline_with_device.argtypes = [p, ctypes.POINTER(p)]
        L.ob_create_pipeline_with_device.restype = p
        L.ob_delete_pipeline.argtypes = [p, ctypes.POINTER(p)]
        L.ob_pipeline_start_with_config.argtypes = [p, p, ctypes.POINTER(p)]
        L.ob_pipeline_stop.argtypes = [p, ctypes.POINTER(p)]
        L.ob_pipeline_wait_for_frameset.argtypes = [p, u32, ctypes.POINTER(p)]
        L.ob_pipeline_wait_for_frameset.restype = p
        L.ob_pipeline_get_camera_param.argtypes = [p, ctypes.POINTER(p)]
        L.ob_pipeline_get_camera_param.restype = OBCameraParam
        L.ob_pipeline_get_stream_profile_list.argtypes = [p, ctypes.c_int, ctypes.POINTER(p)]
        L.ob_pipeline_get_stream_profile_list.restype = p

        L.ob_create_config.argtypes = [ctypes.POINTER(p)]
        L.ob_create_config.restype = p
        L.ob_delete_config.argtypes = [p, ctypes.POINTER(p)]
        L.ob_config_enable_stream.argtypes = [p, ctypes.c_int, ctypes.POINTER(p)]
        L.ob_config_enable_stream_with_stream_profile.argtypes = [p, p, ctypes.POINTER(p)]
        L.ob_config_enable_video_stream.argtypes = [
            p,
            ctypes.c_int,
            u32,
            u32,
            u32,
            ctypes.c_int,
            ctypes.POINTER(p),
        ]
        L.ob_config_set_frame_aggregate_output_mode.argtypes = [p, ctypes.c_int, ctypes.POINTER(p)]
        L.ob_config_set_align_mode.argtypes = [p, ctypes.c_int, ctypes.POINTER(p)]
        L.ob_config_set_depth_scale_after_align_require.argtypes = [p, ctypes.c_bool, ctypes.POINTER(p)]

        L.ob_frameset_get_color_frame.argtypes = [p, ctypes.POINTER(p)]
        L.ob_frameset_get_color_frame.restype = p
        L.ob_frameset_get_depth_frame.argtypes = [p, ctypes.POINTER(p)]
        L.ob_frameset_get_depth_frame.restype = p
        L.ob_frameset_get_ir_frame.argtypes = [p, ctypes.POINTER(p)]
        L.ob_frameset_get_ir_frame.restype = p
        L.ob_frameset_get_frame.argtypes = [p, ctypes.c_int, ctypes.POINTER(p)]
        L.ob_frameset_get_frame.restype = p

        L.ob_delete_frame.argtypes = [p, ctypes.POINTER(p)]
        L.ob_frame_get_type.argtypes = [p, ctypes.POINTER(p)]
        L.ob_frame_get_type.restype = ctypes.c_int
        L.ob_frame_get_format.argtypes = [p, ctypes.POINTER(p)]
        L.ob_frame_get_format.restype = ctypes.c_int
        L.ob_frame_get_index.argtypes = [p, ctypes.POINTER(p)]
        L.ob_frame_get_index.restype = u64
        L.ob_frame_get_timestamp_us.argtypes = [p, ctypes.POINTER(p)]
        L.ob_frame_get_timestamp_us.restype = u64
        L.ob_frame_get_system_timestamp_us.argtypes = [p, ctypes.POINTER(p)]
        L.ob_frame_get_system_timestamp_us.restype = u64
        L.ob_frame_get_data.argtypes = [p, ctypes.POINTER(p)]
        L.ob_frame_get_data.restype = p
        L.ob_frame_get_data_size.argtypes = [p, ctypes.POINTER(p)]
        L.ob_frame_get_data_size.restype = u32
        L.ob_video_frame_get_width.argtypes = [p, ctypes.POINTER(p)]
        L.ob_video_frame_get_width.restype = u32
        L.ob_video_frame_get_height.argtypes = [p, ctypes.POINTER(p)]
        L.ob_video_frame_get_height.restype = u32
        L.ob_depth_frame_get_value_scale.argtypes = [p, ctypes.POINTER(p)]
        L.ob_depth_frame_get_value_scale.restype = ctypes.c_float

        L.ob_delete_stream_profile_list.argtypes = [p, ctypes.POINTER(p)]
        L.ob_stream_profile_list_get_count.argtypes = [p, ctypes.POINTER(p)]
        L.ob_stream_profile_list_get_count.restype = u32
        L.ob_stream_profile_list_get_profile.argtypes = [p, ctypes.c_int, ctypes.POINTER(p)]
        L.ob_stream_profile_list_get_profile.restype = p
        L.ob_delete_stream_profile.argtypes = [p, ctypes.POINTER(p)]
        L.ob_stream_profile_get_type.argtypes = [p, ctypes.POINTER(p)]
        L.ob_stream_profile_get_type.restype = ctypes.c_int
        L.ob_stream_profile_get_format.argtypes = [p, ctypes.POINTER(p)]
        L.ob_stream_profile_get_format.restype = ctypes.c_int
        L.ob_video_stream_profile_get_width.argtypes = [p, ctypes.POINTER(p)]
        L.ob_video_stream_profile_get_width.restype = u32
        L.ob_video_stream_profile_get_height.argtypes = [p, ctypes.POINTER(p)]
        L.ob_video_stream_profile_get_height.restype = u32
        L.ob_video_stream_profile_get_fps.argtypes = [p, ctypes.POINTER(p)]
        L.ob_video_stream_profile_get_fps.restype = u32

        L.ob_get_version.argtypes = []
        L.ob_get_version.restype = ctypes.c_int
        L.ob_get_major_version.argtypes = []
        L.ob_get_major_version.restype = ctypes.c_int
        L.ob_get_minor_version.argtypes = []
        L.ob_get_minor_version.restype = ctypes.c_int
        L.ob_get_patch_version.argtypes = []
        L.ob_get_patch_version.restype = ctypes.c_int
        L.ob_get_stage_version.argtypes = []
        L.ob_get_stage_version.restype = ctypes.c_char_p

        # Device preset API lives in Advanced.h. It is required for Gemini 305
        # Dual Color Streams mode because COLOR_LEFT/COLOR_RIGHT are exposed
        # only after the device preset is switched.
        self._preset_api_available = True
        try:
            L.ob_device_get_current_preset_name.argtypes = [p, ctypes.POINTER(p)]
            L.ob_device_get_current_preset_name.restype = ctypes.c_char_p
            L.ob_device_load_preset.argtypes = [p, ctypes.c_char_p, ctypes.POINTER(p)]
            L.ob_device_load_preset.restype = None
            L.ob_device_get_available_preset_list.argtypes = [p, ctypes.POINTER(p)]
            L.ob_device_get_available_preset_list.restype = p
            L.ob_delete_preset_list.argtypes = [p, ctypes.POINTER(p)]
            L.ob_device_preset_list_get_count.argtypes = [p, ctypes.POINTER(p)]
            L.ob_device_preset_list_get_count.restype = u32
            L.ob_device_preset_list_get_name.argtypes = [p, u32, ctypes.POINTER(p)]
            L.ob_device_preset_list_get_name.restype = ctypes.c_char_p
            L.ob_device_preset_list_has_preset.argtypes = [p, ctypes.c_char_p, ctypes.POINTER(p)]
            L.ob_device_preset_list_has_preset.restype = ctypes.c_bool
        except AttributeError:
            self._preset_api_available = False

    def _err_msg(self, err_ptr: ctypes.c_void_p) -> str:
        if not err_ptr:
            return ''
        msg = self.lib.ob_error_get_message(err_ptr)
        self.lib.ob_delete_error(err_ptr)
        return msg.decode('utf-8', errors='ignore') if msg else 'Unknown SDK error'

    def _check(self, err_ptr: ctypes.c_void_p, api: str) -> None:
        if err_ptr:
            raise RuntimeError(f'{api} failed: {self._err_msg(err_ptr)}')

    def _optional(self, fn, *args):
        e = ctypes.c_void_p()
        out = fn(*args, ctypes.byref(e))
        if e:
            self._err_msg(e)
            return None
        return out

    def create_context(self):
        e = ctypes.c_void_p()
        ctx = self.lib.ob_create_context(ctypes.byref(e))
        self._check(e, 'ob_create_context')
        return ctx

    def delete_context(self, ctx) -> None:
        if not ctx:
            return
        e = ctypes.c_void_p()
        self.lib.ob_delete_context(ctx, ctypes.byref(e))
        if e:
            self._err_msg(e)

    def query_device_list(self, ctx):
        e = ctypes.c_void_p()
        dl = self.lib.ob_query_device_list(ctx, ctypes.byref(e))
        self._check(e, 'ob_query_device_list')
        return dl

    def delete_device_list(self, dl) -> None:
        if not dl:
            return
        e = ctypes.c_void_p()
        self.lib.ob_delete_device_list(dl, ctypes.byref(e))
        if e:
            self._err_msg(e)

    def device_count(self, dl) -> int:
        e = ctypes.c_void_p()
        count = self.lib.ob_device_list_get_count(dl, ctypes.byref(e))
        self._check(e, 'ob_device_list_get_count')
        return int(count)

    def get_device(self, dl, idx=0):
        e = ctypes.c_void_p()
        dev = self.lib.ob_device_list_get_device(dl, ctypes.c_uint32(idx), ctypes.byref(e))
        self._check(e, 'ob_device_list_get_device')
        return dev

    def delete_device(self, dev) -> None:
        if not dev:
            return
        e = ctypes.c_void_p()
        self.lib.ob_delete_device(dev, ctypes.byref(e))
        if e:
            self._err_msg(e)

    def get_device_info(self, dev) -> tuple[str, str]:
        detail = self.get_device_info_detail(dev)
        return detail.get('serial_number', 'UNKNOWN_SN'), detail.get('name', 'UNKNOWN_DEVICE')

    def get_device_info_detail(self, dev) -> dict[str, str]:
        e = ctypes.c_void_p()
        info = self.lib.ob_device_get_device_info(dev, ctypes.byref(e))
        self._check(e, 'ob_device_get_device_info')
        try:
            def read_required_str(fn, api: str, default: str = '') -> str:
                e2 = ctypes.c_void_p()
                value = fn(info, ctypes.byref(e2))
                self._check(e2, api)
                return value.decode('utf-8', errors='ignore') if value else default

            def read_optional_str(fn, api: str, default: str = '') -> str:
                try:
                    return read_required_str(fn, api, default)
                except Exception:
                    return default

            return {
                'serial_number': read_required_str(self.lib.ob_device_info_get_serial_number, 'ob_device_info_get_serial_number', 'UNKNOWN_SN'),
                'name': read_required_str(self.lib.ob_device_info_get_name, 'ob_device_info_get_name', 'UNKNOWN_DEVICE'),
                'firmware_version': read_optional_str(self.lib.ob_device_info_get_firmware_version, 'ob_device_info_get_firmware_version'),
                'hardware_version': read_optional_str(self.lib.ob_device_info_get_hardware_version, 'ob_device_info_get_hardware_version'),
                'supported_min_sdk_version': read_optional_str(
                    self.lib.ob_device_info_get_supported_min_sdk_version,
                    'ob_device_info_get_supported_min_sdk_version',
                ),
            }
        finally:
            e4 = ctypes.c_void_p()
            self.lib.ob_delete_device_info(info, ctypes.byref(e4))
            if e4:
                self._err_msg(e4)

    def get_sdk_version_text(self) -> str:
        try:
            major = int(self.lib.ob_get_major_version())
            minor = int(self.lib.ob_get_minor_version())
            patch = int(self.lib.ob_get_patch_version())
            stage = self.lib.ob_get_stage_version()
            stage_text = stage.decode('utf-8', errors='ignore') if stage else ''
            suffix = f' {stage_text}' if stage_text else ''
            return f'{major}.{minor}.{patch}{suffix}'
        except Exception as ex:
            return f'unknown ({ex})'

    def get_current_preset_name(self, dev) -> str:
        if not self._preset_api_available:
            raise RuntimeError('Orbbec SDK preset API is not available in this DLL.')
        e = ctypes.c_void_p()
        name = self.lib.ob_device_get_current_preset_name(dev, ctypes.byref(e))
        self._check(e, 'ob_device_get_current_preset_name')
        return name.decode('utf-8', errors='ignore') if name else ''

    def get_available_presets(self, dev) -> list[str]:
        if not self._preset_api_available:
            raise RuntimeError('Orbbec SDK preset API is not available in this DLL.')
        e = ctypes.c_void_p()
        preset_list = self.lib.ob_device_get_available_preset_list(dev, ctypes.byref(e))
        self._check(e, 'ob_device_get_available_preset_list')
        names: list[str] = []
        try:
            e = ctypes.c_void_p()
            count = int(self.lib.ob_device_preset_list_get_count(preset_list, ctypes.byref(e)))
            self._check(e, 'ob_device_preset_list_get_count')
            for idx in range(count):
                e = ctypes.c_void_p()
                name = self.lib.ob_device_preset_list_get_name(preset_list, ctypes.c_uint32(idx), ctypes.byref(e))
                self._check(e, 'ob_device_preset_list_get_name')
                names.append(name.decode('utf-8', errors='ignore') if name else '')
        finally:
            if preset_list:
                e2 = ctypes.c_void_p()
                self.lib.ob_delete_preset_list(preset_list, ctypes.byref(e2))
                if e2:
                    self._err_msg(e2)
        return names

    def load_preset(self, dev, preset_name: str) -> None:
        if not self._preset_api_available:
            raise RuntimeError('Orbbec SDK preset API is not available in this DLL.')
        e = ctypes.c_void_p()
        self.lib.ob_device_load_preset(dev, preset_name.encode('utf-8'), ctypes.byref(e))
        self._check(e, f'ob_device_load_preset({preset_name})')

    def set_bool_property_try(self, dev, property_id: int, value: bool, name: str) -> tuple[bool, str]:
        e = ctypes.c_void_p()
        self.lib.ob_device_set_bool_property(dev, ctypes.c_int(property_id), ctypes.c_bool(bool(value)), ctypes.byref(e))
        if e:
            return False, self._err_msg(e)
        return True, f'{name}={bool(value)}'

    def set_int_property_try(self, dev, property_id: int, value: int, name: str) -> tuple[bool, str]:
        e = ctypes.c_void_p()
        self.lib.ob_device_set_int_property(dev, ctypes.c_int(property_id), ctypes.c_int32(int(value)), ctypes.byref(e))
        if e:
            return False, self._err_msg(e)
        return True, f'{name}={int(value)}'

    def set_float_property_try(self, dev, property_id: int, value: float, name: str) -> tuple[bool, str]:
        e = ctypes.c_void_p()
        self.lib.ob_device_set_float_property(dev, ctypes.c_int(property_id), ctypes.c_float(float(value)), ctypes.byref(e))
        if e:
            return False, self._err_msg(e)
        return True, f'{name}={float(value)}'

    def create_pipeline(self, dev):
        e = ctypes.c_void_p()
        pipe = self.lib.ob_create_pipeline_with_device(dev, ctypes.byref(e))
        self._check(e, 'ob_create_pipeline_with_device')
        return pipe

    def delete_pipeline(self, pipe) -> None:
        if not pipe:
            return
        e = ctypes.c_void_p()
        self.lib.ob_delete_pipeline(pipe, ctypes.byref(e))
        if e:
            self._err_msg(e)

    def create_config(self):
        e = ctypes.c_void_p()
        cfg = self.lib.ob_create_config(ctypes.byref(e))
        self._check(e, 'ob_create_config')
        return cfg

    def delete_config(self, cfg) -> None:
        if not cfg:
            return
        e = ctypes.c_void_p()
        self.lib.ob_delete_config(cfg, ctypes.byref(e))
        if e:
            self._err_msg(e)

    def enable_stream(self, cfg, stream_type: int) -> None:
        e = ctypes.c_void_p()
        self.lib.ob_config_enable_stream(cfg, ctypes.c_int(stream_type), ctypes.byref(e))
        self._check(e, f'ob_config_enable_stream({stream_type})')

    def enable_stream_profile(self, cfg, profile) -> None:
        e = ctypes.c_void_p()
        self.lib.ob_config_enable_stream_with_stream_profile(cfg, profile, ctypes.byref(e))
        self._check(e, 'ob_config_enable_stream_with_stream_profile')

    def enable_video_stream(self, cfg, stream_type: int, width: int, height: int, fps: int, fmt: int) -> None:
        e = ctypes.c_void_p()
        self.lib.ob_config_enable_video_stream(
            cfg,
            ctypes.c_int(stream_type),
            ctypes.c_uint32(width),
            ctypes.c_uint32(height),
            ctypes.c_uint32(fps),
            ctypes.c_int(fmt),
            ctypes.byref(e),
        )
        self._check(e, f'ob_config_enable_video_stream({stream_type}, {width}x{height}@{fps}, fmt={fmt})')

    def set_aggregate_all_type(self, cfg) -> None:
        e = ctypes.c_void_p()
        self.lib.ob_config_set_frame_aggregate_output_mode(
            cfg, ctypes.c_int(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE), ctypes.byref(e)
        )
        self._check(e, 'ob_config_set_frame_aggregate_output_mode')

    def set_align_mode_try(self, cfg, mode: int) -> bool:
        e = ctypes.c_void_p()
        self.lib.ob_config_set_align_mode(cfg, ctypes.c_int(mode), ctypes.byref(e))
        if e:
            self._err_msg(e)
            return False
        return True

    def set_depth_scale_after_align(self, cfg, enable: bool = True) -> bool:
        e = ctypes.c_void_p()
        self.lib.ob_config_set_depth_scale_after_align_require(cfg, ctypes.c_bool(enable), ctypes.byref(e))
        if e:
            self._err_msg(e)
            return False
        return True

    def start_pipeline(self, pipe, cfg) -> None:
        e = ctypes.c_void_p()
        self.lib.ob_pipeline_start_with_config(pipe, cfg, ctypes.byref(e))
        self._check(e, 'ob_pipeline_start_with_config')

    def list_video_stream_profiles(self, pipe, sensor_type: int) -> list[dict]:
        e = ctypes.c_void_p()
        profile_list = self.lib.ob_pipeline_get_stream_profile_list(pipe, ctypes.c_int(sensor_type), ctypes.byref(e))
        self._check(e, f'ob_pipeline_get_stream_profile_list({sensor_type})')
        profiles: list[dict] = []
        try:
            e = ctypes.c_void_p()
            count = int(self.lib.ob_stream_profile_list_get_count(profile_list, ctypes.byref(e)))
            self._check(e, 'ob_stream_profile_list_get_count')
            for idx in range(count):
                profile = 0
                try:
                    e = ctypes.c_void_p()
                    profile = self.lib.ob_stream_profile_list_get_profile(profile_list, ctypes.c_int(idx), ctypes.byref(e))
                    self._check(e, 'ob_stream_profile_list_get_profile')
                    e = ctypes.c_void_p()
                    stream_type = int(self.lib.ob_stream_profile_get_type(profile, ctypes.byref(e)))
                    self._check(e, 'ob_stream_profile_get_type')
                    e = ctypes.c_void_p()
                    fmt = int(self.lib.ob_stream_profile_get_format(profile, ctypes.byref(e)))
                    self._check(e, 'ob_stream_profile_get_format')
                    e = ctypes.c_void_p()
                    width = int(self.lib.ob_video_stream_profile_get_width(profile, ctypes.byref(e)))
                    self._check(e, 'ob_video_stream_profile_get_width')
                    e = ctypes.c_void_p()
                    height = int(self.lib.ob_video_stream_profile_get_height(profile, ctypes.byref(e)))
                    self._check(e, 'ob_video_stream_profile_get_height')
                    e = ctypes.c_void_p()
                    fps = int(self.lib.ob_video_stream_profile_get_fps(profile, ctypes.byref(e)))
                    self._check(e, 'ob_video_stream_profile_get_fps')
                    profiles.append({
                        'stream_type': stream_type,
                        'format': fmt,
                        'width': width,
                        'height': height,
                        'fps': fps,
                    })
                finally:
                    if profile:
                        e2 = ctypes.c_void_p()
                        self.lib.ob_delete_stream_profile(profile, ctypes.byref(e2))
                        if e2:
                            self._err_msg(e2)
        finally:
            if profile_list:
                e = ctypes.c_void_p()
                self.lib.ob_delete_stream_profile_list(profile_list, ctypes.byref(e))
                if e:
                    self._err_msg(e)
        return profiles

    def stop_pipeline(self, pipe) -> None:
        if not pipe:
            return
        e = ctypes.c_void_p()
        self.lib.ob_pipeline_stop(pipe, ctypes.byref(e))
        if e:
            self._err_msg(e)

    def wait_frameset(self, pipe, timeout_ms=200):
        e = ctypes.c_void_p()
        fs = self.lib.ob_pipeline_wait_for_frameset(pipe, ctypes.c_uint32(timeout_ms), ctypes.byref(e))
        self._check(e, 'ob_pipeline_wait_for_frameset')
        return fs if fs else None

    def get_camera_param(self, pipe) -> OBCameraParam:
        e = ctypes.c_void_p()
        cp = self.lib.ob_pipeline_get_camera_param(pipe, ctypes.byref(e))
        self._check(e, 'ob_pipeline_get_camera_param')
        return cp

    def delete_frame(self, frame) -> None:
        if not frame:
            return
        e = ctypes.c_void_p()
        self.lib.ob_delete_frame(frame, ctypes.byref(e))
        if e:
            self._err_msg(e)

    def get_optional_frame(self, fs, frame_type: int):
        if frame_type == OB_FRAME_COLOR:
            return self._optional(self.lib.ob_frameset_get_color_frame, fs)
        if frame_type == OB_FRAME_DEPTH:
            return self._optional(self.lib.ob_frameset_get_depth_frame, fs)
        if frame_type == OB_FRAME_IR:
            return self._optional(self.lib.ob_frameset_get_ir_frame, fs)
        return self._optional(self.lib.ob_frameset_get_frame, fs, ctypes.c_int(frame_type))

    def extract(self, frame) -> FrameData:
        e = ctypes.c_void_p()
        ft = int(self.lib.ob_frame_get_type(frame, ctypes.byref(e))); self._check(e, 'ob_frame_get_type')
        fm = int(self.lib.ob_frame_get_format(frame, ctypes.byref(e))); self._check(e, 'ob_frame_get_format')
        w = int(self.lib.ob_video_frame_get_width(frame, ctypes.byref(e))); self._check(e, 'ob_video_frame_get_width')
        h = int(self.lib.ob_video_frame_get_height(frame, ctypes.byref(e))); self._check(e, 'ob_video_frame_get_height')
        fi = int(self.lib.ob_frame_get_index(frame, ctypes.byref(e))); self._check(e, 'ob_frame_get_index')
        dts = int(self.lib.ob_frame_get_timestamp_us(frame, ctypes.byref(e))); self._check(e, 'ob_frame_get_timestamp_us')
        sts = int(self.lib.ob_frame_get_system_timestamp_us(frame, ctypes.byref(e))); self._check(e, 'ob_frame_get_system_timestamp_us')
        sz = int(self.lib.ob_frame_get_data_size(frame, ctypes.byref(e))); self._check(e, 'ob_frame_get_data_size')
        ptr = self.lib.ob_frame_get_data(frame, ctypes.byref(e)); self._check(e, 'ob_frame_get_data')
        data = ctypes.string_at(ptr, sz) if ptr and sz > 0 else b''
        depth_scale = None
        if ft == OB_FRAME_DEPTH:
            depth_scale = float(self.lib.ob_depth_frame_get_value_scale(frame, ctypes.byref(e)))
            self._check(e, 'ob_depth_frame_get_value_scale')
        return FrameData(ft, fm, w, h, fi, dts, sts, data, depth_scale)


def decode_color(fd: FrameData) -> Optional[np.ndarray]:
    h, w, fm = fd.height, fd.width, fd.fmt
    b = np.frombuffer(fd.data, dtype=np.uint8)
    try:
        if fm == OB_FORMAT_MJPG:
            return cv2.imdecode(b, cv2.IMREAD_COLOR)
        if fm == OB_FORMAT_RGB:
            return cv2.cvtColor(b.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        if fm == OB_FORMAT_BGR:
            return b.reshape(h, w, 3).copy()
        if fm == OB_FORMAT_BGRA:
            return cv2.cvtColor(b.reshape(h, w, 4), cv2.COLOR_BGRA2BGR)
        if fm == OB_FORMAT_RGBA:
            return cv2.cvtColor(b.reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
        if fm == OB_FORMAT_YUYV:
            return cv2.cvtColor(b.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
        if fm == OB_FORMAT_UYVY:
            return cv2.cvtColor(b.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    except Exception:
        return None
    return None


def decode_depth(fd: FrameData) -> Optional[np.ndarray]:
    if fd.fmt != OB_FORMAT_Y16:
        return None
    try:
        return np.frombuffer(fd.data, dtype=np.uint16).reshape(fd.height, fd.width).copy()
    except Exception:
        return None


def decode_ir(fd: FrameData) -> Optional[np.ndarray]:
    try:
        if fd.fmt == OB_FORMAT_Y16:
            return np.frombuffer(fd.data, dtype=np.uint16).reshape(fd.height, fd.width).copy()
        if fd.fmt in (OB_FORMAT_Y8, OB_FORMAT_GRAY):
            return np.frombuffer(fd.data, dtype=np.uint8).reshape(fd.height, fd.width).copy()
    except Exception:
        return None
    return None


def ir_to_vis(ir_img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if ir_img is None:
        return None
    if ir_img.dtype == np.uint16:
        vis = cv2.normalize(ir_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        vis = ir_img.astype(np.uint8, copy=False)
    return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)


def write_png_file(path: Path, image: np.ndarray) -> None:
    if image is None:
        raise RuntimeError(f'PNG image is None: {path}')
    if image.size <= 0:
        raise RuntimeError(f'PNG image is empty: {path}, shape={image.shape}, dtype={image.dtype}')

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(image)
    compression = int(max(0, min(9, PNG_COMPRESSION)))
    ok, encoded = cv2.imencode('.png', arr, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not ok or encoded is None:
        raise RuntimeError(f'cv2.imencode failed: {path}, shape={arr.shape}, dtype={arr.dtype}')
    path.write_bytes(encoded.tobytes())



def write_array_file(path: Path, image: np.ndarray) -> None:
    if image is None:
        raise RuntimeError(f'Image is None: {path}')
    if image.size <= 0:
        raise RuntimeError(f'Image is empty: {path}, shape={image.shape}, dtype={image.dtype}')

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(image)
    ext = path.suffix.lower()
    if ext == '.png':
        write_png_file(path, arr)
        return
    if ext == '.bmp':
        ok, encoded = cv2.imencode('.bmp', arr)
        if not ok or encoded is None:
            raise RuntimeError(f'cv2.imencode bmp failed: {path}, shape={arr.shape}, dtype={arr.dtype}')
        path.write_bytes(encoded.tobytes())
        return
    if ext == '.npy':
        with path.open('wb') as f:
            np.save(f, arr)
        return
    if ext == '.raw':
        path.write_bytes(arr.tobytes())
        return
    raise RuntimeError(f'Unsupported output format: {path.suffix} for {path}')
def depth_to_vis(depth_u16: np.ndarray) -> np.ndarray:
    if depth_u16 is None or depth_u16.size == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    valid = depth_u16 > 0
    vis = np.zeros(depth_u16.shape, dtype=np.uint8)
    if not np.any(valid):
        return cv2.applyColorMap(vis, cv2.COLORMAP_JET)

    vals = depth_u16[valid]
    if vals.size > 200_000:
        vals = vals[::(vals.size // 200_000) + 1]

    lo = float(np.percentile(vals, 5))
    hi = float(np.percentile(vals, 95))
    if hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())
    if hi <= lo:
        return cv2.applyColorMap(vis, cv2.COLORMAP_JET)

    norm = (depth_u16.astype(np.float32) - lo) * (255.0 / (hi - lo))
    norm = np.maximum(norm, 0.0)
    norm = np.minimum(norm, 255.0)
    vis = norm.astype(np.uint8)
    vis[~valid] = 0
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def make_preview(
    color_img: Optional[np.ndarray],
    depth_vis_img: Optional[np.ndarray],
    ir_left_vis: Optional[np.ndarray] = None,
    ir_right_vis: Optional[np.ndarray] = None,
    color_left_img: Optional[np.ndarray] = None,
    color_right_img: Optional[np.ndarray] = None,
) -> np.ndarray:
    panels = []
    if color_img is not None:
        panels.append(color_img)
    if color_left_img is not None:
        panels.append(color_left_img)
    if color_right_img is not None:
        panels.append(color_right_img)
    if depth_vis_img is not None:
        panels.append(depth_vis_img)
    if ir_left_vis is not None:
        panels.append(ir_left_vis)
    if ir_right_vis is not None:
        panels.append(ir_right_vis)
    if not panels:
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    target_h = 480
    out = []
    for panel in panels:
        h, w = panel.shape[:2]
        nw = max(1, int(w * (target_h / max(1, h))))
        out.append(cv2.resize(panel, (nw, target_h), interpolation=cv2.INTER_AREA))
    return np.hstack(out)


def frame_res_text(fd: Optional[FrameData]) -> str:
    if fd is None:
        return '--'
    return f'{fd.width}x{fd.height}'


def draw_status_sections(img: np.ndarray, sections: list[tuple[str, list[str]]]) -> np.ndarray:
    out = img.copy()
    overlay = out.copy()
    margin = 10
    gap = 8
    card_w = 300
    title_h = 24
    line_h = 18
    inner_pad = 10
    rects: list[tuple[int, int, int, int, str, list[str]]] = []

    for idx, (title, lines) in enumerate(sections[:4]):
        col = idx % 2
        row = idx // 2
        x1 = margin + col * (card_w + gap)
        x2 = x1 + card_w
        card_h = inner_pad * 2 + title_h + max(1, len(lines)) * line_h
        y1 = margin + row * (card_h + gap)
        y2 = y1 + card_h
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 18, 18), thickness=-1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (90, 90, 90), thickness=1)
        rects.append((x1, y1, x2, y2, title, lines))

    out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0.0)
    for x1, y1, x2, y2, title, lines in rects:
        cv2.putText(out, f'[{title}]', (x1 + inner_pad, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        ty = y1 + inner_pad + title_h + 4
        for text in lines:
            cv2.putText(out, text, (x1 + inner_pad, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
            ty += line_h
    return out


def draw_controls(img: np.ndarray, capturing: bool) -> tuple[np.ndarray, dict]:
    out = img.copy()
    h, w = out.shape[:2]
    btn_h = 44
    margin = 12
    y1 = max(margin, h - btn_h - margin)
    y2 = y1 + btn_h

    labels = [
        ('toggle', 'STOP (E/SPACE)' if capturing else 'START (S/SPACE)'),
        ('quit', 'QUIT (Q/ESC)'),
    ]
    buttons = {}
    x = margin
    for name, text in labels:
        tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
        bw = tw + 26
        x2 = min(w - margin, x + bw)
        if x2 <= x:
            continue
        color = (0, 70, 230) if name == 'quit' else ((0, 180, 0) if not capturing else (0, 140, 255))
        cv2.rectangle(out, (x, y1), (x2, y2), color, thickness=-1)
        cv2.rectangle(out, (x, y1), (x2, y2), (255, 255, 255), thickness=1)
        ty = y1 + ((btn_h + th) // 2) - 4
        cv2.putText(out, text, (x + 12, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        buttons[name] = (x, y1, x2, y2)
        x = x2 + margin
    return out, buttons


def on_preview_mouse(event, x, y, flags, state):
    if state is None or event != cv2.EVENT_LBUTTONUP:
        return
    for name, (x1, y1, x2, y2) in state.get('buttons', {}).items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            state['action'] = name
            break


def camera_param_to_dict(cp: OBCameraParam, align_mode_name: str) -> dict:
    return {
        'align_mode': align_mode_name,
        'note': 'camera parameters returned by ob_pipeline_get_camera_param under current pipeline config',
        'depth_intrinsic': {
            'width': int(cp.depthIntrinsic.width),
            'height': int(cp.depthIntrinsic.height),
            'fx': float(cp.depthIntrinsic.fx),
            'fy': float(cp.depthIntrinsic.fy),
            'cx': float(cp.depthIntrinsic.cx),
            'cy': float(cp.depthIntrinsic.cy),
        },
        'color_intrinsic': {
            'width': int(cp.rgbIntrinsic.width),
            'height': int(cp.rgbIntrinsic.height),
            'fx': float(cp.rgbIntrinsic.fx),
            'fy': float(cp.rgbIntrinsic.fy),
            'cx': float(cp.rgbIntrinsic.cx),
            'cy': float(cp.rgbIntrinsic.cy),
        },
        'depth_distortion': {
            'model': int(cp.depthDistortion.model),
            'k1': float(cp.depthDistortion.k1),
            'k2': float(cp.depthDistortion.k2),
            'k3': float(cp.depthDistortion.k3),
            'k4': float(cp.depthDistortion.k4),
            'k5': float(cp.depthDistortion.k5),
            'k6': float(cp.depthDistortion.k6),
            'p1': float(cp.depthDistortion.p1),
            'p2': float(cp.depthDistortion.p2),
        },
        'color_distortion': {
            'model': int(cp.rgbDistortion.model),
            'k1': float(cp.rgbDistortion.k1),
            'k2': float(cp.rgbDistortion.k2),
            'k3': float(cp.rgbDistortion.k3),
            'k4': float(cp.rgbDistortion.k4),
            'k5': float(cp.rgbDistortion.k5),
            'k6': float(cp.rgbDistortion.k6),
            'p1': float(cp.rgbDistortion.p1),
            'p2': float(cp.rgbDistortion.p2),
        },
        'depth_to_color_extrinsic': {
            'rotation_3x3': [
                [float(cp.transform.rot[0]), float(cp.transform.rot[1]), float(cp.transform.rot[2])],
                [float(cp.transform.rot[3]), float(cp.transform.rot[4]), float(cp.transform.rot[5])],
                [float(cp.transform.rot[6]), float(cp.transform.rot[7]), float(cp.transform.rot[8])],
            ],
            'translation_mm': [
                float(cp.transform.trans[0]),
                float(cp.transform.trans[1]),
                float(cp.transform.trans[2]),
            ],
        },
        'is_mirrored': bool(cp.isMirrored),
    }


def distortion_list_to_dict(values: Any, default_model: int = 4) -> dict[str, float]:
    vals = list(values or [])
    vals = vals + [0.0] * (5 - len(vals))
    return {
        'model': int(default_model),
        'k1': float(vals[0]),
        'k2': float(vals[1]),
        'k3': float(vals[4]),
        'k4': 0.0,
        'k5': 0.0,
        'k6': 0.0,
        'p1': float(vals[2]),
        'p2': float(vals[3]),
    }


def intrinsic_has_signal(data: dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    width = int(data.get('width', 0) or 0)
    height = int(data.get('height', 0) or 0)
    fx = abs(float(data.get('fx', 0.0) or 0.0))
    fy = abs(float(data.get('fy', 0.0) or 0.0))
    return width > 0 and height > 0 and (fx > 1e-6 or fy > 1e-6)


def find_matching_intrinsics_reference(
    settings: dict[str, Any],
    device_name: str,
    serial_number: str,
    stereo_pair: bool,
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    refs = (settings.get('intrinsics_reference', {}) or {})
    best_key = None
    best_ref = None
    best_score = -1
    device_name = (device_name or '').strip().lower()
    serial_number = (serial_number or '').strip().lower()
    for key, ref in refs.items():
        if not isinstance(ref, dict):
            continue
        score = 0
        ref_serial = str(ref.get('serial_number', '')).strip().lower()
        ref_name = str(ref.get('device_name', '')).strip().lower()
        if serial_number and ref_serial == serial_number:
            score += 100
        if device_name and ref_name == device_name:
            score += 50
        if stereo_pair and ('left' in ref or 'right' in ref):
            score += 20
        if not stereo_pair and ('color' in ref or 'depth' in ref):
            score += 20
        if score > best_score:
            best_score = score
            best_key = str(key)
            best_ref = ref
    return best_key, copy.deepcopy(best_ref) if best_ref else None


def apply_intrinsics_reference_fallback(
    cam_params: dict[str, Any],
    settings: dict[str, Any],
    stereo_pair: bool,
) -> dict[str, Any]:
    device_name = str(cam_params.get('device_name', '') or '')
    serial_number = str(cam_params.get('serial_number', '') or '')
    ref_key, ref = find_matching_intrinsics_reference(settings, device_name, serial_number, stereo_pair)
    if not ref:
        return cam_params

    report: dict[str, Any] = {
        'used': False,
        'reference_key': ref_key,
        'stereo_pair': bool(stereo_pair),
    }

    if stereo_pair:
        left = copy.deepcopy(ref.get('left') or ref.get('color') or {})
        right = copy.deepcopy(ref.get('right') or ref.get('color') or left)
        baseline_m = float(ref.get('baseline_m', 0.0) or 0.0)
        left_to_right = copy.deepcopy(ref.get('left_to_right') or {})
        rotation = left_to_right.get('rotation_3x3') or [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        translation_m = left_to_right.get('translation_m') or [baseline_m, 0.0, 0.0]

        if not intrinsic_has_signal(cam_params.get('color_intrinsic', {})):
            cam_params['color_intrinsic'] = {
                'width': int(left.get('width', 0)),
                'height': int(left.get('height', 0)),
                'fx': float(left.get('fx', 0.0)),
                'fy': float(left.get('fy', 0.0)),
                'cx': float(left.get('cx', 0.0)),
                'cy': float(left.get('cy', 0.0)),
            }
            cam_params['color_distortion'] = distortion_list_to_dict(left.get('distortion'))
            report['filled_color_intrinsic_from_reference'] = True
            report['used'] = True

        cam_params['stereo_calibration'] = {
            'device_name': device_name,
            'serial_number': serial_number,
            'mode': str(ref.get('mode', 'Dual RGB')),
            'calibration_quality': str(ref.get('calibration_quality', 'reference_only')),
            'rectified': bool(ref.get('rectified', False)),
            'baseline_m': baseline_m,
            'image_width': int(left.get('width', right.get('width', 0))),
            'image_height': int(left.get('height', right.get('height', 0))),
            'left_intrinsic': {
                'width': int(left.get('width', 0)),
                'height': int(left.get('height', 0)),
                'fx': float(left.get('fx', 0.0)),
                'fy': float(left.get('fy', 0.0)),
                'cx': float(left.get('cx', 0.0)),
                'cy': float(left.get('cy', 0.0)),
            },
            'right_intrinsic': {
                'width': int(right.get('width', 0)),
                'height': int(right.get('height', 0)),
                'fx': float(right.get('fx', 0.0)),
                'fy': float(right.get('fy', 0.0)),
                'cx': float(right.get('cx', 0.0)),
                'cy': float(right.get('cy', 0.0)),
            },
            'left_distortion': distortion_list_to_dict(left.get('distortion')),
            'right_distortion': distortion_list_to_dict(right.get('distortion')),
            'left_to_right_extrinsic': {
                'rotation_3x3': rotation,
                'translation_m': [float(v) for v in translation_m],
            },
        }
        report['wrote_stereo_calibration_reference'] = True
        report['used'] = True
    else:
        color = copy.deepcopy(ref.get('color') or {})
        depth = copy.deepcopy(ref.get('depth') or {})
        if color and not intrinsic_has_signal(cam_params.get('color_intrinsic', {})):
            cam_params['color_intrinsic'] = {
                'width': int(color.get('width', 0)),
                'height': int(color.get('height', 0)),
                'fx': float(color.get('fx', 0.0)),
                'fy': float(color.get('fy', 0.0)),
                'cx': float(color.get('cx', 0.0)),
                'cy': float(color.get('cy', 0.0)),
            }
            cam_params['color_distortion'] = distortion_list_to_dict(color.get('distortion'))
            report['filled_color_intrinsic_from_reference'] = True
            report['used'] = True
        if depth and not intrinsic_has_signal(cam_params.get('depth_intrinsic', {})):
            cam_params['depth_intrinsic'] = {
                'width': int(depth.get('width', 0)),
                'height': int(depth.get('height', 0)),
                'fx': float(depth.get('fx', 0.0)),
                'fy': float(depth.get('fy', 0.0)),
                'cx': float(depth.get('cx', 0.0)),
                'cy': float(depth.get('cy', 0.0)),
            }
            cam_params['depth_distortion'] = distortion_list_to_dict(depth.get('distortion'), default_model=3)
            report['filled_depth_intrinsic_from_reference'] = True
            report['used'] = True

    if report['used']:
        cam_params['intrinsics_fallback_report'] = report
        note = str(cam_params.get('note', '') or '').strip()
        suffix = f' fallback intrinsics reference "{ref_key}" applied.'
        cam_params['note'] = (note + suffix).strip()
    return cam_params


class SessionWriter:
    def __init__(
        self,
        root: Path,
        sn: str,
        tag: str,
        target_width: int,
        target_height: int,
        align_mode_name: str,
        role: str = '',
        require_aligned_depth_to_color: bool = True,
        save_color: bool = True,
        save_color_left: bool = False,
        save_color_right: bool = False,
        save_depth: bool = True,
        save_depth_vis: bool = True,
        save_ir_left: bool = False,
        save_ir_right: bool = False,
        minimal_dual_rgb_layout: bool = False,
        color_format: str = 'png',
        depth_raw_format: str = 'png',
        writer_thread_count: int = 1,
        write_queue_maxsize: int = 256,
    ):
        self.root = root
        self.sn = sn
        self.tag = tag.strip().replace(' ', '_')
        self.target_width = int(target_width)
        self.target_height = int(target_height)
        self.align_mode_name = align_mode_name
        self.role = role.strip()
        self.require_aligned_depth_to_color = bool(require_aligned_depth_to_color)
        self.save_color = bool(save_color)
        self.save_color_left = bool(save_color_left)
        self.save_color_right = bool(save_color_right)
        self.save_depth = bool(save_depth)
        self.save_depth_vis = bool(save_depth_vis)
        self.save_ir_left = bool(save_ir_left)
        self.save_ir_right = bool(save_ir_right)
        self.minimal_dual_rgb_layout = bool(minimal_dual_rgb_layout)
        self.color_format = str(color_format or 'png').strip().lower().lstrip('.')
        self.depth_raw_format = str(depth_raw_format or 'png').strip().lower().lstrip('.')
        self.writer_thread_count = max(1, int(writer_thread_count or 1))
        self.write_queue_maxsize = max(1, int(write_queue_maxsize or 256))
        self.active = False

        self.session_dir: Optional[Path] = None
        self.rgb_dir: Optional[Path] = None
        self.color_left_dir: Optional[Path] = None
        self.color_right_dir: Optional[Path] = None
        self.depth_dir: Optional[Path] = None
        self.depth_vis_dir: Optional[Path] = None
        self.ir_left_dir: Optional[Path] = None
        self.ir_right_dir: Optional[Path] = None
        self.timestamps_file = None
        self.timestamps_filename = 'timestamps.txt'
        self.timestamps_csv = False
        self.write_queue: Optional[queue.Queue] = None
        self.writer_thread: Optional[threading.Thread] = None
        self.writer_threads: list[threading.Thread] = []
        self.timestamp_lock = threading.Lock()
        self.enqueue_lock = threading.Lock()
        self.write_error: Optional[str] = None

        self.pair_index = 0
        self.first_ts_us: Optional[int] = None
        self.depth_scale: Optional[float] = None
        self.started_at: Optional[str] = None
        self.ended_at: Optional[str] = None
        self.started_perf: Optional[float] = None
        self.duration_sec: float = 0.0
        self.camera_params = None
        self.out_width: Optional[int] = None
        self.out_height: Optional[int] = None
        self.src_width: Optional[int] = None
        self.src_height: Optional[int] = None

        self.skipped_missing = 0
        self.skipped_format = 0
        self.skipped_resolution = 0
        self.skipped_decode = 0
        self.skipped_sync = 0
        self.last_skip_reason = ''
        self.last_skip_detail = ''
        self.depth_resized_count = 0

    def _writer_loop(self):
        while True:
            item = self.write_queue.get()
            try:
                if item is None:
                    return

                pid = int(item['pid'])
                base = item['base']
                rgb_base = item.get('rgb_base', base)
                depth_base = item.get('depth_base', base)
                ts = float(item['timestamp_s'])
                rgb = item.get('rgb')
                color_left = item.get('color_left')
                color_right = item.get('color_right')
                depth = item.get('depth')
                depth_vis = item.get('depth_vis')
                ir_left = item.get('ir_left')
                ir_right = item.get('ir_right')
                written_paths: list[Path] = []

                try:
                    if self.rgb_dir is not None and rgb is not None:
                        path = self.rgb_dir / rgb_base
                        write_array_file(path, rgb)
                        written_paths.append(path)
                    if self.color_left_dir is not None and color_left is not None:
                        path = self.color_left_dir / base
                        write_png_file(path, color_left)
                        written_paths.append(path)
                    if self.color_right_dir is not None and color_right is not None:
                        path = self.color_right_dir / base
                        write_png_file(path, color_right)
                        written_paths.append(path)
                    if self.depth_dir is not None and depth is not None:
                        path = self.depth_dir / depth_base
                        write_array_file(path, depth)
                        written_paths.append(path)
                    if self.depth_vis_dir is not None and depth_vis is not None:
                        path = self.depth_vis_dir / base
                        write_png_file(path, depth_vis)
                        written_paths.append(path)
                    if self.ir_left_dir is not None and ir_left is not None:
                        path = self.ir_left_dir / base
                        write_png_file(path, ir_left)
                        written_paths.append(path)
                    if self.ir_right_dir is not None and ir_right is not None:
                        path = self.ir_right_dir / base
                        write_png_file(path, ir_right)
                        written_paths.append(path)
                except Exception as write_ex:
                    try:
                        for path in written_paths:
                            if path.exists():
                                path.unlink()
                    except Exception:
                        pass
                    raise RuntimeError(
                        f'Failed to write image files for pair {pid}: {write_ex}; '
                        f'base={base}'
                    )

                if self.timestamps_file is not None:
                    with self.timestamp_lock:
                        if self.timestamps_csv:
                            self.timestamps_file.write(f'{pid:06d},{ts:.6f}\n')
                        else:
                            self.timestamps_file.write(f'{pid:06d} {ts:.6f}\n')
            except Exception as ex:
                self.write_error = str(ex)
                return
            finally:
                self.write_queue.task_done()

    def _raise_if_write_failed(self):
        if self.write_error:
            raise RuntimeError(f'Background writer failed: {self.write_error}')

    def _unique_dir(self) -> Path:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = f'capture_{ts}'
        path = self.root / base
        if not path.exists():
            return path
        i = 1
        while True:
            candidate = self.root / f'{base}_{i:02d}'
            if not candidate.exists():
                return candidate
            i += 1

    def _build_camera_info_yaml(self) -> str:
        cp = self.camera_params or {}
        ci = cp.get('color_intrinsic', {})
        di = cp.get('depth_intrinsic', {})
        cd = cp.get('color_distortion', {})
        dd = cp.get('depth_distortion', {})
        ex = cp.get('depth_to_color_extrinsic', {})
        w = int(self.out_width or self.target_width or ci.get('width', 0))
        h = int(self.out_height or self.target_height or ci.get('height', 0))
        src_w = float(self.src_width or ci.get('width', w) or w or 1)
        src_h = float(self.src_height or ci.get('height', h) or h or 1)
        sx = float(w) / src_w if src_w > 0 else 1.0
        sy = float(h) / src_h if src_h > 0 else 1.0
        fx = float(ci.get('fx', 0.0)) * sx
        fy = float(ci.get('fy', 0.0)) * sy
        cx = float(ci.get('cx', 0.0)) * sx
        cy = float(ci.get('cy', 0.0)) * sy
        k1 = float(cd.get('k1', 0.0))
        k2 = float(cd.get('k2', 0.0))
        p1 = float(cd.get('p1', 0.0))
        p2 = float(cd.get('p2', 0.0))
        k3 = float(cd.get('k3', 0.0))
        depth_scale = float(self.depth_scale if self.depth_scale is not None else 1.0)
        rot = ex.get('rotation_3x3') or [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        trans = ex.get('translation_mm') or [0.0, 0.0, 0.0]

        def q(value: object, path_like: bool = False) -> str:
            text = str(value)
            if path_like:
                text = text.replace('\\', '/')
            text = text.replace('\\', '\\\\').replace('"', '\\"')
            return '"' + text + '"'

        def flist(values) -> str:
            return '[' + ', '.join(f'{float(v):.8f}' for v in values) + ']'

        lines = [
            'camera_name: ' + q(f'Orbbec_{self.sn}'),
            'serial_number: ' + q(self.sn),
            'camera_role: ' + q(self.role),
            'session_tag: ' + q(self.tag),
            'created_at: ' + q(self.started_at or ''),
            'ended_at: ' + q(self.ended_at or ''),
            'align_mode: ' + q(self.align_mode_name),
            'capture_config_path: ' + q(cp.get('capture_config_path', ''), path_like=True),
            'rgb_directory: "' + ('color' if self.save_color else '') + '"',
            'color_left_directory: "' + ('left_rgb' if self.save_color_left else '') + '"',
            'color_right_directory: "' + ('right_rgb' if self.save_color_right else '') + '"',
            'depth_directory: "' + ('depth_raw' if self.save_depth else '') + '"',
            'depth_vis_directory: "' + ('depth_vis' if self.save_depth and self.save_depth_vis else '') + '"',
            'ir_left_directory: "' + ('ir_left' if self.save_ir_left else '') + '"',
            'ir_right_directory: "' + ('ir_right' if self.save_ir_right else '') + '"',
            'image_index_start: 1',
            'timestamp_unit: "seconds_from_session_start"',
            'timestamp_file_format: "' + ('frame_id,timestamp_s' if self.timestamps_csv else 'frame_id timestamp_s') + '"',
            f'pair_count: {self.pair_index}',
            f'image_width: {w}',
            f'image_height: {h}',
            f'depth_resized_to_rgb: {str(self.depth_resized_count > 0).lower()}',
            f'depth_resized_pair_count: {self.depth_resized_count}',
            'depth_unit: "millimeter"',
            f'depth_scale_from_raw: {depth_scale:.8f}',
            f'pointcloud_ready: {str(bool(self.require_aligned_depth_to_color and self.depth_resized_count == 0)).lower()}',
            f'require_aligned_depth_to_color: {str(self.require_aligned_depth_to_color).lower()}',
            'camera_model: "pinhole"',
            f'color_fx: {fx:.8f}',
            f'color_fy: {fy:.8f}',
            f'color_cx: {cx:.8f}',
            f'color_cy: {cy:.8f}',
            f'color_k1: {k1:.8f}',
            f'color_k2: {k2:.8f}',
            f'color_p1: {p1:.8f}',
            f'color_p2: {p2:.8f}',
            f'color_k3: {k3:.8f}',
            'color_camera_matrix: ' + flist([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]),
            'color_distortion_coefficients: ' + flist([k1, k2, p1, p2, k3]),
            f'source_color_width: {int(ci.get("width", 0))}',
            f'source_color_height: {int(ci.get("height", 0))}',
            f'source_color_fx: {float(ci.get("fx", 0.0)):.8f}',
            f'source_color_fy: {float(ci.get("fy", 0.0)):.8f}',
            f'source_color_cx: {float(ci.get("cx", 0.0)):.8f}',
            f'source_color_cy: {float(ci.get("cy", 0.0)):.8f}',
            f'source_color_distortion_model: {int(cd.get("model", 0))}',
            f'source_color_k1: {float(cd.get("k1", 0.0)):.8f}',
            f'source_color_k2: {float(cd.get("k2", 0.0)):.8f}',
            f'source_color_k3: {float(cd.get("k3", 0.0)):.8f}',
            f'source_color_k4: {float(cd.get("k4", 0.0)):.8f}',
            f'source_color_k5: {float(cd.get("k5", 0.0)):.8f}',
            f'source_color_k6: {float(cd.get("k6", 0.0)):.8f}',
            f'source_color_p1: {float(cd.get("p1", 0.0)):.8f}',
            f'source_color_p2: {float(cd.get("p2", 0.0)):.8f}',
            f'source_depth_width: {int(di.get("width", 0))}',
            f'source_depth_height: {int(di.get("height", 0))}',
            f'source_depth_fx: {float(di.get("fx", 0.0)):.8f}',
            f'source_depth_fy: {float(di.get("fy", 0.0)):.8f}',
            f'source_depth_cx: {float(di.get("cx", 0.0)):.8f}',
            f'source_depth_cy: {float(di.get("cy", 0.0)):.8f}',
            f'source_depth_distortion_model: {int(dd.get("model", 0))}',
            f'source_depth_k1: {float(dd.get("k1", 0.0)):.8f}',
            f'source_depth_k2: {float(dd.get("k2", 0.0)):.8f}',
            f'source_depth_k3: {float(dd.get("k3", 0.0)):.8f}',
            f'source_depth_k4: {float(dd.get("k4", 0.0)):.8f}',
            f'source_depth_k5: {float(dd.get("k5", 0.0)):.8f}',
            f'source_depth_k6: {float(dd.get("k6", 0.0)):.8f}',
            f'source_depth_p1: {float(dd.get("p1", 0.0)):.8f}',
            f'source_depth_p2: {float(dd.get("p2", 0.0)):.8f}',
            'depth_to_color_rotation_3x3: ' + flist([v for row in rot for v in row]),
            'depth_to_color_translation_mm: ' + flist(trans),
            f'is_mirrored: {str(bool(cp.get("is_mirrored", False))).lower()}',
        ]
        return '\n'.join(lines) + '\n'

    def _build_camera_info_json(self) -> dict[str, Any]:
        cp = self.camera_params or {}
        resolved = cp.get('resolved_capture_config', {}) or {}
        rect_cfg = resolved.get('rectification', {}) or {}
        stereo_cfg = resolved.get('stereo', {}) or {}
        fusion_cfg = resolved.get('fusion', {}) or {}
        pose_source_cfg = resolved.get('pose_source', {}) or {}
        property_report = cp.get('camera_property_apply_report', []) or []
        stereo_calibration = cp.get('stereo_calibration', {}) or {}
        ir_rectify_report = next(
            (item for item in property_report if item.get('name') == 'OB_PROP_IR_RECTIFY_BOOL'),
            None,
        )

        if self.save_color_left and self.save_color_right:
            stereo_modality = 'dual_rgb'
            left_dir = 'left_rgb'
            right_dir = 'right_rgb'
            left_frame = 'OB_FRAME_COLOR_LEFT'
            right_frame = 'OB_FRAME_COLOR_RIGHT'
            note = (
                'Dual RGB mode uses SDK COLOR_LEFT/COLOR_RIGHT frames. '
                'Gemini 305 must be switched to the Dual Color Streams device preset before stream profile query.'
            )
        elif self.save_ir_left and self.save_ir_right:
            stereo_modality = 'stereo_ir'
            left_dir = 'ir_left'
            right_dir = 'ir_right'
            left_frame = 'OB_FRAME_IR_LEFT'
            right_frame = 'OB_FRAME_IR_RIGHT'
            note = (
                'Stereo IR mode uses SDK IR_LEFT/IR_RIGHT frames. '
                'For Fast-FoundationStereo, convert/replicate grayscale to the expected input format if needed.'
            )
        else:
            stereo_modality = 'not_stereo_pair'
            left_dir = ''
            right_dir = ''
            left_frame = ''
            right_frame = ''
            note = 'This session is not a left/right stereo capture session.'

        rectification_requested = bool(rect_cfg.get('try_sdk_rectify', False))
        rectified = bool(rect_cfg.get('assume_rectified', False))
        if rectified:
            rectification_note = (
                'rectified=true comes from config.rectification.assume_rectified. '
                'Use it only after SDK documentation or an epipolar check confirms the saved pairs are rectified.'
            )
        elif rectification_requested:
            rectification_note = (
                'SDK rectification was requested, but this collector cannot verify the saved frames are already '
                'undistorted and epipolar-rectified, so rectified=false is kept for safety.'
            )
        else:
            rectification_note = (
                'Saved stereo frames are treated as raw SDK frames. Run stereo calibration/rectification before '
                'Fast-FoundationStereo if rectified=false.'
            )

        stereo_verified = bool(stereo_cfg.get('left_right_order_verified_by_user', False))
        write_identity_poses = bool(stereo_cfg.get('write_identity_poses_csv', False))
        static_identity_pose = bool(stereo_cfg.get('identity_pose_assumes_static_scene', False))
        fusion_enabled = bool(fusion_cfg.get('enabled', False))
        pose_mode = str(pose_source_cfg.get('mode', 'none') or 'none')

        return {
            'schema': 'orbbec_live_capture_camera_info_v1',
            'created_at': self.started_at or '',
            'ended_at': self.ended_at or '',
            'device': cp.get('device_name', ''),
            'serial_number': self.sn,
            'session_tag': self.tag,
            'total_frames': self.pair_index,
            'duration_sec': round(self.duration_sec, 3),
            'capture_config_path': cp.get('capture_config_path', ''),
            'device_preset': (cp.get('device_preset_report') or {}).get('target_preset', ''),
            'device_preset_report': cp.get('device_preset_report', {}),
            'left_source_frame': left_frame,
            'right_source_frame': right_frame,
            'rectified': rectified,
            'raw_camera_params_file': 'camera_params.json',
            'timestamps_file': self.timestamps_filename,
            'stereo': {
                'modality': stereo_modality,
                'left_directory': left_dir,
                'right_directory': right_dir,
                'left_source_frame': left_frame,
                'right_source_frame': right_frame,
                'left_right_swapped': False,
                'left_right_order_verified_by_user': stereo_verified,
                'stereo_calibration_file': 'stereo_calib.json' if stereo_modality != 'not_stereo_pair' else '',
                'note': note,
            },
            'rectification': {
                'rectified': rectified,
                'rectification_requested': rectification_requested,
                'sdk_ir_rectify_property': ir_rectify_report,
                'note': rectification_note,
            },
            'fast_foundation_stereo': {
                'recommended_left_input': left_dir,
                'recommended_right_input': right_dir,
                'requires_rectified_pairs': True,
                'ready_without_offline_rectification': bool(rectified and left_dir and right_dir),
                'note': (
                    'Fast-FoundationStereo should receive synchronized left/right pairs with the same frame id. '
                    'If rectified=false, do not assume these images are ready for metric stereo until rectified offline.'
                ),
            },
            'poses': {
                'file': 'poses.csv' if write_identity_poses and stereo_modality != 'not_stereo_pair' else '',
                'recorded': False,
                'identity_placeholder': bool(write_identity_poses and stereo_modality != 'not_stereo_pair'),
                'static_scene_assumption': bool(static_identity_pose and write_identity_poses and stereo_modality != 'not_stereo_pair'),
                'pose_source_mode': pose_mode,
                'note': (
                    'poses.csv is an identity placeholder only. Replace it with real poses if the camera or scene moves.'
                    if write_identity_poses and stereo_modality != 'not_stereo_pair'
                    else 'No pose file is generated by default. Configure a real pose source before multi-frame fusion.'
                ),
            },
            'fusion': {
                'enabled': fusion_enabled,
                'ready': bool(fusion_enabled and write_identity_poses and stereo_modality != 'not_stereo_pair'),
                'note': (
                    'Identity poses are only appropriate when the rig and scene stay fixed.'
                    if fusion_enabled and write_identity_poses and stereo_modality != 'not_stereo_pair'
                    else 'Fusion requires a valid poses.csv when enabled.'
                ),
            },
            'image': {
                'width': int(self.out_width or self.target_width or 0),
                'height': int(self.out_height or self.target_height or 0),
                'index_start': 1,
                'filename_format': '000001.png',
            },
            'streams': resolved.get('streams', {}),
            'profiles': resolved.get('stream_profile', {}),
            'stereo_calibration': stereo_calibration,
        }

    def _build_stereo_calibration_json(self) -> dict[str, Any]:
        cp = self.camera_params or {}
        resolved = cp.get('resolved_capture_config', {}) or {}
        rect_cfg = resolved.get('rectification', {}) or {}
        stereo_cfg = resolved.get('stereo', {}) or {}
        calib = cp.get('stereo_calibration', {}) or {}
        if not (self.save_color_left and self.save_color_right):
            return {}
        left = calib.get('left_intrinsic', {}) or cp.get('color_intrinsic', {}) or {}
        right = calib.get('right_intrinsic', {}) or left
        left_dist = calib.get('left_distortion', {}) or cp.get('color_distortion', {}) or {}
        right_dist = calib.get('right_distortion', {}) or left_dist
        extrinsic = calib.get('left_to_right_extrinsic', {}) or {}
        rotation = extrinsic.get('rotation_3x3') or [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        translation_m = extrinsic.get('translation_m') or [float(calib.get('baseline_m', 0.0) or 0.0), 0.0, 0.0]
        return {
            'schema': 'orbbec_stereo_calibration_v1',
            'device_name': cp.get('device_name', ''),
            'serial_number': self.sn,
            'mode': calib.get('mode', 'Dual RGB'),
            'calibration_quality': calib.get('calibration_quality', 'reference_only'),
            'rectified': bool(rect_cfg.get('assume_rectified', False)),
            'left_right_order_verified_by_user': bool(stereo_cfg.get('left_right_order_verified_by_user', False)),
            'image_width': int(self.out_width or self.target_width or left.get('width', 0)),
            'image_height': int(self.out_height or self.target_height or left.get('height', 0)),
            'baseline_m': float(calib.get('baseline_m', 0.0) or 0.0),
            'K_left': [
                [float(left.get('fx', 0.0)), 0.0, float(left.get('cx', 0.0))],
                [0.0, float(left.get('fy', 0.0)), float(left.get('cy', 0.0))],
                [0.0, 0.0, 1.0],
            ],
            'D_left': [
                float(left_dist.get('k1', 0.0)),
                float(left_dist.get('k2', 0.0)),
                float(left_dist.get('p1', 0.0)),
                float(left_dist.get('p2', 0.0)),
                float(left_dist.get('k3', 0.0)),
            ],
            'K_right': [
                [float(right.get('fx', 0.0)), 0.0, float(right.get('cx', 0.0))],
                [0.0, float(right.get('fy', 0.0)), float(right.get('cy', 0.0))],
                [0.0, 0.0, 1.0],
            ],
            'D_right': [
                float(right_dist.get('k1', 0.0)),
                float(right_dist.get('k2', 0.0)),
                float(right_dist.get('p1', 0.0)),
                float(right_dist.get('p2', 0.0)),
                float(right_dist.get('k3', 0.0)),
            ],
            'R_left_right': [[float(v) for v in row] for row in rotation],
            'T_left_right_m': [float(v) for v in translation_m],
            'note': (
                'Generated by the collector. If calibration_quality=reference_only or rectified=false, '
                'run offline stereo calibration/rectification before Fast-FoundationStereo.'
            ),
        }

    def _build_stereo_calibration_yaml(self) -> str:
        data = self._build_stereo_calibration_json()
        if not data:
            return ''
        try:
            import yaml
        except ImportError:
            return json.dumps(data, ensure_ascii=False, indent=2) + '\n'
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)

    def _build_identity_poses_csv(self) -> str:
        lines = [
            'frame_id,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33'
        ]
        for idx in range(1, self.pair_index + 1):
            lines.append(f'{idx:06d},1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1')
        return '\n'.join(lines) + '\n'

    def _build_pose_source_note(self) -> str:
        cp = self.camera_params or {}
        stereo_cfg = ((cp.get('resolved_capture_config') or {}).get('stereo', {}) or {})
        lines = [
            'Pose source note',
            '',
            'poses.csv was generated as an identity placeholder by the collector.',
            'Use it only when the camera rig and the observed scene remain static across frames.',
            'If there is any camera or object motion, replace poses.csv with real per-frame poses.',
            f'left_right_order_verified_by_user: {str(bool(stereo_cfg.get("left_right_order_verified_by_user", False))).lower()}',
        ]
        return '\n'.join(lines) + '\n'

    def _build_pose_note(self) -> str:
        lines = [
            'Pose note',
            '',
            'No pose data is recorded by this collector.',
            'Enabled image streams are paired by the same six-digit frame id.',
            f'{self.timestamps_filename} format: '
            + ('frame_id,timestamp_s CSV' if self.timestamps_csv else '<frame_id> <timestamp_seconds_from_session_start>.'),
            'Add one pose per frame here if an external tracker is used.',
        ]
        if self.tag:
            lines.append(f'Session tag: {self.tag}')
        if self.role:
            lines.append(f'Camera role: {self.role}')
        return '\n'.join(lines) + '\n'

    def start(self, camera_params: dict, session_dir: Optional[Path] = None):
        if self.active:
            raise RuntimeError('Session already active')

        self.root.mkdir(parents=True, exist_ok=True)
        self.session_dir = session_dir if session_dir is not None else self._unique_dir()
        self.rgb_dir = self.session_dir / 'color' if self.save_color else None
        self.color_left_dir = self.session_dir / 'left_rgb' if self.save_color_left else None
        self.color_right_dir = self.session_dir / 'right_rgb' if self.save_color_right else None
        self.depth_dir = self.session_dir / 'depth_raw' if self.save_depth else None
        self.depth_vis_dir = self.session_dir / 'depth_vis' if self.save_depth and self.save_depth_vis else None
        self.ir_left_dir = self.session_dir / 'ir_left' if self.save_ir_left else None
        self.ir_right_dir = self.session_dir / 'ir_right' if self.save_ir_right else None
        for folder in (self.rgb_dir, self.color_left_dir, self.color_right_dir, self.depth_dir, self.depth_vis_dir, self.ir_left_dir, self.ir_right_dir):
            if folder is not None:
                folder.mkdir(parents=True, exist_ok=True)

        self.timestamps_csv = bool(self.save_color_left and self.save_color_right and not self.save_depth)
        self.timestamps_filename = 'timestamps.csv' if self.timestamps_csv else 'timestamps.txt'
        self.timestamps_file = (self.session_dir / self.timestamps_filename).open('w', encoding='utf-8', buffering=1)
        if self.timestamps_csv:
            self.timestamps_file.write('frame_id,timestamp_s\n')
        self.timestamps_file.flush()
        if not self.minimal_dual_rgb_layout:
            (self.session_dir / 'pose_note.txt').write_text(self._build_pose_note(), encoding='utf-8')

        self.pair_index = 0
        self.first_ts_us = None
        self.depth_scale = None
        self.started_at = now_str()
        self.ended_at = None
        self.started_perf = time.perf_counter()
        self.duration_sec = 0.0
        self.camera_params = camera_params
        self.out_width = None
        self.out_height = None
        self.src_width = None
        self.src_height = None
        self.write_error = None
        self.write_queue = queue.Queue(maxsize=self.write_queue_maxsize)
        self.writer_threads = []
        for idx in range(self.writer_thread_count):
            thread = threading.Thread(target=self._writer_loop, name=f'SessionWriterThread-{idx + 1}', daemon=True)
            thread.start()
            self.writer_threads.append(thread)
        self.writer_thread = self.writer_threads[0] if self.writer_threads else None
        print(f'[{now_str()}] Background writer: threads={self.writer_thread_count}, queue_maxsize={self.write_queue_maxsize}')

        self.skipped_missing = 0
        self.skipped_format = 0
        self.skipped_resolution = 0
        self.skipped_decode = 0
        self.skipped_sync = 0
        self.last_skip_reason = ''
        self.last_skip_detail = ''
        self.depth_resized_count = 0
        self.active = True
        return self.session_dir

    def mark_skip(self, reason: str, detail: str = '') -> None:
        if not self.active:
            return
        if reason == 'missing':
            self.skipped_missing += 1
        elif reason == 'format':
            self.skipped_format += 1
        elif reason == 'resolution':
            self.skipped_resolution += 1
        elif reason == 'decode':
            self.skipped_decode += 1
        elif reason == 'sync':
            self.skipped_sync += 1
        self.last_skip_reason = reason
        self.last_skip_detail = detail

    def skip_summary(self) -> str:
        return (
            f'saved={self.pair_index}, '
            f'missing={self.skipped_missing}, format={self.skipped_format}, '
            f'resolution={self.skipped_resolution}, decode={self.skipped_decode}, sync={self.skipped_sync}, '
            f'depth_resized={self.depth_resized_count}'
        )

    def save_pair(self, color_fd: FrameData, color_img: np.ndarray, depth_fd: FrameData, depth_raw: np.ndarray):
        if not self.active:
            return
        self._raise_if_write_failed()

        color_out = color_img
        if self.src_width is None or self.src_height is None:
            self.src_width = int(color_img.shape[1])
            self.src_height = int(color_img.shape[0])

        depth_scale = float(depth_fd.depth_scale if depth_fd.depth_scale is not None else 1.0)
        depth_mm_f = depth_raw.astype(np.float32) * depth_scale
        depth_out = np.clip(np.rint(depth_mm_f), 0, np.iinfo(np.uint16).max).astype(np.uint16)

        if self.target_width > 0 and self.target_height > 0:
            src_ratio = float(color_out.shape[1]) / float(max(1, color_out.shape[0]))
            dst_ratio = float(self.target_width) / float(max(1, self.target_height))
            if abs(src_ratio - dst_ratio) > 1e-3:
                raise RuntimeError(
                    f'Aspect ratio mismatch: source={color_out.shape[1]}x{color_out.shape[0]}, '
                    f'target={self.target_width}x{self.target_height}. '
                    'Use a target with the same aspect ratio as the aligned stream.'
                )
            if color_out.shape[1] != self.target_width or color_out.shape[0] != self.target_height:
                color_out = cv2.resize(color_out, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
            if depth_out.shape[1] != self.target_width or depth_out.shape[0] != self.target_height:
                depth_out = cv2.resize(depth_out, (self.target_width, self.target_height), interpolation=cv2.INTER_NEAREST)

        if depth_out.shape[1] != color_out.shape[1] or depth_out.shape[0] != color_out.shape[0]:
            if self.require_aligned_depth_to_color:
                raise RuntimeError(
                    f'RGB/Depth resolution mismatch after D2C alignment: '
                    f'color={color_out.shape[1]}x{color_out.shape[0]}, '
                    f'depth={depth_out.shape[1]}x{depth_out.shape[0]}. '
                    'Point cloud conversion requires aligned depth. '
                    'Fix pipeline.align_mode/stream resolution, or set pointcloud.require_aligned_depth_to_color=false.'
                )
            depth_out = cv2.resize(depth_out, (color_out.shape[1], color_out.shape[0]), interpolation=cv2.INTER_NEAREST)
            self.depth_resized_count += 1

        depth_vis = None
        if self.save_depth_vis:
            try:
                depth_vis = depth_to_vis(depth_out)
            except Exception:
                depth_vis = None

        self.out_height, self.out_width = color_out.shape[0], color_out.shape[1]
        pid = self.pair_index + 1
        base = f'{pid:06d}.png'
        rgb_base = f'{pid:06d}.{self.color_format}'
        depth_base = f'{pid:06d}.{self.depth_raw_format}'

        ts_us = int((int(color_fd.dev_ts) + int(depth_fd.dev_ts)) // 2)
        if self.first_ts_us is None:
            self.first_ts_us = ts_us
        ts = max(0.0, (ts_us - self.first_ts_us) / 1_000_000.0)

        with self.enqueue_lock:
            if not self.active:
                return
            self.write_queue.put({
                'pid': pid,
                'base': base,
                'rgb_base': rgb_base,
                'depth_base': depth_base,
                'timestamp_s': ts,
                'rgb': color_out,
                'depth': depth_out,
                'depth_vis': depth_vis,
            })
            self.pair_index = pid
            self.depth_scale = depth_scale

    def save_stereo_pair(
        self,
        left_fd: FrameData,
        left_img: np.ndarray,
        right_fd: FrameData,
        right_img: np.ndarray,
    ):
        if not self.active:
            return
        self._raise_if_write_failed()

        if self.src_width is None or self.src_height is None:
            self.src_width = int(left_img.shape[1])
            self.src_height = int(left_img.shape[0])
        self.out_height, self.out_width = int(left_img.shape[0]), int(left_img.shape[1])

        pid = self.pair_index + 1
        base = f'{pid:06d}.png'
        ts_us = int((int(left_fd.dev_ts) + int(right_fd.dev_ts)) // 2)
        if self.first_ts_us is None:
            self.first_ts_us = ts_us
        ts = max(0.0, (ts_us - self.first_ts_us) / 1_000_000.0)

        with self.enqueue_lock:
            if not self.active:
                return
            self.write_queue.put({
                'pid': pid,
                'base': base,
                'timestamp_s': ts,
                'ir_left': left_img,
                'ir_right': right_img,
            })
            self.pair_index = pid

    def save_dual_color_pair(
        self,
        left_fd: FrameData,
        left_img: np.ndarray,
        right_fd: FrameData,
        right_img: np.ndarray,
    ):
        if not self.active:
            return
        self._raise_if_write_failed()

        if self.src_width is None or self.src_height is None:
            self.src_width = int(left_img.shape[1])
            self.src_height = int(left_img.shape[0])
        self.out_height, self.out_width = int(left_img.shape[0]), int(left_img.shape[1])

        pid = self.pair_index + 1
        base = f'{pid:06d}.png'
        ts_us = int((int(left_fd.dev_ts) + int(right_fd.dev_ts)) // 2)
        if self.first_ts_us is None:
            self.first_ts_us = ts_us
        ts = max(0.0, (ts_us - self.first_ts_us) / 1_000_000.0)

        with self.enqueue_lock:
            if not self.active:
                return
            self.write_queue.put({
                'pid': pid,
                'base': base,
                'timestamp_s': ts,
                'color_left': left_img,
                'color_right': right_img,
            })
            self.pair_index = pid

    def stop(self) -> None:
        if not self.active:
            return

        self.ended_at = now_str()
        if self.started_perf is not None:
            self.duration_sec = max(0.0, time.perf_counter() - self.started_perf)
        if self.write_queue is not None:
            if self.writer_threads and not self.write_error:
                for _ in self.writer_threads:
                    self.write_queue.put(None)
            for thread in self.writer_threads:
                thread.join()
            self.write_queue = None
            self.writer_threads = []
            self.writer_thread = None

        if self.timestamps_file:
            self.timestamps_file.flush()
            self.timestamps_file.close()
            self.timestamps_file = None

        stereo_calib = self._build_stereo_calibration_json()
        if stereo_calib:
            k = stereo_calib.get('K_left', [])
            baseline_m = float(stereo_calib.get('baseline_m', 0.0))
            if len(k) == 3:
                flat = [float(v) for row in k for v in row]
                k_text = ' '.join(f'{v:.8f}' for v in flat) + '\n' + f'{baseline_m:.9f}\n'
                (self.session_dir / 'K.txt').write_text(k_text, encoding='utf-8')

        if not self.minimal_dual_rgb_layout:
            (self.session_dir / 'camera_info.yaml').write_text(self._build_camera_info_yaml(), encoding='utf-8')
            (self.session_dir / 'camera_info.json').write_text(
                json.dumps(self._build_camera_info_json(), ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
            (self.session_dir / 'camera_params.json').write_text(
                json.dumps(self.camera_params or {}, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
            if stereo_calib:
                (self.session_dir / 'stereo_calib.json').write_text(
                    json.dumps(stereo_calib, ensure_ascii=False, indent=2),
                    encoding='utf-8',
                )
                (self.session_dir / 'stereo_calib.yaml').write_text(
                    self._build_stereo_calibration_yaml(),
                    encoding='utf-8',
                )
                (self.session_dir / 'baseline.txt').write_text(
                    f'{float(stereo_calib.get("baseline_m", 0.0)):.9f}\n',
                    encoding='utf-8',
                )

            stereo_cfg = ((self.camera_params or {}).get('resolved_capture_config', {}) or {}).get('stereo', {}) or {}
            if self.save_color_left and self.save_color_right and bool(stereo_cfg.get('write_identity_poses_csv', False)):
                (self.session_dir / 'poses.csv').write_text(self._build_identity_poses_csv(), encoding='utf-8')
                (self.session_dir / 'poses_source.txt').write_text(self._build_pose_source_note(), encoding='utf-8')

            metadata = {
                'device': (self.camera_params or {}).get('device_name', ''),
                'serial_number': self.sn,
                'total_frames': self.pair_index,
                'duration_sec': round(self.duration_sec, 3),
                'streams': (self.camera_params or {}).get('resolved_capture_config', {}).get('streams', {}),
                'pointcloud': (self.camera_params or {}).get('resolved_capture_config', {}).get('pointcloud', {}),
                'align_mode': self.align_mode_name,
                'output_dirs': {
                    'color': 'color' if self.save_color else None,
                    'left_rgb': 'left_rgb' if self.save_color_left else None,
                    'right_rgb': 'right_rgb' if self.save_color_right else None,
                    'color_left': 'left_rgb' if self.save_color_left else None,
                    'color_right': 'right_rgb' if self.save_color_right else None,
                    'depth_raw': 'depth_raw' if self.save_depth else None,
                    'depth_vis': 'depth_vis' if self.save_depth and self.save_depth_vis else None,
                    'ir_left': 'ir_left' if self.save_ir_left else None,
                    'ir_right': 'ir_right' if self.save_ir_right else None,
                },
                'camera_info_file': 'camera_info.yaml',
                'camera_info_json_file': 'camera_info.json',
                'timestamps_file': self.timestamps_filename,
                'stereo_calibration_file': 'stereo_calib.json' if stereo_calib else None,
                'poses_file': 'poses.csv' if self.save_color_left and self.save_color_right and bool(stereo_cfg.get('write_identity_poses_csv', False)) else None,
            }
            (self.session_dir / 'metadata.json').write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
        self.active = False
        self._raise_if_write_failed()


def format_name(fmt: int) -> str:
    return FORMAT_NAMES.get(int(fmt), str(fmt))


def summarize_1280_profiles(profiles: list[dict]) -> str:
    grouped: dict[tuple[int, int], dict[str, set]] = {}
    for prof in profiles:
        if int(prof['width']) != 1280:
            continue
        key = (int(prof['width']), int(prof['height']))
        item = grouped.setdefault(key, {'fps': set(), 'formats': set()})
        item['fps'].add(int(prof['fps']))
        item['formats'].add(format_name(int(prof['format'])))
    if not grouped:
        return 'none'
    parts = []
    for (width, height), item in sorted(grouped.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
        fps_text = '/'.join(str(v) for v in sorted(item['fps'], reverse=True))
        fmt_text = ','.join(sorted(item['formats']))
        parts.append(f'{width}x{height} fps={fps_text} fmt={fmt_text}')
    return '; '.join(parts)


def summarize_profiles(profiles: list[dict], limit: int = 12) -> str:
    if not profiles:
        return 'none'
    grouped: dict[tuple[int, int], dict[str, set]] = {}
    for prof in profiles:
        key = (int(prof['width']), int(prof['height']))
        item = grouped.setdefault(key, {'fps': set(), 'formats': set()})
        item['fps'].add(int(prof['fps']))
        item['formats'].add(format_name(int(prof['format'])))
    parts = []
    for (width, height), item in sorted(grouped.items(), key=lambda kv: (-kv[0][0] * kv[0][1], -kv[0][0], -kv[0][1])):
        fps_text = '/'.join(str(v) for v in sorted(item['fps'], reverse=True))
        fmt_text = ','.join(sorted(item['formats']))
        parts.append(f'{width}x{height} fps={fps_text} fmt={fmt_text}')
        if len(parts) >= limit:
            break
    return '; '.join(parts)


def print_all_stream_profiles(sdk: SDK, dev, device_name: str, title: str = 'stream profiles') -> None:
    sensor_items = [
        ('COLOR', OB_SENSOR_COLOR),
        ('COLOR_LEFT', OB_SENSOR_COLOR_LEFT),
        ('COLOR_RIGHT', OB_SENSOR_COLOR_RIGHT),
        ('DEPTH', OB_SENSOR_DEPTH),
        ('IR_LEFT', OB_SENSOR_IR_LEFT),
        ('IR_RIGHT', OB_SENSOR_IR_RIGHT),
        ('IR', OB_SENSOR_IR),
    ]
    pipe = 0
    print(f'[{now_str()}] {device_name}: {title}')
    try:
        pipe = sdk.create_pipeline(dev)
        for label, sensor_type in sensor_items:
            try:
                profiles = sdk.list_video_stream_profiles(pipe, sensor_type)
                print(f'[{now_str()}]   {label}: {summarize_profiles(profiles, limit=32)}')
            except Exception as ex:
                print(f'[{now_str()}]   {label}: unavailable ({ex})')
    finally:
        if pipe:
            sdk.delete_pipeline(pipe)


def is_dual_rgb_settings(settings: dict[str, Any]) -> bool:
    streams_cfg = settings.get('streams', {}) or {}
    return bool(
        streams_cfg.get('color_left', False)
        and streams_cfg.get('color_right', False)
        and not streams_cfg.get('depth', True)
    )


def switch_device_preset_if_configured(sdk: SDK, dev, device_name: str, settings: dict[str, Any]) -> dict[str, Any]:
    preset_cfg = settings.get('device_preset', {}) or {}
    enabled = bool(preset_cfg.get('enabled', False))
    target = str(preset_cfg.get('name') or '').strip()
    settle_ms = int(preset_cfg.get('settle_ms', 800) or 0)
    report: dict[str, Any] = {
        'requested': False,
        'target_preset': target,
        'before_preset': '',
        'after_preset': '',
        'available_presets': [],
        'ok': False,
    }

    if is_dual_rgb_settings(settings) and not target:
        target = DUAL_COLOR_PRESET_NAME
        report['target_preset'] = target

    if not enabled or not target:
        return report

    report['requested'] = True
    if target == DUAL_COLOR_PRESET_NAME:
        fail_msg = 'Failed to switch Gemini 305 to Dual Color Streams. Please check SDK/firmware/Viewer support.'
    else:
        fail_msg = f'Failed to switch {device_name} to device preset "{target}". Please check SDK/firmware/Viewer support.'
    try:
        before = sdk.get_current_preset_name(dev)
        presets = sdk.get_available_presets(dev)
        report['before_preset'] = before
        report['available_presets'] = presets
        print(f'[{now_str()}] Current device preset: {before or "unknown"}')
        print(f'[{now_str()}] Available device presets: {", ".join(presets) if presets else "none"}')
        if target not in presets:
            raise RuntimeError(f'target preset "{target}" not in available presets: {presets}')
        if before != target:
            print(f'[{now_str()}] Switching device preset to: {target}')
            sdk.load_preset(dev, target)
            if settle_ms > 0:
                time.sleep(settle_ms / 1000.0)
        after = sdk.get_current_preset_name(dev)
        report['after_preset'] = after
        if after != target:
            raise RuntimeError(f'after switch current preset is "{after}", expected "{target}"')
        report['ok'] = True
        print(f'[{now_str()}] Device preset switched OK: {after}')
        return report
    except Exception as ex:
        report['error'] = str(ex)
        print(f'[{now_str()}] ERROR {fail_msg}')
        print(f'[{now_str()}] ERROR preset detail: {ex}')
        raise RuntimeError(f'{fail_msg} Detail: {ex}') from ex


def choose_format(profiles: list[dict], preferred: list[int]) -> Optional[dict]:
    by_format = {int(prof['format']): prof for prof in profiles}
    for fmt in preferred:
        if fmt in by_format:
            return by_format[fmt]
    return profiles[0] if profiles else None


def choose_profile_from_config(profiles: list[dict], cfg: dict[str, Any], preferred_formats: list[int]) -> Optional[dict]:
    width = int(cfg.get('width') or 0)
    height = int(cfg.get('height') or 0)
    fps = int(cfg.get('fps') or 0)
    formats = format_candidates_from_config(cfg.get('formats', cfg.get('format')), preferred_formats)

    candidates = []
    for prof in profiles:
        if width > 0 and int(prof['width']) != width:
            continue
        if height > 0 and int(prof['height']) != height:
            continue
        if fps > 0 and int(prof['fps']) != fps:
            continue
        candidates.append(prof)
    return choose_format(candidates, formats)


def unique_ints(values: list[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        value = int(value)
        if value not in result:
            result.append(value)
    return result


def choose_fallback_profile(profiles: list[dict], cfg: dict[str, Any], preferred_formats: list[int]) -> Optional[dict]:
    if not profiles:
        return None
    target_w = int(cfg.get('width') or 0)
    target_h = int(cfg.get('height') or 0)
    target_fps = int(cfg.get('fps') or 0)
    target_area = target_w * target_h
    formats = format_candidates_from_config(cfg.get('formats', cfg.get('format')), preferred_formats)
    formats = unique_ints(formats + preferred_formats)

    def score(prof: dict) -> tuple[int, int, int, int, int]:
        fmt = int(prof['format'])
        width = int(prof['width'])
        height = int(prof['height'])
        fps = int(prof['fps'])
        area = width * height
        fmt_score = len(formats) - formats.index(fmt) if fmt in formats else 0
        fps_score = -abs(fps - target_fps) if target_fps > 0 else fps
        area_score = -abs(area - target_area) if target_area > 0 else area
        exact_res = 1 if target_w > 0 and target_h > 0 and width == target_w and height == target_h else 0
        exact_fps = 1 if target_fps > 0 and fps == target_fps else 0
        return (exact_res, exact_fps, fmt_score, area_score, fps_score)

    return sorted(profiles, key=score, reverse=True)[0]


def select_configured_stream_profile(sdk: SDK, pipe, device_name: str, stream_cfg: dict[str, Any]) -> Optional[dict]:
    color_profiles = sdk.list_video_stream_profiles(pipe, OB_SENSOR_COLOR)
    depth_profiles = sdk.list_video_stream_profiles(pipe, OB_SENSOR_DEPTH)
    print(f'[{now_str()}] {device_name} SDK COLOR 1280 profiles: {summarize_1280_profiles(color_profiles)}')
    print(f'[{now_str()}] {device_name} SDK DEPTH 1280 profiles: {summarize_1280_profiles(depth_profiles)}')

    color_cfg = stream_cfg.get('color', {}) or {}
    depth_cfg = stream_cfg.get('depth', {}) or {}
    color = choose_profile_from_config(
        color_profiles,
        color_cfg,
        [OB_FORMAT_BGR, OB_FORMAT_RGB, OB_FORMAT_YUYV, OB_FORMAT_MJPG, OB_FORMAT_BGRA, OB_FORMAT_RGBA, OB_FORMAT_UYVY],
    )
    depth = choose_profile_from_config(depth_profiles, depth_cfg, [OB_FORMAT_Y16])

    if not color or not depth:
        requested = (
            f'color {color_cfg.get("width", 0)}x{color_cfg.get("height", 0)}@{color_cfg.get("fps", 0)} '
            f'{color_cfg.get("formats", color_cfg.get("format", ""))}, '
            f'depth {depth_cfg.get("width", 0)}x{depth_cfg.get("height", 0)}@{depth_cfg.get("fps", 0)} '
            f'{depth_cfg.get("formats", depth_cfg.get("format", ""))}'
        )
        print(f'[{now_str()}] ERROR {device_name}: SDK did not expose requested RGB-D profile: {requested}')
        return None

    label = (
        f'color {color["width"]}x{color["height"]}@{color["fps"]} {format_name(color["format"])}, '
        f'depth {depth["width"]}x{depth["height"]}@{depth["fps"]} {format_name(depth["format"])}'
    )
    print(f'[{now_str()}] {device_name}: selected configured RGB-D profile: {label}')
    return {
        'label': label,
        'color_width': int(color['width']),
        'color_height': int(color['height']),
        'color_fps': int(color['fps']),
        'color_format': int(color['format']),
        'depth_width': int(depth['width']),
        'depth_height': int(depth['height']),
        'depth_fps': int(depth['fps']),
        'depth_format': int(depth['format']),
    }


def select_ir_stereo_profile(sdk: SDK, pipe, device_name: str, stream_cfg: dict[str, Any]) -> Optional[dict]:
    ir_cfg = stream_cfg.get('ir', {}) or {}
    preferred = [OB_FORMAT_Y16, OB_FORMAT_Y8, OB_FORMAT_GRAY]
    try:
        left_profiles = sdk.list_video_stream_profiles(pipe, OB_SENSOR_IR_LEFT)
    except Exception as ex:
        print(f'[{now_str()}] WARN {device_name}: query IR_LEFT profiles failed: {ex}')
        left_profiles = []
    try:
        right_profiles = sdk.list_video_stream_profiles(pipe, OB_SENSOR_IR_RIGHT)
    except Exception as ex:
        print(f'[{now_str()}] WARN {device_name}: query IR_RIGHT profiles failed: {ex}')
        right_profiles = []

    print(f'[{now_str()}] {device_name} SDK IR_LEFT profiles: {summarize_profiles(left_profiles)}')
    print(f'[{now_str()}] {device_name} SDK IR_RIGHT profiles: {summarize_profiles(right_profiles)}')

    left = choose_profile_from_config(left_profiles, ir_cfg, preferred)
    right = choose_profile_from_config(right_profiles, ir_cfg, preferred)
    exact = bool(left and right)
    if not left:
        left = choose_fallback_profile(left_profiles, ir_cfg, preferred)
    if not right:
        right = choose_fallback_profile(right_profiles, ir_cfg, preferred)
    if not left or not right:
        print(f'[{now_str()}] ERROR {device_name}: SDK did not expose usable IR_LEFT/IR_RIGHT profiles.')
        return None

    if not exact:
        requested = f'{ir_cfg.get("width", 0)}x{ir_cfg.get("height", 0)}@{ir_cfg.get("fps", 0)} {ir_cfg.get("formats", ir_cfg.get("format", ""))}'
        selected = (
            f'left {left["width"]}x{left["height"]}@{left["fps"]} {format_name(left["format"])}, '
            f'right {right["width"]}x{right["height"]}@{right["fps"]} {format_name(right["format"])}'
        )
        print(f'[{now_str()}] WARN requested stereo IR profile not available: {requested}; selected fallback: {selected}')

    label = (
        f'ir_left {left["width"]}x{left["height"]}@{left["fps"]} {format_name(left["format"])}, '
        f'ir_right {right["width"]}x{right["height"]}@{right["fps"]} {format_name(right["format"])}'
    )
    print(f'[{now_str()}] {device_name}: selected stereo IR profile: {label}')
    return {
        'label': label,
        'ir_left_width': int(left['width']),
        'ir_left_height': int(left['height']),
        'ir_left_fps': int(left['fps']),
        'ir_left_format': int(left['format']),
        'ir_right_width': int(right['width']),
        'ir_right_height': int(right['height']),
        'ir_right_fps': int(right['fps']),
        'ir_right_format': int(right['format']),
    }


def select_dual_color_profile(sdk: SDK, pipe, device_name: str, stream_cfg: dict[str, Any]) -> Optional[dict]:
    color_cfg = stream_cfg.get('dual_color', {}) or {}
    preferred = [OB_FORMAT_BGR, OB_FORMAT_RGB, OB_FORMAT_YUYV, OB_FORMAT_MJPG, OB_FORMAT_BGRA, OB_FORMAT_RGBA, OB_FORMAT_UYVY]
    try:
        left_profiles = sdk.list_video_stream_profiles(pipe, OB_SENSOR_COLOR_LEFT)
    except Exception as ex:
        print(f'[{now_str()}] WARN {device_name}: query COLOR_LEFT profiles failed: {ex}')
        left_profiles = []
    try:
        right_profiles = sdk.list_video_stream_profiles(pipe, OB_SENSOR_COLOR_RIGHT)
    except Exception as ex:
        print(f'[{now_str()}] WARN {device_name}: query COLOR_RIGHT profiles failed: {ex}')
        right_profiles = []

    print(f'[{now_str()}] {device_name} SDK COLOR_LEFT profiles: {summarize_profiles(left_profiles)}')
    print(f'[{now_str()}] {device_name} SDK COLOR_RIGHT profiles: {summarize_profiles(right_profiles)}')

    left = choose_profile_from_config(left_profiles, color_cfg, preferred)
    right = choose_profile_from_config(right_profiles, color_cfg, preferred)
    exact = bool(left and right)
    if not left:
        left = choose_fallback_profile(left_profiles, color_cfg, preferred)
    if not right:
        right = choose_fallback_profile(right_profiles, color_cfg, preferred)
    if not left or not right:
        print(f'[{now_str()}] ERROR {device_name}: SDK did not expose usable COLOR_LEFT/COLOR_RIGHT profiles.')
        return None

    if not exact:
        requested = f'{color_cfg.get("width", 0)}x{color_cfg.get("height", 0)}@{color_cfg.get("fps", 0)} {color_cfg.get("formats", color_cfg.get("format", ""))}'
        selected = (
            f'left {left["width"]}x{left["height"]}@{left["fps"]} {format_name(left["format"])}, '
            f'right {right["width"]}x{right["height"]}@{right["fps"]} {format_name(right["format"])}'
        )
        print(f'[{now_str()}] WARN requested dual RGB profile not available: {requested}; selected fallback: {selected}')

    label = (
        f'color_left {left["width"]}x{left["height"]}@{left["fps"]} {format_name(left["format"])}, '
        f'color_right {right["width"]}x{right["height"]}@{right["fps"]} {format_name(right["format"])}'
    )
    print(f'[{now_str()}] {device_name}: selected dual RGB profile: {label}')
    return {
        'label': label,
        'color_left_width': int(left['width']),
        'color_left_height': int(left['height']),
        'color_left_fps': int(left['fps']),
        'color_left_format': int(left['format']),
        'color_right_width': int(right['width']),
        'color_right_height': int(right['height']),
        'color_right_fps': int(right['fps']),
        'color_right_format': int(right['format']),
    }


def fixed_1280x800_30_model(device_name: str) -> bool:
    name = device_name.lower()
    return 'gemini 305' in name or 'gemini 335l' in name


def select_fixed_1280x800_30_profile(sdk: SDK, pipe, device_name: str) -> Optional[dict]:
    if not fixed_1280x800_30_model(device_name):
        return None

    color_profiles = sdk.list_video_stream_profiles(pipe, OB_SENSOR_COLOR)
    depth_profiles = sdk.list_video_stream_profiles(pipe, OB_SENSOR_DEPTH)
    print(f'[{now_str()}] {device_name} SDK COLOR 1280 profiles: {summarize_1280_profiles(color_profiles)}')
    print(f'[{now_str()}] {device_name} SDK DEPTH 1280 profiles: {summarize_1280_profiles(depth_profiles)}')

    decodable_color = {OB_FORMAT_YUYV, OB_FORMAT_MJPG, OB_FORMAT_BGR, OB_FORMAT_RGB, OB_FORMAT_BGRA, OB_FORMAT_RGBA, OB_FORMAT_UYVY}
    color_candidates = [
        p for p in color_profiles
        if p['width'] == 1280 and p['height'] == 800 and p['fps'] == 30 and p['format'] in decodable_color
    ]
    depth_candidates = [
        p for p in depth_profiles
        if p['width'] == 1280 and p['height'] == 800 and p['fps'] == 30 and p['format'] == OB_FORMAT_Y16
    ]

    color = choose_format(
        color_candidates,
        [OB_FORMAT_BGR, OB_FORMAT_RGB, OB_FORMAT_YUYV, OB_FORMAT_MJPG, OB_FORMAT_BGRA, OB_FORMAT_RGBA, OB_FORMAT_UYVY],
    )
    depth = depth_candidates[0] if depth_candidates else None

    if not color or not depth:
        print(f'[{now_str()}] ERROR {device_name}: SDK did not expose required RGB-D 1280x800@30fps profile.')
        return None

    label = (
        f'color {color["width"]}x{color["height"]}@{color["fps"]} {format_name(color["format"])}, '
        f'depth {depth["width"]}x{depth["height"]}@{depth["fps"]} {format_name(depth["format"])}'
    )
    print(f'[{now_str()}] {device_name}: selected fixed RGB-D profile: {label}')

    return {
        'label': label,
        'color_width': int(color['width']),
        'color_height': int(color['height']),
        'color_fps': int(color['fps']),
        'color_format': int(color['format']),
        'depth_width': int(depth['width']),
        'depth_height': int(depth['height']),
        'depth_fps': int(depth['fps']),
        'depth_format': int(depth['format']),
    }


def apply_camera_properties(sdk: SDK, dev, settings: dict[str, Any]) -> list[dict[str, Any]]:
    prop_cfg = settings.get('camera_properties', {}) or {}
    if not bool(prop_cfg.get('enabled', True)):
        print(f'[{now_str()}] Camera property config disabled; keep current camera values.')
        return []

    strict = bool(prop_cfg.get('strict', False))
    values = prop_cfg.get('values', {}) or {}
    report: list[dict[str, Any]] = []
    for name, value in values.items():
        if value is None:
            continue
        spec = PROPERTY_SPECS.get(str(name))
        if not spec:
            item = {'name': str(name), 'value': value, 'ok': False, 'message': 'unknown property name'}
            report.append(item)
            print(f'[{now_str()}] WARN camera property skipped: {name}, unknown property name')
            if strict:
                raise RuntimeError(f'Unknown camera property in config: {name}')
            continue

        property_id, value_type = spec
        try:
            if value_type == 'bool':
                parsed = bool_or_none(value)
                if parsed is None:
                    continue
                ok, message = sdk.set_bool_property_try(dev, property_id, parsed, str(name))
            elif value_type == 'int':
                parsed = int(value)
                ok, message = sdk.set_int_property_try(dev, property_id, parsed, str(name))
            elif value_type == 'float':
                parsed = float(value)
                ok, message = sdk.set_float_property_try(dev, property_id, parsed, str(name))
            else:
                ok, message = False, f'unsupported property value type: {value_type}'
        except Exception as ex:
            ok, message = False, str(ex)

        report.append({'name': str(name), 'value': value, 'type': value_type, 'ok': ok, 'message': message})
        level = 'OK' if ok else 'WARN'
        print(f'[{now_str()}] {level} camera property {name}: {message}')
        if strict and not ok:
            raise RuntimeError(f'Failed to apply camera property {name}: {message}')

    if not report:
        print(f'[{now_str()}] No camera properties applied; all configured values are None or disabled.')
    return report


def enable_configured_video_stream(
    sdk: SDK,
    cfg,
    stream_type: int,
    stream_name: str,
    profile_cfg: dict[str, Any],
    default_formats: list[int],
) -> None:
    width = int(profile_cfg.get('width') or 0)
    height = int(profile_cfg.get('height') or 0)
    fps = int(profile_cfg.get('fps') or 0)
    formats = format_candidates_from_config(profile_cfg.get('formats', profile_cfg.get('format')), default_formats)
    fmt = formats[0]
    if width > 0 and height > 0 and fps > 0:
        try:
            sdk.enable_video_stream(cfg, stream_type, width, height, fps, fmt)
            print(f'[{now_str()}] Enabled {stream_name}: {width}x{height}@{fps} {format_name(fmt)}')
            return
        except Exception as ex:
            print(f'[{now_str()}] WARN enable {stream_name} exact profile failed: {ex}; fallback to SDK default stream.')
    sdk.enable_stream(cfg, stream_type)
    print(f'[{now_str()}] Enabled {stream_name}: SDK default')


def build_capture_config(
    sdk: SDK,
    align_mode: int,
    name: str,
    stream_profile: Optional[dict] = None,
    settings: Optional[dict[str, Any]] = None,
):
    align_cfg = (settings or {}).get('align', {}) or {}
    streams_cfg = (settings or {}).get('streams', {}) or {}
    configured_profiles = (settings or {}).get('stream_profile', {}) or {}
    save_color = bool(streams_cfg.get('color', True))
    save_color_left = bool(streams_cfg.get('color_left', False))
    save_color_right = bool(streams_cfg.get('color_right', False))
    save_depth = bool(streams_cfg.get('depth', True))
    save_ir_left = bool(streams_cfg.get('ir_left', False))
    save_ir_right = bool(streams_cfg.get('ir_right', False))
    stereo_ir_mode = bool(save_ir_left and save_ir_right and not save_depth)
    dual_rgb_mode = bool(save_color_left and save_color_right and not save_depth)
    cfg = sdk.create_config()
    try:
        if stream_profile and save_color and save_depth:
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
            if save_color:
                enable_configured_video_stream(
                    sdk,
                    cfg,
                    OB_STREAM_COLOR,
                    'COLOR',
                    configured_profiles.get('color', {}),
                    [OB_FORMAT_BGR, OB_FORMAT_RGB, OB_FORMAT_YUYV, OB_FORMAT_MJPG, OB_FORMAT_BGRA, OB_FORMAT_RGBA, OB_FORMAT_UYVY],
                )
            if save_color_left:
                if stream_profile and 'color_left_width' in stream_profile:
                    sdk.enable_video_stream(
                        cfg,
                        OB_STREAM_COLOR_LEFT,
                        stream_profile['color_left_width'],
                        stream_profile['color_left_height'],
                        stream_profile['color_left_fps'],
                        stream_profile['color_left_format'],
                    )
                    print(
                        f'[{now_str()}] Enabled COLOR_LEFT: '
                        f'{stream_profile["color_left_width"]}x{stream_profile["color_left_height"]}@{stream_profile["color_left_fps"]} '
                        f'{format_name(stream_profile["color_left_format"])}'
                    )
                elif dual_rgb_mode:
                    sdk.enable_stream(cfg, OB_STREAM_COLOR_LEFT)
                    print(f'[{now_str()}] Enabled COLOR_LEFT: SDK default')
                else:
                    enable_configured_video_stream(
                        sdk,
                        cfg,
                        OB_STREAM_COLOR_LEFT,
                        'COLOR_LEFT',
                        configured_profiles.get('dual_color', {}),
                        [OB_FORMAT_BGR, OB_FORMAT_RGB, OB_FORMAT_YUYV, OB_FORMAT_MJPG, OB_FORMAT_BGRA, OB_FORMAT_RGBA, OB_FORMAT_UYVY],
                    )
            if save_color_right:
                if stream_profile and 'color_right_width' in stream_profile:
                    sdk.enable_video_stream(
                        cfg,
                        OB_STREAM_COLOR_RIGHT,
                        stream_profile['color_right_width'],
                        stream_profile['color_right_height'],
                        stream_profile['color_right_fps'],
                        stream_profile['color_right_format'],
                    )
                    print(
                        f'[{now_str()}] Enabled COLOR_RIGHT: '
                        f'{stream_profile["color_right_width"]}x{stream_profile["color_right_height"]}@{stream_profile["color_right_fps"]} '
                        f'{format_name(stream_profile["color_right_format"])}'
                    )
                elif dual_rgb_mode:
                    sdk.enable_stream(cfg, OB_STREAM_COLOR_RIGHT)
                    print(f'[{now_str()}] Enabled COLOR_RIGHT: SDK default')
                else:
                    enable_configured_video_stream(
                        sdk,
                        cfg,
                        OB_STREAM_COLOR_RIGHT,
                        'COLOR_RIGHT',
                        configured_profiles.get('dual_color', {}),
                        [OB_FORMAT_BGR, OB_FORMAT_RGB, OB_FORMAT_YUYV, OB_FORMAT_MJPG, OB_FORMAT_BGRA, OB_FORMAT_RGBA, OB_FORMAT_UYVY],
                    )
            if save_depth:
                enable_configured_video_stream(
                    sdk,
                    cfg,
                    OB_STREAM_DEPTH,
                    'DEPTH',
                    configured_profiles.get('depth', {}),
                    [OB_FORMAT_Y16],
                )
        if save_ir_left:
            if stream_profile and 'ir_left_width' in stream_profile:
                sdk.enable_video_stream(
                    cfg,
                    OB_STREAM_IR_LEFT,
                    stream_profile['ir_left_width'],
                    stream_profile['ir_left_height'],
                    stream_profile['ir_left_fps'],
                    stream_profile['ir_left_format'],
                )
                print(
                    f'[{now_str()}] Enabled IR_LEFT: '
                    f'{stream_profile["ir_left_width"]}x{stream_profile["ir_left_height"]}@{stream_profile["ir_left_fps"]} '
                    f'{format_name(stream_profile["ir_left_format"])}'
                )
            else:
                if stereo_ir_mode:
                    sdk.enable_stream(cfg, OB_STREAM_IR_LEFT)
                    print(f'[{now_str()}] Enabled IR_LEFT: SDK default')
                else:
                    enable_configured_video_stream(
                        sdk,
                        cfg,
                        OB_STREAM_IR_LEFT,
                        'IR_LEFT',
                        configured_profiles.get('ir', {}),
                        [OB_FORMAT_Y16, OB_FORMAT_Y8, OB_FORMAT_GRAY],
                    )
        if save_ir_right:
            if stream_profile and 'ir_right_width' in stream_profile:
                sdk.enable_video_stream(
                    cfg,
                    OB_STREAM_IR_RIGHT,
                    stream_profile['ir_right_width'],
                    stream_profile['ir_right_height'],
                    stream_profile['ir_right_fps'],
                    stream_profile['ir_right_format'],
                )
                print(
                    f'[{now_str()}] Enabled IR_RIGHT: '
                    f'{stream_profile["ir_right_width"]}x{stream_profile["ir_right_height"]}@{stream_profile["ir_right_fps"]} '
                    f'{format_name(stream_profile["ir_right_format"])}'
                )
            else:
                if stereo_ir_mode:
                    sdk.enable_stream(cfg, OB_STREAM_IR_RIGHT)
                    print(f'[{now_str()}] Enabled IR_RIGHT: SDK default')
                else:
                    enable_configured_video_stream(
                        sdk,
                        cfg,
                        OB_STREAM_IR_RIGHT,
                        'IR_RIGHT',
                        configured_profiles.get('ir', {}),
                        [OB_FORMAT_Y16, OB_FORMAT_Y8, OB_FORMAT_GRAY],
                    )

        if save_depth and not sdk.set_align_mode_try(cfg, align_mode):
            raise RuntimeError(f'Unable to enable {name}')
        if save_depth and bool(align_cfg.get('depth_scale_after_align', True)):
            sdk.set_depth_scale_after_align(cfg, True)
        sdk.set_aggregate_all_type(cfg)
        return cfg
    except Exception:
        sdk.delete_config(cfg)
        raise


def start_rgbd_pipeline(sdk: SDK, dev, device_name: str = '', settings: Optional[dict[str, Any]] = None):
    settings = settings or {}
    stream_cfg = settings.get('stream_profile', {}) or {}
    streams_cfg = settings.get('streams', {}) or {}
    stream_profile = None
    stereo_ir_mode = bool(streams_cfg.get('ir_left', False) and streams_cfg.get('ir_right', False) and not streams_cfg.get('depth', True))
    dual_rgb_mode = bool(streams_cfg.get('color_left', False) and streams_cfg.get('color_right', False) and not streams_cfg.get('depth', True))
    fixed_required = (
        bool(stream_cfg.get('enabled', True))
        and bool(stream_cfg.get('use_fixed_profile_for_305_335l', True))
        and fixed_1280x800_30_model(device_name)
        and not stereo_ir_mode
        and not dual_rgb_mode
    )
    if dual_rgb_mode:
        probe_pipe = 0
        try:
            probe_pipe = sdk.create_pipeline(dev)
            stream_profile = select_dual_color_profile(sdk, probe_pipe, device_name, stream_cfg)
        except Exception as ex:
            print(f'[{now_str()}] WARN {device_name}: failed to query dual RGB profiles: {ex}')
            stream_profile = None
        finally:
            if probe_pipe:
                sdk.delete_pipeline(probe_pipe)
        if stream_profile is None:
            raise RuntimeError(
                f'{device_name} still does not expose usable COLOR_LEFT/COLOR_RIGHT profiles after selecting '
                f'the "{DUAL_COLOR_PRESET_NAME}" device preset. Please check SDK/firmware/Orbbec Viewer support.'
            )
    if stereo_ir_mode:
        probe_pipe = 0
        try:
            probe_pipe = sdk.create_pipeline(dev)
            stream_profile = select_ir_stereo_profile(sdk, probe_pipe, device_name, stream_cfg)
        except Exception as ex:
            print(f'[{now_str()}] WARN {device_name}: failed to query stereo IR profiles: {ex}')
            stream_profile = None
        finally:
            if probe_pipe:
                sdk.delete_pipeline(probe_pipe)
        if stream_profile is None and not bool(stream_cfg.get('fallback_to_sdk_default', True)):
            raise RuntimeError(f'{device_name} does not provide usable stereo IR profile in this SDK/device mode.')
    if fixed_required:
        probe_pipe = 0
        try:
            probe_pipe = sdk.create_pipeline(dev)
            stream_profile = select_configured_stream_profile(sdk, probe_pipe, device_name, stream_cfg)
        except Exception as ex:
            print(f'[{now_str()}] WARN {device_name}: failed to query SDK stream profiles: {ex}')
            stream_profile = None
        finally:
            if probe_pipe:
                sdk.delete_pipeline(probe_pipe)
        if stream_profile is None:
            if bool(stream_cfg.get('fallback_to_sdk_default', False)):
                print(f'[{now_str()}] WARN fallback_to_sdk_default=True; using SDK default streams.')
            else:
                raise RuntimeError(f'{device_name} does not provide requested RGB-D profile in this SDK/device mode.')
    attempts: list[Optional[dict]] = [stream_profile] if stream_profile else [None]
    align_modes = align_modes_from_settings(settings)

    for stream_profile in attempts:
        for mode, name in align_modes:
            pipe = 0
            cfg = 0
            try:
                pipe = sdk.create_pipeline(dev)
                cfg = build_capture_config(sdk, mode, name, stream_profile, settings)
                sdk.start_pipeline(pipe, cfg)
                if stream_profile:
                    print(f'[{now_str()}] Started with profile {stream_profile["label"]}')
                return pipe, cfg, name
            except Exception as ex:
                profile_name = stream_profile['label'] if stream_profile else 'SDK default'
                print(f'[{now_str()}] WARN start with {name}, profile={profile_name} failed: {ex}')
                if pipe:
                    try:
                        sdk.stop_pipeline(pipe)
                    except Exception:
                        pass
                    sdk.delete_pipeline(pipe)
                if cfg:
                    sdk.delete_config(cfg)
    raise RuntimeError('Unable to start requested video pipeline.')


def parse_args():
    p = argparse.ArgumentParser(description='Orbbec strict RGB-D collector')
    p.add_argument('--config', default=str(DEFAULT_CONFIG_PATH), help='YAML/Python config file, default: config.yaml')
    p.add_argument('--sdk-bin', default=r'D:\OrbbecSDK_v2\bin', help='Folder containing OrbbecSDK.dll')
    p.add_argument('--output-root', default=r'D:\OrbbecLiveCollector\captures', help='Root output folder')
    p.add_argument('--tag', default='', help='Optional tag recorded in pose_note.txt and camera_info.yaml')
    p.add_argument('--width', type=int, default=0, help='Output width for rgb/depth PNG, 0 means keep native RGB width')
    p.add_argument('--height', type=int, default=0, help='Output height for rgb/depth PNG, 0 means keep native RGB height')
    p.add_argument('--max-sync-diff-ms', type=float, default=15.0, help='Max allowed |rgb_ts-depth_ts| in ms')
    p.add_argument('--device-index', type=int, default=None, help='Orbbec device index to use when multiple cameras are connected')
    p.add_argument('--serial', default='', help='Orbbec camera serial number to use when multiple cameras are connected')
    p.add_argument('--model-hint', default='', help='Select first device whose model name contains this text, e.g. 335L or 305')
    p.add_argument('--preset', default='', help='Override device preset name, e.g. Default or Dual Color Streams')
    p.add_argument('--auto-start', action='store_true', help='Start saving automatically after camera pipeline is ready')
    p.add_argument('--auto-start-at', type=float, default=0.0, help='Unix timestamp seconds for scheduled auto start')
    p.add_argument('--stop-file', default='', help='Stop saving when this file exists, used by multi-camera controller')
    return p.parse_args()


def normalize_device_match_text(text: str) -> str:
    return ''.join(ch for ch in str(text or '').lower() if ch.isalnum())


def select_device(sdk: SDK, dl, serial: str, device_index: Optional[int], model_hint: str = ''):
    count = sdk.device_count(dl)
    if count < 1:
        raise RuntimeError('No Orbbec device found.')
    if device_index is not None and (device_index < 0 or device_index >= count):
        raise RuntimeError(f'--device-index {device_index} is out of range, found {count} device(s).')

    serial_filter = serial.strip().lower()
    if serial_filter in ('auto', 'none'):
        serial_filter = ''
    model_filter = normalize_device_match_text(model_hint)
    print(f'[{now_str()}] Found {count} Orbbec device(s).')
    selected_dev = 0
    selected_sn = ''
    selected_name = ''
    selected_by_model = False
    for idx in range(count):
        dev = 0
        try:
            dev = sdk.get_device(dl, idx)
            sn, name = sdk.get_device_info(dev)
            print(f'[{now_str()}] Device[{idx}]: {name}, SN: {sn}')
            if serial_filter == 'any':
                selected = idx == 0
            elif serial_filter:
                selected = sn.strip().lower() == serial_filter
            elif device_index is not None:
                selected = idx == device_index
            elif model_filter:
                selected = model_filter in normalize_device_match_text(name)
                selected_by_model = selected
            else:
                selected = idx == 0
            if selected:
                selected_dev = dev
                selected_sn = sn
                selected_name = name
                dev = 0
                break
        finally:
            if dev:
                sdk.delete_device(dev)

    if not selected_dev:
        if serial_filter and serial_filter != 'any':
            selector = f'SN {serial}'
        elif device_index is not None:
            selector = f'index {device_index}'
        elif model_filter:
            selector = f'model hint {model_hint}'
        else:
            selector = 'first device'
        raise RuntimeError(f'No Orbbec device matched {selector}.')
    if selected_by_model:
        print(f'[{now_str()}] Auto-selected by model hint "{model_hint}". SN: {selected_sn}')
    return selected_dev, selected_sn, selected_name


def main() -> int:
    global PNG_COMPRESSION
    args = parse_args()
    config_path = Path(args.config).expanduser()
    settings = load_capture_config(config_path)
    apply_config_defaults_to_args(args, settings)
    if str(args.preset or '').strip():
        preset_cfg = settings.setdefault('device_preset', {})
        preset_cfg['enabled'] = True
        preset_cfg['name'] = str(args.preset).strip()
        preset_cfg['required'] = True
    validate_capture_settings(settings)
    PNG_COMPRESSION = int((settings.get('output', {}) or {}).get('png_compression', PNG_COMPRESSION))
    print(f'[{now_str()}] Capture config: {config_path}')
    print(f'[{now_str()}] PNG compression: {PNG_COMPRESSION}')

    sdk = SDK(Path(args.sdk_bin))
    ctx = dl = dev = pipe = cfg = 0
    writer = None
    property_report: list[dict[str, Any]] = []
    device_preset_report: dict[str, Any] = {}
    device_info_detail: dict[str, str] = {}
    capturing = False
    current_session = ''
    last_color = None
    last_color_left = None
    last_color_right = None
    last_depth_vis = None
    last_ir_left_vis = None
    last_ir_right_vis = None
    fps = 0.0
    fps_window_start = time.perf_counter()
    fps_window_frames = 0
    frame_timeout_ms = int((settings.get('pipeline', {}) or {}).get('frame_timeout_ms', 200) or 200)
    pointcloud_cfg = settings.get('pointcloud', {}) or {}
    require_aligned_depth_to_color = bool(pointcloud_cfg.get('require_aligned_depth_to_color', True))
    streams_cfg = settings.get('streams', {}) or {}
    output_cfg = settings.get('output', {}) or {}
    save_color = bool(streams_cfg.get('color', True))
    save_color_left = bool(streams_cfg.get('color_left', False))
    save_color_right = bool(streams_cfg.get('color_right', False))
    save_depth = bool(streams_cfg.get('depth', True))
    save_depth_vis = bool(output_cfg.get('save_depth_vis', True))
    save_ir_left = bool(streams_cfg.get('ir_left', False))
    save_ir_right = bool(streams_cfg.get('ir_right', False))
    minimal_dual_rgb_layout = bool(
        output_cfg.get('minimal_dual_rgb_layout', False)
        and save_color_left
        and save_color_right
        and not save_depth
    )

    try:
        sdk_version = sdk.get_sdk_version_text()
        print(f'[{now_str()}] Orbbec SDK version: {sdk_version}')
        ctx = sdk.create_context()
        dl = sdk.query_device_list(ctx)
        dev, sn, dev_name = select_device(sdk, dl, args.serial, args.device_index, args.model_hint)
        print(f'[{now_str()}] Device: {dev_name}, SN: {sn}')
        try:
            device_info_detail = sdk.get_device_info_detail(dev)
            print(f'[{now_str()}] Firmware version: {device_info_detail.get("firmware_version", "") or "unknown"}')
            print(f'[{now_str()}] Hardware version: {device_info_detail.get("hardware_version", "") or "unknown"}')
            print(f'[{now_str()}] Supported min SDK: {device_info_detail.get("supported_min_sdk_version", "") or "unknown"}')
        except Exception as ex:
            print(f'[{now_str()}] WARN device info detail failed: {ex}')
            device_info_detail = {}
        try:
            print(f'[{now_str()}] Current device preset: {sdk.get_current_preset_name(dev) or "unknown"}')
            presets = sdk.get_available_presets(dev)
            print(f'[{now_str()}] Available device presets: {", ".join(presets) if presets else "none"}')
        except Exception as ex:
            print(f'[{now_str()}] WARN query device preset failed: {ex}')

        device_preset_report = switch_device_preset_if_configured(sdk, dev, dev_name, settings)
        print_all_stream_profiles(sdk, dev, dev_name, 'available stream profiles after preset selection')
        property_report = apply_camera_properties(sdk, dev, settings)

        pipe, cfg, align_mode_name = start_rgbd_pipeline(sdk, dev, dev_name, settings)
        try:
            cam_params = camera_param_to_dict(sdk.get_camera_param(pipe), align_mode_name)
        except Exception as ex:
            print(f'[{now_str()}] WARN get camera params failed: {ex}')
            cam_params = {'align_mode': align_mode_name, 'camera_param_error': str(ex)}
        cam_params['device_name'] = dev_name
        cam_params['serial_number'] = sn
        cam_params['device_info_detail'] = device_info_detail
        cam_params['sdk_version'] = sdk_version
        cam_params['capture_config_path'] = str(config_path)
        cam_params['device_preset_report'] = device_preset_report
        cam_params['camera_property_apply_report'] = property_report
        cam_params['resolved_capture_config'] = settings
        cam_params = apply_intrinsics_reference_fallback(
            cam_params,
            settings,
            bool(save_color_left and save_color_right and not save_depth),
        )
        if not minimal_dual_rgb_layout:
            write_latest_camera_intrinsics(Path(args.output_root), cam_params)
        print(f'[{now_str()}] Started. Align mode: {align_mode_name}')
        print('Keys in preview window: S=start | E=end | SPACE=toggle | Q/ESC=quit')
        print('Mouse in preview window: click START/STOP/QUIT buttons')

        writer = SessionWriter(
            Path(args.output_root),
            sn,
            args.tag,
            target_width=args.width,
            target_height=args.height,
            align_mode_name=align_mode_name,
            require_aligned_depth_to_color=require_aligned_depth_to_color,
            save_color=save_color,
            save_color_left=save_color_left,
            save_color_right=save_color_right,
            save_depth=save_depth,
            save_depth_vis=save_depth_vis,
            save_ir_left=save_ir_left,
            save_ir_right=save_ir_right,
            minimal_dual_rgb_layout=minimal_dual_rgb_layout,
            writer_thread_count=int(output_cfg.get('writer_threads', 1) or 1),
            write_queue_maxsize=int(output_cfg.get('write_queue_maxsize', 256) or 256),
            color_format=str(output_cfg.get('color_format', 'png') or 'png'),
            depth_raw_format=str(output_cfg.get('depth_raw_format', 'png') or 'png'),
        )

        window_name = 'Orbbec Live Capture'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        preview_cfg = settings.get('preview', {}) or {}
        cv2.resizeWindow(
            window_name,
            int(preview_cfg.get('window_width', 1400) or 1400),
            int(preview_cfg.get('window_height', 800) or 800),
        )
        cv2.moveWindow(window_name, 80, 80)
        ui_state = {'buttons': {}, 'action': None}
        cv2.setMouseCallback(window_name, on_preview_mouse, ui_state)
        preview_every_n = max(1, int(preview_cfg.get('preview_every_n_frames', 1) or 1))
        preview_depth_every_n = max(1, int(preview_cfg.get('depth_vis_every_n_frames', 1) or 1))
        preview_frame_index = 0

        def start_capture():
            nonlocal capturing, current_session
            if capturing:
                return
            session_dir = writer.start(cam_params)
            if not writer.minimal_dual_rgb_layout:
                write_capture_config_snapshot(session_dir, settings, config_path)
            current_session = str(session_dir)
            capturing = True
            print(f'[{now_str()}] Session started: {session_dir}')

        def stop_capture():
            nonlocal capturing
            if not capturing:
                return
            writer.stop()
            capturing = False
            print(f'[{now_str()}] Session ended: {writer.session_dir}')
            print(f'[{now_str()}] Save summary: {writer.skip_summary()}')
            if writer.pair_index == 0:
                detail = f' ({writer.last_skip_detail})' if writer.last_skip_detail else ''
                print(f'[{now_str()}] WARN no RGB-D pairs saved. Last skip: {writer.last_skip_reason or "none"}{detail}')

        if args.auto_start:
            if args.auto_start_at > 0:
                wait_s = max(0.0, float(args.auto_start_at) - time.time())
                print(f'[{now_str()}] Auto-start scheduled at {args.auto_start_at:.3f}, wait {wait_s:.3f}s')
                while wait_s > 0:
                    time.sleep(min(0.05, wait_s))
                    wait_s = max(0.0, float(args.auto_start_at) - time.time())
            try:
                start_capture()
            except Exception as ex:
                print(f'[{now_str()}] WARN auto-start failed: {ex}')

        while True:
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    if capturing and writer and writer.active:
                        stop_capture()
                    break
            except Exception:
                pass

            fs = sdk.wait_frameset(pipe, frame_timeout_ms)
            color_fd = None
            color_left_fd = None
            color_right_fd = None
            depth_fd = None
            ir_left_fd = None
            ir_right_fd = None
            if fs:
                ptrs = []
                try:
                    f_color = sdk.get_optional_frame(fs, OB_FRAME_COLOR) if save_color else None
                    f_color_left = sdk.get_optional_frame(fs, OB_FRAME_COLOR_LEFT) if save_color_left else None
                    f_color_right = sdk.get_optional_frame(fs, OB_FRAME_COLOR_RIGHT) if save_color_right else None
                    f_depth = sdk.get_optional_frame(fs, OB_FRAME_DEPTH) if save_depth else None
                    f_ir_left = sdk.get_optional_frame(fs, OB_FRAME_IR_LEFT) if save_ir_left else None
                    f_ir_right = sdk.get_optional_frame(fs, OB_FRAME_IR_RIGHT) if save_ir_right else None
                    if f_color:
                        ptrs.append(f_color)
                        color_fd = sdk.extract(f_color)
                    if f_color_left:
                        ptrs.append(f_color_left)
                        color_left_fd = sdk.extract(f_color_left)
                    if f_color_right:
                        ptrs.append(f_color_right)
                        color_right_fd = sdk.extract(f_color_right)
                    if f_depth:
                        ptrs.append(f_depth)
                        depth_fd = sdk.extract(f_depth)
                    if f_ir_left:
                        ptrs.append(f_ir_left)
                        ir_left_fd = sdk.extract(f_ir_left)
                    if f_ir_right:
                        ptrs.append(f_ir_right)
                        ir_right_fd = sdk.extract(f_ir_right)
                finally:
                    for fp in ptrs:
                        sdk.delete_frame(fp)
                    sdk.delete_frame(fs)

            has_any_frame = (
                color_fd is not None or color_left_fd is not None or color_right_fd is not None
                or depth_fd is not None or ir_left_fd is not None or ir_right_fd is not None
            )
            if has_any_frame:
                preview_frame_index += 1
                fps_window_frames += 1
                now_perf = time.perf_counter()
                elapsed = now_perf - fps_window_start
                if elapsed >= 1.0:
                    fps = fps_window_frames / elapsed
                    fps_window_frames = 0
                    fps_window_start = now_perf

            color_img = decode_color(color_fd) if color_fd else None
            color_left_img = decode_color(color_left_fd) if color_left_fd else None
            color_right_img = decode_color(color_right_fd) if color_right_fd else None
            depth_raw = decode_depth(depth_fd) if depth_fd else None
            ir_left_img = decode_ir(ir_left_fd) if ir_left_fd else None
            ir_right_img = decode_ir(ir_right_fd) if ir_right_fd else None
            if color_img is not None:
                last_color = color_img
            if color_left_img is not None:
                last_color_left = color_left_img
            if color_right_img is not None:
                last_color_right = color_right_img
            if depth_raw is not None and preview_frame_index % preview_depth_every_n == 0:
                try:
                    last_depth_vis = depth_to_vis(depth_raw)
                except Exception as ex:
                    print(f'[{now_str()}] WARN depth preview failed: {ex}')
            if ir_left_img is not None:
                last_ir_left_vis = ir_to_vis(ir_left_img)
            if ir_right_img is not None:
                last_ir_right_vis = ir_to_vis(ir_right_img)

            if capturing and writer and writer.active:
                if save_color_left and save_color_right and not save_depth:
                    if color_left_fd is None or color_right_fd is None:
                        writer.mark_skip('missing', f'color_left={"yes" if color_left_fd is not None else "no"}, color_right={"yes" if color_right_fd is not None else "no"}')
                    elif color_left_img is None or color_right_img is None:
                        writer.mark_skip('decode', f'color_left={"yes" if color_left_img is not None else "no"}, color_right={"yes" if color_right_img is not None else "no"}')
                    elif abs(int(color_left_fd.dev_ts) - int(color_right_fd.dev_ts)) > int(args.max_sync_diff_ms * 1000.0):
                        diff_ms = abs(int(color_left_fd.dev_ts) - int(color_right_fd.dev_ts)) / 1000.0
                        writer.mark_skip('sync', f'color_left-color_right diff_ms={diff_ms:.3f}, max_ms={args.max_sync_diff_ms:.3f}')
                    else:
                        writer.save_dual_color_pair(color_left_fd, color_left_img, color_right_fd, color_right_img)
                elif save_ir_left and save_ir_right and not save_depth:
                    if ir_left_fd is None or ir_right_fd is None:
                        writer.mark_skip('missing', f'ir_left={"yes" if ir_left_fd is not None else "no"}, ir_right={"yes" if ir_right_fd is not None else "no"}')
                    elif ir_left_img is None or ir_right_img is None:
                        writer.mark_skip('decode', f'ir_left={"yes" if ir_left_img is not None else "no"}, ir_right={"yes" if ir_right_img is not None else "no"}')
                    elif abs(int(ir_left_fd.dev_ts) - int(ir_right_fd.dev_ts)) > int(args.max_sync_diff_ms * 1000.0):
                        diff_ms = abs(int(ir_left_fd.dev_ts) - int(ir_right_fd.dev_ts)) / 1000.0
                        writer.mark_skip('sync', f'ir_left-ir_right diff_ms={diff_ms:.3f}, max_ms={args.max_sync_diff_ms:.3f}')
                    else:
                        writer.save_stereo_pair(ir_left_fd, ir_left_img, ir_right_fd, ir_right_img)
                elif color_fd is None or depth_fd is None:
                    writer.mark_skip('missing', f'color={"yes" if color_fd is not None else "no"}, depth={"yes" if depth_fd is not None else "no"}')
                elif depth_fd.fmt != OB_FORMAT_Y16:
                    writer.mark_skip('format', f'depth_fmt={depth_fd.fmt}, expected={OB_FORMAT_Y16}')
                elif color_img is None or depth_raw is None:
                    writer.mark_skip('decode', f'color_img={"yes" if color_img is not None else "no"}, depth_raw={"yes" if depth_raw is not None else "no"}')
                elif require_aligned_depth_to_color and (
                    depth_raw.shape[1] != color_img.shape[1] or depth_raw.shape[0] != color_img.shape[0]
                ):
                    writer.mark_skip(
                        'resolution',
                        f'pointcloud requires aligned depth, color={color_img.shape[1]}x{color_img.shape[0]}, '
                        f'depth={depth_raw.shape[1]}x{depth_raw.shape[0]}',
                    )
                elif abs(int(color_fd.dev_ts) - int(depth_fd.dev_ts)) > int(args.max_sync_diff_ms * 1000.0):
                    diff_ms = abs(int(color_fd.dev_ts) - int(depth_fd.dev_ts)) / 1000.0
                    writer.mark_skip('sync', f'diff_ms={diff_ms:.3f}, max_ms={args.max_sync_diff_ms:.3f}')
                else:
                    writer.save_pair(color_fd, color_img, depth_fd, depth_raw)

            if preview_frame_index % preview_every_n == 0:
                preview = make_preview(last_color, last_depth_vis, last_ir_left_vis, last_ir_right_vis, last_color_left, last_color_right)
                dual_rgb_on = bool(save_color_left and save_color_right and not save_depth)
                rgbd_on = bool(save_color and save_depth)
                pair_count = writer.pair_index if writer and writer.active else 0
                skip_text = '--'
                if writer and writer.active:
                    skip_text = (
                        f'M{writer.skipped_missing} F{writer.skipped_format} '
                        f'R{writer.skipped_resolution} D{writer.skipped_decode} S{writer.skipped_sync}'
                    )
                depth_scale_text = '--'
                if rgbd_on and depth_fd is not None and depth_fd.depth_scale is not None:
                    depth_scale_text = f'{float(depth_fd.depth_scale):.6f}'
                session_name = Path(current_session).name if current_session else '--'
                sections = [
                    (
                        'Device',
                        [
                            f'{dev_name} | SN: {sn}',
                        ],
                    ),
                    (
                        'Dual RGB',
                        [
                            f'Status: {"ON" if dual_rgb_on else "OFF"}',
                            f'RGB_L: {frame_res_text(color_left_fd) if dual_rgb_on else "--"}',
                            f'RGB_R: {frame_res_text(color_right_fd) if dual_rgb_on else "--"}',
                            f'FPS: {fps:.1f}' if dual_rgb_on else 'FPS: --',
                            f'Pair count: {pair_count}' if dual_rgb_on else 'Pair count: --',
                            f'Skips: {skip_text}' if dual_rgb_on else 'Skips: --',
                        ],
                    ),
                    (
                        'RGB-D',
                        [
                            f'Status: {"ON" if rgbd_on else "OFF"}',
                            f'Color: {frame_res_text(color_fd) if rgbd_on else "--"}',
                            f'Depth: {frame_res_text(depth_fd) if rgbd_on else "--"}',
                            f'Align: {align_mode_name if rgbd_on else "--"}',
                            f'Depth scale: {depth_scale_text}',
                        ],
                    ),
                    (
                        'Capture',
                        [
                            f'Status: {"RUNNING" if capturing else "IDLE"}',
                            f'Preview: 1/{preview_every_n} frames',
                            f'Session: {session_name}',
                        ],
                    ),
                ]

                preview = draw_status_sections(preview, sections)
                preview, buttons = draw_controls(preview, capturing)
                ui_state['buttons'] = buttons
                cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            click_action = ui_state.get('action')
            ui_state['action'] = None
            external_stop_req = bool(args.stop_file) and Path(args.stop_file).exists()

            quit_req = (key in (ord('q'), ord('Q'), 27)) or click_action == 'quit'
            start_req = (key in (ord('s'), ord('S'))) or (click_action == 'toggle' and not capturing)
            stop_req = external_stop_req or (key in (ord('e'), ord('E'))) or (click_action == 'toggle' and capturing)
            toggle_req = key in (32,)
            if toggle_req:
                if capturing:
                    stop_req = True
                else:
                    start_req = True
            if quit_req:
                if capturing and writer and writer.active:
                    stop_capture()
                break
            if start_req and not capturing:
                try:
                    start_capture()
                except Exception as ex:
                    print(f'[{now_str()}] WARN start failed: {ex}')
            if stop_req and capturing:
                try:
                    stop_capture()
                except Exception as ex:
                    print(f'[{now_str()}] WARN stop failed: {ex}')

        cv2.destroyAllWindows()
        print(f'[{now_str()}] Exit.')
        return 0
    except Exception as ex:
        print(f'[ERROR] {ex}')
        return 1
    finally:
        if writer and writer.active:
            try:
                writer.stop()
            except Exception:
                pass
        if pipe:
            try:
                sdk.stop_pipeline(pipe)
            except Exception:
                pass
        if cfg:
            sdk.delete_config(cfg)
        if pipe:
            sdk.delete_pipeline(pipe)
        if dev:
            sdk.delete_device(dev)
        if dl:
            sdk.delete_device_list(dl)
        if ctx:
            sdk.delete_context(ctx)


if __name__ == '__main__':
    sys.exit(main())






