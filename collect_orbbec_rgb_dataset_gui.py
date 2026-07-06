#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Portable Tkinter launcher for the Orbbec capture scripts in this repository.

Run it from the desired Python/conda environment. The launched collector
inherits that environment, so Orbbec SDK/OpenCV packages stay consistent.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
CONFIG_DIR = ROOT / "config"
SETTINGS_FILE = CONFIG_DIR / "orbbec_rgb_dataset_gui_settings.json"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SCRIPTS = {
    "rgb_dataset": SCRIPTS_DIR / "collect_orbbec_rgb_dataset.py",
    "rgb_interval_305": SCRIPTS_DIR / "capture_305_rgb_interval.py",
    "rgbd_305": SCRIPTS_DIR / "capture_305_rgbd.py",
    "rgbd_config": SCRIPTS_DIR / "orbbec_live_capture.py",
    "merged_dual": SCRIPTS_DIR / "merged_dual_camera_capture.py",
}

MODE_LABELS = {
    "rgb_dataset": "YOLO RGB 数据集采集（335L/305）",
    "rgb_interval_305": "305 单 RGB 间隔采集",
    "rgbd_305": "305 RGB-D 采集",
    "rgbd_335l": "335L RGB-D 采集",
    "dual_rgb_305": "305 双 RGB 采集",
    "merged_dual": "335L + 305 联合采集（单窗口）",
    "merged_rgbd": "335L + 305 RGB-D 联合采集",
}

MODE_DESCRIPTIONS = {
    "rgb_dataset": "只保存 RGB 图片和 metadata.csv，可选 335L/coarse 或 305/precise。",
    "rgb_interval_305": "旧版 305 单路 RGB 单张/间隔保存工具。",
    "rgbd_305": "打开 Gemini 305 普通 RGB-D 预览窗口，按空格/S/E 控制保存。",
    "rgbd_335l": "使用 config/config.yaml 启动普通 RGB-D 采集，保存 color/depth 等配置内启用的数据。",
    "dual_rgb_305": "使用 config/config_dual_rgb.yaml 切到 Dual Color Streams，保存 305 左右双 RGB。",
    "merged_dual": "推荐日常使用：一个窗口同时预览两台相机，按空格/S/E 控制一起保存。",
    "merged_rgbd": "335L 和 305 同时启动 RGB-D，默认按 1280x800@30 请求 color/depth。",
}

MODE_FIELDS = {
    "rgb_dataset": [
        "camera",
        "task",
        "device_preset",
        "width",
        "height",
        "fps",
        "preview_fps",
        "auto_interval",
        "session",
        "output_root",
        "formats",
        "png_compression",
        "start_auto",
        "no_preview",
    ],
    "rgb_interval_305": [
        "device_preset",
        "width",
        "height",
        "fps",
        "preview_fps",
        "save_every_seconds",
        "save_every_frames",
        "max_saves",
        "output_root",
        "formats",
        "start_auto",
        "no_preview",
    ],
    "rgbd_305": [
        "output_root",
        "tag",
        "device_preset",
        "preview_fps",
    ],
    "rgbd_335l": [
        "output_root",
        "tag",
        "device_preset",
        "preview_fps",
    ],
    "dual_rgb_305": [
        "output_root",
        "tag",
        "device_preset",
        "preview_fps",
    ],
    "merged_dual": [
        "preview_fps",
    ],
    "merged_rgbd": [
        "width",
        "height",
        "fps",
        "preview_fps",
        "device_preset",
        "sdk_bin",
        "output_root",
    ],
}

CAMERA_TASKS = {"335L": "coarse", "305": "precise"}
KNOWN_CAMERA_SERIALS = {"CV2L36000024", "CP28563000N0", "CP2N1630005C"}
DUAL_COLOR_PRESET = "Dual Color Streams"
STANDARD_CONFIG_NAMES = {"config.yaml", "config_305_rgbd.yaml", "config_dual_rgb.yaml"}


def is_standard_config_path(value: object) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return True
    path = Path(raw)
    if path.name not in STANDARD_CONFIG_NAMES:
        return False
    parent = path.parent if str(path.parent) != "." else ROOT
    try:
        parent = parent.resolve()
    except OSError:
        pass
    return parent in {ROOT.resolve(), CONFIG_DIR.resolve()}


def normalize_standard_config_path(value: object) -> str:
    raw = str(value or "").strip()
    if is_standard_config_path(raw) and raw:
        return str(CONFIG_DIR / Path(raw).name)
    return raw

PRESET_FALLBACKS = {
    "305": [
        "Default",
        "High Accuracy",
        "Close Range High Accuracy",
        "Close Range Default",
        "Factory Calib",
        DUAL_COLOR_PRESET,
        "Custom",
    ],
    "335L": [
        "Default",
        "Hand",
        "High Accuracy",
        "High Density",
        "Medium Density",
        "Factory Calib",
        "Custom",
    ],
}
MODE_CAMERA_HINTS = {
    "rgb_interval_305": "305",
    "rgbd_305": "305",
    "rgbd_335l": "335L",
    "dual_rgb_305": "305",
    "merged_rgbd": "305",
}
MODE_DEFAULT_PRESETS = {
    "rgb_dataset": "Default",
    "rgb_interval_305": "Default",
    "rgbd_305": "Default",
    "rgbd_335l": "Default",
    "dual_rgb_305": DUAL_COLOR_PRESET,
    "merged_rgbd": "Default",
}

DEFAULTS = {
    "mode": "rgb_dataset",
    "camera": "335L",
    "task": "coarse",
    "width": "1280",
    "height": "800",
    "fps": "30",
    "preview_fps": "",
    "auto_interval": "1.0",
    "save_every_seconds": "1.0",
    "save_every_frames": "0",
    "max_saves": "0",
    "session": "",
    "tag": "",
    "device_preset": "Default",
    "serial": "",
    "device_index": "",
    "sdk_bin": r"D:\OrbbecSDK_v2\bin",
    "output_root": str(ROOT / "captures" / "rgb_dataset"),
    "config_path": str(CONFIG_DIR / "config.yaml"),
    "formats": "BGR RGB MJPG YUYV BGRA RGBA UYVY",
    "png_compression": "3",
    "start_auto": False,
    "no_preview": False,
}


class LauncherApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Orbbec Capture Launcher")
        self.root.minsize(900, 720)
        self.process: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.field_rows: dict[str, ttk.Frame] = {}
        self.detected_devices: list[dict[str, object]] = []
        self.preset_combo: ttk.Combobox | None = None
        self.mode_by_label = {label: key for key, label in MODE_LABELS.items()}

        data = self.load_settings()
        self.vars: dict[str, StringVar | BooleanVar] = {
            "mode": StringVar(value=data.get("mode", DEFAULTS["mode"])),
            "camera": StringVar(value=data.get("camera", DEFAULTS["camera"])),
            "task": StringVar(value=data.get("task", DEFAULTS["task"])),
            "width": StringVar(value=data.get("width", DEFAULTS["width"])),
            "height": StringVar(value=data.get("height", DEFAULTS["height"])),
            "fps": StringVar(value=data.get("fps", DEFAULTS["fps"])),
            "preview_fps": StringVar(value=data.get("preview_fps", DEFAULTS["preview_fps"])),
            "auto_interval": StringVar(value=data.get("auto_interval", DEFAULTS["auto_interval"])),
            "save_every_seconds": StringVar(value=data.get("save_every_seconds", DEFAULTS["save_every_seconds"])),
            "save_every_frames": StringVar(value=data.get("save_every_frames", DEFAULTS["save_every_frames"])),
            "max_saves": StringVar(value=data.get("max_saves", DEFAULTS["max_saves"])),
            "session": StringVar(value=data.get("session", DEFAULTS["session"])),
            "tag": StringVar(value=data.get("tag", DEFAULTS["tag"])),
            "device_preset": StringVar(value=data.get("device_preset", DEFAULTS["device_preset"])),
            "serial": StringVar(value=data.get("serial", DEFAULTS["serial"])),
            "device_index": StringVar(value=data.get("device_index", DEFAULTS["device_index"])),
            "sdk_bin": StringVar(value=data.get("sdk_bin", DEFAULTS["sdk_bin"])),
            "output_root": StringVar(value=data.get("output_root", DEFAULTS["output_root"])),
            "config_path": StringVar(value=data.get("config_path", DEFAULTS["config_path"])),
            "formats": StringVar(value=data.get("formats", DEFAULTS["formats"])),
            "png_compression": StringVar(value=data.get("png_compression", DEFAULTS["png_compression"])),
            "start_auto": BooleanVar(value=bool(data.get("start_auto", DEFAULTS["start_auto"]))),
            "no_preview": BooleanVar(value=bool(data.get("no_preview", DEFAULTS["no_preview"]))),
        }

        if self.vars["mode"].get() not in MODE_LABELS:
            self.vars["mode"].set("rgb_dataset")

        self.mode_label = StringVar(value=MODE_LABELS[self.vars["mode"].get()])
        self.mode_description = StringVar(value="")
        self.device_list_text = StringVar(value="设备列表未读取")
        self.command_preview = StringVar(value="")
        self.status = StringVar(value=f"Ready | Python: {sys.executable}")

        self.build_ui()
        self.on_mode_changed()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.flush_log_queue)
        self.root.after(300, self.refresh_devices)

    def load_settings(self) -> dict:
        if not SETTINGS_FILE.exists():
            return dict(DEFAULTS)
        try:
            with SETTINGS_FILE.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            data = {**DEFAULTS, **loaded}
            data["config_path"] = normalize_standard_config_path(data.get("config_path"))
            return data
        except Exception:
            return dict(DEFAULTS)

    def save_settings(self) -> None:
        data = {}
        for key, var in self.vars.items():
            data[key] = bool(var.get()) if isinstance(var, BooleanVar) else str(var.get())
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        title = ttk.Label(outer, text="Orbbec Capture Launcher", font=("Segoe UI", 15, "bold"))
        title.pack(anchor="w")

        mode_box = ttk.LabelFrame(outer, text="采集类型", padding=12)
        mode_box.pack(fill="x", pady=(12, 10))
        mode_box.columnconfigure(1, weight=1)
        ttk.Label(mode_box, text="类型").grid(row=0, column=0, sticky="w", padx=(0, 8))
        mode_combo = ttk.Combobox(
            mode_box,
            textvariable=self.mode_label,
            values=list(MODE_LABELS.values()),
            state="readonly",
            width=32,
        )
        mode_combo.grid(row=0, column=1, sticky="ew")
        mode_combo.bind("<<ComboboxSelected>>", lambda _event: self.on_mode_label_changed())
        ttk.Label(mode_box, textvariable=self.mode_description, foreground="#555555").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        device_box = ttk.LabelFrame(outer, text="已连接设备", padding=12)
        device_box.pack(fill="x", pady=(0, 10))
        device_box.columnconfigure(0, weight=1)
        ttk.Label(device_box, textvariable=self.device_list_text, justify="left").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(device_box, text="刷新设备", command=self.refresh_devices).grid(
            row=0, column=1, sticky="e", padx=(12, 0)
        )
        ttk.Label(
            device_box,
            text="序列号留空时按型号自动选择；需要固定某一台时再填写 SN 或设备 Index。",
            foreground="#666666",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.param_box = ttk.LabelFrame(outer, text="参数", padding=12)
        self.param_box.pack(fill="x", pady=(0, 10))
        self.param_box.columnconfigure(0, weight=1)

        self.add_combo_row("camera", "相机", ["335L", "305"], self.on_camera_changed, "数据集采集模式使用")
        self.add_combo_row("task", "任务", ["coarse", "precise"], None, "335L=coarse，305=precise")
        self.add_entry_row("width", "宽度")
        self.add_entry_row("height", "高度")
        self.add_entry_row("fps", "FPS")
        self.add_entry_row("preview_fps", "预览帧率", "留空=脚本默认；只限制预览窗口")
        self.add_entry_row("auto_interval", "自动保存间隔(s)")
        self.add_entry_row("save_every_seconds", "间隔保存秒数")
        self.add_entry_row("save_every_frames", "间隔保存帧数", "0 表示不用帧间隔")
        self.add_entry_row("max_saves", "最多保存张数", "0 表示不限")
        self.add_entry_row("session", "Session", "留空则自动使用当前时间")
        self.add_entry_row("tag", "保存文件夹名", "留空=自动使用 capture_时间；重名自动加 _01")
        self.preset_combo = self.add_combo_row(
            "device_preset",
            "设备模式",
            [],
            None,
            "启动采集前加载的 Orbbec device preset",
        )
        self.add_entry_row("serial", "序列号", "留空=按型号自动选；any=第一个设备")
        self.add_entry_row("device_index", "设备 Index", "可留空")
        self.add_path_row("sdk_bin", "SDK bin", browse_dir=True)
        self.add_path_row("output_root", "输出目录", browse_dir=True)
        self.add_path_row("config_path", "配置文件", browse_dir=False)
        self.add_entry_row("formats", "COLOR 格式优先级")
        self.add_entry_row("png_compression", "PNG 压缩", "0 最快，9 最小")
        self.add_check_row("start_auto", "自动保存模式")
        self.add_check_row("no_preview", "无预览窗口")
        self.empty_params_note = ttk.Label(
            self.param_box,
            text="该模式无需额外参数，直接启动即可。",
            foreground="#666666",
        )

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(0, 10))
        self.start_button = ttk.Button(buttons, text="启动采集", command=self.start_capture)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="停止采集", command=self.stop_capture, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="复制命令", command=self.copy_command).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="打开输出目录", command=self.open_output_root).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="用当前时间命名", command=self.reset_time_token).pack(side="right")

        command_box = ttk.LabelFrame(outer, text="将执行的命令", padding=8)
        command_box.pack(fill="x", pady=(0, 10))
        ttk.Label(command_box, textvariable=self.command_preview, wraplength=820).pack(anchor="w")

        log_box = ttk.LabelFrame(outer, text="运行日志", padding=8)
        log_box.pack(fill="both", expand=True)
        self.log = ScrolledText(log_box, height=14, wrap="word", font=("Consolas", 10))
        self.log.pack(fill="both", expand=True)

        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=(8, 0))
        for var in self.vars.values():
            var.trace_add("write", lambda *_: self.update_command_preview())

    def add_field_row(self, key: str, label: str, hint: str = "") -> ttk.Frame:
        row = ttk.Frame(self.param_box)
        row.columnconfigure(1, weight=1, minsize=320)
        ttk.Label(row, text=label, width=18).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        if hint:
            ttk.Label(row, text=hint, foreground="#666666", wraplength=360).grid(
                row=0, column=2, sticky="w", padx=(8, 0), pady=4
            )
        self.field_rows[key] = row
        return row

    def add_entry_row(self, key: str, label: str, hint: str = "") -> None:
        row = self.add_field_row(key, label, hint)
        ttk.Entry(row, textvariable=self.vars[key]).grid(row=0, column=1, sticky="ew", pady=4)

    def add_combo_row(self, key: str, label: str, values: list[str], command=None, hint: str = "") -> ttk.Combobox:
        row = self.add_field_row(key, label, hint)
        combo = ttk.Combobox(row, textvariable=self.vars[key], values=values, state="readonly")
        if command is not None:
            combo.bind("<<ComboboxSelected>>", lambda _event: command())
        combo.grid(row=0, column=1, sticky="ew", pady=4)
        return combo

    def add_check_row(self, key: str, label: str) -> None:
        row = self.add_field_row(key, "")
        ttk.Checkbutton(row, text=label, variable=self.vars[key]).grid(row=0, column=1, sticky="w", pady=4)

    def add_path_row(self, key: str, label: str, browse_dir: bool) -> None:
        row = self.add_field_row(key, label)
        frame = ttk.Frame(row)
        frame.columnconfigure(0, weight=1)
        ttk.Entry(frame, textvariable=self.vars[key]).grid(row=0, column=0, sticky="ew")
        ttk.Button(frame, text="浏览", command=lambda: self.browse_path(key, browse_dir)).grid(row=0, column=1, padx=(8, 0))
        frame.grid(row=0, column=1, sticky="ew", pady=4)

    def refresh_devices(self) -> None:
        self.device_list_text.set("正在读取 Orbbec 设备...")
        threading.Thread(target=self.read_devices_worker, daemon=True).start()

    def read_devices_worker(self) -> None:
        try:
            devices = self.enumerate_orbbec_devices()
            text = self.format_device_list(devices)
        except Exception as ex:
            devices = []
            text = f"读取失败: {ex}"
        self.root.after(0, lambda: self.apply_device_list(devices, text))

    def enumerate_orbbec_devices(self) -> list[dict[str, object]]:
        import orbbec_live_capture as cap

        sdk_bin = str(self.vars["sdk_bin"].get()).strip() or DEFAULTS["sdk_bin"]
        sdk = cap.SDK(Path(sdk_bin))
        ctx = dl = 0
        devices: list[dict[str, object]] = []
        try:
            ctx = sdk.create_context()
            dl = sdk.query_device_list(ctx)
            count = sdk.device_count(dl)
            for idx in range(count):
                dev = 0
                try:
                    dev = sdk.get_device(dl, idx)
                    sn, name = sdk.get_device_info(dev)
                    current_preset = ""
                    presets: list[str] = []
                    preset_error = ""
                    try:
                        current_preset = sdk.get_current_preset_name(dev)
                        presets = sdk.get_available_presets(dev)
                    except Exception as ex:
                        preset_error = str(ex)
                    devices.append(
                        {
                            "index": str(idx),
                            "name": name,
                            "sn": sn,
                            "current_preset": current_preset,
                            "presets": ", ".join(presets),
                            "preset_names": presets,
                            "preset_error": preset_error,
                        }
                    )
                finally:
                    if dev:
                        sdk.delete_device(dev)
        finally:
            if dl:
                sdk.delete_device_list(dl)
            if ctx:
                sdk.delete_context(ctx)
        return devices

    def format_device_list(self, devices: list[dict[str, object]]) -> str:
        if not devices:
            return "未发现 Orbbec 设备"
        lines: list[str] = []
        for d in devices:
            lines.append(f"Device[{d['index']}] {d['name']}  SN:{d['sn']}")
            if d.get("current_preset"):
                lines.append(f"  当前模式: {d['current_preset']}")
            if d.get("presets"):
                lines.append(f"  可用模式: {d['presets']}")
            elif d.get("preset_error"):
                lines.append(f"  模式读取失败: {d['preset_error']}")
        return "\n".join(lines)

    def apply_device_list(self, devices: list[dict[str, object]], text: str) -> None:
        self.detected_devices = devices
        self.device_list_text.set(text)
        if devices:
            self.status.set(f"Ready | detected {len(devices)} Orbbec device(s)")
        self.update_preset_choices()

    def browse_path(self, key: str, browse_dir: bool) -> None:
        initial = str(self.vars[key].get()).strip()
        initial_path = Path(initial) if initial else ROOT
        if browse_dir:
            selected = filedialog.askdirectory(initialdir=str(initial_path if initial_path.exists() else ROOT))
        else:
            selected = filedialog.askopenfilename(initialdir=str(initial_path.parent if initial_path.exists() else ROOT))
        if selected:
            self.vars[key].set(selected)

    def current_mode(self) -> str:
        mode = str(self.vars["mode"].get())
        return mode if mode in MODE_LABELS else "rgb_dataset"

    @staticmethod
    def normalize_device_text(text: str) -> str:
        return "".join(ch for ch in str(text or "").lower() if ch.isalnum())

    @staticmethod
    def ordered_unique(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            text = str(value or "").strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                result.append(text)
        return result

    def camera_hint_for_current_mode(self) -> str:
        mode = self.current_mode()
        if mode == "rgb_dataset":
            return str(self.vars["camera"].get()).strip() or "335L"
        return MODE_CAMERA_HINTS.get(mode, "")

    def is_preset_allowed_for_mode(self, mode: str, preset_name: str) -> bool:
        preset = str(preset_name or "").strip()
        if not preset:
            return False
        if mode == "dual_rgb_305":
            return preset == DUAL_COLOR_PRESET
        if mode in {"rgb_dataset", "rgb_interval_305", "rgbd_305"}:
            return preset != DUAL_COLOR_PRESET
        return True

    def preset_values_for_current_mode(self) -> list[str]:
        mode = self.current_mode()
        camera_hint = self.camera_hint_for_current_mode()
        if not camera_hint:
            return []

        normalized_hint = self.normalize_device_text(camera_hint)
        detected_values: list[str] = []
        for device in self.detected_devices:
            name = str(device.get("name", ""))
            if normalized_hint and normalized_hint not in self.normalize_device_text(name):
                continue
            preset_names = device.get("preset_names", [])
            if isinstance(preset_names, str):
                detected_values.extend([item.strip() for item in preset_names.split(",")])
            elif isinstance(preset_names, list):
                detected_values.extend([str(item).strip() for item in preset_names])

        values = self.ordered_unique(detected_values)
        if not values:
            values = list(PRESET_FALLBACKS.get(camera_hint, ["Default"]))
        values = [value for value in values if self.is_preset_allowed_for_mode(mode, value)]
        if values:
            return values
        default = MODE_DEFAULT_PRESETS.get(mode, "Default")
        return [default] if default else []

    def update_preset_choices(self, force_default: bool = False) -> None:
        values = self.preset_values_for_current_mode()
        if self.preset_combo is not None:
            self.preset_combo.configure(values=values)
        if not values:
            return

        current = str(self.vars["device_preset"].get()).strip()
        default = MODE_DEFAULT_PRESETS.get(self.current_mode(), "Default")
        target = current
        if force_default or not current or current not in values:
            target = default if default in values else values[0]
        if target != current:
            self.vars["device_preset"].set(target)

    def on_mode_label_changed(self) -> None:
        self.vars["mode"].set(self.mode_by_label.get(self.mode_label.get(), "rgb_dataset"))
        self.on_mode_changed()

    def on_mode_changed(self) -> None:
        mode = self.current_mode()
        self.mode_label.set(MODE_LABELS[mode])
        self.mode_description.set(MODE_DESCRIPTIONS[mode])
        if mode == "rgb_dataset":
            self.on_camera_changed()
            self.vars["output_root"].set(str(ROOT / "captures" / "rgb_dataset"))
        elif mode == "rgb_interval_305":
            self.clear_known_serial()
            self.vars["output_root"].set(str(ROOT / "captures"))
        elif mode == "rgbd_305":
            self.clear_known_serial()
            self.vars["output_root"].set(str(ROOT / "captures"))
        elif mode == "rgbd_335l":
            self.clear_known_serial()
            if is_standard_config_path(self.vars["config_path"].get()):
                self.vars["config_path"].set(str(CONFIG_DIR / "config.yaml"))
            self.vars["output_root"].set(str(ROOT / "captures"))
        elif mode == "dual_rgb_305":
            self.clear_known_serial()
            if is_standard_config_path(self.vars["config_path"].get()):
                self.vars["config_path"].set(str(CONFIG_DIR / "config_dual_rgb.yaml"))
            self.vars["output_root"].set(str(ROOT / "captures"))
        elif mode == "merged_rgbd":
            self.clear_known_serial()
            self.vars["width"].set("1280")
            self.vars["height"].set("800")
            self.vars["fps"].set("30")
            self.vars["output_root"].set(str(ROOT / "captures"))
        self.update_preset_choices(force_default=True)

        for row in self.field_rows.values():
            row.grid_remove()
        self.empty_params_note.grid_remove()

        visible = MODE_FIELDS[mode]
        if not visible:
            self.empty_params_note.grid(row=0, column=0, sticky="w", pady=8)
        else:
            for row_index, key in enumerate(visible):
                self.field_rows[key].grid(row=row_index, column=0, sticky="ew")
        self.update_command_preview()

    def on_camera_changed(self) -> None:
        camera = str(self.vars["camera"].get())
        self.vars["task"].set(CAMERA_TASKS.get(camera, "precise"))
        if str(self.vars["serial"].get()).strip() in KNOWN_CAMERA_SERIALS:
            self.vars["serial"].set("")
        self.update_preset_choices(force_default=True)

    def clear_known_serial(self) -> None:
        if str(self.vars["serial"].get()).strip() in KNOWN_CAMERA_SERIALS:
            self.vars["serial"].set("")

    def reset_time_token(self) -> None:
        token = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.vars["session"].set(token)
        self.vars["tag"].set(token)

    def add_common_device_args(self, cmd: list[str]) -> None:
        serial = str(self.vars["serial"].get()).strip()
        device_index = str(self.vars["device_index"].get()).strip()
        if serial:
            cmd.extend(["--serial", serial])
        if device_index:
            cmd.extend(["--device-index", device_index])

    def add_common_capture_args(self, cmd: list[str]) -> None:
        sdk_bin = str(self.vars["sdk_bin"].get()).strip()
        output_root = str(self.vars["output_root"].get()).strip()
        if sdk_bin:
            cmd.extend(["--sdk-bin", sdk_bin])
        if output_root:
            cmd.extend(["--output-root", output_root])

    def add_optional_preview_fps(self, cmd: list[str]) -> None:
        preview_fps = str(self.vars["preview_fps"].get()).strip()
        if preview_fps:
            cmd.extend(["--preview-fps", preview_fps])

    def build_command(self) -> list[str]:
        mode = self.current_mode()
        if mode == "rgb_dataset":
            cmd = [
                sys.executable,
                str(SCRIPTS["rgb_dataset"]),
                "--camera",
                str(self.vars["camera"].get()).strip(),
                "--task",
                str(self.vars["task"].get()).strip(),
                "--width",
                str(self.vars["width"].get()).strip(),
                "--height",
                str(self.vars["height"].get()).strip(),
                "--fps",
                str(self.vars["fps"].get()).strip(),
                "--auto-interval",
                str(self.vars["auto_interval"].get()).strip(),
                "--png-compression",
                str(self.vars["png_compression"].get()).strip(),
            ]
            output_root = str(self.vars["output_root"].get()).strip()
            if output_root:
                cmd.extend(["--output-root", output_root])
            session = str(self.vars["session"].get()).strip()
            if session:
                cmd.extend(["--session", session])
            formats = [item for item in str(self.vars["formats"].get()).split() if item]
            if formats:
                cmd.append("--formats")
                cmd.extend(formats)
            preset = str(self.vars["device_preset"].get()).strip()
            if preset:
                cmd.extend(["--preset", preset])
            self.add_optional_preview_fps(cmd)
            if bool(self.vars["start_auto"].get()):
                cmd.append("--start-auto")
            if bool(self.vars["no_preview"].get()):
                cmd.append("--no-preview")
            return cmd

        if mode == "rgb_interval_305":
            cmd = [
                sys.executable,
                str(SCRIPTS["rgb_interval_305"]),
                "--width",
                str(self.vars["width"].get()).strip(),
                "--height",
                str(self.vars["height"].get()).strip(),
                "--fps",
                str(self.vars["fps"].get()).strip(),
            ]
            output_root = str(self.vars["output_root"].get()).strip()
            if output_root:
                cmd.extend(["--output-root", output_root])
            save_every_frames = str(self.vars["save_every_frames"].get()).strip()
            save_every_seconds = str(self.vars["save_every_seconds"].get()).strip()
            max_saves = str(self.vars["max_saves"].get()).strip()
            if save_every_frames and save_every_frames != "0":
                cmd.extend(["--save-every-frames", save_every_frames])
            if save_every_seconds and save_every_seconds != "0":
                cmd.extend(["--save-every-seconds", save_every_seconds])
            if max_saves and max_saves != "0":
                cmd.extend(["--max-saves", max_saves])
            formats = [item for item in str(self.vars["formats"].get()).split() if item]
            if formats:
                cmd.append("--formats")
                cmd.extend(formats)
            preset = str(self.vars["device_preset"].get()).strip()
            if preset:
                cmd.extend(["--preset", preset])
            self.add_optional_preview_fps(cmd)
            if bool(self.vars["start_auto"].get()):
                cmd.append("--auto")
            if bool(self.vars["no_preview"].get()):
                cmd.append("--no-preview")
            return cmd

        if mode == "rgbd_305":
            cmd = [
                sys.executable,
                str(SCRIPTS["rgbd_305"]),
            ]
            output_root = str(self.vars["output_root"].get()).strip()
            tag = str(self.vars["tag"].get()).strip()
            if output_root:
                cmd.extend(["--output-root", output_root])
            if tag:
                cmd.extend(["--tag", tag])
            preset = str(self.vars["device_preset"].get()).strip()
            if preset:
                cmd.extend(["--preset", preset])
            self.add_optional_preview_fps(cmd)
            return cmd

        if mode in ("rgbd_335l", "dual_rgb_305"):
            cmd = [
                sys.executable,
                str(SCRIPTS["rgbd_config"]),
                "--config",
                str(self.vars["config_path"].get()).strip(),
            ]
            output_root = str(self.vars["output_root"].get()).strip()
            if output_root:
                cmd.extend(["--output-root", output_root])
            tag = str(self.vars["tag"].get()).strip()
            if tag:
                cmd.extend(["--tag", tag])
            preset = str(self.vars["device_preset"].get()).strip()
            if preset:
                cmd.extend(["--preset", preset])
            self.add_optional_preview_fps(cmd)
            return cmd

        if mode == "merged_dual":
            cmd = [sys.executable, str(SCRIPTS["merged_dual"])]
            self.add_optional_preview_fps(cmd)
            return cmd

        if mode == "merged_rgbd":
            cmd = [
                sys.executable,
                str(SCRIPTS["merged_dual"]),
                "--capture-mode",
                "rgbd-rgbd",
                "--config-335l",
                str(CONFIG_DIR / "config.yaml"),
                "--config-305",
                str(CONFIG_DIR / "config_305_rgbd.yaml"),
                "--width",
                str(self.vars["width"].get()).strip(),
                "--height",
                str(self.vars["height"].get()).strip(),
                "--fps",
                str(self.vars["fps"].get()).strip(),
                "--show-depth-preview",
                "--preview-every-n",
                "1",
                "--depth-preview-every-n",
                "5",
            ]
            preset = str(self.vars["device_preset"].get()).strip()
            if preset:
                cmd.extend(["--preset-305", preset])
            self.add_optional_preview_fps(cmd)
            self.add_common_capture_args(cmd)
            return cmd

        raise RuntimeError(f"unknown mode: {mode}")

    def update_command_preview(self) -> None:
        try:
            self.command_preview.set(subprocess.list2cmdline(self.build_command()))
        except Exception as ex:
            self.command_preview.set(f"参数暂不可用: {ex}")

    def validate_before_start(self) -> bool:
        mode = self.current_mode()
        script = self.build_command()[1]
        if not Path(script).exists():
            messagebox.showerror("缺少采集脚本", f"找不到: {script}")
            return False

        numeric_fields = {
            "rgb_dataset": ["width", "height", "fps", "auto_interval", "png_compression"],
            "rgb_interval_305": ["width", "height", "fps", "save_every_seconds", "save_every_frames", "max_saves"],
            "merged_rgbd": ["width", "height", "fps"],
        }.get(mode, [])
        for key in numeric_fields:
            value = str(self.vars[key].get()).strip()
            try:
                number = float(value)
            except ValueError:
                messagebox.showerror("参数错误", f"{key} 必须是数字: {value}")
                return False
            if key in {"width", "height", "fps", "auto_interval"} and number <= 0:
                messagebox.showerror("参数错误", f"{key} 必须大于 0")
                return False
            if key in {"save_every_seconds", "save_every_frames", "max_saves", "png_compression"} and number < 0:
                messagebox.showerror("参数错误", f"{key} 不能小于 0")
                return False

        preview_fps = str(self.vars["preview_fps"].get()).strip()
        if "preview_fps" in MODE_FIELDS[mode] and preview_fps:
            try:
                preview_fps_number = float(preview_fps)
            except ValueError:
                messagebox.showerror("参数错误", f"preview_fps 必须是数字或留空: {preview_fps}")
                return False
            if preview_fps_number <= 0:
                messagebox.showerror("参数错误", "preview_fps 必须大于 0，或留空使用默认值")
                return False

        if "device_index" in MODE_FIELDS[mode] and str(self.vars["device_index"].get()).strip():
            try:
                int(str(self.vars["device_index"].get()).strip())
            except ValueError:
                messagebox.showerror("参数错误", "device_index 必须是整数或留空")
                return False
        if "config_path" in MODE_FIELDS[mode] and not Path(str(self.vars["config_path"].get()).strip()).exists():
            messagebox.showerror("参数错误", f"配置文件不存在: {self.vars['config_path'].get()}")
            return False
        preset = str(self.vars["device_preset"].get()).strip()
        if "device_preset" in MODE_FIELDS[mode] and not self.is_preset_allowed_for_mode(mode, preset):
            messagebox.showerror(
                "设备模式不匹配",
                f"{MODE_LABELS[mode]} 不能使用 {preset}。\n"
                "RGB-D/普通 RGB 请用 Default、High Accuracy 等普通模式；305 双 RGB 才使用 Dual Color Streams。",
            )
            return False
        return True

    def start_capture(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo("正在运行", "采集进程已经在运行。")
            return
        if not self.validate_before_start():
            return

        self.save_settings()
        cmd = self.build_command()
        self.append_log("\n>>> " + subprocess.list2cmdline(cmd) + "\n")
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except Exception as ex:
            messagebox.showerror("启动失败", str(ex))
            self.process = None
            return

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.set("Running | 预览窗口里按 q/Esc 可安全退出")
        threading.Thread(target=self.read_process_output, daemon=True).start()
        self.root.after(500, self.check_process)

    def read_process_output(self) -> None:
        proc = self.process
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self.log_queue.put(line)

    def flush_log_queue(self) -> None:
        while True:
            try:
                text = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.append_log(text)
        self.root.after(100, self.flush_log_queue)

    def append_log(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")

    def check_process(self) -> None:
        if self.process is None:
            return
        code = self.process.poll()
        if code is None:
            self.root.after(500, self.check_process)
            return
        self.append_log(f"\n<<< process exited with code {code}\n")
        self.process = None
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status.set(f"Ready | last exit code: {code}")

    def stop_capture(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.append_log("\n>>> stop requested\n")
        try:
            if os.name == "nt":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.terminate()
        except Exception as ex:
            self.append_log(f"Stop signal failed: {ex}\n")
            try:
                self.process.terminate()
            except Exception:
                pass

    def copy_command(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.command_preview.get())
        self.status.set("Command copied to clipboard")

    def open_output_root(self) -> None:
        path = Path(str(self.vars["output_root"].get())).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))
        except Exception as ex:
            messagebox.showerror("打开失败", str(ex))

    def on_close(self) -> None:
        self.save_settings()
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("采集仍在运行", "采集进程还在运行，是否先发送停止信号并关闭界面？"):
                return
            self.stop_capture()
        self.root.destroy()


def main() -> int:
    root = Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    LauncherApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
