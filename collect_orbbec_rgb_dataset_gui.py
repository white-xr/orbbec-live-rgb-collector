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
SETTINGS_FILE = ROOT / "orbbec_rgb_dataset_gui_settings.json"

SCRIPTS = {
    "rgb_dataset": ROOT / "collect_orbbec_rgb_dataset.py",
    "rgb_interval_305": ROOT / "capture_305_rgb_interval.py",
    "rgbd_config": ROOT / "orbbec_live_capture.py",
    "merged_dual": ROOT / "merged_dual_camera_capture.py",
    "both_controller": ROOT / "run_both_cameras.py",
}

MODE_LABELS = {
    "rgb_dataset": "YOLO RGB 数据集采集",
    "rgb_interval_305": "305 单 RGB 间隔采集",
    "rgbd_335l": "335L RGB-D 采集",
    "dual_rgb_305": "305 双 RGB 采集",
    "merged_dual": "335L + 305 单窗口联合采集",
    "both_controller": "335L + 305 双进程联合采集",
}

MODE_DESCRIPTIONS = {
    "rgb_dataset": "只保存 RGB 图片和 metadata.csv，适合检测/分割数据集采集。",
    "rgb_interval_305": "旧版 305 单路 RGB 单张/间隔保存工具。",
    "rgbd_335l": "使用 config.yaml 启动普通 RGB-D 采集，保存 color/depth 等配置内启用的数据。",
    "dual_rgb_305": "使用 config_dual_rgb.yaml 切到 Dual Color Streams，保存 305 左右双 RGB。",
    "merged_dual": "一个窗口同时打开 335L RGB-D 和 305 双 RGB，适合现场联合预览采集。",
    "both_controller": "启动两个独立采集进程并做软件同步开始/停止，会打开独立控制窗口。",
}

MODE_FIELDS = {
    "rgb_dataset": [
        "camera",
        "task",
        "width",
        "height",
        "fps",
        "auto_interval",
        "session",
        "serial",
        "device_index",
        "sdk_bin",
        "output_root",
        "formats",
        "png_compression",
        "start_auto",
        "no_preview",
    ],
    "rgb_interval_305": [
        "width",
        "height",
        "fps",
        "save_every_seconds",
        "save_every_frames",
        "max_saves",
        "serial",
        "sdk_bin",
        "output_root",
        "formats",
        "start_auto",
        "no_preview",
    ],
    "rgbd_335l": [
        "config_path",
        "sdk_bin",
        "output_root",
        "tag",
        "serial",
        "device_index",
        "start_auto",
    ],
    "dual_rgb_305": [
        "config_path",
        "sdk_bin",
        "output_root",
        "tag",
        "serial",
        "device_index",
        "start_auto",
    ],
    "merged_dual": [],
    "both_controller": ["tag", "delay"],
}

CAMERA_TASKS = {"335L": "coarse", "305": "precise"}
LEGACY_CAMERA_SERIALS = {"CP2N1630005C", "CV2L36000024"}
STANDARD_CONFIGS = {str(ROOT / "config.yaml"), str(ROOT / "config_dual_rgb.yaml"), ""}

DEFAULTS = {
    "mode": "rgb_dataset",
    "camera": "335L",
    "task": "coarse",
    "width": "1280",
    "height": "800",
    "fps": "30",
    "auto_interval": "1.0",
    "save_every_seconds": "1.0",
    "save_every_frames": "0",
    "max_saves": "0",
    "session": "",
    "tag": "",
    "delay": "8.0",
    "serial": "",
    "device_index": "",
    "sdk_bin": r"D:\OrbbecSDK_v2\bin",
    "output_root": str(ROOT / "captures" / "rgb_dataset"),
    "config_path": str(ROOT / "config.yaml"),
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
        self.mode_by_label = {label: key for key, label in MODE_LABELS.items()}

        data = self.load_settings()
        self.vars: dict[str, StringVar | BooleanVar] = {
            "mode": StringVar(value=data.get("mode", DEFAULTS["mode"])),
            "camera": StringVar(value=data.get("camera", DEFAULTS["camera"])),
            "task": StringVar(value=data.get("task", DEFAULTS["task"])),
            "width": StringVar(value=data.get("width", DEFAULTS["width"])),
            "height": StringVar(value=data.get("height", DEFAULTS["height"])),
            "fps": StringVar(value=data.get("fps", DEFAULTS["fps"])),
            "auto_interval": StringVar(value=data.get("auto_interval", DEFAULTS["auto_interval"])),
            "save_every_seconds": StringVar(value=data.get("save_every_seconds", DEFAULTS["save_every_seconds"])),
            "save_every_frames": StringVar(value=data.get("save_every_frames", DEFAULTS["save_every_frames"])),
            "max_saves": StringVar(value=data.get("max_saves", DEFAULTS["max_saves"])),
            "session": StringVar(value=data.get("session", DEFAULTS["session"])),
            "tag": StringVar(value=data.get("tag", DEFAULTS["tag"])),
            "delay": StringVar(value=data.get("delay", DEFAULTS["delay"])),
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
        self.command_preview = StringVar(value="")
        self.status = StringVar(value=f"Ready | Python: {sys.executable}")

        self.build_ui()
        self.on_mode_changed()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.flush_log_queue)

    def load_settings(self) -> dict:
        if not SETTINGS_FILE.exists():
            return dict(DEFAULTS)
        try:
            with SETTINGS_FILE.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            return {**DEFAULTS, **loaded}
        except Exception:
            return dict(DEFAULTS)

    def save_settings(self) -> None:
        data = {}
        for key, var in self.vars.items():
            data[key] = bool(var.get()) if isinstance(var, BooleanVar) else str(var.get())
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

        self.param_box = ttk.LabelFrame(outer, text="参数", padding=12)
        self.param_box.pack(fill="x", pady=(0, 10))
        self.param_box.columnconfigure(0, weight=1)

        self.add_combo_row("camera", "相机", ["335L", "305"], self.on_camera_changed, "数据集采集模式使用")
        self.add_combo_row("task", "任务", ["coarse", "precise"], None, "335L=coarse，305=precise")
        self.add_entry_row("width", "宽度")
        self.add_entry_row("height", "高度")
        self.add_entry_row("fps", "FPS")
        self.add_entry_row("auto_interval", "自动保存间隔(s)")
        self.add_entry_row("save_every_seconds", "间隔保存秒数")
        self.add_entry_row("save_every_frames", "间隔保存帧数", "0 表示不用帧间隔")
        self.add_entry_row("max_saves", "最多保存张数", "0 表示不限")
        self.add_entry_row("session", "Session", "留空则自动使用当前时间")
        self.add_entry_row("tag", "Tag", "用于 RGB-D/双机采集的 session 标记")
        self.add_entry_row("delay", "双机启动延迟(s)")
        self.add_entry_row("serial", "序列号", "留空=使用脚本/配置默认；any=第一个设备")
        self.add_entry_row("device_index", "设备 Index", "可留空")
        self.add_path_row("sdk_bin", "SDK bin", browse_dir=True)
        self.add_path_row("output_root", "输出目录", browse_dir=True)
        self.add_path_row("config_path", "配置文件", browse_dir=False)
        self.add_entry_row("formats", "COLOR 格式优先级")
        self.add_entry_row("png_compression", "PNG 压缩", "0 最快，9 最小")
        self.add_check_row("start_auto", "启动后自动保存")
        self.add_check_row("no_preview", "无预览窗口")

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(0, 10))
        self.start_button = ttk.Button(buttons, text="启动采集", command=self.start_capture)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="停止采集", command=self.stop_capture, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="复制命令", command=self.copy_command).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="打开输出目录", command=self.open_output_root).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="重置时间标记", command=self.reset_time_token).pack(side="right")

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

    def add_field_row(self, key: str, label: str, widget, hint: str = "") -> None:
        row = ttk.Frame(self.param_box)
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text=label, width=18).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        widget.grid(row=0, column=1, sticky="ew", pady=4)
        if hint:
            ttk.Label(row, text=hint, foreground="#666666").grid(row=0, column=2, sticky="w", padx=(8, 0), pady=4)
        self.field_rows[key] = row

    def add_entry_row(self, key: str, label: str, hint: str = "") -> None:
        entry = ttk.Entry(self.param_box, textvariable=self.vars[key])
        self.add_field_row(key, label, entry, hint)

    def add_combo_row(self, key: str, label: str, values: list[str], command=None, hint: str = "") -> None:
        combo = ttk.Combobox(self.param_box, textvariable=self.vars[key], values=values, state="readonly")
        if command is not None:
            combo.bind("<<ComboboxSelected>>", lambda _event: command())
        self.add_field_row(key, label, combo, hint)

    def add_check_row(self, key: str, label: str) -> None:
        check = ttk.Checkbutton(self.param_box, text=label, variable=self.vars[key])
        self.add_field_row(key, "", check)

    def add_path_row(self, key: str, label: str, browse_dir: bool) -> None:
        frame = ttk.Frame(self.param_box)
        frame.columnconfigure(0, weight=1)
        ttk.Entry(frame, textvariable=self.vars[key]).grid(row=0, column=0, sticky="ew")
        ttk.Button(frame, text="浏览", command=lambda: self.browse_path(key, browse_dir)).grid(row=0, column=1, padx=(8, 0))
        self.add_field_row(key, label, frame)

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
            self.vars["serial"].set("CV2L36000024" if self.vars["serial"].get() in ("", "CP2N1630005C") else self.vars["serial"].get())
            self.vars["output_root"].set(str(ROOT / "captures"))
        elif mode == "rgbd_335l":
            if self.vars["config_path"].get() in STANDARD_CONFIGS:
                self.vars["config_path"].set(str(ROOT / "config.yaml"))
            self.vars["output_root"].set(str(ROOT / "captures"))
        elif mode == "dual_rgb_305":
            if self.vars["config_path"].get() in STANDARD_CONFIGS:
                self.vars["config_path"].set(str(ROOT / "config_dual_rgb.yaml"))
            self.vars["output_root"].set(str(ROOT / "captures"))

        visible = set(MODE_FIELDS[mode])
        row_index = 0
        for key, row in self.field_rows.items():
            if key in visible:
                row.grid(row=row_index, column=0, sticky="ew")
                row_index += 1
            else:
                row.grid_remove()
        self.update_command_preview()

    def on_camera_changed(self) -> None:
        camera = str(self.vars["camera"].get())
        self.vars["task"].set(CAMERA_TASKS.get(camera, "precise"))
        if str(self.vars["serial"].get()).strip() in LEGACY_CAMERA_SERIALS:
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
            self.add_common_capture_args(cmd)
            self.add_common_device_args(cmd)
            session = str(self.vars["session"].get()).strip()
            if session:
                cmd.extend(["--session", session])
            formats = [item for item in str(self.vars["formats"].get()).split() if item]
            if formats:
                cmd.append("--formats")
                cmd.extend(formats)
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
            self.add_common_capture_args(cmd)
            serial = str(self.vars["serial"].get()).strip()
            if serial:
                cmd.extend(["--serial", serial])
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
            if bool(self.vars["start_auto"].get()):
                cmd.append("--auto")
            if bool(self.vars["no_preview"].get()):
                cmd.append("--no-preview")
            return cmd

        if mode in ("rgbd_335l", "dual_rgb_305"):
            cmd = [
                sys.executable,
                str(SCRIPTS["rgbd_config"]),
                "--config",
                str(self.vars["config_path"].get()).strip(),
            ]
            self.add_common_capture_args(cmd)
            self.add_common_device_args(cmd)
            tag = str(self.vars["tag"].get()).strip()
            if tag:
                cmd.extend(["--tag", tag])
            if bool(self.vars["start_auto"].get()):
                cmd.append("--auto-start")
            return cmd

        if mode == "merged_dual":
            return [sys.executable, str(SCRIPTS["merged_dual"])]

        if mode == "both_controller":
            cmd = [sys.executable, str(SCRIPTS["both_controller"])]
            delay = str(self.vars["delay"].get()).strip()
            tag = str(self.vars["tag"].get()).strip()
            if delay:
                cmd.extend(["--delay", delay])
            if tag:
                cmd.extend(["--tag", tag])
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
            "both_controller": ["delay"],
        }.get(mode, [])
        for key in numeric_fields:
            value = str(self.vars[key].get()).strip()
            try:
                number = float(value)
            except ValueError:
                messagebox.showerror("参数错误", f"{key} 必须是数字: {value}")
                return False
            if key in {"width", "height", "fps", "auto_interval", "delay"} and number <= 0:
                messagebox.showerror("参数错误", f"{key} 必须大于 0")
                return False
            if key in {"save_every_seconds", "save_every_frames", "max_saves", "png_compression"} and number < 0:
                messagebox.showerror("参数错误", f"{key} 不能小于 0")
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
        pipe_output = self.current_mode() != "both_controller"
        if self.current_mode() == "both_controller" and os.name == "nt":
            creationflags |= subprocess.CREATE_NEW_CONSOLE

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE if pipe_output else None,
                stderr=subprocess.STDOUT if pipe_output else None,
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
        if pipe_output:
            threading.Thread(target=self.read_process_output, daemon=True).start()
        else:
            self.append_log("双进程控制器已在独立窗口中运行，日志请看新窗口。\n")
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
