#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Small portable Tkinter launcher for collect_orbbec_rgb_dataset.py.

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
COLLECTOR = ROOT / "collect_orbbec_rgb_dataset.py"
SETTINGS_FILE = ROOT / "orbbec_rgb_dataset_gui_settings.json"

CAMERA_TASKS = {"335L": "coarse", "305": "precise"}
LEGACY_CAMERA_SERIALS = {"CP2N1630005C", "CV2L36000024"}

DEFAULTS = {
    "camera": "305",
    "task": "precise",
    "width": "1280",
    "height": "800",
    "fps": "30",
    "auto_interval": "1.0",
    "session": "",
    "serial": "",
    "device_index": "",
    "sdk_bin": r"D:\OrbbecSDK_v2\bin",
    "output_root": str(ROOT / "captures" / "rgb_dataset"),
    "formats": "BGR RGB MJPG YUYV BGRA RGBA UYVY",
    "png_compression": "3",
    "start_auto": False,
    "no_preview": False,
}


class LauncherApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Orbbec RGB Dataset Collector")
        self.root.minsize(760, 640)
        self.process: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        data = self.load_settings()
        self.vars: dict[str, StringVar | BooleanVar] = {
            "camera": StringVar(value=data.get("camera", DEFAULTS["camera"])),
            "task": StringVar(value=data.get("task", DEFAULTS["task"])),
            "width": StringVar(value=data.get("width", DEFAULTS["width"])),
            "height": StringVar(value=data.get("height", DEFAULTS["height"])),
            "fps": StringVar(value=data.get("fps", DEFAULTS["fps"])),
            "auto_interval": StringVar(value=data.get("auto_interval", DEFAULTS["auto_interval"])),
            "session": StringVar(value=data.get("session", DEFAULTS["session"])),
            "serial": StringVar(value=data.get("serial", DEFAULTS["serial"])),
            "device_index": StringVar(value=data.get("device_index", DEFAULTS["device_index"])),
            "sdk_bin": StringVar(value=data.get("sdk_bin", DEFAULTS["sdk_bin"])),
            "output_root": StringVar(value=data.get("output_root", DEFAULTS["output_root"])),
            "formats": StringVar(value=data.get("formats", DEFAULTS["formats"])),
            "png_compression": StringVar(value=data.get("png_compression", DEFAULTS["png_compression"])),
            "start_auto": BooleanVar(value=bool(data.get("start_auto", DEFAULTS["start_auto"]))),
            "no_preview": BooleanVar(value=bool(data.get("no_preview", DEFAULTS["no_preview"]))),
        }

        self.status = StringVar(value=f"Ready | Python: {sys.executable}")
        self.build_ui()
        self.on_camera_changed()
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

        title = ttk.Label(outer, text="Orbbec RGB Dataset Collector", font=("Segoe UI", 15, "bold"))
        title.pack(anchor="w")

        grid = ttk.LabelFrame(outer, text="采集参数", padding=12)
        grid.pack(fill="x", pady=(12, 10))
        for col in (1, 3):
            grid.columnconfigure(col, weight=1)

        self.add_combo(grid, "camera", "相机", 0, 0, ["335L", "305"], self.on_camera_changed)
        self.add_combo(grid, "task", "任务", 0, 2, ["coarse", "precise"])
        self.add_entry(grid, "width", "宽度", 1, 0)
        self.add_entry(grid, "height", "高度", 1, 2)
        self.add_entry(grid, "fps", "FPS", 2, 0)
        self.add_entry(grid, "auto_interval", "自动间隔(s)", 2, 2)
        self.add_entry(grid, "session", "Session", 3, 0, hint="留空则采集脚本用当前时间")
        self.add_entry(grid, "png_compression", "PNG压缩", 3, 2)
        self.add_entry(grid, "serial", "序列号", 4, 0, hint="留空=按型号自动选；any=第一个设备")
        self.add_entry(grid, "device_index", "设备Index", 4, 2, hint="可留空")
        self.add_path_row(grid, "sdk_bin", "SDK bin", 5, browse_dir=True)
        self.add_path_row(grid, "output_root", "输出目录", 6, browse_dir=True)
        self.add_entry(grid, "formats", "COLOR格式优先级", 7, 0, columnspan=3)

        options = ttk.Frame(grid)
        options.grid(row=8, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Checkbutton(options, text="启动后自动保存", variable=self.vars["start_auto"]).pack(side="left")
        ttk.Checkbutton(options, text="无预览窗口", variable=self.vars["no_preview"]).pack(side="left", padx=(18, 0))

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(0, 10))
        self.start_button = ttk.Button(buttons, text="启动采集", command=self.start_capture)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="停止采集", command=self.stop_capture, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="复制命令", command=self.copy_command).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="打开输出目录", command=self.open_output_root).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="重置Session", command=self.reset_session).pack(side="right")

        self.command_preview = StringVar(value="")
        command_box = ttk.LabelFrame(outer, text="将执行的命令", padding=8)
        command_box.pack(fill="x", pady=(0, 10))
        ttk.Label(command_box, textvariable=self.command_preview, wraplength=700).pack(anchor="w")

        log_box = ttk.LabelFrame(outer, text="运行日志", padding=8)
        log_box.pack(fill="both", expand=True)
        self.log = ScrolledText(log_box, height=14, wrap="word", font=("Consolas", 10))
        self.log.pack(fill="both", expand=True)

        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=(8, 0))
        self.update_command_preview()
        for var in self.vars.values():
            var.trace_add("write", lambda *_: self.update_command_preview())

    def add_combo(self, parent, key: str, label: str, row: int, col: int, values: list[str], command=None) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=4)
        combo = ttk.Combobox(parent, textvariable=self.vars[key], values=values, state="readonly", width=18)
        combo.grid(row=row, column=col + 1, sticky="ew", pady=4)
        if command is not None:
            combo.bind("<<ComboboxSelected>>", lambda _event: command())

    def add_entry(self, parent, key: str, label: str, row: int, col: int, columnspan: int = 1, hint: str = "") -> None:
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=self.vars[key])
        entry.grid(row=row, column=col + 1, columnspan=columnspan, sticky="ew", pady=4)
        if hint:
            ttk.Label(parent, text=hint, foreground="#666666").grid(row=row, column=col + 2, sticky="w", padx=(8, 0), pady=4)

    def add_path_row(self, parent, key: str, label: str, row: int, browse_dir: bool = True) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=self.vars[key]).grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Button(parent, text="浏览", command=lambda: self.browse_path(key, browse_dir)).grid(row=row, column=3, sticky="e", padx=(8, 0), pady=4)

    def browse_path(self, key: str, browse_dir: bool) -> None:
        initial = str(self.vars[key].get())
        if browse_dir:
            selected = filedialog.askdirectory(initialdir=initial if Path(initial).exists() else str(ROOT))
        else:
            selected = filedialog.askopenfilename(initialdir=str(ROOT))
        if selected:
            self.vars[key].set(selected)

    def on_camera_changed(self) -> None:
        camera = str(self.vars["camera"].get())
        self.vars["task"].set(CAMERA_TASKS.get(camera, "precise"))
        current_serial = str(self.vars["serial"].get()).strip()
        if current_serial in LEGACY_CAMERA_SERIALS:
            self.vars["serial"].set("")

    def reset_session(self) -> None:
        self.vars["session"].set(datetime.now().strftime("%Y%m%d_%H%M%S"))

    def build_command(self) -> list[str]:
        camera = str(self.vars["camera"].get()).strip()
        task = str(self.vars["task"].get()).strip()
        cmd = [
            sys.executable,
            str(COLLECTOR),
            "--camera",
            camera,
            "--task",
            task,
            "--width",
            str(self.vars["width"].get()).strip(),
            "--height",
            str(self.vars["height"].get()).strip(),
            "--fps",
            str(self.vars["fps"].get()).strip(),
            "--auto-interval",
            str(self.vars["auto_interval"].get()).strip(),
            "--sdk-bin",
            str(self.vars["sdk_bin"].get()).strip(),
            "--output-root",
            str(self.vars["output_root"].get()).strip(),
            "--png-compression",
            str(self.vars["png_compression"].get()).strip(),
        ]

        session = str(self.vars["session"].get()).strip()
        serial = str(self.vars["serial"].get()).strip()
        device_index = str(self.vars["device_index"].get()).strip()
        formats = [item for item in str(self.vars["formats"].get()).split() if item]

        if session:
            cmd.extend(["--session", session])
        if serial:
            cmd.extend(["--serial", serial])
        if device_index:
            cmd.extend(["--device-index", device_index])
        if formats:
            cmd.append("--formats")
            cmd.extend(formats)
        if bool(self.vars["start_auto"].get()):
            cmd.append("--start-auto")
        if bool(self.vars["no_preview"].get()):
            cmd.append("--no-preview")
        return cmd

    def update_command_preview(self) -> None:
        try:
            cmd = self.build_command()
            self.command_preview.set(subprocess.list2cmdline(cmd))
        except Exception as ex:
            self.command_preview.set(f"参数暂不可用: {ex}")

    def validate_before_start(self) -> bool:
        if not COLLECTOR.exists():
            messagebox.showerror("缺少采集脚本", f"找不到: {COLLECTOR}")
            return False
        numeric_fields = ["width", "height", "fps", "auto_interval", "png_compression"]
        for key in numeric_fields:
            value = str(self.vars[key].get()).strip()
            try:
                number = float(value)
            except ValueError:
                messagebox.showerror("参数错误", f"{key} 必须是数字: {value}")
                return False
            if number <= 0 and key != "png_compression":
                messagebox.showerror("参数错误", f"{key} 必须大于 0")
                return False
        if str(self.vars["device_index"].get()).strip():
            try:
                int(str(self.vars["device_index"].get()).strip())
            except ValueError:
                messagebox.showerror("参数错误", "device_index 必须是整数或留空")
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
        self.status.set("Running | 在 OpenCV 预览窗口按 q 可安全退出")
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
