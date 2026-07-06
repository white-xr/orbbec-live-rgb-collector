# Orbbec Live Collector

Orbbec Gemini 335L / Gemini 305 data collection utilities for RGB-D capture,
dual RGB capture, and YOLO RGB dataset collection.

## RGB YOLO Dataset Collection

Use the Tkinter launcher after activating the Python environment:

```powershell
conda activate vision-data
cd D:\OrbbecLiveCollector
.\Run-Orbbec-RGB-Dataset-GUI.bat
```

The launcher starts `scripts\collect_orbbec_rgb_dataset.py`, which saves RGB images only.
Depth, IR, D2C, and point clouds are not saved in this dataset mode.

Keyboard controls in the preview window:

- `s` or `Space`: save one RGB frame in manual mode; start/stop saving in auto mode
- `a`: toggle manual/auto mode
- `q` or `Esc`: quit safely

The launcher provides an optional preview FPS field. Leave it empty to use each
script's default preview behavior; filling it only caps the preview window
refresh rate and does not change the camera capture FPS.

Saved images use:

```text
{index:06d}.png
```

Each session also writes `metadata.csv` with filename, timestamp, camera, task,
resolution, FPS, and save mode.

## Command Line Example

```powershell
python scripts\collect_orbbec_rgb_dataset.py --camera 335L --task coarse
python scripts\collect_orbbec_rgb_dataset.py --camera 305 --task precise
```

## 305 RGB-D Capture

```powershell
python scripts\capture_305_rgbd.py
```

This opens the normal RGB-D preview window. It does not save immediately:
use `Space`/`S` to start saving, `Space`/`E` to stop, and `Q`/`Esc` to quit.

Optional serial behavior:

- leave `--serial` empty: select by camera model name
- `--serial any`: use the first Orbbec device
- explicit serial: use the matching device only

## 335L + 305 RGB-D Joint Capture

Use the GUI mode `335L + 305 RGB-D 联合采集`, or run:

```powershell
python scripts\merged_dual_camera_capture.py --capture-mode rgbd-rgbd --width 1280 --height 800 --fps 30
```

If the 305 is enumerated as USB2.x, 1280x800@30 may not be exposed by the SDK;
the script prints the available profiles so the resolution can be adjusted.
This mode opens RGB and depth preview panels by default. The overlay shows live
FPS while saving; sustained 30 FPS still depends on the exposed SDK profile,
USB bandwidth, CPU load, and disk write speed.
The merged preview is decoupled from saving: the window refreshes at a capped
preview FPS, RGB preview uses the latest frames, and depth pseudo-color preview
is generated less frequently to reduce CPU load.
Saving is software-gated across both cameras: one 335L sample and one 305 sample
are saved together, so the two camera folders keep the same image count. This is
count synchronization, not hardware timestamp synchronization. D2C alignment is
disabled by default in the provided RGB-D configs (`pipeline.align_mode: "DISABLE"`).

## Notes

- The scripts expect Orbbec SDK v2 binaries under `D:\OrbbecSDK_v2\bin` by default.
- YAML capture configs are stored under `config/`.
- CLI capture scripts are stored under `scripts/`; helper analysis tools are under `tools/`.
- Captured data under `captures/` is intentionally ignored by git.
- Local GUI settings are stored in `config/orbbec_rgb_dataset_gui_settings.json` and are not committed.
