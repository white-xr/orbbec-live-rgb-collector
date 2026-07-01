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

- `1`: `base`
- `2`: `cover`
- `3`: `both`
- `0`: `empty`
- `s` or `Space`: save current RGB frame
- `a`: toggle automatic saving
- `q` or `Esc`: quit safely

Saved images use:

```text
{camera}_{task}_{tag}_{session}_{index:06d}.png
```

Each session also writes `metadata.csv` with filename, timestamp, camera, task,
resolution, FPS, tag, and save mode.

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

## Notes

- The scripts expect Orbbec SDK v2 binaries under `D:\OrbbecSDK_v2\bin` by default.
- YAML capture configs are stored under `config/`.
- CLI capture scripts are stored under `scripts/`; helper analysis tools are under `tools/`.
- Captured data under `captures/` is intentionally ignored by git.
- Local GUI settings are stored in `config/orbbec_rgb_dataset_gui_settings.json` and are not committed.
