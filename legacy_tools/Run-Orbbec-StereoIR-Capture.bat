@echo off
setlocal
cd /d "%~dp0"
set "CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat"
if exist "%CONDA_ACTIVATE%" (
    call "%CONDA_ACTIVATE%" rgbdseg
) else (
    call conda activate rgbdseg
)
python "%~dp0..\scripts\orbbec_live_capture.py" --config "%~dp0..\config\config_stereo_ir.yaml"
pause
