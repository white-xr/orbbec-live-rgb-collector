@echo off
setlocal
set "SCRIPT_DIR=%~dp0.."
set "ROOT_DIR=%~dp0..\.."
cd /d "%ROOT_DIR%"
set "CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat"
if exist "%CONDA_ACTIVATE%" (
    call "%CONDA_ACTIVATE%" rgbdseg
) else (
    call conda activate rgbdseg
)
python "%SCRIPT_DIR%\orbbec_live_capture.py" --config "%ROOT_DIR%\config\config_dual_rgb.yaml"
pause
