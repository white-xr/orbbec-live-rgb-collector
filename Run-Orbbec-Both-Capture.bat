@echo off
setlocal
cd /d "%~dp0"
set "CONDA_ACTIVATE=%USERPROFILE%\miniconda3\Scripts\activate.bat"
if exist "%CONDA_ACTIVATE%" (
    call "%CONDA_ACTIVATE%" rgbdseg
) else (
    call conda activate rgbdseg
)
python "%~dp0merged_dual_camera_capture.py"
pause
