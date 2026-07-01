@echo off
setlocal
set "SCRIPT_DIR=%~dp0.."
set "ROOT_DIR=%~dp0..\.."
cd /d "%ROOT_DIR%"

set "ENV_PYTHON=%USERPROFILE%\miniconda3\envs\vision-data\python.exe"
if exist "%ENV_PYTHON%" (
    "%ENV_PYTHON%" "%SCRIPT_DIR%\capture_305_rgbd.py" %*
) else (
    python "%SCRIPT_DIR%\capture_305_rgbd.py" %*
)
pause
