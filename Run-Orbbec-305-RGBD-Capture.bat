@echo off
setlocal
cd /d "%~dp0"

set "ENV_PYTHON=%USERPROFILE%\miniconda3\envs\vision-data\python.exe"
if exist "%ENV_PYTHON%" (
    "%ENV_PYTHON%" "%~dp0capture_305_rgbd.py" %*
) else (
    python "%~dp0capture_305_rgbd.py" %*
)
pause
