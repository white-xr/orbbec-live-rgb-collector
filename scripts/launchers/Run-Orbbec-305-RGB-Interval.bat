@echo off
setlocal
set "SCRIPT_DIR=%~dp0.."
set "ROOT_DIR=%~dp0..\.."
cd /d "%ROOT_DIR%"
python "%SCRIPT_DIR%\capture_305_rgb_interval.py" --save-every-seconds 1 %*
pause
