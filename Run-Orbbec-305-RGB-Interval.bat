@echo off
setlocal
cd /d "%~dp0"
python "%~dp0capture_305_rgb_interval.py" --save-every-seconds 1 %*
pause