@echo off
setlocal
cd /d "%~dp0"
python "%~dp0orbbec_dual_live_capture.py" --output-root "%~dp0captures"
pause
