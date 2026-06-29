@echo off
setlocal
cd /d "%~dp0"

set "EXPECTED_ENV=vision-data"

echo Current conda env: %CONDA_DEFAULT_ENV%
if /I not "%CONDA_DEFAULT_ENV%"=="%EXPECTED_ENV%" (
    echo [WARN] This launcher does not activate conda automatically.
    echo [WARN] Expected env: %EXPECTED_ENV%
    echo [WARN] Current env : %CONDA_DEFAULT_ENV%
    echo [WARN] If needed, run: conda activate %EXPECTED_ENV%
)

echo Using Python:
where python

python "%~dp0collect_orbbec_rgb_dataset_gui.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

pause
exit /b %EXIT_CODE%
