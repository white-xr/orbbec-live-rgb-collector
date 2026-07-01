@echo off
setlocal
cd /d "%~dp0"

set "EXPECTED_ENV=vision-data"
set "ENV_PYTHON=%USERPROFILE%\miniconda3\envs\vision-data\python.exe"
set "PYTHON_EXE="

echo Current conda env: %CONDA_DEFAULT_ENV%
if /I "%CONDA_DEFAULT_ENV%"=="%EXPECTED_ENV%" if exist "%CONDA_PREFIX%\python.exe" (
    set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
)

if not defined PYTHON_EXE if exist "%ENV_PYTHON%" (
    set "PYTHON_EXE=%ENV_PYTHON%"
    echo [INFO] Using vision-data Python directly; conda activation is not required here.
)

if not defined PYTHON_EXE (
    set "PYTHON_EXE=python"
    echo [WARN] This launcher does not activate conda automatically.
    echo [WARN] Expected env: %EXPECTED_ENV%
    echo [WARN] Current env : %CONDA_DEFAULT_ENV%
    echo [WARN] If needed, run: conda activate %EXPECTED_ENV%
)

echo Using Python:
"%PYTHON_EXE%" -c "import sys; print(sys.executable)"

"%PYTHON_EXE%" "%~dp0collect_orbbec_rgb_dataset_gui.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

pause
exit /b %EXIT_CODE%
