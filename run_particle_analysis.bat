@echo off
setlocal

chcp 65001 >nul
set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

set "PYTHON_EXE="
set "PYTHON_ARGS="

if exist "%APP_DIR%.venv\Scripts\python.exe" (
    "%APP_DIR%.venv\Scripts\python.exe" -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=%APP_DIR%.venv\Scripts\python.exe"
        set "PYTHON_ARGS="
    )
)

if not defined PYTHON_EXE (
    py -3 -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3"
    )
)

if not defined PYTHON_EXE (
    python -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=python"
        set "PYTHON_ARGS="
    )
)

if not defined PYTHON_EXE (
    echo Python was not found.
    echo Install Python 3 and dependencies from requirements.txt.
    pause
    exit /b 1
)

echo Using Python: %PYTHON_EXE% %PYTHON_ARGS%

"%PYTHON_EXE%" %PYTHON_ARGS% -c "import PyQt5, cv2, numpy" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Missing required packages.
    echo Run this command from the project folder:
    echo "%PYTHON_EXE%" %PYTHON_ARGS% -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

"%PYTHON_EXE%" %PYTHON_ARGS% "%APP_DIR%gui\main_window.py"
if errorlevel 1 (
    echo.
    echo ParticleAnalysis exited with an error.
    pause
    exit /b 1
)

endlocal
