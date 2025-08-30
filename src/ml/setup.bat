@echo off
echo ==============================
echo Hallucination Detection Setup
echo ==============================
echo.

REM Set UTF-8 encoding for Python
set PYTHONIOENCODING=utf-8

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Run the setup script
echo Running Python setup script...
python setup.py

echo.
echo Setup completed. Press any key to exit...
pause >nul
