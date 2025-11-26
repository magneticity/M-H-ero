@echo off
REM Complete setup script for M(H)ero on Windows
REM This script sets up the environment and builds the application

echo ================================================
echo   M(H)ero Build Setup (Windows)
echo ================================================
echo.

REM Check if Python 3 is available
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python 3 is not installed or not in PATH.
    echo   Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo + Python found: %PYTHON_VERSION%
echo.

REM Create virtual environment if it doesn't exist
if not exist "env" (
    echo Creating virtual environment...
    python -m venv env
    echo + Virtual environment created
) else (
    echo + Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call env\Scripts\activate.bat
echo + Virtual environment activated
echo.

REM Install/upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo + pip upgraded
echo.

REM Install dependencies
echo Installing dependencies...
echo   - PySide6
echo   - matplotlib
echo   - numpy
echo   - pandas
pip install PySide6 matplotlib numpy pandas --quiet
echo + Dependencies installed
echo.

REM Install PyInstaller
echo Installing PyInstaller...
pip install pyinstaller --quiet
echo + PyInstaller installed
echo.

REM Clean previous builds
if exist "build" (
    echo Cleaning previous builds...
    rmdir /s /q build
)
if exist "dist" (
    rmdir /s /q dist
)
echo + Build directories cleaned
echo.

REM Build the application
echo ================================================
echo   Building M(H)ero Application
echo ================================================
echo.
echo This may take a few minutes...
echo.

pyinstaller "M(H)ero.spec" --clean

echo.
echo ================================================
echo   Build Complete!
echo ================================================
echo.

if exist "dist\M(H)ero.exe" (
    echo + Windows application built successfully
    echo.
    echo Location: dist\M(H^)ero.exe
    echo.
    echo To test the application:
    echo   dist\M(H^)ero.exe
    echo.
    echo To distribute:
    echo   1. Compress the dist folder to M(H^)ero.zip
    echo   2. Share the zip file
) else (
    echo X Build may have failed. Check output above for errors.
    pause
    exit /b 1
)

echo.
pause
