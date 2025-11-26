#!/bin/bash
# Complete setup script for M(H)ero
# This script sets up the environment and builds the application

echo "================================================"
echo "  M(H)ero Build Setup"
echo "================================================"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv env
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install/upgrade pip
echo "Upgrading pip..."
./env/bin/python -m pip install --upgrade pip --quiet
echo "  ✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "  - PySide6"
echo "  - matplotlib"
echo "  - numpy"
echo "  - pandas"
./env/bin/pip install PySide6 matplotlib numpy pandas --quiet
echo "  ✓ Dependencies installed"
echo ""

# Install PyInstaller
echo "Installing PyInstaller..."
./env/bin/pip install pyinstaller --quiet
echo "  ✓ PyInstaller installed"
echo ""

# Clean previous builds
if [ -d "build" ] || [ -d "dist" ]; then
    echo "Cleaning previous builds..."
    rm -rf build dist
    echo "✓ Build directories cleaned"
    echo ""
fi

# Build the application
echo "================================================"
echo "  Building M(H)ero Application"
echo "================================================"
echo ""
echo "This may take a few minutes..."
echo ""

# Use the venv's pyinstaller explicitly
./env/bin/pyinstaller "M(H)ero.spec" --clean

echo ""
echo "================================================"
echo "  Build Complete!"
echo "================================================"
echo ""

if [ -d "dist/M(H)ero.app" ]; then
    echo "✓ macOS application built successfully"
    echo ""
    echo "Location: dist/M(H)ero.app"
    echo ""
    echo "To test the application:"
    echo "  open dist/M(H)ero.app"
    echo ""
    echo "To distribute:"
    echo "  cd dist && zip -r M(H)ero.zip M(H)ero.app"
elif [ -f "dist/M(H)ero" ]; then
    echo "✓ Application built successfully"
    echo ""
    echo "Location: dist/M(H)ero"
    echo ""
    echo "To test the application:"
    echo "  ./dist/M(H)ero"
else
    echo "❌ Build may have failed. Check output above for errors."
    exit 1
fi

echo ""
