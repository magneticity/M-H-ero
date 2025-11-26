#!/bin/bash
# Build script for M(H)ero application

# Activate virtual environment
source env/bin/activate

# Install PyInstaller if not already installed
pip install pyinstaller

# Clean previous builds
rm -rf build dist

# Build the application
# --name: Application name
# --windowed: Don't show console window
# --icon: Application icon (optional, you'd need to create one)
# --add-data: Include the Logo directory
# --onefile: Create a single executable (alternative: --onedir for faster startup)

pyinstaller --name="M(H)ero" \
    --windowed \
    --add-data="Logo:Logo" \
    --onefile \
    "M(H)ero.py"

echo ""
echo "Build complete! Application is in the 'dist' folder"
echo "You can distribute the entire 'dist/M(H)ero' (or 'dist/M(H)ero.app' on macOS)"
