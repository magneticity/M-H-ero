#!/bin/bash
# Simplified build script using the spec file

# Activate virtual environment
source env/bin/activate

# Install PyInstaller if not already installed
pip install pyinstaller

# Clean previous builds
rm -rf build dist

# Build using the spec file
pyinstaller "M(H)ero.spec"

echo ""
echo "✓ Build complete!"
echo ""
echo "Application location:"
if [ -d "dist/M(H)ero.app" ]; then
    echo "  macOS: dist/M(H)ero.app"
    echo ""
    echo "To test: open dist/M(H)ero.app"
else
    echo "  dist/M(H)ero"
fi
echo ""
echo "To distribute:"
echo "  1. Test the application on a clean machine"
echo "  2. Zip the dist folder: cd dist && zip -r M(H)ero.zip M(H)ero.app"
echo "  3. Share the zip file"
