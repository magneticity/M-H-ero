# M(H)ero - Hysteresis Loop Analysis Tool

A GUI application for analyzing magnetic hysteresis loops with automated drift and background correction.

---

## For End Users

### Download Pre-built Application

**Option 1: Download Release (Recommended)**
1. Go to [Releases](https://github.com/magneticity/M-H-ero/releases)
2. Download the version for your platform:
   - **macOS**: `M(H)ero-macOS.zip`
   - **Windows**: `M(H)ero-Windows.zip`
3. Extract and run:
   - **macOS**: Double-click `M(H)ero.app` (if blocked by security, right-click → Open)
   - **Windows**: Double-click `M(H)ero.exe` (if blocked by SmartScreen, click "More info" → "Run anyway")

**Option 2: Build from Source** (see Developer section below)

### Quick Start
1. Click "📁 Open File" to load hysteresis data
2. Click "⚡ Auto-Process" for automatic drift and background correction
3. Use the Analysis menu for coercivity, remanence, and anisotropy calculations

---

## For Developers

### Building from Source

**Prerequisites:**
- Python 3.8 or higher
- Git

**Quick Build (Recommended):**
```bash
# Clone the repository
git clone https://github.com/magneticity/M-H-ero.git
cd M-H-ero

# macOS/Linux:
./setup_and_build.sh

# Windows:
setup_and_build.bat

# The executable will be in the 'dist' folder
```

**Manual Setup:**
```bash
# Clone the repository
git clone https://github.com/magneticity/M-H-ero.git
cd M-H-ero

# Create and activate virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install PySide6 matplotlib numpy pandas

# Run the application directly
python "M(H)ero.py"
```

**Building Executable (Manual):**
```bash
# Install PyInstaller
pip install pyinstaller

# Run the build script
./build_simple.sh

# The executable will be in the 'dist' folder
```

**Running Directly (without building):**
```bash
# After installing dependencies
python "M(H)ero.py"
```
---

## Project Structure
```
M(H)ero/
├── M(H)ero.py              # Main application
├── M(H)ero.spec            # PyInstaller build configuration
├── Logo/                   # Application logo
│   └── logo_white.png
├── tests/                  # Test suite
├── tools/                  # Utility scripts
├── setup_and_build.sh      # Automated build (macOS/Linux)
├── setup_and_build.bat     # Automated build (Windows)
└── build_simple.sh         # Simple build script

Not in repo (generated locally):
├── env/                    # Virtual environment (created by build scripts)
├── build/                  # PyInstaller temporary files
└── dist/                   # Built executables (.app or .exe)
```

## Dependencies
- PySide6 (Qt for Python)
- matplotlib
- numpy
- pandas