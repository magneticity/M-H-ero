# M(H)ero - Hysteresis Loop Analysis Tool

## For Users

### Running the Application

**macOS:**
1. Download `M(H)ero.zip`
2. Extract the zip file
3. Double-click `M(H)ero.app` to run
4. If macOS blocks it (security), right-click → Open, then click "Open" in the dialog

**Windows:**
1. Download `M(H)ero.zip`
2. Extract the zip file
3. Double-click `M(H)ero.exe` to run

### Quick Start
1. Click "📁 Open File" to load hysteresis data
2. Click "⚡ Auto-Process" for automatic drift and background correction
3. Use the Analysis menu for coercivity, remanence, and anisotropy calculations

---

## For Developers

### Building from Source

**Prerequisites:**
- Python 3.8 or higher
- Virtual environment with dependencies

**Setup:**
```bash
# Clone the repository
git clone https://github.com/magneticity/M-H-ero.git
cd M-H-ero

# Create and activate virtual environment (if not already done)
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install PySide6 matplotlib numpy pandas

# Run the application
python "M(H)ero.py"
```

**Building Executable:**
```bash
# Install PyInstaller
pip install pyinstaller

# Run the build script
./build_app.sh

# The executable will be in the 'dist' folder
```

**Distribution Checklist:**
- [ ] Test the executable on a clean machine (without Python installed)
- [ ] Include sample data files
- [ ] Test auto-process feature
- [ ] Test history export/import
- [ ] Verify logo displays correctly

---

## Project Structure
```
M(H)ero/
├── M(H)ero.py          # Main application
├── Logo/               # Application logo
│   └── logo_white.png
├── env/                # Virtual environment (not distributed)
├── build_app.sh        # Build script
└── README.md           # This file
```

## Dependencies
- PySide6 (Qt for Python)
- matplotlib
- numpy
- pandas

## License
[Add your license here]
