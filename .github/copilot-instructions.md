# SQUiiD Project - AI Agent Instructions

## Project Overview
SQUiiD is a Qt-based scientific data visualization application focused on magnetometry data viewing and analysis. The project uses PySide6 for the GUI and matplotlib for plotting capabilities.

## Key Components

### GUI Architecture
- Main application window (`MainWindow` class in `QT_test.py`)
- Custom plotting canvas (`PlotCanvas` class) extending matplotlib's `FigureCanvasQTAgg`
- Standard Qt components: menus, status bar, and file dialogs

### Data Handling
- Uses pandas DataFrames for data management
- Robust CSV/TSV/DAT file parsing with automatic delimiter detection
- Supports various numeric formats including Fortran-style notation and decimal commas
- Automatic detection of numeric columns for plotting

## Development Patterns

### File Parsing Strategy
The `_read_table_auto` method implements sophisticated data file parsing with these key features:
- Auto-detection of data section start (after headers/comments)
- Smart delimiter detection (`,`, `\t`, `;`, spaces)
- Support for explicit data markers (`[Data]`, `New Section: Section 0:`)
- Handles inline comments and varying number formats

### GUI Updates
- Plot updates use `draw_idle()` for efficient rendering
- Status bar provides user feedback during operations
- Error handling uses Qt's message boxes for user notification

## Dependencies
- PySide6: Qt GUI framework
- matplotlib: Plotting library with Qt backend
- pandas: Data manipulation
- numpy: Numerical operations

## Project-Specific Conventions
1. Error Handling:
   - GUI operations wrap exceptions in user-friendly dialog boxes
   - Status bar updates reflect current operation state

2. Data Processing:
   - All file reading operations use UTF-8 with error ignoring
   - Numeric columns are automatically coerced when possible
   - Empty rows and columns are automatically dropped

3. Plot Management:
   - Plots are automatically sized using `tight_layout`
   - Grid lines are added with 30% transparency
   - First two numeric columns are used as default X/Y data

## Key Files
- `QT_test.py`: Main application file containing all core functionality