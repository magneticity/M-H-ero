import sys
import io
import pandas as pd
import matplotlib
import numpy as np

from PySide6 import QtWidgets, QtGui, QtCore

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.setParent(parent)

    def plot_xy(self, x, y, xlabel="X", ylabel="Y", title=""):
        self.ax.clear()
        self.ax.plot(x, y, linewidth=1.5)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if title:
            self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        self.fig.canvas.draw_idle()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Magnetometry Data Viewer")
        self.resize(1000, 640)

        # --- central layout (controls + plot) ---
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        # Controls row
        controls = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(controls)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        self.xCombo = QtWidgets.QComboBox()
        self.yCombo = QtWidgets.QComboBox()
        self.swapBtn = QtWidgets.QToolButton()
        self.swapBtn.setText("⇄ Swap")
        self.autoRescaleChk = QtWidgets.QCheckBox("Auto-rescale")
        self.autoRescaleChk.setChecked(True)

        h.addWidget(QtWidgets.QLabel("X:"))
        h.addWidget(self.xCombo, 1)
        h.addWidget(QtWidgets.QLabel("Y:"))
        h.addWidget(self.yCombo, 1)
        h.addWidget(self.swapBtn)
        h.addStretch(1)
        h.addWidget(self.autoRescaleChk)

        vbox.addWidget(controls)

        # Plot area
        self.canvas = PlotCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.canvas, 1)

        # Status bar
        self.status = self.statusBar()
        self.status.showMessage("Ready")

        # Menu
        file_menu = self.menuBar().addMenu("&File")
        open_act = QtGui.QAction("&Open…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self.open_file)
        file_menu.addAction(open_act)

        quit_act = QtGui.QAction("&Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        process_menu = self.menuBar().addMenu("&Process")

        center_y_act = QtGui.QAction("Center &Y about 0", self)
        center_y_act.setStatusTip("Subtract mean of current Y column so it is centered at zero")
        center_y_act.triggered.connect(self.center_y_about_zero)
        process_menu.addAction(center_y_act)

        export_menu = self.menuBar().addMenu("&Export")
        export_hist_act = QtGui.QAction("Export &History…", self)
        export_hist_act.triggered.connect(self.export_history)
        export_menu.addAction(export_hist_act)

        # Data
        self.original_df = None    # raw data from file
        self.df = None             # current, modified data
        self.numeric_cols = []
        self.last_path = None
        self.history = []          # list of dicts describing operations
        self._wiring_done = False

        # Wire UI interactions
        self.swapBtn.clicked.connect(self._swap_axes)
        # Delay connecting combos until we populate them (prevents spurious plots)

    # ---------- File loading ----------
    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open data file",
            self.last_path or "",
            "Data files (*.csv *.txt *.dat *.tsv *.vhd);;All files (*)",
        )
        if not path:
            return
        
        # Always clear current state on new file
        self.df = None
        self.original_df = None
        self.history = []

        try:
            self.status.showMessage(f"Loading: {path}")
            df = self._read_table_auto(path)

            # Keep original copy + working copy
            self.original_df = df.copy(deep=True)
            self.df = df

            self.last_path = path

            # Determine numeric columns
            import numpy as np
            self.numeric_cols = [c for c in self.df.columns if np.issubdtype(self.df[c].dtype, np.number)]

            if len(self.numeric_cols) < 2:
                raise ValueError("Need at least two numeric columns to plot.")

            # Populate combos
            self._populate_combos()

            # Initial plot: Y vs X using first two numeric columns
            self._replot()
            self.status.showMessage(f"Loaded: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))
            self.status.showMessage("Error")

    def _populate_combos(self):
        self.xCombo.blockSignals(True)
        self.yCombo.blockSignals(True)

        self.xCombo.clear()
        self.yCombo.clear()
        self.xCombo.addItems([str(c) for c in self.numeric_cols])
        self.yCombo.addItems([str(c) for c in self.numeric_cols])

        # Default: first → X, second → Y (or preserve prior selections if still valid)
        if len(self.numeric_cols) >= 2:
            self.xCombo.setCurrentIndex(0)
            self.yCombo.setCurrentIndex(1)

        self.xCombo.blockSignals(False)
        self.yCombo.blockSignals(False)

        # Connect signals once
        if not self._wiring_done:
            self.xCombo.currentIndexChanged.connect(self._replot)
            self.yCombo.currentIndexChanged.connect(self._replot)
            self.autoRescaleChk.toggled.connect(self._replot)
            self._wiring_done = True
    
    # ---------- History export ----------
    def export_history(self):
        if not self.history:
            QtWidgets.QMessageBox.information(self, "No history",
                                              "There are no operations to export yet.")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save history as JSON",
            self.last_path or "",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return

        import json
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
            self.status.showMessage(f"History exported to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export error", str(e))
            self.status.showMessage("Error exporting history")


    # ---------- Plotting ----------
    def _replot(self):
        if self.df is None or not self.numeric_cols:
            return
        x_name = self.xCombo.currentText()
        y_name = self.yCombo.currentText()
        if not x_name or not y_name:
            return
        if x_name == y_name:
            self.status.showMessage("X and Y are the same column; nothing to plot.")
            self.canvas.ax.clear()
            self.canvas.fig.canvas.draw_idle()
            return

        x = self.df[x_name].to_numpy()
        y = self.df[y_name].to_numpy()

        # Drop NaNs together to keep paired data aligned
        import numpy as np
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        # Draw
        self.canvas.ax.clear()
        self.canvas.ax.plot(x, y, linewidth=1.5)
        self.canvas.ax.set_xlabel(x_name)
        self.canvas.ax.set_ylabel(y_name)
        self.canvas.ax.set_title(self.windowTitle())
        self.canvas.ax.grid(True, alpha=0.3)

        if self.autoRescaleChk.isChecked():
            self.canvas.ax.relim()
            self.canvas.ax.autoscale()

        self.canvas.fig.canvas.draw_idle()
        self.status.showMessage(f"Plotted {y_name} vs {x_name} ({len(x)} points)")

    def _swap_axes(self):
        if self.xCombo.count() == 0 or self.yCombo.count() == 0:
            return
        xi = self.xCombo.currentIndex()
        yi = self.yCombo.currentIndex()
        if xi < 0 or yi < 0:
            return
        self.xCombo.blockSignals(True)
        self.yCombo.blockSignals(True)
        self.xCombo.setCurrentIndex(yi)
        self.yCombo.setCurrentIndex(xi)
        self.xCombo.blockSignals(False)
        self.yCombo.blockSignals(False)
        self._replot()

    def _read_table_auto(self, path):
        import io, re, csv
        import pandas as pd
        import numpy as np

        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()

        # --- 1) Find where numeric data begins ---
        def split_guess(s):
            return [t for t in re.split(r"[,\t; ]+", s.strip()) if t != ""]

        def is_number(tok):
            try:
                float(tok.replace("D", "E").replace(",", "."))  # allow Fortran D and decimal commas
                return True
            except Exception:
                return False

        def numeric_fraction(s):
            toks = split_guess(s)
            if len(toks) < 2: return 0.0
            return sum(is_number(t) for t in toks) / len(toks)

        # Honor explicit markers like [Data], or "new section: section 0:" for the DMS MkII
        data_start = None
        for i, ln in enumerate(lines):
            if ln.strip().lower() == "[data]" or ln.strip().lower() == "new section: section 0:": 
                if ln.strip().lower() == "new section: section 0:":
                    DMS_marker = True # True if DMS MkII style marker found
                print(f"Found data start marker at line {i + 1}")
                data_start = i + 1
                break

        if data_start is None:
            for i, ln in enumerate(lines):
                ls = ln.strip()
                if not ls: 
                    continue
                if ls.startswith(("#", "%", "*", "//", "/*", "@")):
                    continue
                if numeric_fraction(ls) >= 0.8: # If no explicit marker found, use first line with high numeric content
                    print(f"Found data start candidate at line {i}")
                    data_start = i
                    break

        if data_start is None:
            raise ValueError("Could not locate the start of numeric data.")

        # Optional header row just before numeric block?
        header_idx = None
        if data_start > 0:
            if DMS_marker != True: # If not DMK MkII style marker, check previous line for header
                prev = lines[data_start - 1].strip()
                if prev and (numeric_fraction(prev) < 0.5) and not prev.startswith(("#", "%", "*", "//", "/*", "@")):
                    header_idx = data_start - 1
            else: # For DMS MkII style files, use four lines above
                prev = lines[data_start - 4].strip()
                if prev and (numeric_fraction(prev) < 0.5) and not prev.startswith(("#", "%", "*", "//", "/*", "@")):
                    header_idx = data_start - 4

        start_idx = header_idx if header_idx is not None else data_start
        tail = [ln.rstrip("\n") for ln in lines[start_idx:]]

        # --- 2) Pick the best delimiter by consistency over first ~50 data rows ---
        candidates = [",", "\t", ";", r"\s+"]  # treat whitespace as a single delimiter
        def score_sep(sep_regex):
            widths = []
            checked = 0
            for ln in tail:
                s = ln.strip()
                if not s or s.startswith(("#", "%")):
                    continue
                parts = re.split(sep_regex, s)
                # ignore obvious inline comments after a delimiter
                if "#" in parts[-1]:
                    parts[-1] = parts[-1].split("#", 1)[0].strip()
                widths.append(len([p for p in parts if p != ""]))
                checked += 1
                if checked >= 50:
                    break
            if not widths:
                return (0, 0, 0)
            return (np.median(widths), -np.std(widths), -abs(widths[0]-np.median(widths)))  # higher is better, then lower variance

        sep_regex = max(candidates, key=score_sep)
        sep_for_pandas = sep_regex if sep_regex == r"\s+" else sep_regex

        # Heuristic: if we picked ';' and see many commas inside tokens, likely decimal commas → keep ';' as sep
        # If we picked ',' but also see many ';', we might have chosen wrong; the consistency score should have handled this.

        # --- 3) Build text, keep header if detected ---
        text = "\n".join(tail)

        # --- 4) Read robustly: let pandas infer header if we found one; skip malformed lines ---
        # Also trim inline comments and extra spaces after delimiters.
        df = pd.read_csv(
            io.StringIO(text),
            sep=sep_for_pandas,
            engine="python",
            header=0 if header_idx is not None else None,
            comment="#",                # drop trailing inline comments
            skipinitialspace=True,      # "x,  y" → treat like "x,y"
            on_bad_lines="skip",        # skip ragged rows instead of raising
            quoting=csv.QUOTE_MINIMAL
        )

        # Convert decimal commas to dots in object columns that look numeric
        for c in df.columns:
            if df[c].dtype == object:
                # only convert if >=80% of non-null cells look like numbers with comma or dot
                ser = df[c].dropna().astype(str)
                if not ser.empty and (ser.str.contains(r"^[\s\-\+\d\.,EeDd]+$").mean() > 0.8):
                    df[c] = pd.to_numeric(ser.str.replace(",", ".", regex=False), errors="coerce")
            # already numeric → keep
        # Drop empty cols/rows
        df = df.dropna(axis=1, how="all").dropna(how="all")
        if df.empty:
            raise ValueError("Parsed an empty table after handling headers and bad lines.")
        return df
    
    # ---------- Data processing ----------
    
    # Shift about y=0
    def center_y_about_zero(self):
        """Shift the currently selected Y column so its mean is 0."""
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load data before applying corrections.")
            return

        y_name = self.yCombo.currentText()
        if not y_name:
            QtWidgets.QMessageBox.warning(self, "No Y column", "Select a Y column first.")
            return

        col = self.df[y_name]
        # Only operate on numeric data
        if not hasattr(col, "dtype") or not np.issubdtype(col.dtype, np.number):
            QtWidgets.QMessageBox.warning(self, "Not numeric",
                                          f"Column '{y_name}' is not numeric.")
            return

        # Compute offset (mean) and apply
        offset = float(col.mean(skipna=True))
        self.df[y_name] = col - offset

        # Record in history
        self._add_history_entry(
            op="center_y",
            params={
                "column": y_name,
                "offset": offset,
            },
        )

        self.status.showMessage(f"Centered {y_name} about 0 (offset {offset:.4g})")
        self._replot()

    def _add_history_entry(self, op, params):
        """Append an operation to the history log."""
        from datetime import datetime

        entry = {
            "op": op,
            "params": params or {},
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.history.append(entry)
        # optional: print to console for debugging
        # print("HISTORY:", entry)

    

def _postprocess_df(df):
    import pandas as pd
    import numpy as np
    # Coerce numeric where possible (without clobbering mixed text columns)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    # Drop fully-empty cols/rows
    df = df.dropna(axis=1, how="all").dropna(how="all")
    if df.empty:
        raise ValueError("Parsed an empty table after skipping headers.")
    return df


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
