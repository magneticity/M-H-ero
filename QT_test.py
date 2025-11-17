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

        self.bgApplyBtn = QtWidgets.QPushButton("Apply BG")
        self.bgCancelBtn = QtWidgets.QPushButton("Cancel BG")
        self.bgApplyBtn.setVisible(False)
        self.bgCancelBtn.setVisible(False)

        h.addWidget(self.bgApplyBtn)
        h.addWidget(self.bgCancelBtn)

        self.bgApplyBtn.clicked.connect(self._bg_commit)
        self.bgCancelBtn.clicked.connect(self._bg_cancel)

        # Background-mode state
        self.bg_mode_active = False
        self._bg_df_before = None   # df snapshot before starting BG mode
        self._bg_x_col = None
        self._bg_y_col = None
        self._bg_threshold = None

        self._bg_vline = None
        self._bg_cid_press = None
        self._bg_cid_motion = None
        self._bg_cid_release = None

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

        process_menu.addSeparator()

        bg_act = QtGui.QAction("Linear background (high field)…", self)
        bg_act.setStatusTip("Fit a straight line to the high-field region and subtract it")
        bg_act.triggered.connect(self._bg_start_mode)
        process_menu.addAction(bg_act)

        undo_act = QtGui.QAction("&Undo last operation", self)
        undo_act.setShortcut("Ctrl+Z")
        undo_act.setStatusTip("Undo the last data correction")
        undo_act.triggered.connect(self.undo_last_operation)
        process_menu.addAction(undo_act)

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
    
    # ---------- Data operations ----------
    def _apply_operation(self, op, params, record=True):
        """
        Apply a single operation to self.df.

        op      : string, e.g. "center_y"
        params  : dict, operation-specific parameters
        record  : if True, also append to self.history
        """
        if self.df is None:
            return

        if op == "center_y":
            col_name = params.get("column")
            if col_name not in self.df.columns:
                return

            col = self.df[col_name]

            if not hasattr(col, "dtype") or not np.issubdtype(col.dtype, np.number):
                return

            # Use given offset if present, otherwise recompute
            if "offset" in params:
                offset = float(params["offset"])
            else:
                offset = float(col.mean(skipna=True))

            self.df[col_name] = col - offset

            if record:
                self._add_history_entry(
                    op="center_y",
                    params={"column": col_name, "offset": offset},
                )

        elif op == "bg_linear_branches":
            xcol = params.get("x_column")
            ycol = params.get("column")
            m_bg = float(params.get("m_bg", 0.0))

            if xcol not in self.df.columns or ycol not in self.df.columns:
                return

            x = np.asarray(self.df[xcol].to_numpy(), dtype=float)
            y = np.asarray(self.df[ycol].to_numpy(), dtype=float)

            finite = np.isfinite(x) & np.isfinite(y)
            y_corr = y.copy()
            y_corr[finite] = y[finite] - m_bg * x[finite]

            self.df[ycol] = y_corr

            if record:
                self._add_history_entry("bg_linear_branches", params)

        # ... more operations in future
        # elif op == "drift_correct": ...
        # (add more operations here as you implement them)

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

        # Delegate to operation dispatcher (record=True by default)
        self._apply_operation("center_y", {"column": y_name}, record=True)

        # Grab offset of last operation for status text (optional)
        if self.history and self.history[-1]["op"] == "center_y":
            offset = self.history[-1]["params"].get("offset", 0.0)
            self.status.showMessage(f"Centered {y_name} about 0 (offset {offset:.4g})")
        else:
            self.status.showMessage(f"Centered {y_name} about 0")

        self._replot()

    def _compute_bg_corrected(self, df, xcol, ycol, threshold):
        """
        Fit a straight line to the high-field *branches* of a hysteresis loop:

        - Positive branch:   x >=  +threshold
        - Negative branch:   x <=  -threshold

        Each branch is fit separately:
        y_pos ≈ m_pos * x + b_pos
        y_neg ≈ m_neg * x + b_neg

        The background slope m_bg is taken as the average of m_pos and m_neg.
        Only this slope is subtracted:

        y_corr = y - m_bg * x
        """
        x = np.asarray(df[xcol].to_numpy(), dtype=float)
        y = np.asarray(df[ycol].to_numpy(), dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 2:
            raise ValueError("Not enough finite points in the data.")

        # Ensure threshold is positive
        thr = float(abs(threshold))

        mask_pos = finite & (x >= thr)
        mask_neg = finite & (x <= -thr)

        if mask_pos.sum() < 2 or mask_neg.sum() < 2:
            raise ValueError("Not enough high-field points on one or both branches.")

        x_pos = x[mask_pos]
        y_pos = y[mask_pos]
        x_neg = x[mask_neg]
        y_neg = y[mask_neg]

        # Guard against nearly constant field region in either branch
        if np.allclose(x_pos, x_pos[0]):
            raise ValueError("Positive high-field branch has almost constant H; cannot fit a line.")
        if np.allclose(x_neg, x_neg[0]):
            raise ValueError("Negative high-field branch has almost constant H; cannot fit a line.")

        m_pos, b_pos = np.polyfit(x_pos, y_pos, 1)
        m_neg, b_neg = np.polyfit(x_neg, y_neg, 1)

        # Background slope: average of the two branch slopes
        m_bg = 0.5 * (m_pos + m_neg)

        # Subtract only the background slope
        y_corr = y - m_bg * x

        info = {
            "m_pos": float(m_pos),
            "b_pos": float(b_pos),
            "m_neg": float(m_neg),
            "b_neg": float(b_neg),
            "m_bg": float(m_bg),
            "threshold": thr,
        }
        return y_corr, info


    def _bg_start_mode(self):
        """Enter interactive background-subtraction mode."""
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load data before background subtraction.")
            return

        x_name = self.xCombo.currentText()
        y_name = self.yCombo.currentText()
        if not x_name or not y_name:
            QtWidgets.QMessageBox.warning(self, "Select columns",
                                          "Select X and Y columns before background subtraction.")
            return

        # Snapshot current df (this is the state we'll operate on)
        self._bg_df_before = self.df.copy(deep=True)
        self._bg_x_col = x_name
        self._bg_y_col = y_name

        x = self._bg_df_before[x_name].to_numpy()
        if x.size == 0:
            QtWidgets.QMessageBox.warning(self, "No data", "Selected X column is empty.")
            return

        # Choose an initial threshold: e.g. 70% of max |x|
        self._bg_threshold = 0.7 * float(np.nanmax(np.abs(x)))
        
        self.bg_mode_active = True
        self.bgApplyBtn.setVisible(True)
        self.bgCancelBtn.setVisible(True)
        self.status.showMessage("Background mode: drag the vertical line to set high-field threshold, then click 'Apply BG'.")

        # Create vertical line
        ax = self.canvas.ax
        if self._bg_vline is not None:
            self._bg_vline.remove()
        self._bg_vline = ax.axvline(self._bg_threshold, linestyle="--")

        # Connect matplotlib events for dragging
        canvas = self.canvas
        self._bg_cid_press = canvas.mpl_connect("button_press_event", self._bg_on_press)
        self._bg_cid_motion = canvas.mpl_connect("motion_notify_event", self._bg_on_motion)
        self._bg_cid_release = canvas.mpl_connect("button_release_event", self._bg_on_release)
        self._bg_dragging = False

        # Draw initial preview
        self._bg_update_preview()

    def _bg_on_press(self, event):
        if not self.bg_mode_active or event.inaxes != self.canvas.ax:
            return
        if event.xdata is None:
            return

        # Check if click is near the vertical line
        x_line = self._bg_threshold
        tol = 0.02 * (self.canvas.ax.get_xlim()[1] - self.canvas.ax.get_xlim()[0])
        if abs(event.xdata - x_line) < tol:
            self._bg_dragging = True

    def _bg_on_motion(self, event):
        if not self.bg_mode_active or not self._bg_dragging:
            return
        if event.inaxes != self.canvas.ax or event.xdata is None:
            return

        # Update threshold and preview
        self._bg_threshold = float(event.xdata)
        self._bg_update_preview()

    def _bg_on_release(self, event):
        if not self.bg_mode_active:
            return
        self._bg_dragging = False

    def _bg_update_preview(self):
        if not self.bg_mode_active or self._bg_df_before is None:
            return

        df = self._bg_df_before
        xcol = self._bg_x_col
        ycol = self._bg_y_col
        thr = self._bg_threshold

        x = df[xcol].to_numpy()

        ax = self.canvas.ax
        ax.clear()

        try:
            y_corr, info = self._compute_bg_corrected(df, xcol, ycol, thr)
            y_plot = y_corr
            m_bg = info["m_bg"]
            title_extra = f"  (m_bg={m_bg:.3g}, m+={info['m_pos']:.3g}, m-={info['m_neg']:.3g})"
        except Exception as e:
            # If fit fails (threshold too large, etc.), just show original data
            y_plot = df[ycol].to_numpy()
            title_extra = f"  (BG fit invalid: {e})"

        # Plot in *original* H order (so loop shape is preserved)
        ax.plot(x, y_plot, linewidth=1.5)

        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol + " (preview, BG-subtracted)")
        ax.set_title("Preview: background subtraction" + title_extra)
        ax.grid(True, alpha=0.3)

        # Draw threshold as symmetric ±thr guides
        self._bg_vline = ax.axvline(+abs(thr), linestyle="--")
        ax.axvline(-abs(thr), linestyle="--")

        # Optional: shade high-field regions
        x_max = np.nanmax(np.abs(x[np.isfinite(x)]))
        ax.axvspan(+abs(thr), +x_max, alpha=0.1)
        ax.axvspan(-x_max, -abs(thr), alpha=0.1)

        self.canvas.fig.canvas.draw_idle()
        self.status.showMessage(
            f"Background mode: |{xcol}| >= {abs(thr):.4g}; drag line, then 'Apply BG'"
        )

    def _bg_disconnect_events(self):
        if self._bg_cid_press is not None:
            self.canvas.mpl_disconnect(self._bg_cid_press)
        if self._bg_cid_motion is not None:
            self.canvas.mpl_disconnect(self._bg_cid_motion)
        if self._bg_cid_release is not None:
            self.canvas.mpl_disconnect(self._bg_cid_release)
        self._bg_cid_press = self._bg_cid_motion = self._bg_cid_release = None

    def _bg_commit(self):
        if not self.bg_mode_active or self._bg_df_before is None:
            return

        df_before = self._bg_df_before
        xcol = self._bg_x_col
        ycol = self._bg_y_col
        thr = self._bg_threshold

        try:
            y_corr, info = self._compute_bg_corrected(df_before, xcol, ycol, thr)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Cannot apply background", str(e))
            return

        # Commit: replace current df with BG-subtracted version
        self.df = df_before.copy(deep=True)
        self.df[ycol] = y_corr

        # Log operation so undo/replay works
        self._add_history_entry(
            op="bg_linear_branches",
            params={
                "x_column": xcol,
                "column": ycol,
                "threshold": info["threshold"],
                "m_pos": info["m_pos"],
                "b_pos": info["b_pos"],
                "m_neg": info["m_neg"],
                "b_neg": info["b_neg"],
                "m_bg": info["m_bg"],
            },
        )

        self._bg_exit_mode()
        self.status.showMessage(
            f"Applied BG: |{xcol}|>={info['threshold']:.4g}, m_bg={info['m_bg']:.3g}"
        )
        self._replot()

    def _bg_cancel(self):
        """User clicks 'Cancel BG' – discard preview and restore df."""
        if not self.bg_mode_active:
            return

        # Just restore df_before and exit
        if self._bg_df_before is not None:
            self.df = self._bg_df_before

        self._bg_exit_mode()
        self.status.showMessage("Background subtraction canceled.")
        self._replot()

    def _bg_exit_mode(self):
        """Common cleanup for leaving background mode."""
        self.bg_mode_active = False
        self.bgApplyBtn.setVisible(False)
        self.bgCancelBtn.setVisible(False)
        self._bg_disconnect_events()

        self._bg_df_before = None
        self._bg_x_col = None
        self._bg_y_col = None
        self._bg_threshold = None

        if self._bg_vline is not None:
            try:
                self._bg_vline.remove()
            except Exception:
                pass
            self._bg_vline = None


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

    def undo_last_operation(self):
        """Remove the last operation from history and rebuild df from scratch."""
        if not self.history:
            QtWidgets.QMessageBox.information(self, "Nothing to undo",
                                              "There are no operations to undo.")
            return

        # Remove last operation
        last = self.history.pop()

        # Rebuild df from original + remaining history
        self._rebuild_df_from_history()

        # Update status + plot
        self.status.showMessage(f"Undid: {last['op']}")
        self._replot()

    def _rebuild_df_from_history(self):
        """Reset df to original_df and replay all history entries."""
        if self.original_df is None:
            return

        # Start from a clean copy of the raw data
        self.df = self.original_df.copy(deep=True)

        # Recompute numeric columns
        self.numeric_cols = [
            c for c in self.df.columns if np.issubdtype(self.df[c].dtype, np.number)
        ]

        # Try to keep current X/Y choices if they still exist
        old_x = self.xCombo.currentText() if self.xCombo.count() > 0 else None
        old_y = self.yCombo.currentText() if self.yCombo.count() > 0 else None

        # Re-apply all operations without re-recording them
        for entry in self.history:
            self._apply_operation(entry["op"], entry["params"], record=False)

        # Repopulate combos if needed
        # (e.g. after operations that add/remove columns in the future)
        if not self.numeric_cols:
            return

        # Ensure combos list the right columns
        self._populate_combos()

        # Try to restore previous x/y selection when still valid
        def restore_combo(combo, name):
            if name and combo.findText(name) != -1:
                combo.setCurrentText(name)

        restore_combo(self.xCombo, old_x)
        restore_combo(self.yCombo, old_y)


    



def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
