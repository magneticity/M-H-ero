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

        self.chkShowMarkers = QtWidgets.QCheckBox("Show markers")
        self.chkShowMarkers.setChecked(True)   # default: ON
        h.addWidget(self.chkShowMarkers)
        self.chkShowMarkers.toggled.connect(self._replot)

        vbox.addWidget(controls)

        # Plot area
        self.canvas = PlotCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.canvas, 1)

        # Status bar
        self.status = self.statusBar()
        self.status.showMessage("Ready")

        # Physical parameters dock
        self.paramDock = QtWidgets.QDockWidget("Physical Parameters", self)
        self.paramDock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )

        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(4)

        self.lblHc = QtWidgets.QLabel("—")
        self.lblHcPlus = QtWidgets.QLabel("—")
        self.lblHcMinus = QtWidgets.QLabel("—")

        self.lblMr = QtWidgets.QLabel("—")
        self.lblMrPlus = QtWidgets.QLabel("—")
        self.lblMrMinus = QtWidgets.QLabel("—")

        self.lblMs = QtWidgets.QLabel("—")
        self.lblBgSlope = QtWidgets.QLabel("—")

        form.addRow("Hc (avg):", self.lblHc)
        form.addRow("Hc+:", self.lblHcPlus)
        form.addRow("Hc−:", self.lblHcMinus)

        form.addRow("Mr (mag):", self.lblMr)
        form.addRow("Mr+:", self.lblMrPlus)
        form.addRow("Mr−:", self.lblMrMinus)

        form.addRow("M\u209B (sat):", self.lblMs)      # Mₛ
        form.addRow("BG slope:", self.lblBgSlope)


        self.paramDock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.paramDock)

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
        center_y_act.setShortcut("Ctrl+Y")
        center_y_act.setStatusTip("Subtract mean of current Y column so it is centered at zero")
        center_y_act.triggered.connect(self.center_y_about_zero)
        process_menu.addAction(center_y_act)

        bg_act = QtGui.QAction("Linear background (high field)…", self)
        bg_act.setShortcut("Ctrl+B")
        bg_act.setStatusTip("Fit a straight line to the high-field region and subtract it")
        bg_act.triggered.connect(self._bg_start_mode)
        process_menu.addAction(bg_act)

        process_menu.addSeparator()

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

                # Draw markers if enabled
        if self.chkShowMarkers.isChecked():
            self._draw_feature_markers(x, y)

        self.canvas.fig.canvas.draw_idle()
        self.status.showMessage(f"Plotted {y_name} vs {x_name} ({len(x)} points)")
        
        self._update_parameters()

        self.canvas.fig.canvas.draw_idle()
        self.status.showMessage(f"Plotted {y_name} vs {x_name} ({len(x)} points)")
        self._update_parameters()

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
        Fit a straight line to the high-field branches of a hysteresis loop:

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
        """Recompute and display BG-subtracted data."""
        if not self.bg_mode_active or self._bg_df_before is None:
            return

        df = self._bg_df_before
        xcol = self._bg_x_col
        ycol = self._bg_y_col
        thr = self._bg_threshold

        x = np.asarray(df[xcol].to_numpy(), dtype=float)
        y = np.asarray(df[ycol].to_numpy(), dtype=float)

        ax = self.canvas.ax
        ax.clear()

        try:
            y_corr, info = self._compute_bg_corrected(df, xcol, ycol, thr)
            m_pos = info["m_pos"]
            b_pos = info["b_pos"]
            m_neg = info["m_neg"]
            b_neg = info["b_neg"]
            m_bg  = info["m_bg"]
            thr_abs = info["threshold"]

            title_extra = (
                f"  (m_bg={m_bg:.3g}, m+={m_pos:.3g}, m-={m_neg:.3g}, |H|≥{thr_abs:.3g})"
            )
        except Exception as e:
            # If fit fails, just show raw loop and bail on extras
            ax.plot(x, y, linewidth=1.5, alpha=0.6)
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.set_title(f"Preview: background subtraction (fit invalid: {e})")
            ax.grid(True, alpha=0.3)
            self.canvas.fig.canvas.draw_idle()
            return

        # 1) Raw loop (faint, in the background)
        ax.plot(x, y, linewidth=1.0, alpha=0.3, label="raw loop")

        # 2) BG-subtracted loop (preview)
        ax.plot(x, y_corr, linewidth=1.5, label="BG-subtracted (preview)")

        # 3) Dashed fit lines on ± high-field branches (over raw data)
        finite = np.isfinite(x) & np.isfinite(y)
        thr_abs = float(abs(thr_abs))

        mask_pos = finite & (x >= thr_abs)
        mask_neg = finite & (x <= -thr_abs)

        if mask_pos.sum() >= 2:
            x_pos = x[mask_pos]
            # Draw fit line only over the region actually used for the fit
            x_pos_line = np.linspace(x_pos.min(), x_pos.max(), 100)
            y_pos_fit = m_pos * x_pos_line + b_pos
            ax.plot(x_pos_line, y_pos_fit, linestyle="--", linewidth=1.0, label="fit (+H)")

        if mask_neg.sum() >= 2:
            x_neg = x[mask_neg]
            x_neg_line = np.linspace(x_neg.min(), x_neg.max(), 100)
            y_neg_fit = m_neg * x_neg_line + b_neg
            ax.plot(x_neg_line, y_neg_fit, linestyle="--", linewidth=1.0, label="fit (−H)")

        # 4) Symmetric vertical threshold guides + optional shading
        self._bg_vline = ax.axvline(+thr_abs, linestyle="--")
        ax.axvline(-thr_abs, linestyle="--")

        x_finite = x[finite]
        if x_finite.size > 0:
            x_max = float(np.nanmax(np.abs(x_finite)))
            ax.axvspan(+thr_abs, +x_max, alpha=0.05)
            ax.axvspan(-x_max, -thr_abs, alpha=0.05)

        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol + " / BG-corrected (preview)")
        ax.set_title("Preview: background subtraction" + title_extra)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        if self.chkShowMarkers.isChecked():
            self._draw_feature_markers(x, y_corr)

        self.canvas.fig.canvas.draw_idle()
        self.status.showMessage(
            f"Background mode: |{xcol}| ≥ {thr_abs:.4g}; drag line, then 'Apply BG'"
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


    def _compute_coercivity(self, x, y):
        """
        Estimate coercive fields from zero-crossings of y(x):

          - Hc+ : smallest positive field where M crosses zero
          - Hc− : largest negative field where M crosses zero
          - Hc  : average half-width, ~ (Hc+ - Hc−)/2

        Uses simple linear interpolation between sign changes.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]

        if x.size < 2:
            return None, None, None

        hc_candidates = []

        for i in range(len(x) - 1):
            y0, y1 = y[i], y[i + 1]
            x0, x1 = x[i], x[i + 1]

            # Exact zero at a point
            if y0 == 0:
                hc_candidates.append(x0)
                continue
            if y1 == 0:
                hc_candidates.append(x1)
                continue

            # Sign change between points
            if y0 * y1 < 0:
                # Linear interpolation to M=0
                x_zero = x0 - y0 * (x1 - x0) / (y1 - y0)
                hc_candidates.append(x_zero)

        if not hc_candidates:
            return None, None, None

        hc_candidates = np.array(hc_candidates, dtype=float)
        pos = hc_candidates[hc_candidates > 0]
        neg = hc_candidates[hc_candidates < 0]

        hc_plus = float(pos.min()) if pos.size else None
        hc_minus = float(neg.max()) if neg.size else None

        hc_avg = None
        if hc_plus is not None and hc_minus is not None:
            hc_avg = 0.5 * (hc_plus - hc_minus)
        elif hc_plus is not None:
            hc_avg = abs(hc_plus)
        elif hc_minus is not None:
            hc_avg = abs(hc_minus)

        return hc_plus, hc_minus, hc_avg

    def _compute_remanence(self, x, y, fallback_window_fraction=0.02):
        """
        Estimate remanent magnetisations using branch-specific interpolation
        at H = 0.

        Strategy:
          - Scan consecutive points for sign changes in H (x).
          - For each pair that brackets H = 0, linearly interpolate M at H = 0.
          - Use dH = x[i+1] - x[i] to decide branch:
                dH < 0  → descending branch (from +H to -H) → Mr+
                dH > 0  → ascending branch (from -H to +H) → Mr−
          - Mr (mag) ≈ (Mr+ - Mr−)/2 for a roughly symmetric loop.

        If no zero-crossings are found, falls back to a simple |H| window
        around 0 with width set by fallback_window_fraction.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]

        if x.size < 2:
            return None, None, None

        # Ensure we actually span H = 0
        if not (np.nanmin(x) <= 0.0 <= np.nanmax(x)):
            # No zero field in range → nothing meaningful to do
            return None, None, None

        Mr_plus_candidates = []   # descending branch (dH < 0)
        Mr_minus_candidates = []  # ascending branch (dH > 0)

        for i in range(len(x) - 1):
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[i], y[i + 1]

            # Skip degenerate segment
            if not np.isfinite(x0) or not np.isfinite(x1):
                continue
            if x1 == x0:
                continue

            # Does this segment bracket H=0?
            # (includes exact zeros and sign changes)
            if x0 == 0.0:
                x_zero = 0.0
                y_zero = float(y0)
            elif x1 == 0.0:
                x_zero = 0.0
                y_zero = float(y1)
            elif x0 * x1 < 0.0:
                # Linear interpolation to H=0
                x_zero = 0.0
                y_zero = float(y0 - x0 * (y1 - y0) / (x1 - x0))
            else:
                continue

            dH = x1 - x0
            if dH < 0:
                Mr_plus_candidates.append(y_zero)   # descending branch (from +H)
            elif dH > 0:
                Mr_minus_candidates.append(y_zero)  # ascending branch (from -H)
            # dH==0 already excluded above

        Mr_plus = np.mean(Mr_plus_candidates) if Mr_plus_candidates else None
        Mr_minus = np.mean(Mr_minus_candidates) if Mr_minus_candidates else None

        # If for some reason we didn’t find both branches, fall back to window method
        if Mr_plus is None or Mr_minus is None:
            max_abs_H = np.nanmax(np.abs(x))
            if max_abs_H > 0:
                h_window = fallback_window_fraction * max_abs_H
                mask = np.abs(x) <= h_window
                if mask.sum() >= 2:
                    y_window = y[mask]
                    Mr_plus_fb = float(np.nanmax(y_window))
                    Mr_minus_fb = float(np.nanmin(y_window))
                    if Mr_plus is None:
                        Mr_plus = Mr_plus_fb
                    if Mr_minus is None:
                        Mr_minus = Mr_minus_fb

        if Mr_plus is None and Mr_minus is None:
            return None, None, None

        Mr_mag = None
        if Mr_plus is not None and Mr_minus is not None:
            Mr_mag = 0.5 * (Mr_plus - Mr_minus)
        elif Mr_plus is not None:
            Mr_mag = abs(Mr_plus)
        elif Mr_minus is not None:
            Mr_mag = abs(Mr_minus)

        return (
            float(Mr_plus) if Mr_plus is not None else None,
            float(Mr_minus) if Mr_minus is not None else None,
            float(Mr_mag) if Mr_mag is not None else None,
        )


    def _update_parameters(self):
        """
        Recompute and display physical parameters for the selected
        X/Y columns from the current df and history.

        - Coercivity (Hc+, Hc−, Hc) whenever possible.
        - Background slope and M_sat only after a BG correction
          (last bg_linear_branches op for these columns).
        """
        # Defaults when nothing is available
        for lbl in [self.lblHc, self.lblHcPlus, self.lblHcMinus,
                    self.lblMr, self.lblMrPlus, self.lblMrMinus,
                    self.lblMs, self.lblBgSlope]:
            lbl.setText("—")

        if self.df is None:
            return

        x_name = self.xCombo.currentText()
        y_name = self.yCombo.currentText()
        if not x_name or not y_name:
            return

        x_series = self.df.get(x_name)
        y_series = self.df.get(y_name)
        if x_series is None or y_series is None:
            return

        if not (np.issubdtype(x_series.dtype, np.number) and
                np.issubdtype(y_series.dtype, np.number)):
            return

        x = np.asarray(x_series.to_numpy(), dtype=float)
        y = np.asarray(y_series.to_numpy(), dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]

        if x.size < 2:
            return

        # --- 1) Coercivity from current loop ---
        hc_plus, hc_minus, hc_avg = self._compute_coercivity(x, y)
        if hc_plus is not None:
            self.lblHcPlus.setText(f"{hc_plus:.4g}")
        if hc_minus is not None:
            self.lblHcMinus.setText(f"{hc_minus:.4g}")
        if hc_avg is not None:
            self.lblHc.setText(f"{hc_avg:.4g}")

        # --- 2) Remanence from current loop ---
        Mr_plus, Mr_minus, Mr_mag = self._compute_remanence(x, y)
        if Mr_plus is not None:
            self.lblMrPlus.setText(f"{Mr_plus:.4g}")
        if Mr_minus is not None:
            self.lblMrMinus.setText(f"{Mr_minus:.4g}")
        if Mr_mag is not None:
            self.lblMr.setText(f"{Mr_mag:.4g}")

        # --- 3) Background slope and M_sat from the last BG operation ---
        last_bg = self._get_last_bg_info_for_current_axes()
        if last_bg is None:
            # No BG correction yet for this X/Y → leave Ms, BG slope as "—"
            return
        # Background slope m_bg
        m_bg = last_bg.get("m_bg", None)
        if m_bg is not None:
            self.lblBgSlope.setText(f"{m_bg:.4g}")

        # --- 4) M_sat: from high-field intercepts of BG fits
        b_pos = last_bg.get("b_pos", None)
        b_neg = last_bg.get("b_neg", None)
        if b_pos is not None and b_neg is not None:
            # For a symmetric loop, M_sat ≈ (b_pos - b_neg) / 2
            msat = 0.5 * (b_pos - b_neg)
            self.lblMs.setText(f"{msat:.4g}")

    def _get_last_bg_info_for_current_axes(self):
        """
        Find the most recent bg_linear_branches operation that matches
        the currently selected X/Y columns. Returns its params dict or None.
        """
        x_name = self.xCombo.currentText()
        y_name = self.yCombo.currentText()
        if not x_name or not y_name:
            return None

        for entry in reversed(self.history):
            if entry.get("op") == "bg_linear_branches":
                params = entry.get("params", {})
                if (params.get("x_column") == x_name and
                        params.get("column") == y_name):
                    return params
        return None

    def _draw_feature_markers(self, x, y):
        """
        Draw feature markers (Mr, Hc, etc.) on the current axes,
        based on the given loop x, y.

        Called only when the 'Show markers' checkbox is checked.
        """
        ax = self.canvas.ax

        # --- Remanence markers (at H = 0) ---
        Mr_plus, Mr_minus, Mr_mag = self._compute_remanence(x, y)

        if Mr_plus is not None:
            ax.scatter([0.0], [Mr_plus], s=30, marker="o")
            ax.annotate(
                "Mr+",
                xy=(0.0, Mr_plus),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        if Mr_minus is not None:
            ax.scatter([0.0], [Mr_minus], s=30, marker="o")
            ax.annotate(
                "Mr−",
                xy=(0.0, Mr_minus),
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=8,
            )

        # --- Coercivity markers (at M = 0) ---
        hc_plus, hc_minus, hc_avg = self._compute_coercivity(x, y)

        if hc_plus is not None:
            ax.scatter([hc_plus], [0.0], s=30, marker="s")
            ax.annotate(
                "Hc+",
                xy=(hc_plus, 0.0),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        if hc_minus is not None:
            ax.scatter([hc_minus], [0.0], s=30, marker="s")
            ax.annotate(
                "Hc−",
                xy=(hc_minus, 0.0),
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=8,
            )
         # --- Ms as dashed horizontal lines (only after BG correction) ---
        last_bg = self._get_last_bg_info_for_current_axes()
        if last_bg is not None:
            b_pos = last_bg.get("b_pos", None)
            b_neg = last_bg.get("b_neg", None)
            if b_pos is not None and b_neg is not None:
                msat = 0.5 * (b_pos - b_neg)

                # Compute x-range for the lines
                x_arr = np.asarray(x, dtype=float)
                x_finite = x_arr[np.isfinite(x_arr)]
                if x_finite.size > 0:
                    x_min, x_max = x_finite.min(), x_finite.max()
                else:
                    x_min, x_max = -1.0, 1.0

                # Draw Ms lines (±Ms) as dashed horizontals
                ax.hlines(msat, x_min, x_max, linestyles="--")
                ax.hlines(-msat, x_min, x_max, linestyles="--")

                # Annotate the upper one
                ax.annotate(
                    "Ms",
                    xy=(x_min, msat),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        # Later: we can add H_sat, etc. here using the same pattern.


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
