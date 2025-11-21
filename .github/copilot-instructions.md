# SQUiiD / M(H)ero — AI Agent Instructions (concise)

Quick summary
- Main entry: `QT_test.py` (creates `MainWindow`, `PlotCanvas` and runs `main()`).

Big picture
- Data flow: file -> `_read_table_auto(path)` -> pandas `DataFrame` (`original_df` / `df`) -> plotting via `PlotCanvas` and `_replot()`.
- Transformations are applied through `_apply_operation(op, params)` and recorded in `history` for replay/undo.
- Interactive preview modes (background/drift) work from snapshots: `_bg_df_before` and `_drift_df_before`.

Key files & symbols to read first
- `QT_test.py`: everything lives here (GUI, parsing, operations, history, plotting).
- `_read_table_auto`: robust parsing heuristics (decimal commas, Fortran `D`, delimiter scoring, header autodetect).
- `_apply_operation` and individual op names: `center_y`, `bg_linear_branches`, `drift_linear_tails`, `drift_linear_loopclosure`, `unit_convert`, `volume_normalisation`.
- `_get_last_bg_info_for_current_axes`, `_rebuild_df_from_history`: history/replay semantics.

Developer workflows
- Run locally using bundled venv: `source env/bin/activate` then `python QT_test.py` (or run `env/bin/python QT_test.py`).
- Debugging: run the script in the venv, use prints or inspect `self.history` after operations. UI errors surface via Qt message boxes.
- No test harness present — add unit tests around pure functions (parsers, compute_* helpers) if needed.

Project-specific patterns & conventions
- Numeric-column detection: `numpy.issubdtype(..., np.number)` — default plotting uses the first two numeric columns.
- File parsing: `_read_table_auto` chooses a delimiter by scoring candidate separators, converts object columns with numeric-like strings to numeric (replacing `,` → `.`), and uses `on_bad_lines='skip'`.
- Plotting: use `self.canvas.fig.canvas.draw_idle()` for redraws; interactive events via `canvas.mpl_connect` and `mpl_disconnect`.
- History: every user action that mutates data should call `_add_history_entry(op, params)` so `_rebuild_df_from_history()` can replay operations deterministically.

Where to add features
- To add a new data operation: implement logic in `_apply_operation(op, params)` and append `_add_history_entry` when recording; add a menu/action that calls the operation.
- To add unit conversions, extend `_unit_conversion_factor_for_quantity` / `_unit_conversion_factors_axes`.
- To change axis labels or semantics, update `_update_y_quantity_labels`, `_format_x_axis_label`, `_format_y_axis_label`.

Run snippets
```
source env/bin/activate
python QT_test.py
```

Notes for agents
- Preserve user-visible semantics: history entries must remain stable (store numeric factors explicitly when computing conversions or normalisations).
- Use existing helpers (compute_bg, compute_drift, compute_remanence, compute_coercivity) rather than reimplementing math.
- Keep UI patterns consistent: use Qt dialogs (QDialog) for user input, signal/slot connections for actions, and snapshot/preview pattern for interactive edits.

If any of these areas are unclear or you want more examples (e.g., a small unit-test for `_read_table_auto`), tell me which part to expand.