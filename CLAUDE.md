# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

BOPPO is a PyQt5 desktop application for power systems engineers to compare time-series results from PSS/E and PSCAD simulations side-by-side, with optional field data overlay. The entire application lives in a single file: `boppo.py`.

## Running the App

```bash
python boppo.py
```

**Prerequisites:**
- Python 3 with `pip install PyQt5 matplotlib numpy openpyxl`
- PSS/E 36.5 installed at `C:\Program Files\PTI\PSSE36\36.5\PSSPY314` (hardcoded at line 21)
- Optionally `pip install mhi.psout` for PSCAD v5 `.psout` support

There is no test suite — all testing is manual via the GUI.

## Architecture

The application is structured in layers within `boppo.py`:

### Data Layer (lines ~268–699)
Format-agnostic backends, all exposing `get(channel_name) → (time_array, data_array)`:
- **PSS/E:** `PSSEDataset` / `PSSEFolder` — reads `.out`/`.outx` via `dyntools`
- **PSCAD legacy:** `PSCADFolder` / `PSCADDataset` / `PSCADSimResult` — reads `.inf` metadata + split `_01.out`, `_02.out`... files
- **PSCAD v5:** `PSCADPsoutFolder` — reads `.psout` via `mhi.psout`
- **Field data:** `FieldDataset` — reads CSV/XLSX with a datetime first column, auto-converted to elapsed seconds

### Visualization Layer (lines ~57–221, ~804–1651)
- Standalone helpers: `_autoscale_y_to_xlim`, `_compute_signal_metrics` (rise time 10–90%, settle time), `_draw_analysis_overlay`
- `PlotWidget` — a single matplotlib panel; handles drag-drop, rectangle-zoom, hover crosshair
- `PlotGrid` — resizable N×M grid of `PlotWidget`s; `render_page()` renders off-screen for batch export
- `PlotConfigDialog` — per-plot settings (titles, axis labels, limits, per-channel transforms and custom legend labels)

### UI Layer (lines ~709–2435)
- `ChannelBrowser` (~709–802) — tree widget (PSSE blue / PSCAD red / Field Data teal) with drag-source MIME data
- `MainWindow` — two independent tabs with near-identical logic; Tab 2 methods use `_t2_` prefix throughout
- `ExportDialog` (~1657–1772) — batch export, paging through PSSE AND PSCAD files together (paired by position -- see below) → PDF (combined or per-page) or PNG, with optional XLSX title template

### Persistence (lines ~2209–2396)
- `.boppo` files — JSON-serialized plot layouts (save/load templates), including `page_titles` (manual, filename-keyed) and `page_title_sequence` (positional, e.g. AECST-generated — see below)
- Global settings: grid dimensions. (The toolbar's global X-axis limits and time offset are NOT saved in templates.)
- `page_xlim_sequence`: optional per-page `[xmin, xmax]` windows, by position (see below)
- Per-plot settings: axis limits, legends, analysis parameters

## CLI: pre-loading a PSCAD folder and/or template at startup

`main()` accepts two optional arguments, used by AECST (`Akaysha Energy
Connection Studies Tool`, a sibling repo) to launch BOPPO pre-configured
for a simulation-set run:

```bash
python boppo.py --pscad-folder "C:\path\to\pscad\output"
python boppo.py --template "C:\path\to\preset.boppo"
python boppo.py --template "C:\path\to\preset.boppo" --pscad-folder "C:\path\to\pscad\output"
```

- `--pscad-folder <path>` alone calls `ComparisonTabController._add_dataset('PSCAD',
  path)` on the first tab -- loading the folder without a file dialog.
  `_add_dataset(kind, path)` is the shared tail of `_load_folder()` (which
  now just resolves a path via dialog, then calls `_add_dataset`).
- `--template <path>` calls `ComparisonTabController.load_template_file(path,
  pscad_folder_override=...)` on the first tab. If the file doesn't exist
  yet (a brand-new AECST preset that's never had a template saved), nothing
  is loaded, but `self._current_template_path` is still set to that path --
  so the next `_save_template()` (the existing "Save Template" button/menu
  action) writes there directly with **no save dialog**, bootstrapping the
  template file for that preset. If the file already exists, it's loaded
  via `_apply_template()` (the same logic `_load_template()`'s dialog path
  uses, now extracted into a shared method).
- When both flags are given, `--pscad-folder` overrides the `path` of any
  `PSCAD`-kind dataset entry inside the loaded template, rather than using
  the (likely stale) path the template was originally saved with -- so one
  shared template's plot layout/channel selections can be reused against a
  fresh run's output folder every time.
- `_save_template()` now checks `self._current_template_path` first: if
  set (from either `--template` or a prior "Load Template" dialog), it
  overwrites that file directly with no prompt; only falls back to the
  save-file dialog (`_save_template_as()`) if no path is known yet. This is
  what keeps an AECST preset's linked template in sync: launch with
  `--template <preset path>`, edit the layout, hit Save Template, done --
  no need to re-pick the file each time.

See AECST's `aecst_gui.py`/`CLAUDE.md` for how it launches BOPPO with these
flags (via `subprocess.Popen`).

### Dynamic per-page titles from AECST (`page_title_sequence`)

`ComparisonTabController.page_title_sequence: List[str]` is an ordered list
of full page-title strings, persisted in `.boppo` templates alongside the
existing `page_titles` dict (`_write_template_file()` writes it,
`_apply_template()` reads it via `layout.get('page_title_sequence', [])`).
Unlike `page_titles` (keyed by output filename stem, edited manually via
`PageTitleEditorDialog`), `page_title_sequence` is matched to `_export()`'s
batch-export pages by **position**: `page_title_sequence[i]` for the `i`-th
page. In `_page_title(fallback, index)`, if `page_title_sequence[index]` is
non-empty it's used **verbatim** as the entire page title -- bypassing
`title_prefix`/the filename-keyed `page_titles` lookup/`title_outfile`/
`title_date` entirely, since the caller (AECST) already built the complete
string. Falls back to the pre-existing assembly when the sequence is empty
or doesn't cover that index.

AECST populates this via `aecst_presets.write_title_sequence_to_template()`
right before launching BOPPO with `--template` -- it renders each RANK.txt
row's own column values (plus a `{TestNumber}` counter from a run-time
offset) into a per-preset title format, so a DMAT-style test campaign gets
one descriptive title per page automatically. This is deliberately matched
by **row/page order, not by PSCAD's actual output filename** -- there is no
code anywhere that inspects PSCAD's per-run output naming convention for
this feature; it relies on the same file-order assumption the
PSSE/PSCAD-pairing fix above already depends on.

### Per-page X windows from AECST (`page_xlim_sequence`)

`ComparisonTabController.page_xlim_sequence: List[[xmin, xmax]]` (each end
may be `null`) is persisted in `.boppo` templates exactly like
`page_title_sequence`. It's loaded in `_apply_template()` and written in
`_write_template_file()`, with `[]` for older templates. Both batch loops
(`_export()` and the headless `export_batch_pdf()`) call the module-level
`_page_xlim(seq, i, fallback_min, fallback_max)` per page index `i`. Each
end comes from `seq[i]` if set, otherwise from the fallback (the toolbar's
global X limits, or the CLI `--xmin`/`--xmax`). The result is passed as
`render_page()`'s `global_xmin`/`global_xmax`, so it also beats per-plot
X limits, and the signal-analysis window follows it. Y autoscaling
already uses only the visible X window, so each page's Y fits its own
window unless a plot has fixed Y limits. AECST populates this from its
per-test "X min (s)"/"X max (s)" table columns
(`aecst_presets.build_xlim_sequence()`).

### Batch rise/settle metrics export (CSV + DOCX)

`render_page(..., metrics_out=list)` appends one dict per analysed channel
(plots with Analysis enabled), holding `_compute_signal_metrics()`'s
output plus plot title, channel (its legend label if set), source,
settle %, and the X window used. Both batch loops collect these per page,
adding `page`/`file`, then call `_write_metrics_files(rows, base_path)`:
- `<base>.csv` always gets the full column set (`_METRICS_COLUMNS`, UTF-8 with BOM for Excel).
- `<base>.docx` gets a landscape Word table with the key subset (`_METRICS_DOCX_KEYS`) via python-docx. It's skipped with a reported problem if python-docx isn't installed.

Channels with no detectable step leave the numeric cells blank and get
the note "No step detected in window". Nothing is written if no plot has
Analysis enabled. File names:
- interactive batch export: `BOPPO_metrics.*` in the output folder;
- headless `--export-pdf X.pdf`: `X_metrics.*` next to the PDF.

Write failures are added to the export's error list and never stop the
PDF. `boppo.spec` bundles python-docx's data files.

### Headless PDF export CLI

`--export-pdf <path>` (with `--pscad-folder`/`--template`) renders the
combined batch PDF via `export_batch_pdf()` without showing the window,
then exits. `--page-size <key>` is a `PAGE_SIZES` key, and `--xmin`/`--xmax`
set a one-off global X window for that export (never saved). Per-page
`page_xlim_sequence` entries override `--xmin`/`--xmax`. AECST launches this
in the background after every run.

(Note: the current code uses a `ComparisonTabController` class managed by
`MainWindow.tabs`, not the older `MainWindow` two-tab-with-`_t2_`-prefix
structure described elsewhere in this file -- that description appears to
predate a refactor and no longer matches `boppo.py` as it stands.)

## Key Implementation Details

- **Batch export pages through both folders together, paired by position** (fixed a bug where it only iterated one side): `ComparisonTabController._export()`'s batch section builds `psse_list`/`pscad_list` from whichever folders were selected, then for `i in range(n_pages)` (`n_pages = max(len(psse_list), len(pscad_list))`) looks up `_at(datasets, i)` for each side. `_at()` returns the single dataset unconditionally if a side has exactly one file (reused as a fixed reference on every page, preserving the old single-loop-with-fixed-reference behavior), the `i`-th dataset if within range, or `None` if that side ran out (producing a partial page using only the side that still has data). The old `ExportDialog._loop_combo` ("Loop PSSE files" / "Loop PSCAD files") is gone — there's no longer a manual choice of which side to iterate, since both are now paired automatically. `_loop_filenames()` (used for the page-title editor) now returns whichever folder's filename list is longer, since that determines the actual page count. If both folders have more than one file and the counts differ, the completion dialog notes that pages beyond the shorter list's length only contain data from the longer side.
- **Per-channel transforms** use a restricted `eval` with a limited math namespace — safe expression evaluation for things like `y*50+10` or `abs(y)`.
- **Custom per-channel legend labels**: `_channel_legend_label(ch, cfg, seen_sources=None)` (~line 66) checks `ch.get('legend_label')` first — if non-empty, that literal text is used as the legend entry, bypassing the plot's `legend_mode` setting ('name' vs 'source') entirely. Set via `PlotConfigDialog`'s channel table (the same table used for transforms), which now has 4 columns: Channel (read-only), Expression, **Legend** (blank = fall back to the plot's default label), and a remove button. `result_assigned()` reads column 2 into `ch_copy['legend_label']` on dialog accept. Since `legend_label` lives on the channel dict itself, it round-trips automatically through `PlotWidget.snapshot()`/`restore()` (deep-copies `self.assigned`) and therefore through `.boppo` template save/load — no separate persistence code needed.
- **Y-axis auto-scaling** considers only the visible X window and enforces a minimum span to avoid zoom issues on flat signals.
- **PSCAD split-file routing** computes file index and column offset dynamically from a channel's overall index across split `.out` files.
- **Lazy loading:** `PSSEDataset._load()` is called on first access, not at startup.
- The default XLSX title template directory is hardcoded to `C:\Users\CamSmith\Documents\Claude Working Folder\Benchmarking Tool` (line ~1661).
