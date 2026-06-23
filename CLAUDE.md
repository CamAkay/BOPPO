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

### Visualization Layer (lines ~57–221, ~981–1430)
- `PlotWidget` — a single matplotlib panel; handles drag-drop, rectangle-zoom, hover crosshair
- `PlotGrid` — resizable N×M grid of `PlotWidget`s; `render_page()` renders off-screen for batch export
- `PlotConfigDialog` — per-plot settings (titles, axis labels, limits, per-channel transforms)
- Signal analysis: rise time (10–90%), settle time (configurable band %), ±10% error bands

### UI Layer (lines ~709–1778)
- `ChannelBrowser` — tree widget (PSSE blue / PSCAD red / Field Data teal) with drag-source MIME data
- `MainWindow` — two tabs: *PSSE vs PSCAD* and *Field Data Overlay*
- Batch export loop over PSSE or PSCAD files → PDF (combined or per-page) or PNG, with optional XLSX title template

### Persistence (lines ~2209–2396)
- `.boppo` files — JSON-serialized plot layouts (save/load templates)
- Global settings: time offset, X-axis limits, grid dimensions
- Per-plot settings: axis limits, legends, analysis parameters

## Key Implementation Details

- **Per-channel transforms** use a restricted `eval` with a limited math namespace — safe expression evaluation for things like `y*50+10` or `abs(y)`.
- **Y-axis auto-scaling** considers only the visible X window and enforces a minimum span to avoid zoom issues on flat signals.
- **PSCAD split-file routing** computes file index and column offset dynamically from a channel's overall index across split `.out` files.
- **Lazy loading:** `PSSEDataset._load()` is called on first access, not at startup.
- The default XLSX title template directory is hardcoded to `C:\Users\CamSmith\Documents\Claude Working Folder\Benchmarking Tool` (line ~1661).
