# BOPPO — Benchmarking Tool

A PyQt5 desktop application for power systems engineers to compare time-series results from PSS/E and PSCAD simulations side-by-side, with optional field data overlay.

## Requirements

- Python 3
- `pip install PyQt5 matplotlib numpy openpyxl`
- PSS/E 36.5 installed at `C:\Program Files\PTI\PSSE36\36.5\PSSPY314`
- Optional: `pip install mhi.psout` for PSCAD v5 `.psout` file support

## Running

```bash
python boppo.py
```

## Supported File Formats

| Format | Files | Notes |
|---|---|---|
| PSS/E | `.out`, `.outx` | Read via `dyntools` |
| PSCAD legacy | `.inf` + `_01.out`, `_02.out`, ... | Metadata + split channel files |
| PSCAD v5 | `.psout` | Requires `mhi.psout` |
| Field data | `.csv`, `.xlsx` | Datetime first column, auto-converted to elapsed seconds |

## Features

- Side-by-side comparison of PSS/E and PSCAD simulation results
- Field data overlay against simulation results
- Interactive plots: drag-and-drop channels, rectangle zoom, hover crosshair with interpolated readouts
- Per-channel mathematical transforms (e.g. `y*50+10`, `abs(y)`)
- Signal analysis: rise time (10–90%), settle time (configurable band), ±10% error bands
- Batch export to PDF or PNG, with optional XLSX title template per page
- Save/load plot layouts as `.boppo` template files
