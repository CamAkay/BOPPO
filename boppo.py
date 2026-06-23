#!/usr/bin/env python3
"""
BOPPO – Benchmarking Of PSSE and PSCAD Outputs
================================================
Compare time-series results from PSS/E and PSCAD simulations.

Requirements:
    pip install PyQt5 matplotlib numpy
    PSS/E dyntools must be importable to read .out files from PSS/E.

Usage:
    python boppo.py
"""

import sys
import os
import re
import json
import copy
import logging
PSSE_LOCATION = r"C:\Program Files\PTI\PSSE36\36.5\PSSPY314"
sys.path.append(PSSE_LOCATION)
os.environ['PATH'] = os.environ['PATH'] + ';' +  PSSE_LOCATION 
# PSSE 34 imports
import psse3605
import psspy
# import redirect
import dyntools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QTreeWidget, QTreeWidgetItem, QAbstractItemView, QFrame,
    QScrollArea, QComboBox, QLineEdit, QFormLayout, QDialog,
    QDialogButtonBox, QFileDialog, QMessageBox, QProgressDialog,
    QSizePolicy, QSplitter, QAction, QToolBar, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
)
from matplotlib.widgets import RectangleSelector
from PyQt5.QtCore import Qt, QMimeData, QByteArray, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QDrag, QCursor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ─── y-axis auto-scale helper ────────────────────────────────────────────────

def _autoscale_y_to_xlim(ax, lines, margin_frac: float = 0.05):
    """
    Tighten the y-axis to only the data visible within the current x limits.
    `lines` is a list of matplotlib Line2D objects already plotted on `ax`.
    Does nothing if there is no finite data in the visible window.
    """
    xlo, xhi = ax.get_xlim()
    chunks = []
    for line in lines:
        xd = np.asarray(line.get_xdata(), dtype=float)
        yd = np.asarray(line.get_ydata(), dtype=float)
        if xd.size == 0:
            continue
        mask = (xd >= xlo) & (xd <= xhi)
        if mask.any():
            chunks.append(yd[mask])
    if not chunks:
        return
    visible = np.concatenate(chunks)
    finite  = visible[np.isfinite(visible)]
    if finite.size == 0:
        return
    y_lo, y_hi = float(finite.min()), float(finite.max())
    span   = y_hi - y_lo
    margin = span * margin_frac if span > 0 else (abs(y_lo) * margin_frac or 0.1)

    # Enforce a minimum visible span so flat/near-flat signals don't zoom in
    # excessively.  Use 2 % of |centre| as the floor, with an absolute fallback
    # of 0.1 so zero-centred signals still get a sensible range.
    centre      = (y_lo + y_hi) / 2.0
    min_span    = max(abs(centre) * 0.02, 0.1)
    total_span  = (y_hi + margin) - (y_lo - margin)
    if total_span < min_span:
        half = min_span / 2.0
        ax.set_ylim(centre - half, centre + half)
    else:
        ax.set_ylim(y_lo - margin, y_hi + margin)


def _compute_signal_metrics(
    t: np.ndarray,
    y: np.ndarray,
    settle_pct: float = 2.0,
    avg_window: int = 10,
    xlim: Optional[Tuple[float, float]] = None,
) -> dict:
    """
    Compute rise time (10 %→90 % of step) and settle time for a step response.
    Only data within xlim (the current visible x window) is used.
    Settle time is measured from t0 — the step inception (first sample where the
    signal deviates by >5 % of the step magnitude from the initial value).

    Returns a dict with keys:
        y_initial, y_final, magnitude,
        y_10, y_90,
        t_0, t_10, t_90, rise_time,
        t_settle, settle_time,
        valid (bool)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # Clip to visible x window
    if xlim is not None:
        vis = (t >= xlim[0]) & (t <= xlim[1])
        t, y = t[vis], y[vis]

    if len(t) < 20 or not np.isfinite(y).any():
        return {'valid': False}

    n_avg = max(1, min(avg_window, len(y) // 10))

    # y_initial: mean of the first n_avg samples in the clipped window.
    # xmin is set by the user to exclude startup transients, so the first
    # samples here represent the genuine pre-step steady state.
    y_initial = float(np.mean(y[:n_avg]))

    # Post-step final value: mean of last n_avg samples (signal should be settled)
    y_final   = float(np.mean(y[-n_avg:]))
    magnitude = y_final - y_initial
    if abs(magnitude) < 1e-12:
        return {'valid': False}

    y_10 = y_initial + 0.10 * magnitude
    y_90 = y_initial + 0.90 * magnitude

    # t0 — step inception: first sample where signal moves >2 % of magnitude
    # from y_initial, scanning from the beginning of the clipped window.
    threshold_0 = abs(magnitude) * 0.02
    if magnitude > 0:
        idx_0 = np.where(y >= y_initial + threshold_0)[0]
    else:
        idx_0 = np.where(y <= y_initial - threshold_0)[0]
    if idx_0.size == 0:
        return {'valid': False}
    t_0 = float(t[idx_0[0]])

    # All threshold searches run from t0 onward
    y_step = y[idx_0[0]:]
    t_step = t[idx_0[0]:]
    if magnitude > 0:
        idx_10 = np.where(y_step >= y_10)[0]
        idx_90 = np.where(y_step >= y_90)[0]
    else:
        idx_10 = np.where(y_step <= y_10)[0]
        idx_90 = np.where(y_step <= y_90)[0]

    if idx_10.size == 0 or idx_90.size == 0:
        return {'valid': False}

    t_10 = float(t_step[idx_10[0]])
    t_90 = float(t_step[idx_90[0]])
    rise_time = abs(t_90 - t_10)

    # Settle time — last time signal leaves ±settle_pct % band (post-step only),
    # measured from t0
    band = abs(magnitude) * settle_pct / 100.0
    outside = np.where(np.abs(y_step - y_final) > band)[0]
    t_settle = float(t_step[outside[-1]]) if outside.size > 0 else t_0
    settle_time = max(0.0, t_settle - t_0)

    return {
        'valid':       True,
        'y_initial':   y_initial,
        'y_final':     y_final,
        'magnitude':   magnitude,
        'y_10':        y_10,
        'y_90':        y_90,
        't_0':         t_0,
        't_10':        t_10,
        't_90':        t_90,
        'rise_time':   rise_time,
        't_settle':    t_settle,
        'settle_time': settle_time,
    }


def _draw_analysis_overlay(ax, metrics: dict, color: str, label_prefix: str = ''):
    """Draw rise/settle annotation lines and return a text summary string."""
    if not metrics.get('valid'):
        return None

    m = metrics
    lw, alpha = 0.9, 0.7

    # Horizontal lines at 10 % and 90 % levels
    ax.axhline(m['y_10'], color=color, linewidth=1.4, linestyle='--', alpha=0.85)
    ax.axhline(m['y_90'], color=color, linewidth=1.4, linestyle='--', alpha=0.85)

    # Vertical lines: t0 (step start), t_10, t_90, t_settle
    ax.axvline(m['t_0'],     color=color, linewidth=lw, linestyle='-',  alpha=alpha)
    ax.axvline(m['t_10'],    color=color, linewidth=lw, linestyle='-.', alpha=alpha)
    ax.axvline(m['t_90'],    color=color, linewidth=lw, linestyle='-.', alpha=alpha)
    ax.axvline(m['t_settle'], color=color, linewidth=lw, linestyle='--', alpha=alpha)

    prefix = f"{label_prefix}: " if label_prefix else ''
    return (
        f"{prefix}Rise = {m['rise_time']:.4f} s  "
        f"({m['t_10']:.4f}→{m['t_90']:.4f} s)\n"
        f"{prefix}Settle = {m['settle_time']:.4f} s  "
        f"(t0={m['t_0']:.4f} s, settled at {m['t_settle']:.4f} s)"
    )


# ─── xlsx title helper ────────────────────────────────────────────────────────

def read_title_xlsx(path: str) -> List[str]:
    """
    Read an xlsx where row 1 is variable names and rows 2+ are values.
    Returns one title string per data row; empty cells are omitted.

    Example xlsx:
        SCR   | Pmax | Qmax
        1.5   | 100  |
        2.0   |      | 50

    Returns:
        ['SCR = 1.5,  Pmax = 100', 'SCR = 2.0,  Qmax = 50']
    """
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        return []
    headers = [str(h).strip() if h is not None else '' for h in rows[0]]
    titles = []
    for row in rows[1:]:
        parts = []
        for header, val in zip(headers, row):
            if header and val is not None and str(val).strip():
                parts.append(f"{header} = {val}")
        titles.append(',  '.join(parts))
    return titles

# ─── Constants ────────────────────────────────────────────────────────────────

CHANNEL_MIME = 'application/x-boppo-channel'
PSSE_COLOR   = '#1f77b4'
PSCAD_COLOR  = '#d62728'
LINE_COLORS  = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

def _parse_pscad_inf(inf_path: str) -> Dict[int, dict]:
    """
    Parse a PSCAD .inf file.  Returns {1-based_index: {'desc', 'group', 'units'}}.

    Expected format per line:
        PGB(N)   Output   Desc="..."   Group="..."   Max=...   Min=...   Units="..."
    """
    channels: Dict[int, dict] = {}
    rx = re.compile(
        r'PGB\((\d+)\)\s+\S+\s+Desc="([^"]*)"\s+Group="([^"]*)"\s+'
        r'Max=\S+\s+Min=\S+\s+Units="([^"]*)"'
    )
    with open(inf_path, 'r') as fh:
        for line in fh:
            m = rx.search(line)
            if m:
                channels[int(m.group(1))] = {
                    'desc':  m.group(2),
                    'group': m.group(3),
                    'units': m.group(4),
                }
    return channels


class PSCADDataset:
    """One PSCAD simulation output (.out file)."""

    def __init__(self, path: str, channels: Dict[int, dict]):
        self.path     = path
        self.name     = Path(path).stem
        self.channels = channels          # {1-based idx -> metadata}
        self._data: Optional[np.ndarray] = None

    # ── Data access ──────────────────────────────────────────────────────────

    def load(self):
        if self._data is None:
            self._data = np.loadtxt(self.path)

    def time(self) -> np.ndarray:
        self.load()
        return self._data[:, 0]

    def get(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (time, signal) for channel with matching description."""
        self.load()
        for idx, meta in self.channels.items():
            if meta['desc'] == name:
                return self._data[:, 0], self._data[:, idx]
        raise KeyError(f"PSCAD channel '{name}' not found")


class PSCADSimResult:
    """
    One PSCAD simulation result: one .inf file and its associated .out channel-split files.

    PSCAD names output files as  <stem>_01.out, <stem>_02.out, …  where <stem>
    matches the .inf file stem.  All .out files that start with that stem are
    treated as channel-split files belonging to this result.
    """

    def __init__(self, inf_path: Path):
        self.name = inf_path.stem
        self.channels: Dict[int, dict] = _parse_pscad_inf(str(inf_path))
        folder = inf_path.parent
        stem   = inf_path.stem
        # Match <stem>_*.out  (channel splits for this run)
        out_files = sorted(folder.glob(f'{stem}_*.out'))
        if not out_files:
            # Fallback: if no stem-prefixed files, take all .out in folder
            out_files = sorted(folder.glob('*.out'))
        self._out_datasets: List[PSCADDataset] = [
            PSCADDataset(str(f), self.channels) for f in out_files
        ]

    def channel_names(self) -> List[str]:
        return [m['desc'] for m in self.channels.values()]

    def channel_units(self, name: str) -> str:
        for m in self.channels.values():
            if m['desc'] == name:
                return m['units']
        return ''

    def get(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (time, signal), routing to the correct .out channel-split file."""
        target_idx = None
        for idx, meta in self.channels.items():
            if meta['desc'] == name:
                target_idx = idx
                break
        if target_idx is None:
            raise KeyError(f"PSCAD channel '{name}' not found in .inf")

        first = self._out_datasets[0]
        first.load()
        cols_per_file = first._data.shape[1] - 1   # subtract time column

        file_idx = (target_idx - 1) // cols_per_file
        col      = (target_idx - 1) %  cols_per_file + 1

        if file_idx >= len(self._out_datasets):
            raise IndexError(
                f"Channel '{name}' (PGB {target_idx}) requires file #{file_idx + 1} "
                f"but only {len(self._out_datasets)} file(s) are present."
            )

        ds = self._out_datasets[file_idx]
        ds.load()
        return ds._data[:, 0], ds._data[:, col]


class PSCADFolder:
    """All PSCAD simulation results inside a single folder.

    Each .inf file is one simulation result; its associated .out channel-split
    files are matched by stem prefix (e.g. project_001.inf → project_001_01.out).
    """

    def __init__(self, folder: str):
        self.folder = Path(folder)
        inf_files = sorted(self.folder.glob('*.inf'))
        if not inf_files:
            raise FileNotFoundError(f"No .inf file found in: {folder}")
        self.datasets: List[PSCADSimResult] = [
            PSCADSimResult(f) for f in inf_files
        ]
        if not any(ds._out_datasets for ds in self.datasets):
            raise FileNotFoundError(f"No .out files found in: {folder}")

    def channel_names(self) -> List[str]:
        return self.datasets[0].channel_names()

    def channel_units(self, name: str) -> str:
        return self.datasets[0].channel_units(name)

    def get(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.datasets[0].get(name)


class _PsoutDatasetProxy:
    """Minimal stub so PSCADPsoutFolder.datasets entries have a .name attribute."""
    def __init__(self, path: Path):
        self.name = path.stem


class PSCADPsoutFolder:
    """
    Reads PSCAD v5 .psout output files via the mhi.psout library.
    Exposes the same interface as PSCADFolder so the rest of the app
    needs no changes.
    """

    def __init__(self, folder: str):
        try:
            from mhi.psout import File as PsoutFile
        except ImportError:
            raise ImportError(
                "mhi.psout is not installed. Install it with:  pip install mhi.psout"
            )

        self.folder = Path(folder)
        psout_files = sorted(self.folder.glob('*.psout'))
        if not psout_files:
            raise FileNotFoundError(f"No .psout files found in: {folder}")

        self.datasets = [_PsoutDatasetProxy(f) for f in psout_files]

        # Open all files; build a unified channel catalogue from the first file
        # (all files in a sweep typically share the same channels)
        self._files: Dict[str, object] = {}   # stem → PsoutFile
        for f in psout_files:
            self._files[f.stem] = PsoutFile(str(f))

        # Build channel catalogue from the first file.
        # mhi.psout structure: File → Run → Trace, traversed via call-tree paths.
        first = next(iter(self._files.values()))
        self._channel_names: List[str] = []   # full call-tree path used as key
        self._channel_units: Dict[str, str] = {}
        try:
            run0 = first.run(0)
        except Exception as exc:
            raise RuntimeError(f"No runs found in .psout file: {exc}") from exc

        for path in first.paths("**"):
            try:
                call = first.call(path, sep="/")
                trace = run0.trace(call)
                _ = trace.data          # raises if this node has no trace data
                self._channel_names.append(path)
                units = call.get("Units", "") or ""
                self._channel_units[path] = str(units)
            except Exception:
                continue

    def channel_names(self) -> List[str]:
        return list(self._channel_names)

    def channel_units(self, name: str) -> str:
        return self._channel_units.get(name, '')

    def get(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (time, signal). name is the full call-tree path."""
        for f in self._files.values():
            try:
                call = f.call(name, sep="/")
                run = f.run(0)
                trace = run.trace(call)
                domain = trace.domain
                if domain is not None:
                    t = np.asarray(list(domain.data), dtype=float)
                else:
                    t = np.array([], dtype=float)
                y = np.asarray(list(trace.data), dtype=float)
                return t, y
            except Exception:
                continue
        raise KeyError(f"PSCAD channel '{name}' not found in any .psout file")


class PSSEDataset:
    """One PSS/E simulation output (.out file read via dyntools)."""

    def __init__(self, path: str):
        self.path  = path
        self.name  = Path(path).stem
        self._channels: Optional[List[str]] = None
        self._time: Optional[np.ndarray]    = None
        self._data: Optional[Dict[str, np.ndarray]] = None

    def _load(self):
        try:
            from dyntools import CHNF  # noqa: F401  (PSS/E package)
        except ImportError:
            raise ImportError(
                "The 'dyntools' package is required to read PSS/E output files.\n\n"
                "dyntools is distributed with PSS/E.  Make sure the PSS/E Python\n"
                "environment is active, or install a standalone dyntools package."
            )

        from dyntools import CHNF
        chnf = CHNF(self.path)
        # PSS/E 36 dyntools API:
        #   sh_ttl : str  – short case title (NOT a lookup dict)
        #   ch_id  : dict – {'time': 'Time(s)', 1: 'channel_name', 2: ..., n: ...}
        #   ch_data: dict – {'time': [v,...], 1: [v,...], 2: [v,...], ...}
        sh_ttl, ch_id, ch_data = chnf.get_data()

        self._time = np.asarray(ch_data['time'], dtype=float)
        self._data = {}
        self._channels = []
        for k, arr in ch_data.items():
            if k == 'time':
                continue
            label = str(ch_id.get(k, k)).strip()
            self._data[label] = np.asarray(arr, dtype=float)
            self._channels.append(label)

    @property
    def channels(self) -> List[str]:
        if self._channels is None:
            self._load()
        return self._channels

    def time(self) -> np.ndarray:
        if self._data is None:
            self._load()
        return self._time

    def get(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        if self._data is None:
            self._load()
        if name not in self._data:
            raise KeyError(f"PSSE channel '{name}' not found")
        return self._time, self._data[name]


class PSSEFolder:
    """All PSS/E .out files inside a folder."""

    def __init__(self, folder: str):
        self.folder = Path(folder)
        out_files = sorted(
            list(self.folder.glob('*.out')) + list(self.folder.glob('*.outx')),
            key=lambda p: p.name,
        )
        if not out_files:
            raise FileNotFoundError(f"No .out or .outx files found in: {folder}")
        self.datasets: List[PSSEDataset] = [PSSEDataset(str(f)) for f in out_files]

    def channel_names(self) -> List[str]:
        if self.datasets:
            return self.datasets[0].channels
        return []


# ══════════════════════════════════════════════════════════════════════════════
# FIELD DATA (CSV / XLSX with date-time first column)
# ══════════════════════════════════════════════════════════════════════════════

_DATETIME_FMTS = [
    '%Y-%m-%d %H:%M:%S.%f',
    '%Y-%m-%d %H:%M:%S',
    '%d/%m/%Y %H:%M:%S.%f',
    '%d/%m/%Y %H:%M:%S',
    '%d/%m/%Y %H:%M',
    '%H:%M:%S.%f',
    '%H:%M:%S',
]

def _parse_datetime_col(raw_values):
    """Convert a list of raw cell values to float seconds since the first entry.
    Tries multiple datetime formats; falls back to treating values as plain floats."""
    from datetime import datetime as _dt

    # Try to parse as datetime strings
    parsed = []
    fmt_used = None
    for val in raw_values:
        s = str(val).strip() if val is not None else ''
        if fmt_used is None:
            for fmt in _DATETIME_FMTS:
                try:
                    parsed.append(_dt.strptime(s, fmt))
                    fmt_used = fmt
                    break
                except ValueError:
                    continue
            else:
                # No datetime format matched — treat entire column as floats
                try:
                    return np.array([float(v) for v in raw_values], dtype=float)
                except (TypeError, ValueError):
                    return np.arange(len(raw_values), dtype=float)
        else:
            try:
                parsed.append(_dt.strptime(s, fmt_used))
            except ValueError:
                parsed.append(parsed[-1])   # repeat last on parse failure

    if not parsed:
        return np.arange(len(raw_values), dtype=float)

    t0 = parsed[0]
    return np.array([(p - t0).total_seconds() for p in parsed], dtype=float)


class FieldDataset:
    """
    Loads a single CSV or XLSX file for field data overlay.
    - Row 0 (or header row): column names
    - Column 0: date/time axis (converted to seconds since first entry)
    - Remaining columns: signal data
    """

    def __init__(self, path: str):
        self.path  = path
        self.name  = Path(path).stem
        self._time:     Optional[np.ndarray] = None
        self._data:     Dict[str, np.ndarray] = {}
        self._channels: List[str] = []

    def _load(self):
        if self._time is not None:
            return
        ext = Path(self.path).suffix.lower()
        if ext == '.xlsx':
            self._load_xlsx()
        else:
            self._load_csv()

    def _load_xlsx(self):
        import openpyxl
        wb = openpyxl.load_workbook(self.path, data_only=True, read_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
        if len(rows) < 2:
            raise ValueError("Field data XLSX must have a header row and at least one data row.")
        headers = [str(h).strip() if h is not None else f'col{i}'
                   for i, h in enumerate(rows[0])]
        data_rows = rows[1:]
        self._parse_rows(headers, data_rows)

    def _load_csv(self):
        import csv
        with open(self.path, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)
        if len(rows) < 2:
            raise ValueError("Field data CSV must have a header row and at least one data row.")
        headers = [h.strip() for h in rows[0]]
        data_rows = rows[1:]
        # Convert strings to native types
        converted = []
        for row in data_rows:
            converted.append([_try_float(v) for v in row])
        self._parse_rows(headers, converted)

    def _parse_rows(self, headers: List[str], data_rows):
        n_cols = len(headers)
        columns = [[] for _ in range(n_cols)]
        for row in data_rows:
            for ci in range(n_cols):
                columns[ci].append(row[ci] if ci < len(row) else None)

        self._time = _parse_datetime_col(columns[0])

        for ci in range(1, n_cols):
            name = headers[ci] if ci < len(headers) else f'col{ci}'
            try:
                arr = np.array([float(v) if v is not None else np.nan
                                for v in columns[ci]], dtype=float)
            except (TypeError, ValueError):
                continue
            self._data[name] = arr
            self._channels.append(name)

    @property
    def channels(self) -> List[str]:
        self._load()
        return list(self._channels)

    def get(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        self._load()
        if name not in self._data:
            raise KeyError(f"Field channel '{name}' not found.")
        return self._time, self._data[name]


def _try_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v   # keep as string (e.g. datetime column)


# ══════════════════════════════════════════════════════════════════════════════
# CHANNEL BROWSER (left panel)
# ══════════════════════════════════════════════════════════════════════════════

class _ChannelItem(QTreeWidgetItem):
    """A draggable leaf item representing one signal channel."""

    def __init__(self, parent: QTreeWidgetItem, source: str,
                 name: str, units: str = ''):
        display = f"{name}  [{units}]" if units else name
        super().__init__(parent, [display])
        self.channel_source = source
        self.channel_name   = name
        self.channel_units  = units
        self.setToolTip(0, f"[{source}]  {name}" + (f"  ({units})" if units else ''))


FIELD_COLOR = '#2a9d8f'   # teal for field data

class ChannelBrowser(QTreeWidget):
    """Tree widget with three top-level groups (PSSE / PSCAD / Field Data).
    Channels are drag-source items carrying JSON MIME data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setColumnCount(1)
        self.setAlternatingRowColors(True)

        bold = QFont(); bold.setBold(True)

        self._psse_root  = QTreeWidgetItem(self, ['PSSE'])
        self._psse_root.setFont(0, bold)
        self._psse_root.setForeground(0, QColor(PSSE_COLOR))

        self._pscad_root = QTreeWidgetItem(self, ['PSCAD'])
        self._pscad_root.setFont(0, bold)
        self._pscad_root.setForeground(0, QColor(PSCAD_COLOR))

        self._field_root = QTreeWidgetItem(self, ['Field Data'])
        self._field_root.setFont(0, bold)
        self._field_root.setForeground(0, QColor(FIELD_COLOR))

        self.expandAll()

    # ── Populate ─────────────────────────────────────────────────────────────

    def set_psse_channels(self, names: List[str]):
        self._psse_root.takeChildren()
        for name in names:
            _ChannelItem(self._psse_root, 'PSSE', name)
        self._psse_root.setExpanded(True)

    def set_pscad_channels(self, names: List[str], units_fn=None):
        self._pscad_root.takeChildren()
        for name in names:
            units = units_fn(name) if units_fn else ''
            _ChannelItem(self._pscad_root, 'PSCAD', name, units)
        self._pscad_root.setExpanded(True)

    def set_field_channels(self, names: List[str]):
        self._field_root.takeChildren()
        for name in names:
            _ChannelItem(self._field_root, 'Field', name)
        self._field_root.setExpanded(True)

    def filter_text(self, text: str):
        text = text.lower()
        for root in (self._psse_root, self._pscad_root, self._field_root):
            for i in range(root.childCount()):
                item = root.child(i)
                item.setHidden(bool(text) and text not in item.text(0).lower())

    # ── Drag ─────────────────────────────────────────────────────────────────

    def startDrag(self, supported_actions):
        items = [i for i in self.selectedItems() if isinstance(i, _ChannelItem)]
        if not items:
            return
        payload = json.dumps([
            {'source': i.channel_source,
             'name':   i.channel_name,
             'units':  i.channel_units}
            for i in items
        ]).encode()
        mime = QMimeData()
        mime.setData(CHANNEL_MIME, QByteArray(payload))
        mime.setText(', '.join(i.channel_name for i in items))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec_(Qt.CopyAction)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT CONFIG DIALOG
# ══════════════════════════════════════════════════════════════════════════════

class PlotConfigDialog(QDialog):
    """Edit title, axis labels, legend, axis limits, bands, and per-channel transforms."""

    def __init__(self, config: dict, assigned: List[dict], parent=None):
        super().__init__(parent)
        self._assigned = assigned   # reference – read-only in dialog
        self.setWindowTitle("Configure Plot")
        self.setMinimumWidth(520)
        layout = QFormLayout(self)
        layout.setSpacing(6)

        self._title  = QLineEdit(config.get('title',  ''))
        self._xlabel = QLineEdit(config.get('xlabel', 'Time (s)'))
        self._ylabel = QLineEdit(config.get('ylabel', ''))
        self._legend = QCheckBox()
        self._legend.setChecked(config.get('legend', True))

        layout.addRow("Title:",   self._title)
        layout.addRow("X label:", self._xlabel)
        layout.addRow("Y label:", self._ylabel)
        layout.addRow("Legend:",  self._legend)

        # Axis limits – leave blank for auto-scale
        def _fmt(v):
            return '' if v is None else str(v)

        lim_note = QLabel("Leave blank for auto-scale")
        lim_note.setStyleSheet("color: #888; font-size: 9px;")
        layout.addRow("", lim_note)

        xrow = QHBoxLayout()
        self._xmin = QLineEdit(_fmt(config.get('xmin')))
        self._xmax = QLineEdit(_fmt(config.get('xmax')))
        self._xmin.setPlaceholderText("min")
        self._xmax.setPlaceholderText("max")
        xrow.addWidget(QLabel("min:")); xrow.addWidget(self._xmin)
        xrow.addWidget(QLabel("max:")); xrow.addWidget(self._xmax)
        layout.addRow("X limits:", xrow)

        yrow = QHBoxLayout()
        self._ymin = QLineEdit(_fmt(config.get('ymin')))
        self._ymax = QLineEdit(_fmt(config.get('ymax')))
        self._ymin.setPlaceholderText("min")
        self._ymax.setPlaceholderText("max")
        yrow.addWidget(QLabel("min:")); yrow.addWidget(self._ymin)
        yrow.addWidget(QLabel("max:")); yrow.addWidget(self._ymax)
        layout.addRow("Y limits:", yrow)

        # ── ±10 % bands ───────────────────────────────────────────────────────
        band_sep = QLabel("─── ±10 % bands ──────────────────────────────────")
        band_sep.setStyleSheet("color: #888; font-size: 9px;")
        layout.addRow(band_sep)

        self._bands_enabled = QCheckBox()
        self._bands_enabled.setChecked(config.get('bands_enabled', False))
        layout.addRow("Enable ±10% bands:", self._bands_enabled)

        self._bands_source = QComboBox()
        self._bands_source.addItems(['PSCAD', 'PSSE', 'Both'])
        saved = config.get('bands_source', 'PSCAD')
        idx = self._bands_source.findText(saved)
        if idx >= 0:
            self._bands_source.setCurrentIndex(idx)
        self._bands_enabled.toggled.connect(self._bands_source.setEnabled)
        self._bands_source.setEnabled(self._bands_enabled.isChecked())
        layout.addRow("Apply bands to:", self._bands_source)

        # ── Channel transforms ────────────────────────────────────────────────
        xfm_sep = QLabel("─── Channel transforms ───────────────────────────")
        xfm_sep.setStyleSheet("color: #888; font-size: 9px;")
        layout.addRow(xfm_sep)

        xfm_note = QLabel(
            "Use  y  for the raw signal,  t  for time.  "
            "Examples:  y * 50 + 10    y / 1000    abs(y)    y * t"
        )
        xfm_note.setStyleSheet("color: #666; font-size: 9px;")
        xfm_note.setWordWrap(True)
        layout.addRow("", xfm_note)

        n = len(assigned)
        self._xfm_table = QTableWidget(n, 2)
        self._xfm_table.setHorizontalHeaderLabels(['Channel', 'Expression'])
        self._xfm_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._xfm_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._xfm_table.verticalHeader().setVisible(False)
        self._xfm_table.setMinimumHeight(min(n * 28 + 28, 180))
        for i, ch in enumerate(assigned):
            name_item = QTableWidgetItem(f"[{ch['source']}]  {ch['name']}")
            name_item.setFlags(Qt.ItemIsEnabled)   # read-only
            expr_item = QTableWidgetItem(ch.get('transform', 'y'))
            self._xfm_table.setItem(i, 0, name_item)
            self._xfm_table.setItem(i, 1, expr_item)
        if n:
            layout.addRow("Transforms:", self._xfm_table)

        # ── Signal analysis ───────────────────────────────────────────────────
        ana_sep = QLabel("─── Signal analysis ──────────────────────────────")
        ana_sep.setStyleSheet("color: #888; font-size: 9px;")
        layout.addRow(ana_sep)

        self._ana_enabled = QCheckBox()
        self._ana_enabled.setChecked(config.get('analysis_enabled', False))
        layout.addRow("Enable analysis:", self._ana_enabled)

        self._ana_source = QComboBox()
        self._ana_source.addItems(['PSSE', 'PSCAD', 'Field', 'Both', 'All'])
        saved_src = config.get('analysis_source', 'Both')
        aidx = self._ana_source.findText(saved_src)
        if aidx >= 0:
            self._ana_source.setCurrentIndex(aidx)
        layout.addRow("Analyse trace(s):", self._ana_source)

        self._ana_settle = QDoubleSpinBox()
        self._ana_settle.setRange(0.1, 50.0)
        self._ana_settle.setSingleStep(0.5)
        self._ana_settle.setDecimals(1)
        self._ana_settle.setSuffix(" %")
        self._ana_settle.setValue(config.get('analysis_settle_pct', 2.0))
        layout.addRow("Settle band (±):", self._ana_settle)

        def _toggle_ana(checked):
            self._ana_source.setEnabled(checked)
            self._ana_settle.setEnabled(checked)
        self._ana_enabled.toggled.connect(_toggle_ana)
        _toggle_ana(self._ana_enabled.isChecked())

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    @staticmethod
    def _parse(text: str) -> Optional[float]:
        try:
            return float(text.strip())
        except ValueError:
            return None

    def result_config(self) -> dict:
        return {
            'title':              self._title.text(),
            'xlabel':             self._xlabel.text(),
            'ylabel':             self._ylabel.text(),
            'legend':             self._legend.isChecked(),
            'xmin':               self._parse(self._xmin.text()),
            'xmax':               self._parse(self._xmax.text()),
            'ymin':               self._parse(self._ymin.text()),
            'ymax':               self._parse(self._ymax.text()),
            'bands_enabled':      self._bands_enabled.isChecked(),
            'bands_source':       self._bands_source.currentText(),
            'analysis_enabled':   self._ana_enabled.isChecked(),
            'analysis_source':    self._ana_source.currentText(),
            'analysis_settle_pct': self._ana_settle.value(),
        }

    def result_assigned(self) -> List[dict]:
        """Return the assigned list with updated transform expressions."""
        updated = []
        for i, ch in enumerate(self._assigned):
            ch_copy = dict(ch)
            if i < self._xfm_table.rowCount():
                item = self._xfm_table.item(i, 1)
                expr = item.text().strip() if item else ''
                ch_copy['transform'] = expr if expr else 'y'
            updated.append(ch_copy)
        return updated


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL PLOT WIDGET
# ══════════════════════════════════════════════════════════════════════════════

class PlotWidget(QFrame):
    """
    A single matplotlib panel supporting:
    - channel drag-and-drop
    - rectangle drag-to-zoom with reset
    - hover crosshair with interpolated readout for all curves
    - configurable axis limits, labels, legend
    """

    def __init__(self, row: int, col: int, parent=None):
        super().__init__(parent)
        self.row, self.col = row, col
        self.setFrameShape(QFrame.Box)
        self.setFrameShadow(QFrame.Plain)
        self.setAcceptDrops(True)
        self.setMinimumSize(220, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ── State ─────────────────────────────────────────────────────────────
        self.plot_config: dict = {
            'title': '', 'xlabel': 'Time (s)', 'ylabel': '', 'legend': True,
            'xmin': None, 'xmax': None, 'ymin': None, 'ymax': None,
            'bands_enabled': False, 'bands_source': 'PSCAD',
            'analysis_enabled': False, 'analysis_source': 'Both', 'analysis_settle_pct': 2.0,
        }
        self.assigned: List[dict] = []
        self._psse_ds:   Optional[PSSEDataset]  = None
        self._pscad_src: Optional[PSCADFolder]  = None
        self._field_ds:  Optional[FieldDataset] = None
        self._time_offset: float = 0.0
        self._global_xmin: Optional[float] = None
        self._global_xmax: Optional[float] = None

        # hover / zoom transient state
        self._lines      = []          # Line2D objects from last refresh
        self._vline      = None        # crosshair vertical line
        self._hover_ann  = None        # annotation box
        self._rect_sel   = None        # RectangleSelector instance
        self._zoom_active = False

        # ── Layout ────────────────────────────────────────────────────────────
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(2, 2, 2, 2)
        vbox.setSpacing(0)

        self.fig = Figure(tight_layout=True)
        self.ax  = self.fig.add_subplot(111)
        self.ax.yaxis.get_major_formatter().set_useOffset(False)
        self.ax.yaxis.get_major_formatter().set_scientific(False)
        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(self.canvas, 1)

        # Bottom mini-toolbar
        bar = QHBoxLayout()
        bar.setContentsMargins(2, 0, 2, 2)
        bar.setSpacing(2)

        self._info_lbl = QLabel("")
        self._info_lbl.setStyleSheet("font-size: 9px; color: #666;")
        bar.addWidget(self._info_lbl, 1)

        self._zoom_btn = QPushButton("🔍 Zoom")
        self._zoom_btn.setCheckable(True)
        self._zoom_btn.setFixedHeight(20)
        self._zoom_btn.setStyleSheet("font-size: 9px;")
        self._zoom_btn.setToolTip("Drag a rectangle to zoom in")
        self._zoom_btn.toggled.connect(self._toggle_zoom)

        reset_btn = QPushButton("⊡ Reset")
        reset_btn.setFixedHeight(20)
        reset_btn.setStyleSheet("font-size: 9px;")
        reset_btn.setToolTip("Reset zoom to configured limits")
        reset_btn.clicked.connect(self._reset_zoom)

        cfg_btn = QPushButton("⚙ Config")
        cfg_btn.setFixedHeight(20)
        cfg_btn.setStyleSheet("font-size: 9px;")
        cfg_btn.setToolTip("Configure title / axis labels / limits / legend")
        cfg_btn.clicked.connect(self._open_config)

        clr_btn = QPushButton("✕ Clear")
        clr_btn.setFixedHeight(20)
        clr_btn.setStyleSheet("font-size: 9px;")
        clr_btn.setToolTip("Remove all channels from this plot")
        clr_btn.clicked.connect(self.clear)

        for w in (self._zoom_btn, reset_btn, cfg_btn, clr_btn):
            bar.addWidget(w)
        vbox.addLayout(bar)

        # ── Mouse events ──────────────────────────────────────────────────────
        self.canvas.mpl_connect('motion_notify_event', self._on_hover)
        self.canvas.mpl_connect('axes_leave_event',    self._on_axes_leave)

        self._draw_placeholder()

    # ── Drop handling ─────────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(CHANNEL_MIME):
            event.acceptProposedAction()
            self.setStyleSheet(
                "QFrame { border: 2px solid #0078d7; background: #e8f4fd; }"
            )

    def dragLeaveEvent(self, event):
        self.setStyleSheet("")

    def dropEvent(self, event):
        self.setStyleSheet("")
        if not event.mimeData().hasFormat(CHANNEL_MIME):
            return
        payload = json.loads(bytes(event.mimeData().data(CHANNEL_MIME)).decode())
        for ch in payload:
            if ch not in self.assigned:
                self.assigned.append(ch)
        event.acceptProposedAction()
        self.refresh()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_datasets(self, psse: Optional[PSSEDataset],
                     pscad: Optional[PSCADFolder]):
        self._psse_ds, self._pscad_src = psse, pscad
        self.refresh()

    def set_field_ds(self, ds: Optional['FieldDataset']):
        self._field_ds = ds
        self.refresh()

    def set_time_offset(self, offset: float):
        self._time_offset = offset
        self.refresh()

    def clear(self):
        self.assigned.clear()
        self.refresh()

    def snapshot(self) -> dict:
        return {
            'config':   copy.deepcopy(self.plot_config),
            'assigned': copy.deepcopy(self.assigned),
        }

    def restore(self, snap: dict):
        self.plot_config = copy.deepcopy(snap.get('config', self.plot_config))
        self.assigned    = copy.deepcopy(snap.get('assigned', []))

    # ── Rendering ─────────────────────────────────────────────────────────────

    def refresh(self):
        # Clear transient hover/zoom state before cla()
        self._vline     = None
        self._hover_ann = None
        self._rect_sel  = None

        self.ax.cla()
        self.ax.yaxis.get_major_formatter().set_useOffset(False)
        self.ax.yaxis.get_major_formatter().set_scientific(False)
        self._lines = []
        errors = []

        cfg = self.plot_config
        bands_on  = cfg.get('bands_enabled', False)
        bands_src = cfg.get('bands_source', 'PSCAD')

        for ci, ch in enumerate(self.assigned):
            color = LINE_COLORS[ci % len(LINE_COLORS)]
            try:
                t, y = self._fetch(ch)
                if ch['source'] == 'PSSE':
                    t = t + self._time_offset
                transform = ch.get('transform', 'y')
                if transform.strip() not in ('y', ''):
                    _safe = {'y': y, 't': t, 'np': np,
                             'abs': np.abs, 'sqrt': np.sqrt, 'log': np.log,
                             'exp': np.exp, 'sin': np.sin, 'cos': np.cos,
                             'pi': np.pi, '__builtins__': {}}
                    y = eval(transform, _safe)   # noqa: S307 – user-controlled expression
                label = f"{ch['name']}_{ch['source']}"
                if ch.get('units'):
                    label += f"  [{ch['units']}]"
                line, = self.ax.plot(t, y, color=color, linewidth=1.1, label=label)
                self._lines.append(line)
                # ±10 % bands
                if bands_on and (bands_src == 'Both' or ch['source'] == bands_src):
                    self.ax.plot(t, y * 1.1, color=color, linewidth=0.8,
                                 linestyle='--', alpha=0.65, label='_nolegend_')
                    self.ax.plot(t, y * 0.9, color=color, linewidth=0.8,
                                 linestyle='--', alpha=0.65, label='_nolegend_')
            except Exception as ex:
                errors.append(str(ex))
        if cfg.get('title'):
            self.ax.set_title(cfg['title'], fontsize=9, pad=3)
        if cfg.get('xlabel'):
            self.ax.set_xlabel(cfg['xlabel'], fontsize=8)
        if cfg.get('ylabel'):
            self.ax.set_ylabel(cfg['ylabel'], fontsize=8)
        if self._lines and cfg.get('legend'):
            self.ax.legend(fontsize=7, loc='best')
        self.ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

        # ── Signal analysis overlay ───────────────────────────────────────────
        if cfg.get('analysis_enabled') and self._lines:
            ana_src    = cfg.get('analysis_source', 'Both')
            settle_pct = cfg.get('analysis_settle_pct', 2.0)
            summaries  = []
            for ci, (ch, line) in enumerate(zip(self.assigned, self._lines)):
                if not (ana_src == 'All'
                        or (ana_src == 'Both' and ch['source'] != 'Field')
                        or ch['source'] == ana_src):
                    continue
                color   = line.get_color()
                t_data  = np.asarray(line.get_xdata(), dtype=float)
                y_data  = np.asarray(line.get_ydata(), dtype=float)
                ax_xlim = self.ax.get_xlim()
                xmin_cfg = (self._global_xmin if self._global_xmin is not None
                            else cfg.get('xmin'))
                xmax_cfg = (self._global_xmax if self._global_xmax is not None
                            else cfg.get('xmax'))
                xlim = (xmin_cfg if xmin_cfg is not None else ax_xlim[0],
                        xmax_cfg if xmax_cfg is not None else ax_xlim[1])
                metrics = _compute_signal_metrics(t_data, y_data, settle_pct=settle_pct,
                                                  xlim=xlim)
                txt = _draw_analysis_overlay(self.ax, metrics, color,
                                             label_prefix=ch['source'])
                if txt:
                    summaries.append(txt)
            if summaries:
                self.ax.text(
                    0.98, 0.02, '\n'.join(summaries),
                    transform=self.ax.transAxes, ha='right', va='bottom',
                    fontsize=7, family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5ff',
                              edgecolor='#aaaacc', alpha=0.92),
                )

        if not self.assigned:
            self._draw_placeholder()
        else:
            self._apply_limits()
        if errors:
            self.ax.text(
                0.02, 0.98, '\n'.join(errors[:3]),
                transform=self.ax.transAxes, ha='left', va='top',
                fontsize=7, color='red',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff3f3', alpha=0.8),
            )

        self.ax.tick_params(labelsize=7)
        self.canvas.draw_idle()
        n = len(self.assigned)
        self._info_lbl.setText(f"{n} channel{'s' if n != 1 else ''}")

        # Re-arm zoom selector if it was active
        if self._zoom_active:
            self._arm_selector()

    def set_global_xlim(self, xmin: Optional[float], xmax: Optional[float]):
        self._global_xmin = xmin
        self._global_xmax = xmax
        self.refresh()

    def _apply_limits(self):
        cfg = self.plot_config
        # Global x limits override per-plot x limits
        xmin = self._global_xmin if self._global_xmin is not None else cfg.get('xmin')
        xmax = self._global_xmax if self._global_xmax is not None else cfg.get('xmax')
        ymin_cfg, ymax_cfg = cfg.get('ymin'), cfg.get('ymax')

        if xmin is not None or xmax is not None:
            cur = self.ax.get_xlim()
            self.ax.set_xlim(
                xmin if xmin is not None else cur[0],
                xmax if xmax is not None else cur[1],
            )

        if ymin_cfg is not None or ymax_cfg is not None:
            # Explicit y limits — honour them exactly
            cur = self.ax.get_ylim()
            self.ax.set_ylim(
                ymin_cfg if ymin_cfg is not None else cur[0],
                ymax_cfg if ymax_cfg is not None else cur[1],
            )
        elif self._lines:
            # No explicit y limits — auto-scale to visible x window
            _autoscale_y_to_xlim(self.ax, self._lines)

    def _fetch(self, ch: dict) -> Tuple[np.ndarray, np.ndarray]:
        if ch['source'] == 'PSSE':
            if self._psse_ds is None:
                raise RuntimeError("No PSSE dataset loaded")
            return self._psse_ds.get(ch['name'])
        elif ch['source'] == 'Field':
            if self._field_ds is None:
                raise RuntimeError("No field data loaded")
            return self._field_ds.get(ch['name'])
        else:
            if self._pscad_src is None:
                raise RuntimeError("No PSCAD dataset loaded")
            return self._pscad_src.get(ch['name'])

    def _draw_placeholder(self):
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1)
        self.ax.text(0.5, 0.5, "Drop channels here",
                     transform=self.ax.transAxes, ha='center', va='center',
                     fontsize=10, color='#bbb', style='italic')
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.canvas.draw_idle()

    # ── Config dialog ─────────────────────────────────────────────────────────

    def _open_config(self):
        dlg = PlotConfigDialog(self.plot_config, self.assigned, self)
        if dlg.exec_() == QDialog.Accepted:
            self.plot_config.update(dlg.result_config())
            self.assigned = dlg.result_assigned()
            self.refresh()

    # ── Zoom ──────────────────────────────────────────────────────────────────

    def _toggle_zoom(self, checked: bool):
        self._zoom_active = checked
        if checked:
            self._arm_selector()
            self._info_lbl.setText("Drag to zoom  |  Reset to undo")
        else:
            if self._rect_sel:
                self._rect_sel.set_active(False)
                self._rect_sel = None
            n = len(self.assigned)
            self._info_lbl.setText(f"{n} channel{'s' if n != 1 else ''}")

    def _arm_selector(self):
        self._rect_sel = RectangleSelector(
            self.ax,
            self._on_zoom_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False,
        )

    def _on_zoom_select(self, eclick, erelease):
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        y1, y2 = sorted([eclick.ydata, erelease.ydata])
        if x1 == x2 or y1 == y2:
            return
        self.ax.set_xlim(x1, x2)
        self.ax.set_ylim(y1, y2)
        self.canvas.draw_idle()

    def _reset_zoom(self):
        """Return to configured x limits (or full extent) and re-scale y."""
        self.ax.relim()
        self.ax.autoscale()
        self._apply_limits()   # re-applies x limits then auto-scales y
        self.canvas.draw_idle()

    # ── Hover crosshair ───────────────────────────────────────────────────────

    def _on_hover(self, event):
        if event.inaxes is not self.ax or not self._lines:
            self._clear_hover()
            return

        x = event.xdata
        if x is None:
            self._clear_hover()
            return

        # Remove previous transient artists
        if self._vline is not None:
            try:
                self._vline.remove()
            except Exception:
                pass
        if self._hover_ann is not None:
            try:
                self._hover_ann.remove()
            except Exception:
                pass

        self._vline = self.ax.axvline(
            x=x, color='#555', linewidth=0.8, linestyle='--', alpha=0.6,
            zorder=5,
        )

        rows = [f"t = {x:.5g}"]
        for line in self._lines:
            xd = line.get_xdata()
            yd = line.get_ydata()
            if len(xd) < 2:
                continue
            if x < xd[0] or x > xd[-1]:
                continue
            y_val = float(np.interp(x, xd, yd))
            lbl = line.get_label()
            # Truncate long labels to keep the box compact
            if len(lbl) > 30:
                lbl = lbl[:28] + '…'
            rows.append(f"{lbl}: {y_val:.5g}")

        if len(rows) > 1:
            # Position: follow the mouse in data coords,
            # but clamp so the box stays inside the axes
            ax_xlim = self.ax.get_xlim()
            ax_ylim = self.ax.get_ylim()
            x_frac = (x - ax_xlim[0]) / max(ax_xlim[1] - ax_xlim[0], 1e-30)
            ha = 'left' if x_frac < 0.65 else 'right'
            ann_x = 0.02 if ha == 'left' else 0.98

            self._hover_ann = self.ax.annotate(
                '\n'.join(rows),
                xy=(ann_x, 0.98), xycoords='axes fraction',
                ha=ha, va='top', fontsize=7,
                zorder=6,
                bbox=dict(
                    boxstyle='round,pad=0.35',
                    facecolor='lightyellow',
                    edgecolor='#aaa',
                    alpha=0.92,
                ),
            )

        self.canvas.draw_idle()

    def _on_axes_leave(self, event):
        self._clear_hover()

    def _clear_hover(self):
        changed = False
        if self._vline is not None:
            try:
                self._vline.remove()
            except Exception:
                pass
            self._vline = None
            changed = True
        if self._hover_ann is not None:
            try:
                self._hover_ann.remove()
            except Exception:
                pass
            self._hover_ann = None
            changed = True
        if changed:
            self.canvas.draw_idle()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT GRID
# ══════════════════════════════════════════════════════════════════════════════

class PlotGrid(QWidget):
    """Resizable grid of PlotWidget instances."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = 2
        self._cols = 3
        self._plots: List[List[PlotWidget]] = []
        self._grid_layout = QGridLayout(self)
        self._grid_layout.setSpacing(4)
        self._psse_ds:    Optional[PSSEDataset]  = None
        self._pscad_src:  Optional[PSCADFolder]  = None
        self._field_ds:   Optional[FieldDataset] = None
        self._time_offset = 0.0
        self._global_xmin: Optional[float] = None
        self._global_xmax: Optional[float] = None
        self._rebuild()

    # ── Grid management ──────────────────────────────────────────────────────

    def set_grid(self, rows: int, cols: int):
        if rows == self._rows and cols == self._cols:
            return
        old_snaps = [pw.snapshot() for pw in self._flat()]
        self._rows, self._cols = rows, cols
        self._rebuild()
        for pw, snap in zip(self._flat(), old_snaps):
            pw.restore(snap)
            pw.refresh()

    def _rebuild(self):
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._plots = []
        for r in range(self._rows):
            row = []
            for c in range(self._cols):
                pw = PlotWidget(r, c)
                pw.set_datasets(self._psse_ds, self._pscad_src)
                pw.set_field_ds(self._field_ds)
                pw.set_time_offset(self._time_offset)
                pw.set_global_xlim(self._global_xmin, self._global_xmax)
                self._grid_layout.addWidget(pw, r, c)
                row.append(pw)
            self._plots.append(row)

    def _flat(self) -> List[PlotWidget]:
        return [pw for row in self._plots for pw in row]

    # ── Dataset / offset propagation ─────────────────────────────────────────

    def set_datasets(self, psse: Optional[PSSEDataset],
                     pscad: Optional[PSCADFolder]):
        self._psse_ds, self._pscad_src = psse, pscad
        for pw in self._flat():
            pw.set_datasets(psse, pscad)

    def set_field_ds(self, ds: Optional[FieldDataset]):
        self._field_ds = ds
        for pw in self._flat():
            pw.set_field_ds(ds)

    def set_time_offset(self, offset: float):
        self._time_offset = offset
        for pw in self._flat():
            pw.set_time_offset(offset)

    def set_global_xlim(self, xmin: Optional[float], xmax: Optional[float]):
        self._global_xmin = xmin
        self._global_xmax = xmax
        for pw in self._flat():
            pw.set_global_xlim(xmin, xmax)

    # ── Layout snapshot ──────────────────────────────────────────────────────

    def get_layout_config(self) -> dict:
        return {
            'rows': self._rows,
            'cols': self._cols,
            'plots': [pw.snapshot() for pw in self._flat()],
        }

    # ── Off-screen render for export ─────────────────────────────────────────

    def render_page(
        self,
        psse_ds:     Optional[PSSEDataset],
        pscad_ds:    Optional[PSCADFolder],
        time_offset: float,
        layout:      dict,
        page_title:  str = '',
        global_xmin: Optional[float] = None,
        global_xmax: Optional[float] = None,
        field_ds:    Optional[FieldDataset] = None,
    ) -> Figure:
        rows   = layout.get('rows', self._rows)
        cols   = layout.get('cols', self._cols)
        snaps  = layout.get('plots', [])
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * 5.0, rows * 3.5),
            tight_layout=True,
        )
        # Normalise axes array to 2-D
        axes = np.atleast_2d(axes if cols > 1 else np.array(axes)[:, np.newaxis]
                             if rows > 1 else np.array([[axes]]))

        flat_axes = [axes[r, c] for r in range(rows) for c in range(cols)]
        for ax in flat_axes:
            ax.yaxis.get_major_formatter().set_useOffset(False)
            ax.yaxis.get_major_formatter().set_scientific(False)
        for ax, snap in zip(flat_axes, snaps):
            assigned  = snap.get('assigned', [])
            cfg       = snap.get('config', {})
            bands_on  = cfg.get('bands_enabled', False)
            bands_src = cfg.get('bands_source', 'PSCAD')
            plotted   = []   # (ch, t, y, color) for analysis
            for ci, ch in enumerate(assigned):
                color = LINE_COLORS[ci % len(LINE_COLORS)]
                try:
                    if ch['source'] == 'PSSE':
                        if psse_ds is None:
                            continue
                        t, y = psse_ds.get(ch['name'])
                        t = t + time_offset
                    elif ch['source'] == 'Field':
                        if field_ds is None:
                            continue
                        t, y = field_ds.get(ch['name'])
                    else:
                        if pscad_ds is None:
                            continue
                        t, y = pscad_ds.get(ch['name'])
                    transform = ch.get('transform', 'y')
                    if transform.strip() not in ('y', ''):
                        _safe = {'y': y, 't': t, 'np': np,
                                 'abs': np.abs, 'sqrt': np.sqrt, 'log': np.log,
                                 'exp': np.exp, 'sin': np.sin, 'cos': np.cos,
                                 'pi': np.pi, '__builtins__': {}}
                        y = eval(transform, _safe)   # noqa: S307
                    label = f"{ch['name']}_{ch['source']}"
                    if ch.get('units'):
                        label += f"  [{ch['units']}]"
                    ax.plot(t, y, color=color, linewidth=1.0, label=label)
                    plotted.append((ch, t, y, color))
                    if bands_on and (bands_src == 'Both' or ch['source'] == bands_src):
                        ax.plot(t, y * 1.1, color=color, linewidth=0.7,
                                linestyle='--', alpha=0.65, label='_nolegend_')
                        ax.plot(t, y * 0.9, color=color, linewidth=0.7,
                                linestyle='--', alpha=0.65, label='_nolegend_')
                except Exception as ex:
                    ax.text(0.5, 0.5, str(ex), transform=ax.transAxes,
                            ha='center', va='center', fontsize=7, color='red')
            if cfg.get('title'):
                ax.set_title(cfg['title'], fontsize=9, pad=3)
            if cfg.get('xlabel'):
                ax.set_xlabel(cfg['xlabel'], fontsize=8)
            if cfg.get('ylabel'):
                ax.set_ylabel(cfg['ylabel'], fontsize=8)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
            if assigned and cfg.get('legend'):
                ax.legend(fontsize=7, loc='best')
            if not assigned:
                ax.text(0.5, 0.5, '(empty)', transform=ax.transAxes,
                        ha='center', va='center', color='#ccc', fontsize=9)
            # Per-plot limits, then global x override
            xmin = global_xmin if global_xmin is not None else cfg.get('xmin')
            xmax = global_xmax if global_xmax is not None else cfg.get('xmax')
            ymin_cfg, ymax_cfg = cfg.get('ymin'), cfg.get('ymax')
            if xmin is not None or xmax is not None:
                cur = ax.get_xlim()
                ax.set_xlim(xmin if xmin is not None else cur[0],
                            xmax if xmax is not None else cur[1])
            if ymin_cfg is not None or ymax_cfg is not None:
                cur = ax.get_ylim()
                ax.set_ylim(ymin_cfg if ymin_cfg is not None else cur[0],
                            ymax_cfg if ymax_cfg is not None else cur[1])
            else:
                _autoscale_y_to_xlim(ax, ax.get_lines())
            ax.tick_params(labelsize=7)
            # Signal analysis overlay — after limits so xlim is correct
            if cfg.get('analysis_enabled') and plotted:
                ana_src    = cfg.get('analysis_source', 'Both')
                settle_pct = cfg.get('analysis_settle_pct', 2.0)
                ax_xlim    = ax.get_xlim()
                xmin_cfg   = global_xmin if global_xmin is not None else cfg.get('xmin')
                xmax_cfg   = global_xmax if global_xmax is not None else cfg.get('xmax')
                xlim       = (xmin_cfg if xmin_cfg is not None else ax_xlim[0],
                              xmax_cfg if xmax_cfg is not None else ax_xlim[1])
                summaries  = []
                for ch, t, y, color in plotted:
                    if not (ana_src == 'All'
                            or (ana_src == 'Both' and ch['source'] != 'Field')
                            or ch['source'] == ana_src):
                        continue
                    metrics = _compute_signal_metrics(t, y, settle_pct=settle_pct,
                                                      xlim=xlim)
                    txt = _draw_analysis_overlay(ax, metrics, color,
                                                 label_prefix=ch['source'])
                    if txt:
                        summaries.append(txt)
                if summaries:
                    ax.text(
                        0.98, 0.02, '\n'.join(summaries),
                        transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=6, family='monospace',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5ff',
                                  edgecolor='#aaaacc', alpha=0.92),
                    )

        if page_title:
            fig.suptitle(page_title, fontsize=12, y=1.01)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT DIALOG
# ══════════════════════════════════════════════════════════════════════════════

class ExportDialog(QDialog):
    """Configure folders, output path, format, and title xlsx for batch export."""

    # Default location to look for the title xlsx
    _DEFAULT_XLSX_DIR = r"C:\Users\CamSmith\Documents\Claude Working Folder\Benchmarking Tool"

    def __init__(self, psse_dir: str, pscad_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export All Results")
        self.setMinimumWidth(560)
        layout = QFormLayout(self)
        layout.setSpacing(8)

        # PSSE folder
        self._psse_edit = QLineEdit(psse_dir)
        btn_p = QPushButton("Browse…")
        btn_p.clicked.connect(lambda: self._pick_dir(self._psse_edit))
        h1 = QHBoxLayout(); h1.addWidget(self._psse_edit); h1.addWidget(btn_p)
        layout.addRow("PSSE folder:", h1)

        # PSCAD folder
        self._pscad_edit = QLineEdit(pscad_dir)
        btn_q = QPushButton("Browse…")
        btn_q.clicked.connect(lambda: self._pick_dir(self._pscad_edit))
        h2 = QHBoxLayout(); h2.addWidget(self._pscad_edit); h2.addWidget(btn_q)
        layout.addRow("PSCAD folder:", h2)

        # Output folder
        self._out_edit = QLineEdit()
        btn_o = QPushButton("Browse…")
        btn_o.clicked.connect(self._pick_out)
        h3 = QHBoxLayout(); h3.addWidget(self._out_edit); h3.addWidget(btn_o)
        layout.addRow("Output folder:", h3)

        # Format
        self._fmt = QComboBox()
        self._fmt.addItems(['PDF (combined)', 'PDF (per page)', 'PNG (per page)'])
        layout.addRow("Output format:", self._fmt)

        # Loop mode
        self._loop_combo = QComboBox()
        self._loop_combo.addItems([
            'Loop PSSE files  (one page per PSSE result)',
            'Loop PSCAD files (one page per PSCAD result)',
        ])
        layout.addRow("Iteration:", self._loop_combo)

        # ── Title xlsx ────────────────────────────────────────────────────────
        sep = QLabel("─── Page titles ──────────────────────────────────────")
        sep.setStyleSheet("color: #888; font-size: 9px;")
        layout.addRow(sep)

        # Auto-detect an xlsx in the default folder
        default_xlsx = self._find_default_xlsx()
        self._xlsx_edit = QLineEdit(default_xlsx)
        self._xlsx_edit.setPlaceholderText("(leave blank to use dataset filename)")
        btn_x = QPushButton("Browse…")
        btn_x.clicked.connect(self._pick_xlsx)
        h4 = QHBoxLayout(); h4.addWidget(self._xlsx_edit); h4.addWidget(btn_x)
        layout.addRow("Title xlsx:", h4)

        note = QLabel(
            "Row 1 = variable names  |  Rows 2+ = one row per exported page\n"
            "Empty cells are omitted from the title."
        )
        note.setStyleSheet("color: #666; font-size: 9px;")
        layout.addRow("", note)

        self._title_prefix = QLineEdit()
        self._title_prefix.setPlaceholderText("optional prefix added before xlsx values")
        layout.addRow("Title prefix:", self._title_prefix)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    @classmethod
    def _find_default_xlsx(cls) -> str:
        """Return the first xlsx found in the default directory, or empty string."""
        try:
            for f in Path(cls._DEFAULT_XLSX_DIR).glob('*.xlsx'):
                return str(f)
        except Exception:
            pass
        return ''

    def _pick_dir(self, edit: QLineEdit):
        folder = QFileDialog.getExistingDirectory(self, "Select folder", edit.text())
        if folder:
            edit.setText(folder)

    def _pick_out(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self._out_edit.setText(folder)

    def _pick_xlsx(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select title xlsx",
            self._DEFAULT_XLSX_DIR,
            "Excel files (*.xlsx *.xls)",
        )
        if path:
            self._xlsx_edit.setText(path)

    def params(self) -> dict:
        return {
            'psse_folder':    self._psse_edit.text().strip(),
            'pscad_folder':   self._pscad_edit.text().strip(),
            'out_folder':     self._out_edit.text().strip(),
            'format':         self._fmt.currentText(),
            'loop_psse':      self._loop_combo.currentIndex() == 0,
            'title_xlsx':     self._xlsx_edit.text().strip(),
            'title_prefix':   self._title_prefix.text().strip(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BOPPO – Benchmarking Of PSSE and PSCAD Outputs")
        self.resize(1440, 900)

        self._psse_folder:  Optional[PSSEFolder]  = None
        self._pscad_folder: Optional[PSCADFolder] = None

        # Tab 2 state
        self._t2_psse_folder:  Optional[PSSEFolder]  = None
        self._t2_pscad_folder: Optional[PSCADFolder] = None
        self._t2_field_ds:     Optional[FieldDataset] = None

        self._build_ui()
        self._build_menu()
        self.statusBar().showMessage(
            "Load a PSSE folder and a PSCAD folder, then drag channels onto plots."
        )

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        tabs.addTab(self._build_comparison_tab(), "PSSE vs PSCAD")
        tabs.addTab(self._build_field_tab(),      "Field Data Overlay")

    def _build_comparison_tab(self) -> QWidget:
        splitter = QSplitter(Qt.Horizontal)

        # ── Left panel ────────────────────────────────────────────────────────
        left = QWidget()
        left.setMaximumWidth(300)
        left.setMinimumWidth(200)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        lv.setSpacing(6)

        psse_btn = QPushButton("⬇  Load PSSE folder…")
        psse_btn.setStyleSheet(
            f"color: white; background: {PSSE_COLOR}; font-weight: bold; padding: 5px;"
        )
        psse_btn.clicked.connect(self._load_psse)
        lv.addWidget(psse_btn)

        pscad_btn = QPushButton("⬇  Load PSCAD folder…")
        pscad_btn.setStyleSheet(
            f"color: white; background: {PSCAD_COLOR}; font-weight: bold; padding: 5px;"
        )
        pscad_btn.clicked.connect(self._load_pscad)
        lv.addWidget(pscad_btn)

        # Dataset selectors
        ds_form = QFormLayout()
        self._psse_sel  = QComboBox()
        self._pscad_sel = QComboBox()
        self._psse_sel.currentIndexChanged.connect(self._on_psse_sel_changed)
        self._pscad_sel.currentIndexChanged.connect(self._on_pscad_sel_changed)
        ds_form.addRow("Preview PSSE:", self._psse_sel)
        ds_form.addRow("Preview PSCAD:", self._pscad_sel)
        lv.addLayout(ds_form)

        # Search
        self._search = QLineEdit()
        self._search.setPlaceholderText("🔍  Filter channels…")
        self._search.textChanged.connect(self._on_search)
        lv.addWidget(self._search)

        # Channel tree
        self._channel_browser = ChannelBrowser()
        lv.addWidget(self._channel_browser, 1)

        splitter.addWidget(left)

        # ── Right panel ───────────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 4, 4, 4)
        rv.setSpacing(6)

        # Toolbar row
        toolbar = QHBoxLayout()

        toolbar.addWidget(QLabel("Grid:"))
        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(1, 10)
        self._rows_spin.setValue(2)
        self._rows_spin.setFixedWidth(50)
        toolbar.addWidget(self._rows_spin)
        toolbar.addWidget(QLabel("×"))
        self._cols_spin = QSpinBox()
        self._cols_spin.setRange(1, 10)
        self._cols_spin.setValue(3)
        self._cols_spin.setFixedWidth(50)
        toolbar.addWidget(self._cols_spin)
        apply_btn = QPushButton("Apply")
        apply_btn.setFixedWidth(60)
        apply_btn.clicked.connect(self._apply_grid)
        toolbar.addWidget(apply_btn)

        toolbar.addSpacing(24)
        toolbar.addWidget(QLabel("PSSE time offset (s):"))
        self._offset_spin = QDoubleSpinBox()
        self._offset_spin.setRange(-99999, 99999)
        self._offset_spin.setSingleStep(0.01)
        self._offset_spin.setDecimals(4)
        self._offset_spin.setValue(0.0)
        self._offset_spin.setFixedWidth(100)
        self._offset_spin.setToolTip(
            "Shift the PSSE time axis by this value (seconds).\n"
            "Positive = PSSE data shifted later."
        )
        self._offset_spin.valueChanged.connect(self._on_offset_changed)
        toolbar.addWidget(self._offset_spin)

        toolbar.addSpacing(24)
        toolbar.addWidget(QLabel("X limits:"))

        def _xlim_edit(placeholder):
            w = QLineEdit()
            w.setPlaceholderText(placeholder)
            w.setFixedWidth(72)
            w.setToolTip("Global x-axis limit applied to all plots (leave blank for auto)")
            return w

        self._gxmin_edit = _xlim_edit("min")
        self._gxmax_edit = _xlim_edit("max")
        toolbar.addWidget(self._gxmin_edit)
        toolbar.addWidget(QLabel("–"))
        toolbar.addWidget(self._gxmax_edit)

        apply_xlim_btn = QPushButton("Apply")
        apply_xlim_btn.setFixedWidth(52)
        apply_xlim_btn.setToolTip("Apply global x-axis limits to all plots")
        apply_xlim_btn.clicked.connect(self._on_global_xlim_changed)
        toolbar.addWidget(apply_xlim_btn)

        toolbar.addSpacing(24)
        save_tpl_btn = QPushButton("💾  Save Template")
        save_tpl_btn.clicked.connect(self._save_template)
        toolbar.addWidget(save_tpl_btn)
        load_tpl_btn = QPushButton("📂  Load Template")
        load_tpl_btn.clicked.connect(self._load_template)
        toolbar.addWidget(load_tpl_btn)

        toolbar.addStretch()
        export_btn = QPushButton("📄  Export All…")
        export_btn.setStyleSheet("font-weight: bold; padding: 5px 16px;")
        export_btn.clicked.connect(self._export)
        toolbar.addWidget(export_btn)
        rv.addLayout(toolbar)

        # Scrollable plot grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._plot_grid = PlotGrid()
        scroll.setWidget(self._plot_grid)
        rv.addWidget(scroll, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        return splitter

    # ── Field Data tab ────────────────────────────────────────────────────────

    def _build_field_tab(self) -> QWidget:
        splitter = QSplitter(Qt.Horizontal)

        # ── Left panel ────────────────────────────────────────────────────────
        left = QWidget()
        left.setMaximumWidth(300)
        left.setMinimumWidth(200)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        lv.setSpacing(6)

        psse_btn = QPushButton("⬇  Load PSSE folder…")
        psse_btn.setStyleSheet(
            f"color: white; background: {PSSE_COLOR}; font-weight: bold; padding: 5px;"
        )
        psse_btn.clicked.connect(self._t2_load_psse)
        lv.addWidget(psse_btn)

        pscad_btn = QPushButton("⬇  Load PSCAD folder…")
        pscad_btn.setStyleSheet(
            f"color: white; background: {PSCAD_COLOR}; font-weight: bold; padding: 5px;"
        )
        pscad_btn.clicked.connect(self._t2_load_pscad)
        lv.addWidget(pscad_btn)

        field_btn = QPushButton("⬇  Load Field Data (CSV/XLSX)…")
        field_btn.setStyleSheet(
            f"color: white; background: {FIELD_COLOR}; font-weight: bold; padding: 5px;"
        )
        field_btn.clicked.connect(self._t2_load_field)
        lv.addWidget(field_btn)

        self._t2_field_label = QLabel("No field data loaded")
        self._t2_field_label.setStyleSheet("color: #888; font-size: 9px;")
        self._t2_field_label.setWordWrap(True)
        lv.addWidget(self._t2_field_label)

        # Dataset selectors
        ds_form = QFormLayout()
        self._t2_psse_sel  = QComboBox()
        self._t2_pscad_sel = QComboBox()
        self._t2_psse_sel.currentIndexChanged.connect(self._t2_on_psse_sel_changed)
        self._t2_pscad_sel.currentIndexChanged.connect(self._t2_on_pscad_sel_changed)
        ds_form.addRow("Preview PSSE:",  self._t2_psse_sel)
        ds_form.addRow("Preview PSCAD:", self._t2_pscad_sel)
        lv.addLayout(ds_form)

        self._t2_search = QLineEdit()
        self._t2_search.setPlaceholderText("🔍  Filter channels…")
        self._t2_search.textChanged.connect(self._t2_on_search)
        lv.addWidget(self._t2_search)

        self._t2_browser = ChannelBrowser()
        lv.addWidget(self._t2_browser, 1)

        splitter.addWidget(left)

        # ── Right panel ───────────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 4, 4, 4)
        rv.setSpacing(6)

        toolbar = QHBoxLayout()

        toolbar.addWidget(QLabel("Grid:"))
        self._t2_rows_spin = QSpinBox()
        self._t2_rows_spin.setRange(1, 10)
        self._t2_rows_spin.setValue(2)
        self._t2_rows_spin.setFixedWidth(50)
        toolbar.addWidget(self._t2_rows_spin)
        toolbar.addWidget(QLabel("×"))
        self._t2_cols_spin = QSpinBox()
        self._t2_cols_spin.setRange(1, 10)
        self._t2_cols_spin.setValue(3)
        self._t2_cols_spin.setFixedWidth(50)
        toolbar.addWidget(self._t2_cols_spin)
        apply_btn = QPushButton("Apply")
        apply_btn.setFixedWidth(60)
        apply_btn.clicked.connect(self._t2_apply_grid)
        toolbar.addWidget(apply_btn)

        toolbar.addSpacing(24)
        toolbar.addWidget(QLabel("PSSE time offset (s):"))
        self._t2_offset_spin = QDoubleSpinBox()
        self._t2_offset_spin.setRange(-99999, 99999)
        self._t2_offset_spin.setSingleStep(0.01)
        self._t2_offset_spin.setDecimals(4)
        self._t2_offset_spin.setValue(0.0)
        self._t2_offset_spin.setFixedWidth(100)
        self._t2_offset_spin.valueChanged.connect(self._t2_on_offset_changed)
        toolbar.addWidget(self._t2_offset_spin)

        toolbar.addSpacing(24)
        toolbar.addWidget(QLabel("X limits:"))

        def _xlim_edit(ph):
            w = QLineEdit()
            w.setPlaceholderText(ph)
            w.setFixedWidth(72)
            return w

        self._t2_gxmin_edit = _xlim_edit("min")
        self._t2_gxmax_edit = _xlim_edit("max")
        toolbar.addWidget(self._t2_gxmin_edit)
        toolbar.addWidget(QLabel("–"))
        toolbar.addWidget(self._t2_gxmax_edit)
        apply_xlim_btn = QPushButton("Apply")
        apply_xlim_btn.setFixedWidth(52)
        apply_xlim_btn.clicked.connect(self._t2_on_global_xlim_changed)
        toolbar.addWidget(apply_xlim_btn)

        toolbar.addSpacing(24)
        save_tpl_btn = QPushButton("💾  Save Template")
        save_tpl_btn.clicked.connect(self._t2_save_template)
        toolbar.addWidget(save_tpl_btn)
        load_tpl_btn = QPushButton("📂  Load Template")
        load_tpl_btn.clicked.connect(self._t2_load_template)
        toolbar.addWidget(load_tpl_btn)

        toolbar.addStretch()
        export_btn = QPushButton("📄  Export All…")
        export_btn.setStyleSheet("font-weight: bold; padding: 5px 16px;")
        export_btn.clicked.connect(self._t2_export)
        toolbar.addWidget(export_btn)
        rv.addLayout(toolbar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._t2_plot_grid = PlotGrid()
        scroll.setWidget(self._t2_plot_grid)
        rv.addWidget(scroll, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        return splitter

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        fm.addAction("Load PSSE folder…",  self._load_psse,  "Ctrl+Shift+P")
        fm.addAction("Load PSCAD folder…", self._load_pscad, "Ctrl+Shift+C")
        fm.addSeparator()
        fm.addAction("Export All…", self._export, "Ctrl+E")
        fm.addSeparator()
        fm.addAction("Quit", self.close, "Ctrl+Q")

        hm = mb.addMenu("&Help")
        hm.addAction("About BOPPO", self._about)

    # ── Load handlers ─────────────────────────────────────────────────────────

    def _load_psse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select PSS/E results folder"
        )
        if not folder:
            return
        try:
            self._psse_folder = PSSEFolder(folder)
        except Exception as ex:
            QMessageBox.critical(self, "Load Error", f"PSSE:\n{ex}")
            return

        self._psse_sel.blockSignals(True)
        self._psse_sel.clear()
        for ds in self._psse_folder.datasets:
            self._psse_sel.addItem(ds.name)
        self._psse_sel.blockSignals(False)

        try:
            names = self._psse_folder.channel_names()
            self._channel_browser.set_psse_channels(names)
            self.statusBar().showMessage(
                f"PSSE: {len(self._psse_folder.datasets)} file(s), {len(names)} channels  ·  "
                + self.statusBar().currentMessage().split("·")[-1].strip()
            )
        except Exception as ex:
            QMessageBox.warning(
                self, "PSSE Channel Warning",
                f"Could not read channel names:\n{ex}\n\n"
                "You can still configure plots, but data will load on first use."
            )

        self._refresh_preview_datasets()

    def _load_pscad(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select PSCAD results folder"
        )
        if not folder:
            return
        try:
            if list(Path(folder).glob('*.psout')):
                self._pscad_folder = PSCADPsoutFolder(folder)
            else:
                self._pscad_folder = PSCADFolder(folder)
        except Exception as ex:
            QMessageBox.critical(self, "Load Error", f"PSCAD:\n{ex}")
            return

        self._pscad_sel.blockSignals(True)
        self._pscad_sel.clear()
        for ds in self._pscad_folder.datasets:
            self._pscad_sel.addItem(ds.name)
        self._pscad_sel.blockSignals(False)

        names = self._pscad_folder.channel_names()
        self._channel_browser.set_pscad_channels(
            names, units_fn=self._pscad_folder.channel_units
        )
        self.statusBar().showMessage(
            self.statusBar().currentMessage().split("·")[0].strip()
            + f"  ·  PSCAD: {len(names)} channels"
        )
        self._refresh_preview_datasets()

    def _on_psse_sel_changed(self, idx: int):
        self._refresh_preview_datasets()

    def _on_pscad_sel_changed(self, idx: int):
        self._refresh_preview_datasets()

    def _refresh_preview_datasets(self):
        psse_ds = (
            self._psse_folder.datasets[self._psse_sel.currentIndex()]
            if self._psse_folder and self._psse_folder.datasets
            and 0 <= self._psse_sel.currentIndex() < len(self._psse_folder.datasets)
            else None
        )
        pscad_idx = self._pscad_sel.currentIndex()
        pscad_src = (
            self._pscad_folder.datasets[pscad_idx]
            if self._pscad_folder and self._pscad_folder.datasets
            and 0 <= pscad_idx < len(self._pscad_folder.datasets)
            else self._pscad_folder
        )
        self._plot_grid.set_datasets(psse_ds, pscad_src)

    # ── Grid / offset ─────────────────────────────────────────────────────────

    def _apply_grid(self):
        self._plot_grid.set_grid(
            self._rows_spin.value(), self._cols_spin.value()
        )

    def _on_offset_changed(self, val: float):
        self._plot_grid.set_time_offset(val)

    def _on_global_xlim_changed(self):
        def _parse(text):
            try:
                return float(text.strip())
            except ValueError:
                return None
        xmin = _parse(self._gxmin_edit.text())
        xmax = _parse(self._gxmax_edit.text())
        self._plot_grid.set_global_xlim(xmin, xmax)

    # ── Templates ─────────────────────────────────────────────────────────────

    def _save_template(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Template", "", "BOPPO Template (*.boppo);;All files (*)"
        )
        if not path:
            return
        if not path.endswith('.boppo'):
            path += '.boppo'
        layout = self._plot_grid.get_layout_config()
        try:
            import json
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(layout, f, indent=2)
        except Exception as ex:
            QMessageBox.critical(self, "Save Template", f"Failed to save:\n{ex}")

    def _load_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Template", "", "BOPPO Template (*.boppo);;All files (*)"
        )
        if not path:
            return
        try:
            import json
            with open(path, 'r', encoding='utf-8') as f:
                layout = json.load(f)
        except Exception as ex:
            QMessageBox.critical(self, "Load Template", f"Failed to load:\n{ex}")
            return
        rows = layout.get('rows', 2)
        cols = layout.get('cols', 3)
        self._rows_spin.setValue(rows)
        self._cols_spin.setValue(cols)
        self._plot_grid.set_grid(rows, cols)
        for pw, snap in zip(self._plot_grid._flat(), layout.get('plots', [])):
            pw.restore(snap)
            pw.refresh()

    # ── Search ────────────────────────────────────────────────────────────────

    def _on_search(self, text: str):
        self._channel_browser.filter_text(text)

    # ── Tab 2: Field Data Overlay handlers ───────────────────────────────────

    def _t2_load_psse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select PSS/E results folder")
        if not folder:
            return
        try:
            self._t2_psse_folder = PSSEFolder(folder)
        except Exception as ex:
            QMessageBox.critical(self, "Load Error", f"PSSE:\n{ex}")
            return
        self._t2_psse_sel.blockSignals(True)
        self._t2_psse_sel.clear()
        for ds in self._t2_psse_folder.datasets:
            self._t2_psse_sel.addItem(ds.name)
        self._t2_psse_sel.blockSignals(False)
        try:
            names = self._t2_psse_folder.channel_names()
            self._t2_browser.set_psse_channels(names)
        except Exception as ex:
            QMessageBox.warning(self, "PSSE Channel Warning", str(ex))
        self._t2_refresh_datasets()

    def _t2_load_pscad(self):
        folder = QFileDialog.getExistingDirectory(self, "Select PSCAD results folder")
        if not folder:
            return
        try:
            if list(Path(folder).glob('*.psout')):
                self._t2_pscad_folder = PSCADPsoutFolder(folder)
            else:
                self._t2_pscad_folder = PSCADFolder(folder)
        except Exception as ex:
            QMessageBox.critical(self, "Load Error", f"PSCAD:\n{ex}")
            return
        self._t2_pscad_sel.blockSignals(True)
        self._t2_pscad_sel.clear()
        for ds in self._t2_pscad_folder.datasets:
            self._t2_pscad_sel.addItem(ds.name)
        self._t2_pscad_sel.blockSignals(False)
        names = self._t2_pscad_folder.channel_names()
        self._t2_browser.set_pscad_channels(
            names, units_fn=self._t2_pscad_folder.channel_units
        )
        self._t2_refresh_datasets()

    def _t2_load_field(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select field data file", "",
            "Field Data (*.csv *.xlsx);;All files (*)"
        )
        if not path:
            return
        try:
            ds = FieldDataset(path)
            names = ds.channels   # trigger load + validate
        except Exception as ex:
            QMessageBox.critical(self, "Load Error", f"Field data:\n{ex}")
            return
        self._t2_field_ds = ds
        self._t2_field_label.setText(Path(path).name)
        self._t2_browser.set_field_channels(names)
        self._t2_plot_grid.set_field_ds(ds)

    def _t2_on_psse_sel_changed(self, idx: int):
        self._t2_refresh_datasets()

    def _t2_on_pscad_sel_changed(self, idx: int):
        self._t2_refresh_datasets()

    def _t2_refresh_datasets(self):
        psse_ds = (
            self._t2_psse_folder.datasets[self._t2_psse_sel.currentIndex()]
            if self._t2_psse_folder and self._t2_psse_folder.datasets
            and 0 <= self._t2_psse_sel.currentIndex() < len(self._t2_psse_folder.datasets)
            else None
        )
        pscad_idx = self._t2_pscad_sel.currentIndex()
        pscad_src = (
            self._t2_pscad_folder.datasets[pscad_idx]
            if self._t2_pscad_folder and self._t2_pscad_folder.datasets
            and 0 <= pscad_idx < len(self._t2_pscad_folder.datasets)
            else self._t2_pscad_folder
        )
        self._t2_plot_grid.set_datasets(psse_ds, pscad_src)

    def _t2_apply_grid(self):
        self._t2_plot_grid.set_grid(
            self._t2_rows_spin.value(), self._t2_cols_spin.value()
        )

    def _t2_on_offset_changed(self, val: float):
        self._t2_plot_grid.set_time_offset(val)

    def _t2_on_global_xlim_changed(self):
        def _parse(text):
            try:
                return float(text.strip())
            except ValueError:
                return None
        self._t2_plot_grid.set_global_xlim(
            _parse(self._t2_gxmin_edit.text()),
            _parse(self._t2_gxmax_edit.text()),
        )

    def _t2_on_search(self, text: str):
        self._t2_browser.filter_text(text)

    def _t2_save_template(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Template", "", "BOPPO Template (*.boppo);;All files (*)"
        )
        if not path:
            return
        if not path.endswith('.boppo'):
            path += '.boppo'
        try:
            import json
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._t2_plot_grid.get_layout_config(), f, indent=2)
        except Exception as ex:
            QMessageBox.critical(self, "Save Template", f"Failed to save:\n{ex}")

    def _t2_load_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Template", "", "BOPPO Template (*.boppo);;All files (*)"
        )
        if not path:
            return
        try:
            import json
            with open(path, 'r', encoding='utf-8') as f:
                layout = json.load(f)
        except Exception as ex:
            QMessageBox.critical(self, "Load Template", f"Failed to load:\n{ex}")
            return
        rows = layout.get('rows', 2)
        cols = layout.get('cols', 3)
        self._t2_rows_spin.setValue(rows)
        self._t2_cols_spin.setValue(cols)
        self._t2_plot_grid.set_grid(rows, cols)
        for pw, snap in zip(self._t2_plot_grid._flat(), layout.get('plots', [])):
            pw.restore(snap)
            pw.refresh()

    def _t2_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Field Data Plot", "",
            "PDF (*.pdf);;PNG (*.png);;All files (*)"
        )
        if not path:
            return

        layout = self._t2_plot_grid.get_layout_config()
        offset = self._t2_offset_spin.value()

        # Use whichever preview datasets are currently selected
        psse_ds = (
            self._t2_psse_folder.datasets[self._t2_psse_sel.currentIndex()]
            if self._t2_psse_folder and self._t2_psse_folder.datasets
            and 0 <= self._t2_psse_sel.currentIndex() < len(self._t2_psse_folder.datasets)
            else None
        )
        pscad_ds = self._t2_pscad_folder if self._t2_pscad_folder else None

        try:
            fig = self._t2_plot_grid.render_page(
                psse_ds, pscad_ds, offset, layout,
                global_xmin=self._t2_plot_grid._global_xmin,
                global_xmax=self._t2_plot_grid._global_xmax,
                field_ds=self._t2_field_ds,
            )
            if path.lower().endswith('.png'):
                fig.savefig(path, dpi=150, bbox_inches='tight')
            else:
                if not path.lower().endswith('.pdf'):
                    path += '.pdf'
                fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            QMessageBox.information(self, "Export Complete", f"Saved to:\n{path}")
        except Exception as ex:
            plt.close('all')
            QMessageBox.critical(self, "Export Error", str(ex))

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        psse_dir  = str(self._psse_folder.folder)  if self._psse_folder  else ''
        pscad_dir = str(self._pscad_folder.folder) if self._pscad_folder else ''

        dlg = ExportDialog(psse_dir, pscad_dir, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        p = dlg.params()

        if not p['out_folder']:
            QMessageBox.warning(self, "Export", "Please select an output folder.")
            return

        layout = self._plot_grid.get_layout_config()
        offset = self._offset_spin.value()

        # Load source folders
        try:
            psse_folder  = PSSEFolder(p['psse_folder'])  if p['psse_folder']  else None
            pscad_folder = PSCADFolder(p['pscad_folder']) if p['pscad_folder'] else None
        except Exception as ex:
            QMessageBox.critical(self, "Export Error", str(ex))
            return

        if p['loop_psse']:
            primary_list = psse_folder.datasets if psse_folder else []
            fixed_pscad  = pscad_folder          # pass whole folder for cross-file routing
            fixed_psse   = None
        else:
            primary_list = pscad_folder.datasets if pscad_folder else []
            fixed_psse   = psse_folder.datasets[0] if psse_folder and psse_folder.datasets else None
            fixed_pscad  = pscad_folder

        if not primary_list:
            QMessageBox.warning(self, "Export", "No result files found to loop over.")
            return

        out_dir = Path(p['out_folder'])
        fmt     = p['format']
        prefix  = p.get('title_prefix', '').strip()

        # Load per-page titles from xlsx if provided
        xlsx_titles: List[str] = []
        if p.get('title_xlsx') and os.path.isfile(p['title_xlsx']):
            try:
                xlsx_titles = read_title_xlsx(p['title_xlsx'])
            except Exception as ex:
                QMessageBox.warning(
                    self, "Title xlsx Warning",
                    f"Could not read title xlsx — falling back to filenames.\n\n{ex}"
                )

        def _page_title(i: int, fallback: str) -> str:
            parts = []
            if prefix:
                parts.append(prefix)
            if i < len(xlsx_titles) and xlsx_titles[i]:
                parts.append(xlsx_titles[i])
            elif not prefix:
                parts.append(fallback)
            return '   '.join(parts)

        # Progress
        prog = QProgressDialog("Exporting pages…", "Cancel", 0, len(primary_list), self)
        prog.setWindowModality(Qt.WindowModal)
        prog.show()

        pdf_combined: Optional[PdfPages] = None
        if fmt == 'PDF (combined)':
            pdf_combined = PdfPages(str(out_dir / 'BOPPO_results.pdf'))

        errors = []
        n_done = 0
        for i, ds in enumerate(primary_list):
            if prog.wasCanceled():
                break
            prog.setValue(i)
            QApplication.processEvents()

            psse_ds  = ds if p['loop_psse'] else fixed_psse
            pscad_ds = fixed_pscad if p['loop_psse'] else ds

            try:
                fig = self._plot_grid.render_page(
                    psse_ds, pscad_ds, offset, layout,
                    page_title=_page_title(i, ds.name),
                    global_xmin=self._plot_grid._global_xmin,
                    global_xmax=self._plot_grid._global_xmax,
                )
                if fmt == 'PDF (combined)' and pdf_combined:
                    pdf_combined.savefig(fig, bbox_inches='tight')
                elif fmt == 'PDF (per page)':
                    fig.savefig(str(out_dir / f"{ds.name}.pdf"), bbox_inches='tight')
                elif fmt == 'PNG (per page)':
                    fig.savefig(str(out_dir / f"{ds.name}.png"),
                                dpi=150, bbox_inches='tight')
                plt.close(fig)
                n_done += 1
            except Exception as ex:
                errors.append(f"{ds.name}: {ex}")
                plt.close('all')

        if pdf_combined:
            pdf_combined.close()

        prog.setValue(len(primary_list))
        msg = f"Exported {n_done} page(s) to:\n{out_dir}"
        if errors:
            msg += f"\n\n{len(errors)} error(s):\n" + '\n'.join(errors[:5])
        QMessageBox.information(self, "Export Complete", msg)

    # ── About ─────────────────────────────────────────────────────────────────

    def _about(self):
        QMessageBox.about(
            self, "About BOPPO",
            "<h2>BOPPO</h2>"
            "<p><b>Benchmarking Of PSSE and PSCAD Outputs</b></p>"
            "<p>Load PSS/E and PSCAD results folders, drag channels onto "
            "the plot grid, then export comparison pages for every result file.</p>"
            "<p><b>Requirements:</b><br>"
            "PyQt5 · matplotlib · numpy · dyntools (for PSS/E files)</p>"
            "Developed by Akaysha PS Team, 2026</p>"
        )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
