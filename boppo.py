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
import datetime
PSSE_LOCATION = r"C:\Program Files\PTI\PSSE36\36.5\PSSPY314"
PSSE_AVAILABLE = False
if os.path.isdir(PSSE_LOCATION):
    sys.path.append(PSSE_LOCATION)
    os.environ['PATH'] = os.environ['PATH'] + ';' + PSSE_LOCATION
    try:
        import psse3605   # noqa: F401
        import psspy       # noqa: F401
        import dyntools    # noqa: F401
        PSSE_AVAILABLE = True
    except Exception:
        pass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QTreeWidget, QTreeWidgetItem, QAbstractItemView, QFrame,
    QScrollArea, QComboBox, QLineEdit, QFormLayout, QDialog,
    QDialogButtonBox, QFileDialog, QMessageBox, QProgressDialog, QColorDialog,
    QSizePolicy, QSplitter, QAction, QToolBar, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
    QListWidget, QListWidgetItem, QInputDialog,
)
from matplotlib.widgets import RectangleSelector
from PyQt5.QtCore import Qt, QMimeData, QByteArray, QEvent, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QDrag, QCursor, QKeySequence

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ─── legend label helper ─────────────────────────────────────────────────────

def _channel_legend_label(ch: dict, cfg: dict, seen_sources: Optional[set] = None) -> str:
    """Build a legend label for a plotted channel. If the channel has a
    non-empty 'legend_label' (set per-channel in PlotConfigDialog's channel
    table), that custom text is used verbatim, overriding everything else.
    Otherwise honours the per-plot 'legend_mode' setting: 'name' (channel
    name + source) or 'source' (just the source, e.g. 'PSSE' / 'PSCAD' /
    'Field'). When grouping by source, `seen_sources` (shared across a
    plot's channels) is used to suppress duplicate legend entries for
    repeated sources."""
    custom = (ch.get('legend_label') or '').strip()
    if custom:
        return custom
    if cfg.get('legend_mode', 'name') == 'source':
        key = ch.get('dataset_id', ch['source'])
        if seen_sources is not None:
            if key in seen_sources:
                return '_nolegend_'
            seen_sources.add(key)
        return ch['source']
    label = f"{ch['source']}: {ch['name']}"
    if ch.get('units'):
        label += f"  [{ch['units']}]"
    return label


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


# ─── page title helper ────────────────────────────────────────────────────────

def format_title_vars(vars_: dict) -> str:
    """
    Join a {variable: value} dict into a title fragment, e.g.
    {'SCR': 1.5, 'Pmax': 100} -> 'SCR = 1.5,  Pmax = 100'.
    Empty/None values are omitted.
    """
    parts = [f"{k} = {v}" for k, v in vars_.items() if v not in (None, '')]
    return ',  '.join(parts)

# ─── Constants ────────────────────────────────────────────────────────────────

CHANNEL_MIME = 'application/x-boppo-channel'
PSSE_COLOR   = '#1f77b4'
PSCAD_COLOR  = '#d62728'
LINE_COLORS  = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
]

# Selectable line styles: (display name, matplotlib linestyle).
LINE_STYLES: List[Tuple[str, str]] = [
    ('Solid',    '-'),
    ('Dashed',   '--'),
    ('Dash-dot', '-.'),
    ('Dotted',   ':'),
]
DEFAULT_LINE_WIDTH = 0.0   # 0 => use the drawing context's own default


def channel_style(ch: dict, ci: int, default_width: float) -> Tuple[str, str, float]:
    """Resolve a channel's (color, linestyle, linewidth).

    Per-channel overrides live on the channel dict itself ('color',
    'linestyle', 'linewidth'), so they round-trip through PlotWidget
    snapshot/restore and .boppo templates alongside 'transform' and
    'legend_label'. Blank/zero means fall back to the automatic colour
    cycle and the caller's default width.
    """
    color = (ch.get('color') or '').strip() or LINE_COLORS[ci % len(LINE_COLORS)]
    style = (ch.get('linestyle') or '').strip() or '-'
    try:
        width = float(ch.get('linewidth') or 0)
    except (TypeError, ValueError):
        width = 0.0
    return color, style, (width if width > 0 else default_width)


def draw_ref_lines(ax, cfg: dict):
    """Draw the plot's horizontal/vertical reference lines.

    Each entry is {'orient': 'h'|'v', 'value': float, 'label': str,
    'color': str, 'linestyle': str, 'linewidth': float}. A blank label
    is kept out of the legend via matplotlib's '_nolegend_' sentinel.
    """
    for rl in cfg.get('ref_lines', []) or []:
        try:
            value = float(rl.get('value'))
        except (TypeError, ValueError):
            continue
        label = (rl.get('label') or '').strip() or '_nolegend_'
        try:
            width = float(rl.get('linewidth') or 0)
        except (TypeError, ValueError):
            width = 0.0
        kw = dict(color=(rl.get('color') or '').strip() or '#444444',
                  linestyle=(rl.get('linestyle') or '').strip() or '--',
                  linewidth=width if width > 0 else 1.0,
                  label=label)
        if rl.get('orient') == 'v':
            ax.axvline(value, **kw)
        else:
            ax.axhline(value, **kw)

# Standard page sizes for PDF/PNG export, in inches (width, height).
# 'Auto (fit grid)' preserves the original behaviour of sizing the page
# to the plot grid dimensions rather than a fixed paper size.
PAGE_SIZES: Dict[str, Optional[Tuple[float, float]]] = {
    'Auto (fit grid)':   None,
    'A4 Landscape':      (11.69, 8.27),
    'A4 Portrait':       (8.27, 11.69),
    'A3 Landscape':      (16.54, 11.69),
    'A3 Portrait':       (11.69, 16.54),
    'Letter Landscape':  (11.0, 8.5),
    'Letter Portrait':   (8.5, 11.0),
}


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

    # PSCAD lets multiple PGBs share the same Desc (e.g. array/profile outputs
    # recorded once per group). Disambiguate with the group name so each is a
    # distinct, individually-selectable channel instead of colliding on lookup.
    desc_counts: Dict[str, int] = {}
    for meta in channels.values():
        desc_counts[meta['desc']] = desc_counts.get(meta['desc'], 0) + 1
    for meta in channels.values():
        if desc_counts[meta['desc']] > 1:
            meta['name'] = f"{meta['desc']} [{meta['group']}]"
        else:
            meta['name'] = meta['desc']

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
        """Return (time, signal) for channel with matching name."""
        self.load()
        for idx, meta in self.channels.items():
            if meta['name'] == name:
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
        return [m['name'] for m in self.channels.values()]

    def channel_units(self, name: str) -> str:
        for m in self.channels.values():
            if m['name'] == name:
                return m['units']
        return ''

    def get(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (time, signal), routing to the correct .out channel-split file."""
        target_idx = None
        for idx, meta in self.channels.items():
            if meta['name'] == name:
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
            list(self.folder.rglob('*.out')) + list(self.folder.rglob('*.outx')),
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
# DATASET REGISTRY (per-tab, multi-dataset)
# ══════════════════════════════════════════════════════════════════════════════

KIND_COLOR = {'PSSE': PSSE_COLOR, 'PSCAD': PSCAD_COLOR, 'Field': '#2a9d8f'}


class LoadedDataset:
    """One loaded folder/file within a tab's registry."""

    def __init__(self, kind: str, label: str, folder_obj, dataset_id: str):
        self.kind = kind              # 'PSSE' | 'PSCAD' | 'Field'
        self.label = label            # user-visible, defaults to folder.name
        self.folder_obj = folder_obj  # PSSEFolder/PSCADFolder/PSCADPsoutFolder/FieldDataset
        self.id = dataset_id          # stable short id, e.g. 'psse-0'
        self.sel_index = 0            # active internal sub-dataset
        self.path = ''

    def active_source(self):
        """Return the object exposing get(name) for the active sub-dataset."""
        ds_list = getattr(self.folder_obj, 'datasets', None)
        if ds_list and 0 <= self.sel_index < len(ds_list):
            return ds_list[self.sel_index]
        return self.folder_obj


class DatasetRegistry:
    """Holds every dataset loaded into one tab, keyed by a stable id.

    Ids are generated as ``f"{kind.lower()}-{n}"`` where n increments a per-kind
    counter that is never reused, so old assigned-channel dicts referencing a
    removed id fail gracefully rather than binding to an unrelated folder."""

    def __init__(self):
        self._datasets: "Dict[str, LoadedDataset]" = {}
        self._counters: Dict[str, int] = {'PSSE': 0, 'PSCAD': 0, 'Field': 0}

    def _next_id(self, kind: str, forced_id: Optional[str] = None) -> str:
        if forced_id is not None:
            try:
                n = int(str(forced_id).rsplit('-', 1)[-1])
                if n >= self._counters.get(kind, 0):
                    self._counters[kind] = n + 1
            except ValueError:
                pass
            return forced_id
        n = self._counters.get(kind, 0)
        self._counters[kind] = n + 1
        return f"{kind.lower()}-{n}"

    def _unique_label(self, kind: str, label: str) -> str:
        existing = {d.label for d in self._datasets.values() if d.kind == kind}
        if label not in existing:
            return label
        n = 2
        while f"{label} ({n})" in existing:
            n += 1
        return f"{label} ({n})"

    def _add(self, kind, folder_obj, path, forced_id, label) -> str:
        if not label:
            label = Path(path).name
        label = self._unique_label(kind, label)
        ds_id = self._next_id(kind, forced_id)
        entry = LoadedDataset(kind, label, folder_obj, ds_id)
        entry.path = str(path)
        self._datasets[ds_id] = entry
        return ds_id

    def add_psse(self, folder_path, forced_id=None, label=None) -> str:
        return self._add('PSSE', PSSEFolder(folder_path), folder_path, forced_id, label)

    def add_pscad(self, folder_path, forced_id=None, label=None) -> str:
        if list(Path(folder_path).glob('*.psout')):
            folder = PSCADPsoutFolder(folder_path)
        else:
            folder = PSCADFolder(folder_path)
        return self._add('PSCAD', folder, folder_path, forced_id, label)

    def add_field(self, path, forced_id=None, label=None) -> str:
        ds = FieldDataset(path)
        _ = ds.channels   # trigger load + validate
        return self._add('Field', ds, path, forced_id, label)

    def remove(self, dataset_id: str):
        self._datasets.pop(dataset_id, None)

    def get(self, dataset_id: Optional[str]) -> Optional[LoadedDataset]:
        return self._datasets.get(dataset_id) if dataset_id else None

    def all(self) -> "List[LoadedDataset]":
        return list(self._datasets.values())

    def all_of_kind(self, kind: str) -> "List[LoadedDataset]":
        return [d for d in self._datasets.values() if d.kind == kind]

    def first_of_kind(self, kind: str) -> Optional[LoadedDataset]:
        for d in self._datasets.values():
            if d.kind == kind:
                return d
        return None

    def export_manifest(self) -> List[dict]:
        return [{'id': d.id, 'kind': d.kind, 'label': d.label, 'path': d.path}
                for d in self._datasets.values()]

    @classmethod
    def from_single(cls, psse_ds=None, pscad_ds=None, field_ds=None) -> "DatasetRegistry":
        """Build a throwaway single-dataset-per-kind registry for batch export."""
        reg = cls()
        for kind, obj in (('PSSE', psse_ds), ('PSCAD', pscad_ds), ('Field', field_ds)):
            if obj is not None:
                ds_id = reg._next_id(kind)
                reg._datasets[ds_id] = LoadedDataset(kind, kind, obj, ds_id)
        return reg


def _resolve_dataset(registry, ch: dict) -> Optional[LoadedDataset]:
    """Resolve an assigned channel dict to its LoadedDataset. Falls back to the
    first loaded dataset of the channel's kind (legacy schema-v1 templates and
    batch export both rely on this)."""
    if registry is None:
        return None
    entry = registry.get(ch.get('dataset_id'))
    if entry is None:
        entry = registry.first_of_kind(ch['source'])
    return entry


# ══════════════════════════════════════════════════════════════════════════════
# CHANNEL BROWSER (left panel)
# ══════════════════════════════════════════════════════════════════════════════

FIELD_COLOR = '#2a9d8f'   # teal for field data


class _DatasetNode(QTreeWidgetItem):
    """A tree node grouping all channels of one loaded dataset."""

    def __init__(self, parent: QTreeWidgetItem, dataset_id: str, label: str):
        super().__init__(parent, [label])
        self.dataset_id = dataset_id
        self.dataset_label = label


class _ChannelItem(QTreeWidgetItem):
    """A draggable leaf item representing one signal channel."""

    def __init__(self, parent: QTreeWidgetItem, source: str,
                 name: str, units: str = '', dataset_id: str = '',
                 dataset_label: str = ''):
        display = f"{name}  [{units}]" if units else name
        super().__init__(parent, [display])
        self.channel_source = source
        self.channel_name   = name
        self.channel_units  = units
        self.channel_dataset_id = dataset_id
        self.channel_dataset_label = dataset_label
        tip = f"[{source}]  {name}" + (f"  ({units})" if units else '')
        if dataset_label:
            tip += f"\nDataset: {dataset_label}"
        self.setToolTip(0, tip)


class ChannelBrowser(QTreeWidget):
    """Tree widget grouped format -> dataset -> channel.
    Channels are drag-source items carrying JSON MIME data (incl. dataset id)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setColumnCount(1)
        self.setAlternatingRowColors(True)

        bold = QFont(); bold.setBold(True)

        self._format_roots: Dict[str, QTreeWidgetItem] = {}
        for kind, label in (('PSSE', 'PSSE'), ('PSCAD', 'PSCAD'),
                            ('Field', 'Field Data')):
            root = QTreeWidgetItem(self, [label])
            root.setFont(0, bold)
            root.setForeground(0, QColor(KIND_COLOR[kind]))
            self._format_roots[kind] = root

        self.expandAll()

    # ── Populate ─────────────────────────────────────────────────────────────

    def _find_or_create_dataset_node(self, root, dataset_id, label) -> _DatasetNode:
        for i in range(root.childCount()):
            node = root.child(i)
            if isinstance(node, _DatasetNode) and node.dataset_id == dataset_id:
                return node
        bold = QFont(); bold.setBold(True)
        node = _DatasetNode(root, dataset_id, label)
        node.setFont(0, bold)
        root.setExpanded(True)
        return node

    def set_dataset_channels(self, kind: str, dataset_id: str, label: str,
                             names: List[str], units_fn=None):
        root = self._format_roots[kind]
        node = self._find_or_create_dataset_node(root, dataset_id, label)
        node.takeChildren()
        for name in names:
            units = units_fn(name) if units_fn else ''
            _ChannelItem(node, kind, name, units, dataset_id, label)
        node.setExpanded(True)

    def remove_dataset(self, kind: str, dataset_id: str):
        root = self._format_roots.get(kind)
        if root is None:
            return
        for i in range(root.childCount()):
            node = root.child(i)
            if isinstance(node, _DatasetNode) and node.dataset_id == dataset_id:
                root.takeChild(i)
                return

    def filter_text(self, text: str):
        text = text.lower()
        for root in self._format_roots.values():
            for i in range(root.childCount()):
                node = root.child(i)
                any_visible = False
                for j in range(node.childCount()):
                    item = node.child(j)
                    hidden = bool(text) and text not in item.text(0).lower()
                    item.setHidden(hidden)
                    any_visible = any_visible or not hidden
                node.setHidden(bool(text) and not any_visible)

    # ── Drag ─────────────────────────────────────────────────────────────────

    def startDrag(self, supported_actions):
        items = [i for i in self.selectedItems() if isinstance(i, _ChannelItem)]
        if not items:
            return
        payload = json.dumps([
            {'source':        i.channel_source,
             'dataset_id':    i.channel_dataset_id,
             'dataset_label': i.channel_dataset_label,
             'name':          i.channel_name,
             'units':         i.channel_units}
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

class ColorButton(QPushButton):
    """A swatch button that opens a colour picker. Empty value == automatic."""

    def __init__(self, color: str = '', parent=None):
        super().__init__(parent)
        self.setFixedWidth(52)
        self._color = (color or '').strip()
        self.clicked.connect(self._pick)
        self._refresh()

    def _refresh(self):
        if self._color:
            self.setText('')
            self.setStyleSheet(
                f"background-color: {self._color}; border: 1px solid #888;")
            self.setToolTip(f"{self._color} — click to change, right-click to reset")
        else:
            self.setText("auto")
            self.setStyleSheet("color: #888; font-size: 9px;")
            self.setToolTip("Automatic colour — click to choose one")

    def _pick(self):
        initial = QColor(self._color) if self._color else QColor('#1f77b4')
        c = QColorDialog.getColor(initial, self, "Trace colour")
        if c.isValid():
            self._color = c.name()
            self._refresh()

    def contextMenuEvent(self, event):
        # Right-click clears back to the automatic colour cycle.
        self._color = ''
        self._refresh()

    def color(self) -> str:
        return self._color


class StyleCombo(QComboBox):
    """Line-style picker backed by LINE_STYLES."""

    def __init__(self, style: str = '-', parent=None):
        super().__init__(parent)
        for name, code in LINE_STYLES:
            self.addItem(name, code)
        idx = self.findData((style or '-').strip() or '-')
        self.setCurrentIndex(idx if idx >= 0 else 0)

    def style_code(self) -> str:
        return self.currentData()


class WidthSpin(QDoubleSpinBox):
    """Line-width picker; 0 shows as 'auto' and means 'use the default'."""

    def __init__(self, width: float = 0.0, parent=None):
        super().__init__(parent)
        self.setRange(0.0, 10.0)
        self.setSingleStep(0.1)
        self.setDecimals(1)
        self.setSpecialValueText("auto")
        try:
            self.setValue(float(width or 0))
        except (TypeError, ValueError):
            self.setValue(0.0)


class PlotConfigDialog(QDialog):
    """Edit title, axis labels, legend, axis limits, bands, per-channel
    transforms and trace styling, reference lines, and signal analysis.

    Laid out as tabs rather than one long form -- the settings fall into
    four fairly independent groups and a single form had grown tall
    enough to need scrolling.
    """

    def __init__(self, config: dict, assigned: List[dict], parent=None):
        super().__init__(parent)
        self._assigned = assigned   # reference – read-only in dialog
        self.setWindowTitle("Configure Plot")
        self.setMinimumWidth(640)

        outer = QVBoxLayout(self)
        tabs = QTabWidget()
        outer.addWidget(tabs)

        tabs.addTab(self._build_axes_tab(config), "Axes && Legend")
        tabs.addTab(self._build_channels_tab(config, assigned), "Channels")
        tabs.addTab(self._build_ref_lines_tab(config), "Reference Lines")
        tabs.addTab(self._build_analysis_tab(config), "Analysis")

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    # ── Tab 1: axes, labels, legend, limits, bands ────────────────────────
    def _build_axes_tab(self, config: dict) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        layout.setSpacing(6)

        self._title  = QLineEdit(config.get('title',  ''))
        self._xlabel = QLineEdit(config.get('xlabel', 'Time (s)'))
        self._ylabel = QLineEdit(config.get('ylabel', ''))
        self._legend = QCheckBox()
        self._legend.setChecked(config.get('legend', True))

        self._legend_mode = QComboBox()
        self._legend_mode.addItem("Channel name", 'name')
        self._legend_mode.addItem("Source only (PSSE / PSCAD)", 'source')
        mode_idx = self._legend_mode.findData(config.get('legend_mode', 'name'))
        self._legend_mode.setCurrentIndex(mode_idx if mode_idx >= 0 else 0)
        self._legend.toggled.connect(self._legend_mode.setEnabled)
        self._legend_mode.setEnabled(self._legend.isChecked())

        layout.addRow("Title:",   self._title)
        layout.addRow("X label:", self._xlabel)
        layout.addRow("Y label:", self._ylabel)
        layout.addRow("Legend:",  self._legend)
        layout.addRow("Legend labels:", self._legend_mode)

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

        return page

    # ── Tab 2: per-channel transform, legend label, colour/style/width ────
    def _build_channels_tab(self, config: dict, assigned: List[dict]) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        note = QLabel(
            "Expression: use  y  for the raw signal,  t  for time "
            "(e.g.  y * 50 + 10,  abs(y)).  "
            "Legend blank = plot default.  Colour 'auto' = automatic cycle "
            "(right-click a swatch to reset it)."
        )
        note.setStyleSheet("color: #666; font-size: 9px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        self._current_assigned = list(assigned)   # working copy; rows can be removed
        self._xfm_table = QTableWidget(0, 7)
        self._xfm_table.setHorizontalHeaderLabels(
            ['Channel', 'Expression', 'Legend', 'Colour', 'Style', 'Width', ''])
        hh = self._xfm_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        for c in (3, 4, 5, 6):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self._xfm_table.verticalHeader().setVisible(False)
        self._rebuild_xfm_table()
        layout.addWidget(self._xfm_table)

        self._no_channels_label = QLabel("No channels assigned to this plot yet.")
        self._no_channels_label.setStyleSheet("color: #888;")
        self._no_channels_label.setVisible(not self._current_assigned)
        layout.addWidget(self._no_channels_label)

        return page

    # ── Tab 3: horizontal / vertical reference lines ──────────────────────
    def _build_ref_lines_tab(self, config: dict) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        note = QLabel(
            "Horizontal lines are drawn at a Y value; vertical lines at an X "
            "value. A label, if given, appears in the plot legend."
        )
        note.setStyleSheet("color: #666; font-size: 9px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        bar = QHBoxLayout()
        add_h = QPushButton("Add Horizontal")
        add_h.clicked.connect(lambda: self._add_ref_line('h'))
        add_v = QPushButton("Add Vertical")
        add_v.clicked.connect(lambda: self._add_ref_line('v'))
        bar.addWidget(add_h); bar.addWidget(add_v); bar.addStretch(1)
        layout.addLayout(bar)

        self._ref_table = QTableWidget(0, 6)
        self._ref_table.setHorizontalHeaderLabels(
            ['Orientation', 'Value', 'Label', 'Colour', 'Style', ''])
        rh = self._ref_table.horizontalHeader()
        rh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        rh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        rh.setSectionResizeMode(2, QHeaderView.Stretch)
        for c in (3, 4, 5):
            rh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self._ref_table.verticalHeader().setVisible(False)
        layout.addWidget(self._ref_table)

        for rl in (config.get('ref_lines') or []):
            self._append_ref_row(rl)

        return page

    def _add_ref_line(self, orient: str):
        self._append_ref_row({'orient': orient, 'value': '', 'label': '',
                              'color': '#444444', 'linestyle': '--'})

    def _append_ref_row(self, rl: dict):
        r = self._ref_table.rowCount()
        self._ref_table.insertRow(r)

        orient = QComboBox()
        orient.addItem("Horizontal", 'h')
        orient.addItem("Vertical", 'v')
        oidx = orient.findData(rl.get('orient', 'h'))
        orient.setCurrentIndex(oidx if oidx >= 0 else 0)
        self._ref_table.setCellWidget(r, 0, orient)

        val = rl.get('value')
        self._ref_table.setItem(r, 1, QTableWidgetItem('' if val in (None, '') else str(val)))
        self._ref_table.setItem(r, 2, QTableWidgetItem(rl.get('label', '') or ''))
        self._ref_table.setCellWidget(r, 3, ColorButton(rl.get('color', '#444444')))
        self._ref_table.setCellWidget(r, 4, StyleCombo(rl.get('linestyle', '--')))

        rm = QPushButton("✕")
        rm.setFixedWidth(24)
        rm.setToolTip("Remove this reference line")
        rm.clicked.connect(lambda _checked, btn=rm: self._remove_ref_row(btn))
        self._ref_table.setCellWidget(r, 5, rm)

    def _remove_ref_row(self, btn):
        # Resolve the row at click time -- indices shift as rows are removed.
        for r in range(self._ref_table.rowCount()):
            if self._ref_table.cellWidget(r, 5) is btn:
                self._ref_table.removeRow(r)
                return

    # ── Tab 4: signal analysis ────────────────────────────────────────────
    def _build_analysis_tab(self, config: dict) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)

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

        return page

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
            'legend_mode':        self._legend_mode.currentData(),
            'xmin':               self._parse(self._xmin.text()),
            'xmax':               self._parse(self._xmax.text()),
            'ymin':               self._parse(self._ymin.text()),
            'ymax':               self._parse(self._ymax.text()),
            'bands_enabled':      self._bands_enabled.isChecked(),
            'bands_source':       self._bands_source.currentText(),
            'ref_lines':          self.result_ref_lines(),
            'analysis_enabled':   self._ana_enabled.isChecked(),
            'analysis_source':    self._ana_source.currentText(),
            'analysis_settle_pct': self._ana_settle.value(),
        }

    def result_ref_lines(self) -> List[dict]:
        """Reference lines from the table. Rows with an unparseable value are
        dropped, since a line with no position cannot be drawn."""
        out = []
        for r in range(self._ref_table.rowCount()):
            val_item = self._ref_table.item(r, 1)
            value = self._parse(val_item.text()) if val_item else None
            if value is None:
                continue
            label_item = self._ref_table.item(r, 2)
            out.append({
                'orient':    self._ref_table.cellWidget(r, 0).currentData(),
                'value':     value,
                'label':     label_item.text().strip() if label_item else '',
                'color':     self._ref_table.cellWidget(r, 3).color() or '#444444',
                'linestyle': self._ref_table.cellWidget(r, 4).style_code(),
            })
        return out

    def result_assigned(self) -> List[dict]:
        """Return the (possibly reduced) assigned list with updated transform
        expressions, custom legend labels, and per-trace styling, reflecting
        any channels removed via the Remove button."""
        updated = []
        for i, ch in enumerate(self._current_assigned):
            ch_copy = dict(ch)
            expr_item = self._xfm_table.item(i, 1)
            expr = expr_item.text().strip() if expr_item else ''
            ch_copy['transform'] = expr if expr else 'y'
            legend_item = self._xfm_table.item(i, 2)
            ch_copy['legend_label'] = legend_item.text().strip() if legend_item else ''
            ch_copy['color']     = self._xfm_table.cellWidget(i, 3).color()
            ch_copy['linestyle'] = self._xfm_table.cellWidget(i, 4).style_code()
            ch_copy['linewidth'] = self._xfm_table.cellWidget(i, 5).value()
            updated.append(ch_copy)
        return updated

    def _rebuild_xfm_table(self):
        self._xfm_table.setRowCount(len(self._current_assigned))
        self._xfm_table.setMinimumHeight(min(len(self._current_assigned) * 30 + 30, 240))
        for i, ch in enumerate(self._current_assigned):
            name_item = QTableWidgetItem(f"[{ch['source']}]  {ch['name']}")
            name_item.setFlags(Qt.ItemIsEnabled)   # read-only
            expr_item = QTableWidgetItem(ch.get('transform', 'y'))
            legend_item = QTableWidgetItem(ch.get('legend_label', ''))
            legend_item.setToolTip("Blank = use the plot's default legend label")
            self._xfm_table.setItem(i, 0, name_item)
            self._xfm_table.setItem(i, 1, expr_item)
            self._xfm_table.setItem(i, 2, legend_item)
            self._xfm_table.setCellWidget(i, 3, ColorButton(ch.get('color', '')))
            self._xfm_table.setCellWidget(i, 4, StyleCombo(ch.get('linestyle', '-')))
            self._xfm_table.setCellWidget(i, 5, WidthSpin(ch.get('linewidth', 0.0)))
            rm_btn = QPushButton("✕")
            rm_btn.setFixedWidth(24)
            rm_btn.setToolTip("Remove this channel from the plot")
            rm_btn.clicked.connect(lambda _checked, idx=i: self._remove_channel(idx))
            self._xfm_table.setCellWidget(i, 6, rm_btn)

    def _remove_channel(self, idx: int):
        del self._current_assigned[idx]
        self._rebuild_xfm_table()
        if hasattr(self, '_no_channels_label'):
            self._no_channels_label.setVisible(not self._current_assigned)


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
        self._registry: Optional[DatasetRegistry] = None
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

    def set_registry(self, registry: Optional['DatasetRegistry']):
        self._registry = registry
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
        seen_sources: set = set()

        cfg = self.plot_config
        bands_on  = cfg.get('bands_enabled', False)
        bands_src = cfg.get('bands_source', 'PSCAD')

        for ci, ch in enumerate(self.assigned):
            color, lstyle, lwidth = channel_style(ch, ci, 1.1)
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
                label = _channel_legend_label(ch, cfg, seen_sources)
                line, = self.ax.plot(t, y, color=color, linestyle=lstyle,
                                     linewidth=lwidth, label=label)
                self._lines.append(line)
                # ±10 % bands
                if bands_on and (bands_src == 'Both' or ch['source'] == bands_src):
                    self.ax.plot(t, y * 1.1, color=color, linewidth=0.8,
                                 linestyle='--', alpha=0.65, label='_nolegend_')
                    self.ax.plot(t, y * 0.9, color=color, linewidth=0.8,
                                 linestyle='--', alpha=0.65, label='_nolegend_')
            except Exception as ex:
                errors.append(str(ex))
        draw_ref_lines(self.ax, cfg)
        if cfg.get('title'):
            self.ax.set_title(cfg['title'], fontsize=9, pad=3)
        if cfg.get('xlabel'):
            self.ax.set_xlabel(cfg['xlabel'], fontsize=8)
        if cfg.get('ylabel'):
            self.ax.set_ylabel(cfg['ylabel'], fontsize=8)
        # Ref lines can carry a legend label of their own, so a plot with
        # only reference lines still warrants a legend.
        if (self._lines or cfg.get('ref_lines')) and cfg.get('legend'):
            handles, _ = self.ax.get_legend_handles_labels()
            if handles:
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
        entry = _resolve_dataset(self._registry, ch)
        if entry is None:
            raise RuntimeError(
                f"Dataset {ch.get('dataset_id') or ch['source']} not loaded")
        return entry.active_source().get(ch['name'])

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
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._registry: Optional[DatasetRegistry] = None
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
                pw.set_registry(self._registry)
                pw.set_time_offset(self._time_offset)
                pw.set_global_xlim(self._global_xmin, self._global_xmax)
                self._grid_layout.addWidget(pw, r, c)
                row.append(pw)
            self._plots.append(row)
        # Equal stretch on every row/col so a single panel's content
        # (e.g. a legend appearing after a drop) can't resize the grid.
        # Stretch factors on QGridLayout persist per-index even after a
        # row/col's widgets are removed, so clear stale ones (up to the
        # spinbox max) before setting the ones actually in use.
        for r in range(10):
            self._grid_layout.setRowStretch(r, 1 if r < self._rows else 0)
        for c in range(10):
            self._grid_layout.setColumnStretch(c, 1 if c < self._cols else 0)
        self._grid_layout.activate()
        self.updateGeometry()

    def _flat(self) -> List[PlotWidget]:
        return [pw for row in self._plots for pw in row]

    # ── Dataset / offset propagation ─────────────────────────────────────────

    def set_registry(self, registry: Optional[DatasetRegistry]):
        self._registry = registry
        for pw in self._flat():
            pw.set_registry(registry)

    def refresh_all(self):
        for pw in self._flat():
            pw.refresh()

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
        registry:    Optional[DatasetRegistry],
        time_offset: float,
        layout:      dict,
        page_title:  str = '',
        global_xmin: Optional[float] = None,
        global_xmax: Optional[float] = None,
        page_size:   Optional[Tuple[float, float]] = None,
    ) -> Figure:
        rows   = layout.get('rows', self._rows)
        cols   = layout.get('cols', self._cols)
        snaps  = layout.get('plots', [])
        # Build off-screen with a plain Agg canvas rather than pyplot's
        # interactive Qt5Agg backend — creating/destroying Qt5Agg figures
        # from inside a button click reenters the running QApplication's
        # Qt event loop and can crash the app (Qt5Core stack corruption).
        figsize = page_size if page_size is not None else (cols * 5.0, rows * 3.5)
        fig = Figure(figsize=figsize, tight_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(rows, cols)
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
            seen_sources: set = set()
            for ci, ch in enumerate(assigned):
                color, lstyle, lwidth = channel_style(ch, ci, 1.0)
                try:
                    entry = _resolve_dataset(registry, ch)
                    if entry is None:
                        continue
                    t, y = entry.active_source().get(ch['name'])
                    if ch['source'] == 'PSSE':
                        t = t + time_offset
                    transform = ch.get('transform', 'y')
                    if transform.strip() not in ('y', ''):
                        _safe = {'y': y, 't': t, 'np': np,
                                 'abs': np.abs, 'sqrt': np.sqrt, 'log': np.log,
                                 'exp': np.exp, 'sin': np.sin, 'cos': np.cos,
                                 'pi': np.pi, '__builtins__': {}}
                        y = eval(transform, _safe)   # noqa: S307
                    label = _channel_legend_label(ch, cfg, seen_sources)
                    ax.plot(t, y, color=color, linestyle=lstyle,
                            linewidth=lwidth, label=label)
                    plotted.append((ch, t, y, color))
                    if bands_on and (bands_src == 'Both' or ch['source'] == bands_src):
                        ax.plot(t, y * 1.1, color=color, linewidth=0.7,
                                linestyle='--', alpha=0.65, label='_nolegend_')
                        ax.plot(t, y * 0.9, color=color, linewidth=0.7,
                                linestyle='--', alpha=0.65, label='_nolegend_')
                except Exception as ex:
                    ax.text(0.5, 0.5, str(ex), transform=ax.transAxes,
                            ha='center', va='center', fontsize=7, color='red')
            draw_ref_lines(ax, cfg)
            if cfg.get('title'):
                ax.set_title(cfg['title'], fontsize=9, pad=3)
            if cfg.get('xlabel'):
                ax.set_xlabel(cfg['xlabel'], fontsize=8)
            if cfg.get('ylabel'):
                ax.set_ylabel(cfg['ylabel'], fontsize=8)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
            if (assigned or cfg.get('ref_lines')) and cfg.get('legend'):
                handles, _ = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=7, loc='best')
            if not assigned and not cfg.get('ref_lines'):
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

class PageTitleEditorDialog(QDialog):
    """
    Table editor for per-file title variables, keyed by filename.

    Column 0 is the (read-only) filename; every other column is a
    user-defined variable. Values can be typed manually or bulk-filled by
    applying a regex with named capture groups against each filename.
    """

    def __init__(self, filenames: List[str], existing: dict, parent=None,
                 columns: Optional[List[str]] = None):
        super().__init__(parent)
        self.setWindowTitle("Edit Page Titles")
        self.setMinimumSize(560, 420)
        self._filenames = list(filenames)

        v = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        btn_add_col = QPushButton("Add Column")
        btn_add_col.clicked.connect(lambda: self._add_column())
        btn_del_col = QPushButton("Remove Column")
        btn_del_col.clicked.connect(self._remove_column)
        btn_regex = QPushButton("Extract from Filename…")
        btn_regex.clicked.connect(self._extract_from_filename)
        toolbar.addWidget(btn_add_col)
        toolbar.addWidget(btn_del_col)
        toolbar.addWidget(btn_regex)
        toolbar.addStretch(1)
        v.addLayout(toolbar)

        self._table = QTableWidget(len(self._filenames), 1)
        self._table.setHorizontalHeaderLabels(["Filename"])
        self._table.horizontalHeader().setStretchLastSection(True)
        for row, name in enumerate(self._filenames):
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 0, item)
        self._table.installEventFilter(self)
        v.addWidget(self._table)

        note = QLabel("Tip: use \"Extract from Filename…\" with named capture "
                       "groups, e.g.  SCR(?P<SCR>[\\d.]+). "
                       "Ctrl+C / Ctrl+V copies and pastes cells to and from Excel.")
        note.setStyleSheet("color: #666; font-size: 9px;")
        v.addWidget(note)

        # Seed columns/values from any existing entries
        existing_vars = list(columns or [])
        for name in self._filenames:
            for var in (existing.get(name) or {}):
                if var not in existing_vars:
                    existing_vars.append(var)
        for var in existing_vars:
            self._add_column(var)
        for row, name in enumerate(self._filenames):
            for var, val in (existing.get(name) or {}).items():
                col = self._col_for_var(var)
                if col is not None:
                    self._table.setItem(row, col, QTableWidgetItem(str(val)))

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

    def eventFilter(self, obj, event):
        # QTableWidget has no built-in clipboard support.
        if obj is self._table and event.type() == QEvent.KeyPress:
            if event.matches(QKeySequence.Copy):
                self._copy_selection()
                return True
            if event.matches(QKeySequence.Paste):
                self._paste_selection()
                return True
        return super().eventFilter(obj, event)

    def _copy_selection(self):
        ranges = self._table.selectedRanges()
        if not ranges:
            return
        r = ranges[0]
        lines = []
        for row in range(r.topRow(), r.bottomRow() + 1):
            cells = []
            for col in range(r.leftColumn(), r.rightColumn() + 1):
                item = self._table.item(row, col)
                cells.append(item.text() if item else "")
            lines.append("\t".join(cells))
        QApplication.clipboard().setText("\n".join(lines))

    def _paste_selection(self):
        text = QApplication.clipboard().text()
        if not text:
            return
        rows = [line.split("\t") for line in text.replace("\r\n", "\n").rstrip("\n").split("\n")]

        start_row = max(self._table.currentRow(), 0)
        # Column 0 is the read-only filename -- never paste over it.
        start_col = max(self._table.currentColumn(), 1)

        needed = start_col + max(len(r) for r in rows)
        if needed > self._table.columnCount():
            QMessageBox.information(
                self, "Paste",
                "The pasted data is wider than the available variable columns. "
                "Add more columns first -- extra values were dropped.")

        for i, row_vals in enumerate(rows):
            row = start_row + i
            if row >= self._table.rowCount():
                break  # rows are fixed to the filename list
            for j, val in enumerate(row_vals):
                col = start_col + j
                if col >= self._table.columnCount():
                    break
                self._table.setItem(row, col, QTableWidgetItem(val.strip()))

    def _col_for_var(self, var: str) -> Optional[int]:
        for c in range(1, self._table.columnCount()):
            header = self._table.horizontalHeaderItem(c)
            if header and header.text() == var:
                return c
        return None

    def _add_column(self, name: Optional[str] = None) -> Optional[int]:
        if not name:
            name, ok = QInputDialog.getText(self, "Add Column", "Variable name:")
            if not ok or not name.strip():
                return None
            name = name.strip()
        existing_col = self._col_for_var(name)
        if existing_col is not None:
            return existing_col
        col = self._table.columnCount()
        self._table.setColumnCount(col + 1)
        self._table.setHorizontalHeaderItem(col, QTableWidgetItem(name))
        return col

    def _remove_column(self):
        col = self._table.currentColumn()
        if col <= 0:
            QMessageBox.information(self, "Remove Column",
                                     "Select a variable column to remove (not Filename).")
            return
        self._table.removeColumn(col)

    def _extract_from_filename(self):
        pattern, ok = QInputDialog.getText(
            self, "Extract from Filename",
            "Regex with named groups, e.g.  SCR(?P<SCR>[\\d.]+)_Pmax(?P<Pmax>\\d+)"
        )
        if not ok or not pattern.strip():
            return
        try:
            rx = re.compile(pattern)
        except re.error as ex:
            QMessageBox.warning(self, "Invalid Regex", str(ex))
            return
        if not rx.groupindex:
            QMessageBox.warning(self, "Invalid Regex",
                                 "Pattern must contain at least one named group, e.g. (?P<SCR>...).")
            return

        matched = 0
        for row, name in enumerate(self._filenames):
            m = rx.search(name)
            if not m:
                continue
            matched += 1
            for var, val in m.groupdict().items():
                if val is None:
                    continue
                col = self._add_column(var)
                self._table.setItem(row, col, QTableWidgetItem(val))
        if matched == 0:
            QMessageBox.information(self, "Extract from Filename",
                                     "The pattern did not match any filenames.")

    def result_columns(self) -> List[str]:
        """Header names of the variable columns, including ones the user
        added but left empty -- result_titles() alone would drop those."""
        names = []
        for c in range(1, self._table.columnCount()):
            header = self._table.horizontalHeaderItem(c)
            if header and header.text().strip():
                names.append(header.text().strip())
        return names

    def result_titles(self) -> dict:
        titles = {}
        for row, name in enumerate(self._filenames):
            vars_ = {}
            for col in range(1, self._table.columnCount()):
                header = self._table.horizontalHeaderItem(col)
                item = self._table.item(row, col)
                if header and item and item.text().strip():
                    vars_[header.text()] = item.text().strip()
            if vars_:
                titles[name] = vars_
        return titles


class ExportDialog(QDialog):
    """Configure folders, output path, format, and page titles for batch export."""

    def __init__(self, registry: 'DatasetRegistry', page_titles: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.setMinimumWidth(560)
        layout = QFormLayout(self)
        layout.setSpacing(8)

        # PSSE folder — populated from loaded datasets, Browse to override
        self._psse_combo = self._make_ds_combo(registry, 'PSSE')
        btn_p = QPushButton("Browse…")
        btn_p.clicked.connect(lambda: self._pick_combo_dir(self._psse_combo))
        h1 = QHBoxLayout(); h1.addWidget(self._psse_combo, 1); h1.addWidget(btn_p)
        layout.addRow("PSSE folder:", h1)

        # PSCAD folder
        self._pscad_combo = self._make_ds_combo(registry, 'PSCAD')
        btn_q = QPushButton("Browse…")
        btn_q.clicked.connect(lambda: self._pick_combo_dir(self._pscad_combo))
        h2 = QHBoxLayout(); h2.addWidget(self._pscad_combo, 1); h2.addWidget(btn_q)
        layout.addRow("PSCAD folder:", h2)

        # Output folder
        self._out_edit = QLineEdit()
        btn_o = QPushButton("Browse…")
        btn_o.clicked.connect(self._pick_out)
        h3 = QHBoxLayout(); h3.addWidget(self._out_edit); h3.addWidget(btn_o)
        layout.addRow("Output folder:", h3)

        # Format
        self._fmt = QComboBox()
        self._fmt.addItems([
            'Current screen – PDF (single page)',
            'Current screen – PNG (single page)',
            'PDF (combined)',
            'PDF (per page)',
            'PNG (per page)',
        ])
        self._fmt.currentIndexChanged.connect(self._on_fmt_changed)
        layout.addRow("Output format:", self._fmt)

        # Page size
        self._page_size = QComboBox()
        self._page_size.addItems(list(PAGE_SIZES.keys()))
        layout.addRow("Page size:", self._page_size)

        # Single-page filename
        self._single_name_edit = QLineEdit("BOPPO_export")
        self._single_name_label = QLabel("Filename (no ext):")
        layout.addRow(self._single_name_label, self._single_name_edit)

        # Iteration note -- batch export always pages through BOTH folders
        # together now (paired by position: 1st PSSE file with 1st PSCAD
        # file, 2nd with 2nd, ...). If one side has only a single file, that
        # one is reused as a fixed reference on every page. If a side has
        # more files than the other (and the other has more than one), the
        # extra files on the longer side are skipped with a warning.
        self._loop_note = QLabel(
            "Batch export pages through PSSE and PSCAD files together, "
            "paired by position. A folder with only one file is used as a "
            "fixed reference for every page."
        )
        self._loop_note.setWordWrap(True)
        self._loop_note.setStyleSheet("color: #666; font-size: 9px;")
        layout.addRow("Iteration:", self._loop_note)

        # ── Page titles ───────────────────────────────────────────────────────
        self._sep = QLabel("─── Page titles ──────────────────────────────────────")
        self._sep.setStyleSheet("color: #888; font-size: 9px;")
        layout.addRow(self._sep)

        self._page_titles = dict(page_titles or {})
        # Column names persist separately: a column the user added but left
        # blank has no values and so never appears in _page_titles.
        self._page_title_columns = []
        for _vars in self._page_titles.values():
            for _v in _vars:
                if _v not in self._page_title_columns:
                    self._page_title_columns.append(_v)
        self._edit_titles_btn = QPushButton("Edit page titles…")
        self._edit_titles_btn.clicked.connect(self._edit_page_titles)
        layout.addRow("", self._edit_titles_btn)

        self._title_prefix = QLineEdit()
        self._title_prefix.setPlaceholderText("optional prefix added before title variables")
        layout.addRow("Title prefix:", self._title_prefix)

        # ── PDF title options ─────────────────────────────────────────────────
        self._sep2 = QLabel("─── PDF title options ────────────────────────────────")
        self._sep2.setStyleSheet("color: #888; font-size: 9px;")
        layout.addRow(self._sep2)

        self._use_outfile_chk = QCheckBox("Use outfile name as PDF title")
        self._use_outfile_chk.setChecked(True)
        layout.addRow("", self._use_outfile_chk)

        self._use_date_chk = QCheckBox("Include export date in PDF title")
        self._use_date_chk.setChecked(True)
        layout.addRow("", self._use_date_chk)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

        # Apply initial visibility
        self._on_fmt_changed(0)

    def _on_fmt_changed(self, _index: int):
        single = self._fmt.currentText().startswith('Current screen')
        is_pdf = 'PDF' in self._fmt.currentText()
        self._single_name_label.setVisible(single)
        self._single_name_edit.setVisible(single)
        self._loop_note.setVisible(not single)
        self._sep.setVisible(not single)
        self._edit_titles_btn.setVisible(not single)
        self._title_prefix.setVisible(not single)
        self._sep2.setVisible(is_pdf and not single)
        self._use_outfile_chk.setVisible(is_pdf and not single)
        self._use_date_chk.setVisible(is_pdf and not single)

    @staticmethod
    def _scan_folder_stems(folder: str, kind: str) -> List[str]:
        """List filename stems in `folder`, mirroring PSSEFolder/PSCADFolder's
        file discovery for the given kind ('psse' or 'pscad')."""
        if not folder or not os.path.isdir(folder):
            return []
        try:
            folder_path = Path(folder)
            if kind == 'psse':
                files = sorted(
                    list(folder_path.rglob('*.out')) + list(folder_path.rglob('*.outx')),
                    key=lambda p: p.name,
                )
            else:
                files = sorted(folder_path.glob('*.inf'), key=lambda p: p.name)
            return [f.stem for f in files]
        except Exception:
            return []

    def _loop_filenames(self) -> List[str]:
        """List filename stems for page-title editing purposes -- uses
        whichever of the two selected folders has more files (batch export
        now pages through both together, paired by position; the side with
        more files determines how many pages there are)."""
        psse_files = self._scan_folder_stems(self._psse_combo.currentData() or '', 'psse')
        pscad_files = self._scan_folder_stems(self._pscad_combo.currentData() or '', 'pscad')
        return psse_files if len(psse_files) >= len(pscad_files) else pscad_files

    def _edit_page_titles(self):
        filenames = self._loop_filenames()
        if not filenames:
            QMessageBox.information(
                self, "Edit Page Titles",
                "Select a PSSE/PSCAD folder and loop mode first so filenames can be listed."
            )
            return
        dlg = PageTitleEditorDialog(filenames, self._page_titles, self,
                                    columns=self._page_title_columns)
        if dlg.exec_() == QDialog.Accepted:
            self._page_titles = dlg.result_titles()
            self._page_title_columns = dlg.result_columns()

    @staticmethod
    def _make_ds_combo(registry, kind) -> QComboBox:
        combo = QComboBox()
        if registry is not None:
            for d in registry.all_of_kind(kind):
                combo.addItem(d.label, d.path)
        if combo.count() == 0:
            combo.addItem("(none loaded)", '')
        return combo

    def _pick_combo_dir(self, combo: QComboBox):
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder", combo.currentData() or '')
        if folder:
            combo.addItem(folder, folder)
            combo.setCurrentIndex(combo.count() - 1)

    def _pick_out(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self._out_edit.setText(folder)

    def params(self) -> dict:
        return {
            'psse_folder':    (self._psse_combo.currentData() or '').strip(),
            'pscad_folder':   (self._pscad_combo.currentData() or '').strip(),
            'out_folder':     self._out_edit.text().strip(),
            'format':         self._fmt.currentText(),
            'page_size':      PAGE_SIZES.get(self._page_size.currentText()),
            'single_page':    self._fmt.currentText().startswith('Current screen'),
            'single_name':    self._single_name_edit.text().strip() or 'BOPPO_export',
            'page_titles':    dict(self._page_titles),
            'title_prefix':   self._title_prefix.text().strip(),
            'title_outfile':  self._use_outfile_chk.isChecked(),
            'title_date':     self._use_date_chk.isChecked(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class ComparisonTabController(QWidget):
    """One benchmarking tab: multiple PSSE/PSCAD (and optionally Field) datasets
    plotted together on a shared grid. Parameterised by `formats` so a single
    class serves both the 'PSSE vs PSCAD' and 'Field Data Overlay' tabs."""

    _KIND_LABELS = {'PSSE': 'PSSE folder', 'PSCAD': 'PSCAD folder',
                    'Field': 'Field Data (CSV/XLSX)'}

    def __init__(self, formats: List[str], statusbar=None, parent=None):
        super().__init__(parent)
        self.formats = formats
        self._statusbar = statusbar
        self.registry = DatasetRegistry()
        self.page_titles = {}
        self.page_title_sequence = []       # ordered list, matched to exported pages by
                                             # POSITION (not filename) -- see _export()'s
                                             # batch loop; set by AECST via --template
        self._current_template_path = None  # set when a template is loaded (CLI or dialog);
                                             # _save_template() writes back here without prompting
        self._build()
        self.plot_grid.set_registry(self.registry)

    # ── UI construction ───────────────────────────────────────────────────────

    def _make_xlim_edit(self, placeholder: str) -> QLineEdit:
        w = QLineEdit()
        w.setPlaceholderText(placeholder)
        w.setFixedWidth(72)
        w.setToolTip("Global x-axis limit applied to all plots (leave blank for auto)")
        return w

    def _build(self):
        splitter = QSplitter(Qt.Horizontal)

        # ── Left panel ────────────────────────────────────────────────────────
        left = QWidget()
        left.setMaximumWidth(320)
        left.setMinimumWidth(220)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        lv.setSpacing(6)

        for kind in self.formats:
            btn = QPushButton(f"⬇  Load {self._KIND_LABELS[kind]}…")
            btn.setStyleSheet(
                f"color: white; background: {KIND_COLOR[kind]}; "
                "font-weight: bold; padding: 5px;"
            )
            btn.clicked.connect(lambda _c=False, k=kind: self._load_folder(k))
            lv.addWidget(btn)

        # Loaded datasets list + remove button
        lv.addWidget(QLabel("Loaded datasets:"))
        self.ds_list = QListWidget()
        self.ds_list.setMaximumHeight(120)
        self.ds_list.currentItemChanged.connect(self._on_ds_selected)
        lv.addWidget(self.ds_list)

        ds_btns = QHBoxLayout()
        remove_btn = QPushButton("✕  Remove")
        remove_btn.setToolTip("Unload the selected dataset")
        remove_btn.clicked.connect(self._remove_selected_dataset)
        ds_btns.addWidget(remove_btn)
        ds_btns.addStretch()
        lv.addLayout(ds_btns)

        # Sub-dataset selector for the highlighted dataset
        subds_form = QFormLayout()
        self.subds_combo = QComboBox()
        self.subds_combo.setToolTip(
            "Which internal result file is active for the selected dataset")
        self.subds_combo.currentIndexChanged.connect(self._on_subds_changed)
        subds_form.addRow("Active file:", self.subds_combo)
        lv.addLayout(subds_form)

        # Search
        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍  Filter channels…")
        self.search.textChanged.connect(self._on_search)
        lv.addWidget(self.search)

        # Channel tree
        self.browser = ChannelBrowser()
        lv.addWidget(self.browser, 1)

        splitter.addWidget(left)

        # ── Right panel ───────────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 4, 4, 4)
        rv.setSpacing(6)

        toolbar = QHBoxLayout()

        toolbar.addWidget(QLabel("Grid:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 10)
        self.rows_spin.setValue(2)
        self.rows_spin.setFixedWidth(50)
        toolbar.addWidget(self.rows_spin)
        toolbar.addWidget(QLabel("×"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 10)
        self.cols_spin.setValue(3)
        self.cols_spin.setFixedWidth(50)
        toolbar.addWidget(self.cols_spin)
        apply_btn = QPushButton("Apply")
        apply_btn.setFixedWidth(60)
        apply_btn.clicked.connect(self._apply_grid)
        toolbar.addWidget(apply_btn)

        toolbar.addSpacing(24)
        toolbar.addWidget(QLabel("PSSE time offset (s):"))
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-99999, 99999)
        self.offset_spin.setSingleStep(0.01)
        self.offset_spin.setDecimals(4)
        self.offset_spin.setValue(0.0)
        self.offset_spin.setFixedWidth(100)
        self.offset_spin.setToolTip(
            "Shift the PSSE time axis by this value (seconds).\n"
            "Positive = PSSE data shifted later."
        )
        self.offset_spin.valueChanged.connect(self._on_offset_changed)
        toolbar.addWidget(self.offset_spin)

        toolbar.addSpacing(24)
        toolbar.addWidget(QLabel("X limits:"))
        self.gxmin_edit = self._make_xlim_edit("min")
        self.gxmax_edit = self._make_xlim_edit("max")
        toolbar.addWidget(self.gxmin_edit)
        toolbar.addWidget(QLabel("–"))
        toolbar.addWidget(self.gxmax_edit)

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
        toolbar.addWidget(QLabel("Page size:"))
        self.page_size_combo = QComboBox()
        self.page_size_combo.addItems(list(PAGE_SIZES.keys()))
        self.page_size_combo.setToolTip("Page size used by the Export PDF button")
        toolbar.addWidget(self.page_size_combo)
        export_pdf_btn = QPushButton("📄  Export PDF")
        export_pdf_btn.clicked.connect(self._export_current_pdf)
        toolbar.addWidget(export_pdf_btn)
        export_btn = QPushButton("📄  Export All…")
        export_btn.setStyleSheet("font-weight: bold; padding: 5px 16px;")
        export_btn.clicked.connect(self._export)
        toolbar.addWidget(export_btn)
        rv.addLayout(toolbar)

        # Scrollable plot grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.plot_grid = PlotGrid()
        scroll.setWidget(self.plot_grid)
        rv.addWidget(scroll, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

    # ── Dataset list helpers ──────────────────────────────────────────────────

    def _refresh_ds_list(self):
        self.ds_list.blockSignals(True)
        self.ds_list.clear()
        for d in self.registry.all():
            item = QListWidgetItem(f"[{d.kind}]  {d.label}")
            item.setData(Qt.UserRole, d.id)
            item.setForeground(QColor(KIND_COLOR[d.kind]))
            self.ds_list.addItem(item)
        self.ds_list.blockSignals(False)
        self._on_ds_selected(self.ds_list.currentItem(), None)

    def _on_ds_selected(self, current, _previous=None):
        self.subds_combo.blockSignals(True)
        self.subds_combo.clear()
        entry = self.registry.get(current.data(Qt.UserRole)) if current else None
        if entry is not None:
            ds_list = getattr(entry.folder_obj, 'datasets', None)
            if ds_list:
                for ds in ds_list:
                    self.subds_combo.addItem(ds.name)
                self.subds_combo.setCurrentIndex(
                    min(entry.sel_index, len(ds_list) - 1))
        self.subds_combo.setEnabled(self.subds_combo.count() > 0)
        self.subds_combo.blockSignals(False)

    def _on_subds_changed(self, idx: int):
        item = self.ds_list.currentItem()
        entry = self.registry.get(item.data(Qt.UserRole)) if item else None
        if entry is not None and idx >= 0:
            entry.sel_index = idx
            self.plot_grid.refresh_all()

    # ── Load handlers ─────────────────────────────────────────────────────────

    def _load_folder(self, kind: str):
        if kind == 'Field':
            path, _ = QFileDialog.getOpenFileName(
                self, "Select field data file", "",
                "Field Data (*.csv *.xlsx);;All files (*)"
            )
        else:
            path = QFileDialog.getExistingDirectory(
                self, f"Select {kind} results folder")
        if not path:
            return
        self._add_dataset(kind, path)

    def _add_dataset(self, kind: str, path: str):
        """Load a dataset from an already-known path (no file dialog) --
        shared tail of _load_folder(), also used for pre-configuring a
        results folder at startup (e.g. --pscad-folder)."""
        try:
            if kind == 'PSSE':
                ds_id = self.registry.add_psse(path)
            elif kind == 'PSCAD':
                ds_id = self.registry.add_pscad(path)
            else:
                ds_id = self.registry.add_field(path)
        except Exception as ex:
            QMessageBox.critical(self, "Load Error", f"{kind}:\n{ex}")
            return
        self._populate_browser_for(ds_id)
        self._refresh_ds_list()
        self.plot_grid.refresh_all()
        self._update_status()

    def _populate_browser_for(self, ds_id: str):
        entry = self.registry.get(ds_id)
        if entry is None:
            return
        folder = entry.folder_obj
        if entry.kind == 'PSSE':
            try:
                names = folder.channel_names()
            except Exception as ex:
                QMessageBox.warning(
                    self, "PSSE Channel Warning",
                    f"Could not read channel names:\n{ex}\n\n"
                    "You can still configure plots, but data will load on first use."
                )
                names = []
            self.browser.set_dataset_channels('PSSE', ds_id, entry.label, names)
        elif entry.kind == 'PSCAD':
            names = folder.channel_names()
            self.browser.set_dataset_channels(
                'PSCAD', ds_id, entry.label, names,
                units_fn=folder.channel_units)
        else:
            names = folder.channels
            self.browser.set_dataset_channels('Field', ds_id, entry.label, names)

    def _remove_selected_dataset(self):
        item = self.ds_list.currentItem()
        if item is None:
            return
        ds_id = item.data(Qt.UserRole)
        entry = self.registry.get(ds_id)
        if entry is not None:
            self.browser.remove_dataset(entry.kind, ds_id)
        self.registry.remove(ds_id)
        self._refresh_ds_list()
        self.plot_grid.refresh_all()
        self._update_status()

    def _update_status(self):
        if self._statusbar is None:
            return
        counts = {k: len(self.registry.all_of_kind(k))
                  for k in ('PSSE', 'PSCAD', 'Field')}
        parts = [f"{k}: {counts[k]}" for k in self.formats]
        self._statusbar.showMessage("Loaded datasets  ·  " + "  ·  ".join(parts))

    # ── Grid / offset ─────────────────────────────────────────────────────────

    def _apply_grid(self):
        self.plot_grid.set_grid(self.rows_spin.value(), self.cols_spin.value())

    def _on_offset_changed(self, val: float):
        self.plot_grid.set_time_offset(val)

    def _on_global_xlim_changed(self):
        def _parse(text):
            try:
                return float(text.strip())
            except ValueError:
                return None
        self.plot_grid.set_global_xlim(
            _parse(self.gxmin_edit.text()),
            _parse(self.gxmax_edit.text()),
        )

    def _on_search(self, text: str):
        self.browser.filter_text(text)

    # ── Templates ─────────────────────────────────────────────────────────────

    def _save_template(self):
        """
        Save the current layout as a .boppo template. If a template was
        already loaded this session (via the Load Template dialog, or via
        --template on the command line), overwrite that same file directly
        with no prompt -- this is what keeps an AECST rank preset's linked
        template in sync: launch BOPPO with --template pointed at the
        preset's file, edit the layout, hit Save Template, done.
        """
        if self._current_template_path:
            self._write_template_file(self._current_template_path)
        else:
            self._save_template_as()

    def _save_template_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Template", "", "BOPPO Template (*.boppo);;All files (*)"
        )
        if not path:
            return
        if not path.endswith('.boppo'):
            path += '.boppo'
        self._current_template_path = path
        self._write_template_file(path)

    def _write_template_file(self, path):
        layout = self.plot_grid.get_layout_config()
        layout['schema_version'] = 2
        layout['datasets'] = self.registry.export_manifest()
        layout['page_titles'] = self.page_titles
        layout['page_title_sequence'] = self.page_title_sequence
        try:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
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
        self.load_template_file(path)

    def load_template_file(self, path, pscad_folder_override=None, silent=False):
        """
        Load a .boppo template from a known path (no dialog) -- used both
        by _load_template() above and by main()'s --template CLI handling.

        If the file doesn't exist yet (e.g. an AECST rank preset that has
        never had a template saved for it), just remember the path so the
        next _save_template() writes there -- this is how a brand-new
        preset's first BOPPO session bootstraps its template file. In that
        case, pscad_folder_override (if given) is still loaded directly via
        _add_dataset(), since there's no template to apply it to.

        pscad_folder_override, if given, replaces the path of any PSCAD
        dataset entry in the template with this folder instead of the
        stale path the template was saved with -- so a shared template can
        be reused against a fresh run's output folder each time.

        `silent`, if True, prints load failures to stderr instead of
        showing a QMessageBox -- required for headless/background
        invocations (see main()'s --export-pdf), where a modal dialog with
        no one to click it would hang the process forever.
        """
        self._current_template_path = path
        if not os.path.isfile(path):
            if pscad_folder_override:
                self._add_dataset('PSCAD', pscad_folder_override)
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                layout = json.load(f)
        except Exception as ex:
            if silent:
                print(f"BOPPO: failed to load template {path}: {ex}", file=sys.stderr)
            else:
                QMessageBox.critical(self, "Load Template", f"Failed to load:\n{ex}")
            return
        self._apply_template(layout, pscad_folder_override=pscad_folder_override, silent=silent)

    def _apply_template(self, layout, pscad_folder_override=None, silent=False):
        if 'datasets' not in layout or layout.get('schema_version', 1) == 1:
            if silent:
                print("BOPPO: legacy template (single dataset per format) loaded -- "
                      "channels will bind to the first-loaded dataset of each type.",
                      file=sys.stderr)
            else:
                QMessageBox.information(
                    self, "Legacy Template",
                    "This is a legacy template (single dataset per format). Load your "
                    "PSSE/PSCAD/Field folders as usual; channels will bind to the "
                    "first-loaded dataset of each type."
                )
        else:
            missing = []
            for entry in layout.get('datasets', []):
                kind = entry.get('kind')
                fid  = entry.get('id')
                ds_path = entry.get('path', '')
                label = entry.get('label')
                if kind == 'PSCAD' and pscad_folder_override:
                    ds_path = pscad_folder_override
                try:
                    if kind == 'PSSE':
                        self.registry.add_psse(ds_path, forced_id=fid, label=label)
                    elif kind == 'PSCAD':
                        self.registry.add_pscad(ds_path, forced_id=fid, label=label)
                    elif kind == 'Field':
                        self.registry.add_field(ds_path, forced_id=fid, label=label)
                    self._populate_browser_for(fid)
                except Exception:
                    missing.append(f"{ds_path} (dataset {fid})")
            self._refresh_ds_list()
            self._update_status()
            if missing:
                if silent:
                    print(f"BOPPO: {len(missing)} dataset(s) could not be reloaded -- "
                          f"channels from these will show fetch errors: {'; '.join(missing)}",
                          file=sys.stderr)
                else:
                    QMessageBox.warning(
                        self, "Template Load",
                        f"{len(missing)} dataset(s) could not be reloaded — channels "
                        "from these will show fetch errors until reloaded manually:\n\n"
                        + '\n'.join(missing)
                    )

        self.page_titles = layout.get('page_titles', {})
        self.page_title_sequence = layout.get('page_title_sequence', [])

        rows = layout.get('rows', 2)
        cols = layout.get('cols', 3)
        self.rows_spin.setValue(rows)
        self.cols_spin.setValue(cols)
        self.plot_grid.set_grid(rows, cols)
        for pw, snap in zip(self.plot_grid._flat(), layout.get('plots', [])):
            pw.restore(snap)
            pw.refresh()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_current_pdf(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Displayed Data", "",
            "PDF (*.pdf);;PNG (*.png);;All files (*)"
        )
        if not path:
            return
        layout = self.plot_grid.get_layout_config()
        offset = self.offset_spin.value()
        try:
            fig = self.plot_grid.render_page(
                self.registry, offset, layout,
                global_xmin=self.plot_grid._global_xmin,
                global_xmax=self.plot_grid._global_xmax,
                page_size=PAGE_SIZES.get(self.page_size_combo.currentText()),
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

    def _export(self):
        dlg = ExportDialog(self.registry, self.page_titles, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        p = dlg.params()
        self.page_titles = p['page_titles']

        if not p['out_folder']:
            QMessageBox.warning(self, "Export", "Please select an output folder.")
            return

        layout = self.plot_grid.get_layout_config()
        offset = self.offset_spin.value()
        out_dir = Path(p['out_folder'])
        fmt     = p['format']

        # ── Single-page (current screen) export ───────────────────────────────
        if p['single_page']:
            try:
                fig = self.plot_grid.render_page(
                    self.registry, offset, layout,
                    global_xmin=self.plot_grid._global_xmin,
                    global_xmax=self.plot_grid._global_xmax,
                    page_size=p['page_size'],
                )
                name = p['single_name']
                if 'PDF' in fmt:
                    out_path = str(out_dir / f"{name}.pdf")
                    fig.savefig(out_path, bbox_inches='tight')
                else:
                    out_path = str(out_dir / f"{name}.png")
                    fig.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                QMessageBox.information(self, "Export Complete",
                                        f"Saved to:\n{out_path}")
            except Exception as ex:
                plt.close('all')
                QMessageBox.critical(self, "Export Error", str(ex))
            return

        # ── Batch (loop) export ───────────────────────────────────────────────
        # Batch export reloads its own folders by path (orthogonal to registry).
        try:
            psse_folder = PSSEFolder(p['psse_folder']) if p['psse_folder'] else None
            if p['pscad_folder']:
                if list(Path(p['pscad_folder']).glob('*.psout')):
                    pscad_folder = PSCADPsoutFolder(p['pscad_folder'])
                else:
                    pscad_folder = PSCADFolder(p['pscad_folder'])
            else:
                pscad_folder = None
        except Exception as ex:
            QMessageBox.critical(self, "Export Error", str(ex))
            return

        psse_list  = psse_folder.datasets if psse_folder else []
        pscad_list = pscad_folder.datasets if pscad_folder else []

        if not psse_list and not pscad_list:
            QMessageBox.warning(self, "Export", "No result files found to export.")
            return

        # Page through BOTH folders together, paired by position (1st PSSE
        # with 1st PSCAD, 2nd with 2nd, ...). A folder with exactly one file
        # is treated as a fixed reference reused on every page, matching the
        # old single-loop behavior. If both have more than one file but
        # different counts, the extra files on the longer side are skipped
        # (reported in the completion message) rather than silently dropped.
        def _at(datasets: List, i: int):
            if not datasets:
                return None
            if len(datasets) == 1:
                return datasets[0]
            return datasets[i] if i < len(datasets) else None

        n_pages = max(len(psse_list), len(pscad_list))
        mismatched_counts = (
            len(psse_list) > 1 and len(pscad_list) > 1
            and len(psse_list) != len(pscad_list)
        )

        prefix  = p.get('title_prefix', '').strip()
        use_outfile = p.get('title_outfile', True)
        use_date    = p.get('title_date', True)
        export_date = datetime.date.today().strftime('%d %b %Y') if use_date else ''

        page_titles = p.get('page_titles', {})
        # Ordered titles (e.g. from AECST's preset-driven title_format), matched
        # to pages by POSITION, not by filename -- see ComparisonTabController's
        # page_title_sequence and aecst_presets.build_title_sequence(). Takes
        # priority over the filename-keyed page_titles/prefix/date assembly
        # below when present for a given page index.
        page_title_sequence = self.page_title_sequence

        def _page_title(fallback: str, index: int) -> str:
            if index < len(page_title_sequence) and page_title_sequence[index]:
                return page_title_sequence[index]
            parts = []
            if prefix:
                parts.append(prefix)
            vars_ = page_titles.get(fallback)
            if vars_:
                parts.append(format_title_vars(vars_))
            if use_outfile:
                parts.append(fallback)
            if export_date:
                parts.append(export_date)
            return '   '.join(parts) if parts else fallback

        prog = QProgressDialog("Exporting pages…", "Cancel", 0, n_pages, self)
        prog.setWindowModality(Qt.WindowModal)
        prog.show()

        pdf_combined: Optional[PdfPages] = None
        if fmt == 'PDF (combined)':
            pdf_combined = PdfPages(str(out_dir / 'BOPPO_results.pdf'))

        errors = []
        n_done = 0
        for i in range(n_pages):
            if prog.wasCanceled():
                break
            prog.setValue(i)
            QApplication.processEvents()

            psse_ds  = _at(psse_list, i)
            pscad_ds = _at(pscad_list, i)
            if psse_ds is None and pscad_ds is None:
                continue  # one side ran out of files -- skip this page
            page_name = (pscad_ds.name if pscad_ds else None) or (psse_ds.name if psse_ds else f"page_{i+1}")
            iter_reg = DatasetRegistry.from_single(psse_ds=psse_ds, pscad_ds=pscad_ds)

            try:
                fig = self.plot_grid.render_page(
                    iter_reg, offset, layout,
                    page_title=_page_title(page_name, i),
                    global_xmin=self.plot_grid._global_xmin,
                    global_xmax=self.plot_grid._global_xmax,
                    page_size=p['page_size'],
                )
                if fmt == 'PDF (combined)' and pdf_combined:
                    pdf_combined.savefig(fig, bbox_inches='tight')
                elif fmt == 'PDF (per page)':
                    fig.savefig(str(out_dir / f"{page_name}.pdf"), bbox_inches='tight')
                elif fmt == 'PNG (per page)':
                    fig.savefig(str(out_dir / f"{page_name}.png"),
                                dpi=150, bbox_inches='tight')
                plt.close(fig)
                n_done += 1
            except Exception as ex:
                errors.append(f"{page_name}: {ex}")
                plt.close('all')

        if pdf_combined:
            pdf_combined.close()

        prog.setValue(n_pages)
        msg = f"Exported {n_done} page(s) to:\n{out_dir}"
        if mismatched_counts:
            msg += (f"\n\nNote: PSSE folder has {len(psse_list)} file(s) and PSCAD folder "
                    f"has {len(pscad_list)} -- pages beyond the shorter list's length only "
                    f"contain data from the longer side.")
        if errors:
            msg += f"\n\n{len(errors)} error(s):\n" + '\n'.join(errors[:5])
        QMessageBox.information(self, "Export Complete", msg)

    def export_batch_pdf(self, pscad_folder: str, out_pdf_path: str,
                          page_size_key: Optional[str] = None,
                          xmin: Optional[float] = None,
                          xmax: Optional[float] = None):
        """
        Headless equivalent of _export()'s "PDF (combined)" batch path above,
        used by main()'s --export-pdf CLI flag (AECST's automatic post-run
        PDF generation). Pages through every result file in `pscad_folder`
        (paired 1:1 with RANK.txt rows, exactly like the interactive batch
        export), using this tab's CURRENTLY LOADED plot layout/channels and
        page_title_sequence (already populated by load_template_file()) --
        no PSSE side (AECST never has PSSE data), no ExportDialog/
        QProgressDialog (nothing to show a user headlessly).

        xmin/xmax, if given, fix the X-axis window for every page (passed
        straight through to render_page()'s global_xmin/global_xmax) --
        this is a ONE-OFF render parameter for this PDF only, independent
        of (and never written back into) this tab's own global X-limit
        fields/the loaded template's saved state.

        Returns (n_done, errors) -- errors is a list of "page_name: message"
        strings for pages that failed to render, matching _export()'s own
        per-page error collection.
        """
        layout = self.plot_grid.get_layout_config()
        offset = self.offset_spin.value()
        page_size = PAGE_SIZES.get(page_size_key) if page_size_key else None

        if list(Path(pscad_folder).glob('*.psout')):
            pscad_dsfolder = PSCADPsoutFolder(pscad_folder)
        else:
            pscad_dsfolder = PSCADFolder(pscad_folder)
        pscad_list = pscad_dsfolder.datasets

        page_title_sequence = self.page_title_sequence

        def _page_title(fallback: str, index: int) -> str:
            if index < len(page_title_sequence) and page_title_sequence[index]:
                return page_title_sequence[index]
            return fallback

        Path(out_pdf_path).parent.mkdir(parents=True, exist_ok=True)
        pdf_combined = PdfPages(out_pdf_path)
        errors = []
        n_done = 0
        for i, pscad_ds in enumerate(pscad_list):
            page_name = pscad_ds.name
            iter_reg = DatasetRegistry.from_single(pscad_ds=pscad_ds)
            try:
                fig = self.plot_grid.render_page(
                    iter_reg, offset, layout,
                    page_title=_page_title(page_name, i),
                    global_xmin=xmin,
                    global_xmax=xmax,
                    page_size=page_size,
                )
                pdf_combined.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                n_done += 1
            except Exception as ex:
                errors.append(f"{page_name}: {ex}")
                plt.close('all')

        pdf_combined.close()
        return n_done, errors


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BOPPO – Benchmarking Of PSSE and PSCAD Outputs")
        self.resize(1440, 900)

        self.tabs: List[ComparisonTabController] = []
        self._build_ui()
        self._build_menu()
        self.statusBar().showMessage(
            "Load PSSE/PSCAD folders, then drag channels onto plots."
        )

    def _build_ui(self):
        self._tabs_widget = QTabWidget()
        self.setCentralWidget(self._tabs_widget)
        for formats, title in (
            (['PSSE', 'PSCAD'],          "PSSE vs PSCAD"),
            (['PSSE', 'PSCAD', 'Field'], "Field Data Overlay"),
        ):
            ctrl = ComparisonTabController(formats, statusbar=self.statusBar())
            self.tabs.append(ctrl)
            self._tabs_widget.addTab(ctrl, title)

    def _current_tab(self) -> ComparisonTabController:
        return self.tabs[self._tabs_widget.currentIndex()]

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        fm.addAction("Load PSSE folder…",
                     lambda: self._current_tab()._load_folder('PSSE'),
                     "Ctrl+Shift+P")
        fm.addAction("Load PSCAD folder…",
                     lambda: self._current_tab()._load_folder('PSCAD'),
                     "Ctrl+Shift+C")
        fm.addSeparator()
        fm.addAction("Export All…", lambda: self._current_tab()._export(), "Ctrl+E")
        fm.addSeparator()
        fm.addAction("Quit", self.close, "Ctrl+Q")

        hm = mb.addMenu("&Help")
        hm.addAction("About BOPPO", self._about)

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

    # Optional CLI args (used by AECST to open BOPPO pre-configured for a run):
    #   --pscad-folder <path>   pre-load a PSCAD results folder into the first tab
    #   --template <path>       load a .boppo template into the first tab; if
    #                           given together with --pscad-folder, the template's
    #                           own (stale) PSCAD dataset path is overridden with
    #                           --pscad-folder instead of being used as-is
    def _arg_after(flag):
        if flag in sys.argv:
            idx = sys.argv.index(flag)
            if idx + 1 < len(sys.argv):
                return sys.argv[idx + 1]
        return None

    pscad_folder = _arg_after('--pscad-folder')
    template_path = _arg_after('--template')
    export_pdf_path = _arg_after('--export-pdf')
    page_size_key = _arg_after('--page-size')
    xmin_str = _arg_after('--xmin')
    xmax_str = _arg_after('--xmax')
    xmin = float(xmin_str) if xmin_str is not None else None
    xmax = float(xmax_str) if xmax_str is not None else None

    # --export-pdf <path> [--page-size <key>]: headless mode used by AECST's
    # automatic post-run PDF generation -- render --template's plot layout
    # against --pscad-folder's result files into a single combined PDF, then
    # exit immediately. No window is ever shown/no event loop is entered, so
    # this can run silently in the background alongside AECST.
    if export_pdf_path:
        if template_path:
            win.tabs[0].load_template_file(
                template_path, pscad_folder_override=pscad_folder, silent=True)
        elif pscad_folder:
            win.tabs[0]._add_dataset('PSCAD', pscad_folder)
        n_done, errors = win.tabs[0].export_batch_pdf(
            pscad_folder, export_pdf_path, page_size_key, xmin=xmin, xmax=xmax)
        if errors:
            print(f"BOPPO: {len(errors)} error(s) exporting {export_pdf_path}:",
                  file=sys.stderr)
            for err in errors:
                print(f"  {err}", file=sys.stderr)
        print(f"BOPPO: wrote {n_done} page(s) to {export_pdf_path}")
        sys.exit(1 if errors else 0)

    if template_path:
        win.tabs[0].load_template_file(template_path, pscad_folder_override=pscad_folder)
    elif pscad_folder:
        win.tabs[0]._add_dataset('PSCAD', pscad_folder)

    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
