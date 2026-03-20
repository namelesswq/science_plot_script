#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class KPointSpec:
    k: Tuple[float, float, float]
    n: int  # number of points from this k-point to the next one; n==1 means a jump/break


def _format_system_label(label: str, mode: str) -> str:
    """Format chemical formulas like Zr2SC with subscripts for matplotlib."""

    if not label:
        return label
    if mode == "raw":
        return label
    if "$" in label:
        return label
    return re.sub(r"(?<=[A-Za-z\)])(\d+)", r"$_{\1}$", label)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot Quantum ESPRESSO band structure from bands.out.gnu with high-symmetry labels.\n\n"
            "Inputs:\n"
            "- bands.out.gnu: two columns (k-distance, energy), blank lines separate bands\n"
            "- band.in: contains K_POINTS crystal_b with per-segment point counts\n"
            "- KPATH.in (optional): provides labels for high-symmetry points (VASPKIT format)\n\n"
            "Jump points: if a k-point line in band.in has N=1, it is treated as a discontinuity;\n"
            "the tick label at that position is merged as 'A|L' (end|start)."
        )
    )

    p.add_argument(
        "--bands",
        required=True,
        nargs="+",
        help="One or more bands.out.gnu files. If multiple are given, they are overlaid for comparison.",
    )
    p.add_argument(
        "--band-in",
        required=True,
        nargs="+",
        help="One or more band.in files. If a single file is given, it is reused for all datasets.",
    )
    p.add_argument(
        "--kpath",
        default=None,
        nargs="+",
        help="One or more KPATH.in files (VASPKIT). If omitted, labels are guessed. If a single file is given, it is reused.",
    )

    p.add_argument(
        "--fermi",
        default=None,
        nargs="+",
        help=(
            "Fermi energy (eV). If provided, shift energies as E -> E - Ef so Ef is at 0 eV. "
            "Provide one per dataset, or a single value to broadcast. Comma-separated tokens are accepted."
        ),
    )
    p.add_argument("--fermi-line", action="store_true", help="Draw a horizontal line at E=0")

    p.add_argument("--ylim", default=None, help='y limits "ymin,ymax" in eV')

    p.add_argument(
        "--style",
        choices=["prb", "default"],
        default="prb",
        help="Plot style preset. 'prb' uses SciencePlots (science,no-latex). Default: prb.",
    )
    p.add_argument(
        "--figsize",
        default=None,
        help=(
            'Figure size in inches. Use "width,height" (e.g. "7,3"). '
            'You may also pass a single number "width" (e.g. "7"), '
            'in which case height is set automatically (height=0.45*width).'
        ),
    )
    p.add_argument(
        "--lw",
        type=float,
        default=None,
        help=(
            "Line width for band lines. If omitted, keep style defaults "
            "(currently ~1.0 for prb and larger for default)."
        ),
    )
    p.add_argument("--out", default="bands.png", help="Output image path")
    p.add_argument("--show", action="store_true", help="Show interactively")

    p.add_argument(
        "--legend",
        default=None,
        nargs="+",
        help=(
            "Legend label(s) for each dataset. Provide one per dataset, or a single value to broadcast. "
            "If only one label is given, it is broadcast. Comma-separated tokens are also accepted."
        ),
    )
    p.add_argument(
        "--legend-format",
        choices=["chem", "raw"],
        default="raw",
        help="Render --legend text with subscripts (chem) or raw text (raw). Default: raw.",
    )
    p.add_argument(
        "--legend-fontsize",
        type=float,
        default=None,
        help="Font size for legend text. If omitted, uses matplotlib default.",
    )
    p.add_argument(
        "--legend-loc",
        default="upper left",
        help="Legend location (matplotlib legend loc). Default: upper left.",
    )

    p.add_argument(
        "--legend-bbox",
        default=None,
        help=(
            "Optional legend anchor (bbox_to_anchor) in axes coordinates 'x,y'. "
            "If provided, legend placement uses both --legend-loc and this anchor."
        ),
    )

    p.add_argument(
        "--system",
        default=None,
        help="Overall system/material label shown as a separate legend entry (pure text).",
    )
    p.add_argument(
        "--system-format",
        choices=["chem", "raw"],
        default="chem",
        help="Render --system as chemical formula with subscripts (chem) or raw text (raw). Default: chem.",
    )
    p.add_argument(
        "--system-fontsize",
        type=float,
        default=None,
        help="Font size for --system legend text. If omitted, uses an automatic larger size.",
    )
    p.add_argument(
        "--system-loc",
        default="upper left",
        help="Legend location for --system (matplotlib legend loc). Default: upper left.",
    )
    p.add_argument(
        "--system-bbox",
        default=None,
        help=(
            "Optional legend anchor (bbox_to_anchor) in axes coordinates 'x,y'. "
            "If provided, legend placement uses both --system-loc and this anchor."
        ),
    )

    p.add_argument(
        "--norm",
        default=None,
        nargs="+",
        help=(
            "Optional per-dataset normalization factor(s) applied to the energy axis after Fermi shift: "
            "E_plot = (E - Ef)/norm. Provide one per dataset, or a single value to broadcast."
        ),
    )

    p.add_argument("--no-bold", action="store_true", help="Disable bold text in default style")

    return p


def _parse_lim(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = s.split(",", 1)
    return float(a), float(b)


def _parse_figsize(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    s2 = s.strip()
    if "," in s2:
        a, b = s2.split(",", 1)
        w = float(a)
        h = float(b)
    else:
        # Allow passing only width; choose a reasonable default aspect.
        w = float(s2)
        h = 0.45 * w
    if w <= 0 or h <= 0:
        raise SystemExit(f"Invalid --figsize {s!r}: width and height must be > 0")
    return w, h


def _parse_xy(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = str(s).split(",", 1)
    return float(a), float(b)


def _flatten_tokens(tokens: Optional[Sequence[str]]) -> List[str]:
    if not tokens:
        return []
    out: List[str] = []
    for t in tokens:
        if t is None:
            continue
        for s in str(t).split(","):
            s2 = s.strip()
            if s2:
                out.append(s2)
    return out


def _broadcast_list(xs: Sequence[str], n: int, name: str) -> List[str]:
    if len(xs) == n:
        return list(xs)
    if len(xs) == 1:
        return [str(xs[0])] * n
    raise SystemExit(f"{name} expects 1 value or {n} values, but got {len(xs)}")


def _parse_float_list(tokens: Optional[Sequence[str]], *, n: int, name: str) -> List[Optional[float]]:
    flat = _flatten_tokens(tokens)
    if not flat:
        return [None] * n
    flat = _broadcast_list(flat, n, name)
    out: List[Optional[float]] = []
    for t in flat:
        if t is None or str(t).strip() == "":
            out.append(None)
            continue
        try:
            out.append(float(t))
        except ValueError as e:
            raise SystemExit(f"{name} contains non-float token {t!r}") from e
    return out


def _map_x_to_reference(
    x_plot: np.ndarray,
    indices: Sequence[int],
    ref_x_plot: np.ndarray,
    ref_indices: Sequence[int],
) -> np.ndarray:
    """Piecewise linear mapping of this dataset's x axis onto the reference x axis."""

    x = np.asarray(x_plot, dtype=float)
    x_ref = np.asarray(ref_x_plot, dtype=float)
    if len(indices) != len(ref_indices):
        raise SystemExit("Cannot overlay: high-symmetry point count differs.")

    x2 = x.copy()
    for i in range(len(indices) - 1):
        a = int(indices[i])
        b = int(indices[i + 1])
        ra = int(ref_indices[i])
        rb = int(ref_indices[i + 1])
        if b <= a or rb <= ra:
            continue
        if a < 0 or b >= len(x2) or ra < 0 or rb >= len(x_ref):
            continue

        x0 = float(x[a])
        x1 = float(x[b])
        rx0 = float(x_ref[ra])
        rx1 = float(x_ref[rb])

        denom = (x1 - x0)
        if abs(denom) < 1e-14:
            x2[a : b + 1] = rx0
            continue
        scale = (rx1 - rx0) / denom
        x2[a : b + 1] = rx0 + (x[a : b + 1] - x0) * scale
    return x2


def _apply_scienceplots_prb_style() -> None:
    try:
        import scienceplots  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "SciencePlots is required for --style prb but could not be imported.\n"
            "Install it with: pip install SciencePlots\n"
            f"Original error: {e}"
        )
    plt.style.use(["science", "no-latex"])


def _apply_bold(ax, *, bold: bool) -> None:
    if not bold:
        return
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        t.set_fontweight("bold")


def _read_bands_out_gnu(path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return x grid and list of energy arrays (one per band).

    Primary delimiter: blank lines.
    Fallback delimiter: if x decreases significantly, start a new band.
    """

    bands_xy: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = []
    prev_x: Optional[float] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if cur:
                    bands_xy.append(cur)
                    cur = []
                prev_x = None
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue

            # Fallback: if blank line is missing, detect a new band by x reset.
            if prev_x is not None and x < prev_x - 1e-6 and len(cur) > 5:
                bands_xy.append(cur)
                cur = []
            cur.append((x, y))
            prev_x = x

    if cur:
        bands_xy.append(cur)

    if not bands_xy:
        raise SystemExit(f"No band data found in {path!r}")

    x0 = np.asarray([t[0] for t in bands_xy[0]], dtype=float)
    energies: List[np.ndarray] = []

    for i, band in enumerate(bands_xy):
        x = np.asarray([t[0] for t in band], dtype=float)
        e = np.asarray([t[1] for t in band], dtype=float)
        if len(x) != len(x0) or np.max(np.abs(x - x0)) > 1e-7:
            raise SystemExit(
                "bands.out.gnu does not seem to have a common x-grid across bands. "
                f"Band 0 length={len(x0)}, band {i} length={len(x)}."
            )
        energies.append(e)

    return x0, energies


def _read_band_in_kpoints(path: str) -> List[KPointSpec]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()

    # Find the K_POINTS crystal_b section
    kline = None
    for i, line in enumerate(text):
        if re.match(r"^\s*K_POINTS\b", line, flags=re.IGNORECASE) and "crystal_b" in line.lower():
            kline = i
            break

    if kline is None:
        raise SystemExit(f"Cannot find 'K_POINTS crystal_b' in {path}")

    # Next non-empty line should be an integer count
    j = kline + 1
    while j < len(text) and not text[j].strip():
        j += 1
    if j >= len(text):
        raise SystemExit(f"Unexpected end of file after K_POINTS in {path}")

    try:
        nk = int(text[j].strip().split()[0])
    except ValueError as e:
        raise SystemExit(f"Cannot parse number of K points after K_POINTS in {path}: {text[j]!r}") from e

    specs: List[KPointSpec] = []
    j += 1
    while j < len(text) and len(specs) < nk:
        line = text[j].strip()
        j += 1
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            kx, ky, kz = float(parts[0]), float(parts[1]), float(parts[2])
            n = int(float(parts[3]))
        except ValueError:
            continue
        specs.append(KPointSpec(k=(kx, ky, kz), n=n))

    if len(specs) != nk:
        raise SystemExit(f"Parsed {len(specs)} k-points but expected {nk} from {path}")

    return specs


def _read_kpath_labels(path: str) -> List[Tuple[Tuple[float, float, float], str]]:
    """Read VASPKIT-style KPATH.in.

    Returns a list of (k, label). Multiple occurrences of same point are allowed.
    """

    entries: List[Tuple[Tuple[float, float, float], str]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # skip header-ish lines
            if any(line.lower().startswith(s) for s in ["k-path", "line-mode", "reciprocal"]):
                continue
            # First meaningful integer line (e.g., "20") should be ignored
            if re.fullmatch(r"\d+", line):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                kx, ky, kz = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            label = parts[3].strip()
            if label:
                entries.append(((kx, ky, kz), label))

    if not entries:
        raise SystemExit(f"No label entries parsed from {path}")

    return entries


def _normalize_label(label: str) -> str:
    u = label.strip()
    if not u:
        return u
    up = u.upper()
    # Avoid mathtext in tick labels (can break with some style/rcParams combinations).
    # Use a plain Unicode Gamma instead.
    if up in {"GAMMA", "Γ", "G"}:
        return "Γ"
    return u


def _find_label_for_k(k: Tuple[float, float, float], entries: Sequence[Tuple[Tuple[float, float, float], str]]) -> Optional[str]:
    # Match by Euclidean distance in fractional coordinates.
    best: Optional[Tuple[float, str]] = None
    for kk, lab in entries:
        dx = k[0] - kk[0]
        dy = k[1] - kk[1]
        dz = k[2] - kk[2]
        d2 = dx * dx + dy * dy + dz * dz
        if best is None or d2 < best[0]:
            best = (d2, lab)

    if best is None:
        return None

    # Tolerance: allow small floating differences (band.in often has rounded decimals)
    if best[0] <= (1e-3) ** 2:
        return _normalize_label(best[1])
    return None


def _infer_indices(
    specs: Sequence[KPointSpec],
    n_data: int,
) -> Tuple[List[int], str]:
    """Infer mapping from high-symmetry points to indices on the x-grid.

    Two conventions exist for the segment count N:
    - overlap: total points = 1 + sum(N_i - 1)
    - no-overlap: total points = 1 + sum(N_i)

    We try both and pick the one matching the data length.
    """

    def build(*, overlap: bool, break_advances: bool) -> List[int]:
        idx = 0
        out = [0]
        for i in range(len(specs) - 1):
            n = int(specs[i].n)
            if n <= 1:
                if break_advances:
                    idx += 1
                out.append(idx)
                continue
            idx += (n - 1) if overlap else n
            out.append(idx)
        return out

    # Try 4 conventions:
    # - overlap vs no-overlap for N
    # - whether a discontinuity (N=1) advances by one data point (QE often outputs an extra point)
    candidates: List[Tuple[str, List[int]]] = []
    candidates.append(("overlap+break", build(overlap=True, break_advances=True)))
    candidates.append(("no-overlap+break", build(overlap=False, break_advances=True)))
    candidates.append(("overlap", build(overlap=True, break_advances=False)))
    candidates.append(("no-overlap", build(overlap=False, break_advances=False)))

    exact = [(name, ind) for (name, ind) in candidates if (ind[-1] + 1) == n_data]
    if exact:
        return exact[0][1], exact[0][0]

    # Fallback: choose the closest one
    best_name, best_ind = candidates[0]
    best_err = abs((best_ind[-1] + 1) - n_data)
    for name, ind in candidates[1:]:
        err = abs((ind[-1] + 1) - n_data)
        if err < best_err:
            best_name, best_ind, best_err = name, ind, err
    return best_ind, best_name + "*"


def _build_segments(specs: Sequence[KPointSpec], indices: Sequence[int]) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    for i in range(len(specs) - 1):
        n = int(specs[i].n)
        if n <= 1:
            continue
        s = int(indices[i])
        e = int(indices[i + 1])
        if e <= s:
            continue
        segs.append((s, e))
    if not segs:
        raise SystemExit("No continuous segments found (did you set all N=1?)")
    return segs


def _build_ticks_and_labels(
    x: np.ndarray,
    specs: Sequence[KPointSpec],
    indices: Sequence[int],
    labels: Sequence[str],
) -> Tuple[List[float], List[str]]:
    xticks: List[float] = []
    xlabels: List[str] = []

    i = 0
    while i < len(specs):
        pos = float(x[int(indices[i])])
        lab = labels[i]

        if i < len(specs) - 1 and int(specs[i].n) == 1:
            lab2 = labels[i + 1]
            lab = f"{lab}|{lab2}"
            i += 2
        else:
            i += 1

        xticks.append(pos)
        xlabels.append(lab)

    # Remove consecutive duplicates (can happen for repeated points)
    xticks2: List[float] = []
    xlabels2: List[str] = []
    for pos, lab in zip(xticks, xlabels):
        if xticks2 and abs(pos - xticks2[-1]) < 1e-10:
            # merge labels if different
            if lab != xlabels2[-1] and lab not in xlabels2[-1]:
                xlabels2[-1] = f"{xlabels2[-1]}|{lab}"
            continue
        xticks2.append(pos)
        xlabels2.append(lab)

    return xticks2, xlabels2


def _find_overlapping_xticklabels(fig: plt.Figure, ax: plt.Axes) -> List[int]:
    """Return indices (in ax.get_xticklabels() order) that overlap with neighbors."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    ticks = ax.get_xticklabels()
    bboxes: List[Optional[object]] = []
    for t in ticks:
        if not t.get_visible() or not t.get_text():
            bboxes.append(None)
            continue
        bboxes.append(t.get_window_extent(renderer=renderer))

    bad: set[int] = set()
    for i in range(len(ticks) - 1):
        a = bboxes[i]
        b = bboxes[i + 1]
        if a is None or b is None:
            continue
        if a.overlaps(b):
            bad.add(i)
            bad.add(i + 1)
    return sorted(bad)


def _fix_dense_xticklabels(fig: plt.Figure, ax: plt.Axes) -> None:
    """Fix dense high-symmetry tick labels without touching non-overlapping ones.

    Strategy (in order):
    1) shrink only overlapping labels
    2) rotate only overlapping labels (max 45°, no 90°)
    3) shrink further
    4) stagger overlapping labels into two rows using leading newlines
    """

    ticks = ax.get_xticklabels()
    bad = _find_overlapping_xticklabels(fig, ax)
    if not bad:
        return

    bad_ticks = [ticks[i] for i in bad if 0 <= i < len(ticks)]
    bad_ticks = [t for t in bad_ticks if t.get_visible() and t.get_text()]
    if not bad_ticks:
        return

    orig_text: Dict[int, str] = {}
    for i in bad:
        if 0 <= i < len(ticks):
            orig_text[i] = ticks[i].get_text()

    base_fs = float(bad_ticks[0].get_fontsize())

    # 1) Shrink only overlapping labels (no rotation).
    for scale in (0.95, 0.9, 0.85, 0.8):
        for t in bad_ticks:
            t.set_fontsize(base_fs * scale)
        fig.tight_layout()
        if not _find_overlapping_xticklabels(fig, ax):
            return

    # 2) Rotate only overlapping labels (max 45°).
    for t in bad_ticks:
        t.set_rotation(45)
        t.set_ha("right")
        t.set_rotation_mode("anchor")
    fig.tight_layout()
    if not _find_overlapping_xticklabels(fig, ax):
        return

    # 3) Shrink further (keep 45°).
    for scale in (0.75, 0.7, 0.65):
        for t in bad_ticks:
            t.set_fontsize(base_fs * scale)
        fig.tight_layout()
        if not _find_overlapping_xticklabels(fig, ax):
            return

    # 4) Stagger into two rows via leading newlines (try without rotation first).
    for t in bad_ticks:
        t.set_rotation(0)
        t.set_ha("center")
        t.set_rotation_mode("default")
        t.set_fontsize(base_fs * 0.8)

    for j, i in enumerate(bad):
        if i not in orig_text:
            continue
        txt = orig_text[i]
        ticks[i].set_text(txt if (j % 2 == 0) else ("\n" + txt))

    fig.tight_layout()
    if not _find_overlapping_xticklabels(fig, ax):
        return

    # If still overlapping, combine staggering with 45° (overlapping labels only).
    for t in bad_ticks:
        t.set_rotation(45)
        t.set_ha("right")
        t.set_rotation_mode("anchor")
    fig.tight_layout()


def main() -> None:
    args = _build_parser().parse_args()

    ylim = _parse_lim(args.ylim)
    figsize = _parse_figsize(args.figsize)
    legend_bbox = _parse_xy(args.legend_bbox)
    system_bbox = _parse_xy(args.system_bbox)

    bands_paths = [str(x) for x in args.bands]
    n_cases = len(bands_paths)
    band_in_paths = _broadcast_list([str(x) for x in args.band_in], n_cases, "--band-in")

    kpath_paths: List[Optional[str]] = []
    if args.kpath is None:
        kpath_paths = [None] * n_cases
    else:
        kpath_paths = [str(x) for x in _broadcast_list([str(x) for x in args.kpath], n_cases, "--kpath")]

    fermis = _parse_float_list(args.fermi, n=n_cases, name="--fermi")

    legends = _flatten_tokens(args.legend)
    if legends:
        legends = _broadcast_list(legends, n_cases, "--legend")
    else:
        legends = [Path(p).name for p in bands_paths]

    norms = _parse_float_list(args.norm, n=n_cases, name="--norm")
    for i, nv in enumerate(norms):
        if nv is not None and float(nv) == 0.0:
            raise SystemExit(f"--norm must be non-zero (dataset#{i+1})")

    # Reference axis (first dataset)
    ref_xticks: Optional[List[float]] = None
    ref_xticklabels: Optional[List[str]] = None
    ref_xlim: Optional[Tuple[float, float]] = None
    ref_x: Optional[np.ndarray] = None
    ref_indices: Optional[List[int]] = None
    ref_scheme: Optional[str] = None
    ref_n_data: Optional[int] = None

    # Style
    if args.style == "prb":
        _apply_scienceplots_prb_style()

    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()

    if args.lw is not None and float(args.lw) <= 0:
        raise SystemExit("--lw must be > 0")
    lw_band = float(args.lw) if args.lw is not None else (1.0 if args.style == "prb" else 1.4)

    # High-contrast dataset colors: black, red, blue, ...
    case_colors = [
        "black",
        "tab:red",
        "tab:blue",
        "tab:green",
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:cyan",
        "tab:pink",
        "tab:olive",
        "tab:gray",
    ]

    # --- Load & plot each dataset ---
    for ic in range(n_cases):
        x, bands = _read_bands_out_gnu(bands_paths[ic])
        specs = _read_band_in_kpoints(band_in_paths[ic])

        label_entries: Optional[List[Tuple[Tuple[float, float, float], str]]] = None
        kp = kpath_paths[ic]
        if kp:
            label_entries = _read_kpath_labels(kp)

        labels: List[str] = []
        for i, sp in enumerate(specs):
            lab = None
            if label_entries is not None:
                lab = _find_label_for_k(sp.k, label_entries)
            if lab is None:
                lab = f"K{i+1}"
            labels.append(lab)

        indices, scheme = _infer_indices(specs, n_data=len(x))
        segments = _build_segments(specs, indices)
        xticks, xticklabels = _build_ticks_and_labels(x, specs, indices, labels)

        if ref_xticks is None:
            ref_xticks = xticks
            ref_xticklabels = xticklabels
            ref_xlim = (float(x[0]), float(x[-1]))
            ref_x = np.asarray(x, dtype=float)
            ref_indices = list(indices)
            ref_scheme = scheme
            ref_n_data = len(x)
        else:
            if len(xticklabels) != len(ref_xticklabels):
                raise SystemExit("Cannot overlay multiple datasets: high-symmetry tick count differs.")
            if any(a != b for a, b in zip(xticklabels, ref_xticklabels)):
                raise SystemExit("Cannot overlay multiple datasets: high-symmetry labels differ.")

        # Map x to reference axis
        if ref_x is None or ref_indices is None:
            x_mapped = np.asarray(x, dtype=float)
        else:
            x_mapped = _map_x_to_reference(x, indices, ref_x, ref_indices)

        # Energy shift per dataset
        y_arrays = bands
        ef = fermis[ic]
        if ef is not None:
            y_arrays = [b - float(ef) for b in bands]
        nv = norms[ic]
        if nv is not None:
            y_arrays = [b / float(nv) for b in y_arrays]

        col = case_colors[ic % len(case_colors)]
        alpha = 1.0 if ic == 0 else 0.65
        for e in y_arrays:
            for (s, t) in segments:
                ax.plot(x_mapped[s : t + 1], e[s : t + 1], color=col, lw=lw_band, alpha=alpha)

    if ref_xticks is None or ref_xticklabels is None or ref_xlim is None:
        raise SystemExit("No dataset loaded")

    # High-symmetry separators and ticks
    for xpos in ref_xticks:
        ax.axvline(xpos, color="black", lw=0.6, alpha=0.6)

    ax.set_xticks(ref_xticks)
    ax.set_xticklabels(ref_xticklabels)
    # Keep only high-symmetry vertical lines on x-axis; hide tick marks (but keep labels).
    ax.tick_params(axis="x", which="both", bottom=False, top=False, length=0)

    # Dataset legend (colored lines)
    handles_leg: List[Line2D] = []
    for ic in range(n_cases):
        lab = _format_system_label(str(legends[ic]), str(args.legend_format))
        col = case_colors[ic % len(case_colors)]
        handles_leg.append(Line2D([0], [0], color=col, lw=lw_band, label=lab))

    leg_main = None
    if handles_leg:
        kwargs = dict(
            handles=handles_leg,
            loc=str(args.legend_loc),
            frameon=False,
            borderaxespad=0.2,
            handlelength=1.8,
            handletextpad=0.6,
            labelspacing=0.35,
        )
        if args.legend_fontsize is not None:
            kwargs["fontsize"] = float(args.legend_fontsize)
        if legend_bbox is None:
            leg_main = ax.legend(**kwargs)
        else:
            leg_main = ax.legend(
                **kwargs,
                bbox_to_anchor=legend_bbox,
                bbox_transform=ax.transAxes,
            )

    # Global system annotation legend (pure text)
    if args.system is not None and str(args.system).strip():
        sys_lab = _format_system_label(str(args.system), str(args.system_format))
        h = Line2D([], [], color="none", label=sys_lab)

        fs = args.system_fontsize
        if fs is None:
            try:
                fs = float(ax.yaxis.label.get_size()) * 1.15
            except Exception:
                fs = None

        if leg_main is not None:
            ax.add_artist(leg_main)

        if system_bbox is None:
            leg_sys = ax.legend(
                handles=[h],
                loc=str(args.system_loc),
                frameon=False,
                handlelength=0,
                handletextpad=0.0,
                borderaxespad=0.2,
                fontsize=fs,
            )
        else:
            leg_sys = ax.legend(
                handles=[h],
                loc=str(args.system_loc),
                bbox_to_anchor=system_bbox,
                bbox_transform=ax.transAxes,
                frameon=False,
                handlelength=0,
                handletextpad=0.0,
                borderaxespad=0.2,
                fontsize=fs,
            )
        if leg_sys is not None:
            for t in leg_sys.get_texts():
                t.set_fontweight("bold")

    if args.fermi_line:
        ax.axhline(0.0, color="gray", linestyle="--", lw=1.0, alpha=0.8)

    ax.set_xlim(*ref_xlim)
    if ylim:
        ax.set_ylim(*ylim)

    ax.set_ylabel(r"$E - E_{f}$ (eV)" if any(f is not None for f in fermis) else "Energy (eV)")

    # No x-label (standard band plot)

    if args.style == "default":
        ax.grid(True, alpha=0.25)
        if not args.no_bold:
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams["axes.linewidth"] = 2
            _apply_bold(ax, bold=True)
    else:
        # SciencePlots sets its own typography; keep minimal tweaks.
        pass

    fig.tight_layout()
    _fix_dense_xticklabels(fig, ax)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"Saved: {args.out}")
    if ref_scheme is not None:
        ndata = ref_n_data if ref_n_data is not None else "?"
        print(f"K-point indexing convention: {ref_scheme} (data points per band: {ndata})")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
