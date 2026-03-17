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

    p.add_argument("--bands", required=True, help="Path to bands.out.gnu")
    p.add_argument("--band-in", required=True, help="Path to band.in (QE input with K_POINTS crystal_b)")
    p.add_argument("--kpath", default=None, help="Path to KPATH.in (VASPKIT). If omitted, labels are guessed.")

    p.add_argument(
        "--fermi",
        type=float,
        default=None,
        help="Fermi energy in eV. If provided, shift energies as E -> E - Ef so Ef is at 0 eV.",
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
        "--system",
        default=None,
        help="System label shown as a small legend entry (e.g. 'Zr2SeC').",
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

    # Load band data
    x, bands = _read_bands_out_gnu(args.bands)

    # Parse path specification and labels
    specs = _read_band_in_kpoints(args.band_in)

    label_entries: Optional[List[Tuple[Tuple[float, float, float], str]]] = None
    if args.kpath:
        label_entries = _read_kpath_labels(args.kpath)

    labels: List[str] = []
    for i, sp in enumerate(specs):
        lab = None
        if label_entries is not None:
            lab = _find_label_for_k(sp.k, label_entries)
        if lab is None:
            # Fallback: try common special points
            lab = f"K{i+1}"
        labels.append(lab)

    indices, scheme = _infer_indices(specs, n_data=len(x))
    segments = _build_segments(specs, indices)
    xticks, xticklabels = _build_ticks_and_labels(x, specs, indices, labels)

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

    # Energy shift
    y_arrays = bands
    if args.fermi is not None:
        y_arrays = [b - float(args.fermi) for b in bands]

    # Plot each band as black solid lines, but break at jumps by plotting per-segment.
    for e in y_arrays:
        for (s, t) in segments:
            ax.plot(x[s : t + 1], e[s : t + 1], color="black", lw=lw_band)

    # High-symmetry separators and ticks
    for xpos in xticks:
        ax.axvline(xpos, color="black", lw=0.6, alpha=0.6)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    # Keep only high-symmetry vertical lines on x-axis; hide tick marks (but keep labels).
    ax.tick_params(axis="x", which="both", bottom=False, top=False, length=0)

    if args.system:
        sys_lab = _format_system_label(str(args.system), str(args.system_format))
        h = Line2D([], [], color="none", label=sys_lab)
        fs = args.system_fontsize
        if fs is None:
            try:
                fs = float(ax.yaxis.label.get_size()) * 1.15
            except Exception:
                fs = None
        leg = ax.legend(
            handles=[h],
            loc=str(args.system_loc),
            frameon=False,
            handlelength=0,
            handletextpad=0.0,
            borderaxespad=0.2,
            fontsize=fs,
        )
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontweight("bold")

    if args.fermi_line:
        ax.axhline(0.0, color="gray", linestyle="--", lw=1.0, alpha=0.8)

    ax.set_xlim(float(x[0]), float(x[-1]))
    if ylim:
        ax.set_ylim(*ylim)

    ax.set_ylabel(r"$E - E_{f}$ (eV)" if args.fermi is not None else "Energy (eV)")

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
    print(f"K-point indexing convention: {scheme} (data points per band: {len(x)})")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
