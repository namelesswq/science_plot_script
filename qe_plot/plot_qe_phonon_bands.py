#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class KPointSpec:
    q: Tuple[float, float, float]
    n: int  # number of points from this q-point to the next; n==1 means a jump/break


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot Quantum ESPRESSO phonon dispersion from matdyn output (*.freq.gp).\n\n"
            "Inputs:\n"
            "- *.freq.gp: first column is path coordinate (x), each following column is one phonon branch\n"
            "- matdyn.in: contains q-point path in band form (q_in_band_form=.true.) with per-segment point counts\n"
            "- KPATH.in (optional): VASPKIT-style labels for high-symmetry points\n\n"
            "Jump points: if a q-point line in matdyn.in has N=1, it is treated as a discontinuity;\n"
            "the tick label at that position is merged as 'A|L' (end|start), and curves are not connected across it."
        )
    )

    p.add_argument("--freq", required=True, help="Path to *.freq.gp (matdyn band output)")
    p.add_argument("--matdyn-in", required=True, help="Path to matdyn.in")
    p.add_argument("--kpath", default=None, help="Path to KPATH.in (VASPKIT). If omitted, labels are guessed.")

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
        help="Line width for phonon branches. If omitted, keep style defaults (prb~0.8, default larger).",
    )

    p.add_argument("--ylim", default=None, help='y limits "ymin,ymax"')
    p.add_argument(
        "--unit",
        choices=["cm^-1", "THz"],
        default="THz",
        help=(
            "Frequency unit for plotting. QE matdyn *.freq.gp is commonly in cm^-1. "
            "When using THz, the script converts as THz = (cm^-1)/33.35641. [default: THz]"
        ),
    )
    p.add_argument("--ylabel", default=None, help="y-axis label (default depends on --unit)")

    p.add_argument(
        "--keep-jumps",
        action="store_true",
        help="Keep the original x-axis jumps at discontinuities (default: compress jumps so segments meet at one x).",
    )

    p.add_argument("--out", default="phonon_bands.png", help="Output image path")
    p.add_argument("--show", action="store_true", help="Show interactively")
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


def _read_freq_gp(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise SystemExit(f"Unexpected *.freq.gp table: shape={data.shape}")
    x = np.asarray(data[:, 0], dtype=float)
    y = np.asarray(data[:, 1:], dtype=float)  # (nq, nbranch)
    return x, y


def _convert_freq_units(y: np.ndarray, unit: str) -> np.ndarray:
    """Convert matdyn frequencies to the requested unit.

    Assumption: input y from *.freq.gp is in cm^-1 (typical for matdyn.x).
    """

    y2 = np.asarray(y, dtype=float)
    if unit == "cm^-1":
        return y2
    # cm^-1 -> THz
    return y2 / 33.35641


def _read_matdyn_in_qpoints(path: str) -> List[KPointSpec]:
    """Parse q-point path from matdyn.in.

    Expected structure (typical):
      &input ... /
      <Nq>
      qx qy qz  N
      ... (Nq lines)
    """

    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()

    # Find the end of namelist: a line containing only '/'
    end_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*/\s*$", line):
            end_idx = i
            break
    if end_idx is None:
        # fallback: some inputs might omit '/', try to find the first standalone integer line
        end_idx = -1

    j = end_idx + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j >= len(lines):
        raise SystemExit(f"Cannot find q-point count after namelist in {path}")

    try:
        nq = int(lines[j].strip().split()[0])
    except ValueError as e:
        raise SystemExit(f"Cannot parse q-point count in {path}: {lines[j]!r}") from e

    specs: List[KPointSpec] = []
    j += 1
    while j < len(lines) and len(specs) < nq:
        line = lines[j].strip()
        j += 1
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            qx, qy, qz = float(parts[0]), float(parts[1]), float(parts[2])
            n = int(float(parts[3]))
        except ValueError:
            continue
        specs.append(KPointSpec(q=(qx, qy, qz), n=n))

    if len(specs) != nq:
        raise SystemExit(f"Parsed {len(specs)} q-points but expected {nq} from {path}")

    return specs


def _read_kpath_labels(path: str) -> List[Tuple[Tuple[float, float, float], str]]:
    entries: List[Tuple[Tuple[float, float, float], str]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if any(line.lower().startswith(s) for s in ["k-path", "line-mode", "reciprocal"]):
                continue
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
    if up in {"GAMMA", "Γ", "G"}:
        return "Γ"
    return u


def _find_label_for_q(
    q: Tuple[float, float, float],
    entries: Sequence[Tuple[Tuple[float, float, float], str]],
) -> Optional[str]:
    best: Optional[Tuple[float, str]] = None
    for qq, lab in entries:
        dx = q[0] - qq[0]
        dy = q[1] - qq[1]
        dz = q[2] - qq[2]
        d2 = dx * dx + dy * dy + dz * dz
        if best is None or d2 < best[0]:
            best = (d2, lab)

    if best is None:
        return None

    if best[0] <= (1e-3) ** 2:
        return _normalize_label(best[1])
    return None


def _infer_indices(specs: Sequence[KPointSpec], n_data: int) -> Tuple[List[int], str]:
    def build(overlap: bool) -> List[int]:
        idx = 0
        out = [0]
        for i in range(len(specs) - 1):
            n = int(specs[i].n)
            if n <= 1:
                # For a discontinuity (N=1) in matdyn band input, QE typically outputs
                # a new point to start the next segment (with a jump in the x-axis).
                # Treat it as advancing by one data row.
                idx += 1
                out.append(idx)
                continue
            idx += (n - 1) if overlap else n
            out.append(idx)
        return out

    ind1 = build(overlap=True)
    ind2 = build(overlap=False)

    len1 = ind1[-1] + 1
    len2 = ind2[-1] + 1

    if len1 == n_data and len2 != n_data:
        return ind1, "overlap"
    if len2 == n_data and len1 != n_data:
        return ind2, "no-overlap"
    if len1 == n_data and len2 == n_data:
        return ind1, "overlap"

    if abs(len1 - n_data) <= abs(len2 - n_data):
        return ind1, "overlap*"
    return ind2, "no-overlap*"


def _compress_x_jumps_by_specs(x: np.ndarray, specs: Sequence[KPointSpec], indices: Sequence[int]) -> np.ndarray:
    """Remove artificial x-axis gaps at discontinuities (N=1).

    For each i with specs[i].n==1, define gap = x[idx(i+1)] - x[idx(i)].
    Then shift all subsequent points by -gap so x becomes continuous at that break.

    Note: curves are still plotted per continuous segment; this only changes x-coordinates.
    """

    x2 = np.asarray(x, dtype=float).copy()
    if len(indices) != len(specs):
        return x2

    # Apply shifts in the natural order along the path.
    for i in range(len(specs) - 1):
        if int(specs[i].n) != 1:
            continue
        a = int(indices[i])
        b = int(indices[i + 1])
        if a < 0 or b < 0 or a >= len(x2) or b >= len(x2):
            continue
        if b <= a:
            continue
        gap = float(x2[b] - x2[a])
        if gap == 0.0:
            continue
        x2[b:] -= gap
    return x2


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

    xticks2: List[float] = []
    xlabels2: List[str] = []
    for pos, lab in zip(xticks, xlabels):
        if xticks2 and abs(pos - xticks2[-1]) < 1e-10:
            if lab != xlabels2[-1] and lab not in xlabels2[-1]:
                xlabels2[-1] = f"{xlabels2[-1]}|{lab}"
            continue
        xticks2.append(pos)
        xlabels2.append(lab)

    return xticks2, xlabels2


def _find_overlapping_xticklabels(fig: plt.Figure, ax: plt.Axes) -> List[int]:
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

    for scale in (0.95, 0.9, 0.85, 0.8):
        for t in bad_ticks:
            t.set_fontsize(base_fs * scale)
        fig.tight_layout()
        if not _find_overlapping_xticklabels(fig, ax):
            return

    for t in bad_ticks:
        t.set_rotation(45)
        t.set_ha("right")
        t.set_rotation_mode("anchor")
    fig.tight_layout()
    if not _find_overlapping_xticklabels(fig, ax):
        return

    for scale in (0.75, 0.7, 0.65):
        for t in bad_ticks:
            t.set_fontsize(base_fs * scale)
        fig.tight_layout()
        if not _find_overlapping_xticklabels(fig, ax):
            return

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

    for t in bad_ticks:
        t.set_rotation(45)
        t.set_ha("right")
        t.set_rotation_mode("anchor")
    fig.tight_layout()


def main() -> None:
    args = _build_parser().parse_args()

    ylim = _parse_lim(args.ylim)
    figsize = _parse_figsize(args.figsize)

    if args.lw is not None and float(args.lw) <= 0:
        raise SystemExit("--lw must be > 0")

    if args.style == "prb":
        _apply_scienceplots_prb_style()

    x, ymat = _read_freq_gp(args.freq)
    ymat = _convert_freq_units(ymat, args.unit)
    specs = _read_matdyn_in_qpoints(args.matdyn_in)

    label_entries: Optional[List[Tuple[Tuple[float, float, float], str]]] = None
    if args.kpath:
        label_entries = _read_kpath_labels(args.kpath)

    hs_labels: List[str] = []
    for i, sp in enumerate(specs):
        lab = None
        if label_entries is not None:
            lab = _find_label_for_q(sp.q, label_entries)
        if lab is None:
            lab = f"Q{i+1}"
        hs_labels.append(lab)

    indices, scheme = _infer_indices(specs, n_data=len(x))
    segments = _build_segments(specs, indices)

    if args.keep_jumps:
        x_plot = x
    else:
        x_plot = _compress_x_jumps_by_specs(x, specs, indices)

    xticks, xticklabels = _build_ticks_and_labels(x_plot, specs, indices, hs_labels)

    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()

    lw = float(args.lw) if args.lw is not None else (0.8 if args.style == "prb" else 1.4)

    # Plot each phonon branch; break at jumps by plotting per-segment.
    n_branch = int(ymat.shape[1])
    for j in range(n_branch):
        e = ymat[:, j]
        for (s, t) in segments:
            ax.plot(x_plot[s : t + 1], e[s : t + 1], color="black", lw=lw)

    for xpos in xticks:
        ax.axvline(xpos, color="black", lw=0.6, alpha=0.6)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_xlim(float(x_plot[0]), float(x_plot[-1]))
    if ylim:
        ax.set_ylim(*ylim)

    if args.ylabel is not None:
        ylab = args.ylabel
    else:
        ylab = "Frequency (THz)" if args.unit == "THz" else "Frequency (cm^-1)"
    ax.set_ylabel(ylab)

    if args.style == "default":
        ax.grid(True, alpha=0.25)
        if not args.no_bold:
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams["axes.linewidth"] = 2
            _apply_bold(ax, bold=True)

    fig.tight_layout()
    _fix_dense_xticklabels(fig, ax)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)

    print(f"Saved: {args.out}")
    print(f"Q-point indexing convention: {scheme} (data points per branch: {len(x)})")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
