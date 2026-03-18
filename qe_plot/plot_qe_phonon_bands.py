#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class KPointSpec:
    q: Tuple[float, float, float]
    n: int  # number of points from this q-point to the next; n==1 means a jump/break


def _format_system_label(label: str, mode: str) -> str:
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
            "Plot Quantum ESPRESSO phonon dispersion from matdyn output (*.freq.gp).\n\n"
            "Inputs:\n"
            "- *.freq.gp: first column is path coordinate (x), each following column is one phonon branch\n"
            "- matdyn.in: contains q-point path in band form (q_in_band_form=.true.) with per-segment point counts\n"
            "- KPATH.in (optional): VASPKIT-style labels for high-symmetry points\n\n"
            "Jump points: if a q-point line in matdyn.in has N=1, it is treated as a discontinuity;\n"
            "the tick label at that position is merged as 'A|L' (end|start), and curves are not connected across it."
        )
    )

    p.add_argument(
        "--freq",
        required=True,
        nargs="+",
        help="One or more *.freq.gp files (matdyn band output). If multiple are given, they are overlaid for comparison.",
    )
    p.add_argument(
        "--matdyn-in",
        required=True,
        nargs="+",
        help="One or more matdyn.in files. If a single file is given, it is reused for all datasets.",
    )
    p.add_argument(
        "--kpath",
        default=None,
        nargs="+",
        help="One or more KPATH.in files (VASPKIT). If omitted, labels are guessed. If a single file is given, it is reused.",
    )

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

    p.add_argument(
        "--system",
        default=None,
        nargs="+",
        help=(
            "System label(s) used as curve labels. Provide one per dataset, e.g. '--system Zr2SC Zr15S8C8'. "
            "If only one label is given, it is broadcast. Comma-separated tokens are also accepted."
        ),
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
            "Optional legend anchor (bbox_to_anchor) in axes coordinates 'x,y' (e.g. '1.02,1.0' for outside right). "
            "If provided, legend placement uses both --system-loc and this anchor."
        ),
    )

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


def _parse_xy(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = str(s).split(",", 1)
    return float(a), float(b)


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
        hint = ""
        low = str(path).lower()
        if low.endswith(".gp") or "freq" in low:
            hint = (
                "\nHint: --matdyn-in expects a QE matdyn input file (matdyn.in). "
                "It looks like you may have passed a *.freq.gp file by mistake."
            )
        raise SystemExit(f"Cannot parse q-point count in {path}: {lines[j]!r}{hint}") from e

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


def _map_x_to_reference(
    x_plot: np.ndarray,
    indices: Sequence[int],
    ref_x_plot: np.ndarray,
    ref_indices: Sequence[int],
) -> np.ndarray:
    """Piecewise linear mapping of this dataset's x axis onto the reference x axis.

    Mapping is done segment-by-segment between consecutive high-symmetry points.
    This allows overlaying different structures with different raw x coordinates.
    """

    x = np.asarray(x_plot, dtype=float)
    x_ref = np.asarray(ref_x_plot, dtype=float)
    if len(x) != len(x_ref):
        # We still can map as long as indices are compatible; the per-row mapping uses x itself.
        pass

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
    system_bbox = _parse_xy(args.system_bbox)

    if args.lw is not None and float(args.lw) <= 0:
        raise SystemExit("--lw must be > 0")

    if args.style == "prb":
        _apply_scienceplots_prb_style()

    freq_paths = [str(x) for x in args.freq]
    n_cases = len(freq_paths)

    matdyn_paths = _broadcast_list([str(x) for x in args.matdyn_in], n_cases, "--matdyn-in")
    kpath_args = args.kpath
    kpath_paths: List[Optional[str]] = []
    if kpath_args is None:
        kpath_paths = [None] * n_cases
    else:
        kps = _broadcast_list([str(x) for x in kpath_args], n_cases, "--kpath")
        kpath_paths = [str(x) for x in kps]

    systems = _flatten_tokens(args.system)
    if systems:
        systems = _broadcast_list(systems, n_cases, "--system")
    else:
        systems = [""] * n_cases

    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()

    lw = float(args.lw) if args.lw is not None else (0.8 if args.style == "prb" else 1.4)

    # Reference axis (first dataset)
    ref_xticks: Optional[List[float]] = None
    ref_xticklabels: Optional[List[str]] = None
    ref_xlim: Optional[Tuple[float, float]] = None
    ref_x_plot: Optional[np.ndarray] = None
    ref_indices: Optional[List[int]] = None

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

    scheme = None
    for ic in range(n_cases):
        x, ymat = _read_freq_gp(freq_paths[ic])
        ymat = _convert_freq_units(ymat, args.unit)
        specs = _read_matdyn_in_qpoints(matdyn_paths[ic])

        label_entries: Optional[List[Tuple[Tuple[float, float, float], str]]] = None
        kp = kpath_paths[ic]
        if kp:
            label_entries = _read_kpath_labels(kp)

        hs_labels: List[str] = []
        for i, sp in enumerate(specs):
            lab = None
            if label_entries is not None:
                lab = _find_label_for_q(sp.q, label_entries)
            if lab is None:
                lab = f"Q{i+1}"
            hs_labels.append(lab)

        indices, scheme_i = _infer_indices(specs, n_data=len(x))
        scheme = scheme or scheme_i
        segments = _build_segments(specs, indices)

        if args.keep_jumps:
            x_plot = x
        else:
            x_plot = _compress_x_jumps_by_specs(x, specs, indices)

        xticks, xticklabels = _build_ticks_and_labels(x_plot, specs, indices, hs_labels)

        if ref_xticks is None:
            ref_xticks = xticks
            ref_xticklabels = xticklabels
            ref_xlim = (float(x_plot[0]), float(x_plot[-1]))
            ref_x_plot = np.asarray(x_plot, dtype=float)
            ref_indices = list(indices)
        else:
            if len(xticklabels) != len(ref_xticklabels):
                raise SystemExit("Cannot overlay multiple datasets: high-symmetry tick count differs.")
            if any(a != b for a, b in zip(xticklabels, ref_xticklabels)):
                raise SystemExit("Cannot overlay multiple datasets: high-symmetry labels differ.")

        # Map x to reference axis
        if ref_x_plot is None or ref_indices is None:
            x_mapped = np.asarray(x_plot, dtype=float)
        else:
            x_mapped = _map_x_to_reference(x_plot, indices, ref_x_plot, ref_indices)

        col = case_colors[ic % len(case_colors)]
        alpha = 1.0 if ic == 0 else 0.65
        n_branch = int(ymat.shape[1])
        for j in range(n_branch):
            e = ymat[:, j]
            for (s, t) in segments:
                ax.plot(x_mapped[s : t + 1], e[s : t + 1], color=col, lw=lw, alpha=alpha)

    if ref_xticks is None or ref_xticklabels is None or ref_xlim is None:
        raise SystemExit("No dataset loaded")

    for xpos in ref_xticks:
        ax.axvline(xpos, color="black", lw=0.6, alpha=0.6)

    ax.set_xticks(ref_xticks)
    ax.set_xticklabels(ref_xticklabels)
    # Keep only high-symmetry vertical lines on x-axis; hide tick marks (but keep labels).
    ax.tick_params(axis="x", which="both", bottom=False, top=False, length=0)

    ax.set_xlim(*ref_xlim)
    if ylim:
        ax.set_ylim(*ylim)

    # System legend: one entry per dataset
    if any(systems):
        handles: List[Line2D] = []
        for ic in range(n_cases):
            if not systems[ic]:
                continue
            sys_lab = _format_system_label(str(systems[ic]), str(args.system_format))
            col = case_colors[ic % len(case_colors)]
            handles.append(Line2D([0], [0], color=col, lw=lw, label=sys_lab))

        fs = args.system_fontsize
        if fs is None:
            try:
                fs = float(ax.yaxis.label.get_size()) * 1.15
            except Exception:
                fs = None
        if system_bbox is None:
            leg = ax.legend(
                handles=handles,
                loc=str(args.system_loc),
                frameon=False,
                fontsize=fs,
            )
        else:
            leg = ax.legend(
                handles=handles,
                loc=str(args.system_loc),
                bbox_to_anchor=system_bbox,
                bbox_transform=ax.transAxes,
                frameon=False,
                fontsize=fs,
            )
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontweight("bold")

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
    if scheme is not None:
        print(f"Q-point indexing convention: {scheme}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
