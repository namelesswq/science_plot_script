#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.text import Text


_CM1_PER_THz = 33.35641  # THz = (cm^-1)/33.35641


@dataclass(frozen=True)
class KPointSpec:
    q: Tuple[float, float, float]
    n: int  # number of points from this q-point to the next; n==1 means a jump/break


@dataclass(frozen=True)
class AtomSpec:
    index: int  # 1-based
    element: str


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot phonon dispersion together with phonon DOS/PDOS on one figure.\n\n"
            "Layout:\n"
            "- Left panel: phonon dispersion vs q-path (from matdyn *.freq.gp)\n"
            "- Right panel: phonon DOS/PDOS rotated by 90 degrees (DOS on x, frequency on shared y)\n\n"
            "Inputs (bands):\n"
            "- *.freq.gp: first column is path coordinate (x), each following column is one phonon branch\n"
            "- matdyn.in: contains q-point path in band form with per-segment point counts (N=1 means a break)\n"
            "- KPATH.in (optional): VASPKIT-style labels for high-symmetry points\n\n"
            "Inputs (DOS/PDOS):\n"
            "- *.dos: col0 frequency (cm^-1), col1 total DOS, col2.. per-atom PDOS\n"
            "- scf.in: used to map per-atom PDOS columns to atoms via ATOMIC_POSITIONS\n\n"
            "Unit handling:\n"
            "- If --unit THz, converts frequency as THz=(cm^-1)/33.35641.\n"
            "- By default, DOS/PDOS are scaled by Jacobian (×33.35641) so y-unit becomes states/THz/unit cell.\n"
            "  Disable with --no-jacobian."
        )
    )

    # Bands (allow multiple datasets for comparison)
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
        "--keep-jumps",
        action="store_true",
        help="Keep the original x-axis jumps at discontinuities (default: compress jumps so segments meet at one x).",
    )

    # DOS/PDOS (allow multiple datasets for comparison)
    p.add_argument(
        "--dos",
        required=True,
        nargs="+",
        help="One or more phonon DOS files (e.g. zr2sc.dos). If multiple are given, total DOS curves are compared.",
    )
    p.add_argument(
        "--scf-in",
        required=True,
        nargs="+",
        help="One or more QE scf inputs (used to map PDOS columns to atoms). If a single file is given, it is reused.",
    )
    p.add_argument(
        "--dos-norm",
        default=None,
        nargs="+",
        help=(
            "Per-dataset integer normalization factors. Provide one per dataset, e.g. '--dos-norm 1 4'. "
            "Each DOS/PDOS is divided by its factor after reading (useful for supercell->unit-cell). "
            "If only one value is given, it is broadcast to all datasets. "
            "Comma-separated tokens are also accepted (e.g. '--dos-norm 1,4')."
        ),
    )

    p.add_argument(
        "--group",
        choices=["atom", "element"],
        default="element",
        help="Plot PDOS grouped by atom or summed by element [default: element]",
    )
    p.add_argument(
        "--elements",
        default=None,
        help="Comma-separated element filter (applies to PDOS only), e.g. 'Zr,S,C'. Default: all.",
    )
    p.add_argument(
        "--atoms",
        default=None,
        help="Comma-separated atom indices (1-based) to plot when --group atom, e.g. '1,2,5-8'. Default: all.",
    )

    # Shared
    p.add_argument(
        "--unit",
        choices=["cm^-1", "THz"],
        default="THz",
        help=(
            "Frequency unit for plotting. matdyn *.freq.gp and fldos *.dos are commonly in cm^-1. "
            "When using THz, the script converts as THz = (cm^-1)/33.35641. [default: THz]"
        ),
    )
    p.add_argument(
        "--no-jacobian",
        action="store_true",
        help=(
            "Disable DOS Jacobian scaling when converting cm^-1 -> THz. "
            "By default, if --unit THz, DOS/PDOS are multiplied by 33.35641 so the y-unit becomes states/THz/unit cell."
        ),
    )

    p.add_argument("--ylim", default=None, help='Shared frequency limits "ymin,ymax" (in selected unit)')

    p.add_argument(
        "--style",
        choices=["prb", "default"],
        default="prb",
        help="Plot style preset. 'prb' uses SciencePlots (science,no-latex). Default: prb.",
    )

    p.add_argument(
        "--lw",
        type=float,
        default=None,
        help=(
            "Line width for phonon branches and DOS/PDOS curves. If omitted, keep style defaults "
            "(currently ~0.8 for prb and larger for default)."
        ),
    )

    p.add_argument(
        "--label-fontsize",
        type=float,
        default=None,
        help="Font size for axis labels and tick labels. If omitted, keep defaults/style behavior.",
    )

    p.add_argument(
        "--fontsize",
        type=float,
        default=None,
        help=(
            "Global default font size (rcParams). Does not override explicit per-item sizes like "
            "--label-fontsize/--legend-fontsize/--system-fontsize/--dos-legend-fontsize."
        ),
    )

    p.add_argument(
        "--bold-fonts",
        action="store_true",
        help="Force all text in the figure to bold (including for --style prb).",
    )

    p.add_argument(
        "--xtick-step",
        type=float,
        default=None,
        help=(
            "Major tick step for the DOS panel x-axis (states/unit). "
            "Note: the bands panel x-axis uses fixed high-symmetry ticks."
        ),
    )
    p.add_argument(
        "--ytick-step",
        type=float,
        default=None,
        help="Major tick step for the shared frequency y-axis (in the selected unit).",
    )

    p.add_argument(
        "--figsize",
        default=None,
        help='Figure size "width,height" in inches.',
    )
    p.add_argument(
        "--figsize-bands",
        default=None,
        help=(
            'Bands panel size "width,height" in inches (e.g. "7,3"). '
            "When used, you must also provide --figsize-dos."
        ),
    )
    p.add_argument(
        "--figsize-dos",
        default=None,
        help=(
            'DOS panel size "width,height" in inches (e.g. "2,3"). '
            "When used, you must also provide --figsize-bands."
        ),
    )
    p.add_argument(
        "--ratios",
        default="3,1",
        help='Panel width ratios "bands,dos" (default: 3,1).',
    )

    p.add_argument(
        "--dos-xlim",
        default=None,
        help='DOS panel x limits (DOS axis) "xmin,xmax". If omitted, auto from data.',
    )

    p.add_argument(
        "--dos-legend-loc",
        default="best",
        help="Legend location inside the DOS panel (matplotlib loc=...) [default: best]",
    )
    p.add_argument("--dos-legend-fontsize", type=float, default=None, help="DOS legend fontsize")
    p.add_argument(
        "--dos-legend-alpha",
        type=float,
        default=None,
        help="If set, draw the DOS legend with a white semi-transparent frame (0..1).",
    )
    p.add_argument(
        "--dos-legend-bbox",
        default=None,
        help="Optional DOS legend anchor (bbox_to_anchor) in axes coordinates 'x,y'.",
    )

    # Dataset legend (same interface as plot_qe_bands_with_pdos.py)
    p.add_argument(
        "--legend",
        default=None,
        nargs="+",
        help=(
            "Legend label(s) for each dataset. Provide one per dataset, or a single value to broadcast. "
            "Pass a blank label (e.g. --legend ' ') to hide that dataset's legend entry. "
            "If omitted, uses filename stem."
        ),
    )
    p.add_argument(
        "--legend-format",
        choices=["chem", "raw"],
        default="chem",
        help="Render --legend text as chemical formula with subscripts (chem) or raw text (raw). Default: chem.",
    )
    p.add_argument(
        "--legend-fontsize",
        type=float,
        default=None,
        help="Font size for dataset legend text (left panel). If omitted, uses an automatic larger size.",
    )
    p.add_argument(
        "--legend-alpha",
        type=float,
        default=None,
        help="If set, draw the dataset legend with a white semi-transparent frame (0..1).",
    )
    p.add_argument(
        "--legend-loc",
        default="upper left",
        help="Legend location for dataset legend (matplotlib legend loc). Default: upper left.",
    )
    p.add_argument(
        "--legend-bbox",
        default=None,
        help=(
            "Optional dataset legend anchor (bbox_to_anchor) in axes coordinates 'x,y'. "
            "If provided, legend placement uses both --legend-loc and this anchor."
        ),
    )

    p.add_argument("--out", default="phonon_bands_dos.png", help="Output image path")
    p.add_argument("--show", action="store_true", help="Show interactively")
    p.add_argument("--no-bold", action="store_true", help="Disable bold text in default style")

    p.add_argument(
        "--system",
        default=None,
        nargs="+",
        help=(
            "Global system annotation shown on the bands panel (pure text), independent of --legend. "
            "Pass a blank label (e.g. --system ' ') to hide."
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
        default="upper right",
        help="Location for --system annotation (matplotlib legend loc). Default: upper right.",
    )

    p.add_argument(
        "--system-alpha",
        type=float,
        default=None,
        help="If set, draw the system annotation with a white semi-transparent frame (0..1).",
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
    a, b = s.split(",", 1)
    w = float(a)
    h = float(b)
    if w <= 0 or h <= 0:
        raise SystemExit(f"Invalid --figsize {s!r}: width and height must be > 0")
    return w, h


def _parse_xy(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = str(s).split(",", 1)
    return float(a), float(b)


def _parse_ratios(s: str) -> Tuple[float, float]:
    a, b = s.split(",", 1)
    ra = float(a)
    rb = float(b)
    if ra <= 0 or rb <= 0:
        raise SystemExit(f"Invalid --ratios {s!r}: both must be > 0")
    return ra, rb


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


def _apply_global_fontsize(fontsize: Optional[float]) -> None:
    if fontsize is None:
        return
    fs = float(fontsize)
    if fs <= 0:
        raise SystemExit("--fontsize must be > 0")
    plt.rcParams.update(
        {
            "font.size": fs,
            "axes.titlesize": fs,
            "axes.labelsize": fs,
            "xtick.labelsize": fs,
            "ytick.labelsize": fs,
            "legend.fontsize": fs,
        }
    )


def _set_figure_text_weight(fig: plt.Figure, weight: str) -> None:
    for t in fig.findobj(Text):
        try:
            t.set_fontweight(weight)
        except Exception:
            pass


def _apply_legend_frame(leg, *, alpha: float) -> None:
    if leg is None:
        return
    a = float(alpha)
    if not (0.0 <= a <= 1.0):
        raise SystemExit("legend alpha must be in [0, 1]")
    leg.set_frame_on(True)
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(a)
    try:
        frame.set_edgecolor("0.6")
    except Exception:
        pass


def _apply_bold(ax, *, bold: bool) -> None:
    if not bold:
        return
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        t.set_fontweight("bold")


def _format_system_label(label: str, mode: str) -> str:
    if not label:
        return label
    if mode == "raw":
        return label
    if "$" in label:
        return label
    return re.sub(r"(?<=[A-Za-z\)])(\d+)", r"$_{\1}$", label)


def _parse_csv_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    parts = [x.strip() for x in str(s).split(",") if x.strip()]
    return parts if parts else None


def _flatten_tokens(tokens: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Flatten a list of tokens, splitting comma-separated items.

    Accepts both:
    - ['1', '4']
    - ['1,4']
    - ['Zr2SC,Zr15S8C8']
    """

    if tokens is None:
        return None
    out: List[str] = []
    for tok in tokens:
        for part in str(tok).split(","):
            p = part.strip()
            if p:
                out.append(p)
    return out if out else None


def _flatten_tokens_allow_blank(tokens: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Flatten tokens splitting by comma but keep blanks.

    This is useful for legend labels where an explicitly blank token
    (e.g. --system ' ') means "hide legend".
    """

    if tokens is None:
        return None
    out: List[str] = []
    for tok in tokens:
        for part in str(tok).split(","):
            out.append(part.strip())
    return out


def _parse_csv_ints(s: Optional[str]) -> Optional[List[int]]:
    parts = _parse_csv_list(s)
    if parts is None:
        return None
    out: List[int] = []
    for p in parts:
        try:
            v = int(p)
        except ValueError as e:
            raise SystemExit(f"Invalid --dos-norm entry {p!r}: must be integer") from e
        if v <= 0:
            raise SystemExit(f"Invalid --dos-norm entry {p!r}: must be > 0")
        out.append(v)
    return out


def _parse_int_list(tokens: Optional[Sequence[str]], *, opt_name: str) -> Optional[List[int]]:
    flat = _flatten_tokens(tokens)
    if flat is None:
        return None
    out: List[int] = []
    for p in flat:
        try:
            v = int(p)
        except ValueError as e:
            raise SystemExit(f"Invalid {opt_name} entry {p!r}: must be integer") from e
        if v <= 0:
            raise SystemExit(f"Invalid {opt_name} entry {p!r}: must be > 0")
        out.append(v)
    return out


def _broadcast_list(name: str, values: Sequence, n: int) -> List:
    if len(values) == n:
        return list(values)
    if len(values) == 1 and n > 1:
        return [values[0]] * n
    raise SystemExit(f"Length mismatch for {name}: got {len(values)}, expected 1 or {n}")


def _read_freq_gp(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise SystemExit(f"Unexpected *.freq.gp table: shape={data.shape}")
    x = np.asarray(data[:, 0], dtype=float)
    y = np.asarray(data[:, 1:], dtype=float)  # (nq, nbranch)
    return x, y


def _convert_freq_units(freq_cm1: np.ndarray, unit: str) -> np.ndarray:
    freq = np.asarray(freq_cm1, dtype=float)
    if unit == "cm^-1":
        return freq
    return freq / _CM1_PER_THz


def _read_matdyn_in_qpoints(path: str) -> List[KPointSpec]:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()

    end_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*/\s*$", line):
            end_idx = i
            break
    if end_idx is None:
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
    x2 = np.asarray(x, dtype=float).copy()
    if len(indices) != len(specs):
        return x2

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


def _map_x_to_reference(
    x: np.ndarray,
    indices: Sequence[int],
    x_ref: np.ndarray,
    indices_ref: Sequence[int],
) -> np.ndarray:
    """Map x to the reference axis by piecewise linear mapping between high-symmetry points.

    This enables overlaying multiple datasets even if their internal x coordinates differ,
    as long as they share the same *sequence* of high-symmetry points.
    """

    x = np.asarray(x, dtype=float)
    x_ref = np.asarray(x_ref, dtype=float)
    if len(indices) != len(indices_ref):
        raise SystemExit(
            "Cannot overlay multiple datasets: number of high-symmetry points differs "
            f"({len(indices)} vs {len(indices_ref)})."
        )

    x_m = np.empty_like(x, dtype=float)
    x_m.fill(np.nan)

    for i in range(len(indices) - 1):
        s = int(indices[i])
        t = int(indices[i + 1])
        sr = int(indices_ref[i])
        tr = int(indices_ref[i + 1])
        if s < 0 or t < 0 or sr < 0 or tr < 0:
            continue
        if t <= s or tr <= sr:
            continue
        if s >= len(x) or t >= len(x) or sr >= len(x_ref) or tr >= len(x_ref):
            continue

        denom = float(x[t] - x[s])
        if abs(denom) < 1e-14:
            x_m[s : t + 1] = float(x_ref[sr])
            continue

        frac = (x[s : t + 1] - float(x[s])) / denom
        x_m[s : t + 1] = float(x_ref[sr]) + frac * (float(x_ref[tr]) - float(x_ref[sr]))

    # Fill any leftover NaNs by nearest valid value (should be rare)
    if np.any(~np.isfinite(x_m)):
        good = np.isfinite(x_m)
        if not np.any(good):
            raise SystemExit("Failed to map x-axis for overlay: no finite mapped points")
        idx = np.where(good)[0]
        first = int(idx[0])
        last = int(idx[-1])
        x_m[:first] = x_m[first]
        x_m[last + 1 :] = x_m[last]
        # linear interpolate in between
        bad = ~good
        if np.any(bad):
            x_m[bad] = np.interp(np.where(bad)[0], np.where(good)[0], x_m[good])

    return x_m


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


def _read_phonon_dos_table(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise SystemExit(f"Unexpected DOS table in {path!r}: shape={data.shape}")

    freq_cm1 = np.asarray(data[:, 0], dtype=float)
    dos_tot = np.asarray(data[:, 1], dtype=float)
    pdos = np.asarray(data[:, 2:], dtype=float) if data.shape[1] > 2 else np.zeros((len(freq_cm1), 0), dtype=float)
    return freq_cm1, dos_tot, pdos


def _read_nat_from_scf_in(path: str) -> Optional[int]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"\bnat\s*=\s*([0-9]+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    nat = int(m.group(1))
    if nat <= 0:
        return None
    return nat


def _read_atoms_from_scf_in(path: str, *, expected_nat: Optional[int] = None) -> List[AtomSpec]:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()

    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*ATOMIC_POSITIONS\b", line, flags=re.IGNORECASE):
            start = i + 1
            break
    if start is None:
        raise SystemExit(f"Cannot find ATOMIC_POSITIONS in {path}")

    atoms: List[AtomSpec] = []
    idx = 1
    for j in range(start, len(lines)):
        line = lines[j].strip()
        if not line:
            continue
        if re.match(r"^(K_POINTS|CELL_PARAMETERS|ATOMIC_SPECIES|CONSTRAINTS|OCCUPATIONS)\b", line, flags=re.IGNORECASE):
            break
        parts = line.split()
        if len(parts) < 4:
            continue
        el = parts[0]
        el2 = re.sub(r"[^A-Za-z]", "", el)
        if not el2:
            continue
        atoms.append(AtomSpec(index=idx, element=el2))
        idx += 1

        if expected_nat is not None and len(atoms) >= expected_nat:
            break

    if not atoms:
        raise SystemExit(f"No atoms parsed from ATOMIC_POSITIONS in {path}")

    if expected_nat is not None and len(atoms) != expected_nat:
        raise SystemExit(
            f"Parsed {len(atoms)} atoms from ATOMIC_POSITIONS, but nat={expected_nat} in {path}. "
            "Please check the scf.in format."
        )

    return atoms


def _parse_atom_selection(s: Optional[str], n_atoms: int) -> List[int]:
    if not s:
        return list(range(1, n_atoms + 1))

    out: List[int] = []
    parts = [x.strip() for x in s.split(",") if x.strip()]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            ia = int(a)
            ib = int(b)
            if ia <= 0 or ib <= 0:
                raise SystemExit(f"Invalid --atoms range: {part!r}")
            if ia > ib:
                ia, ib = ib, ia
            out.extend(range(ia, ib + 1))
        else:
            out.append(int(part))

    out2: List[int] = []
    for i in out:
        if i < 1 or i > n_atoms:
            raise SystemExit(f"Atom index out of range in --atoms: {i} (1..{n_atoms})")
        if i not in out2:
            out2.append(i)
    return out2


def _convert_dos_unit_for_x(
    dos: np.ndarray,
    pdos: np.ndarray,
    *,
    x_unit: str,
    disable_jacobian: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if x_unit != "THz" or disable_jacobian:
        return dos, pdos
    return dos * _CM1_PER_THz, pdos * _CM1_PER_THz


def main() -> None:
    args = _build_parser().parse_args()

    if args.bold_fonts and args.no_bold:
        raise SystemExit("Do not use --bold-fonts together with --no-bold")

    ylim = _parse_lim(args.ylim)
    dos_xlim = _parse_lim(args.dos_xlim)
    figsize = _parse_figsize(args.figsize)
    figsize_bands = _parse_figsize(args.figsize_bands)
    figsize_dos = _parse_figsize(args.figsize_dos)
    ratios = _parse_ratios(args.ratios)
    system_bbox = _parse_xy(args.system_bbox)
    legend_bbox = _parse_xy(args.legend_bbox)
    dos_legend_bbox = _parse_xy(args.dos_legend_bbox)

    if args.lw is not None and float(args.lw) <= 0:
        raise SystemExit("--lw must be > 0")

    if args.style == "prb":
        _apply_scienceplots_prb_style()

    # Apply global defaults after selecting the base style.
    _apply_global_fontsize(args.fontsize)

    want_bold_fonts = bool(args.bold_fonts) or ((args.style == "default") and (not args.no_bold))
    if want_bold_fonts:
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        if args.style == "default":
            plt.rcParams["axes.linewidth"] = 2

    # --- Multi-dataset setup ---
    freq_paths = list(args.freq)
    matdyn_paths = list(args.matdyn_in)
    dos_paths = list(args.dos)
    scf_paths = list(args.scf_in)
    kpath_paths = list(args.kpath) if args.kpath is not None else []

    norms_in = _parse_int_list(args.dos_norm, opt_name="--dos-norm")
    # Dataset labels: prefer --legend; otherwise use filename stem.
    legends_in = _flatten_tokens_allow_blank(args.legend) if (args.legend is not None) else None

    # Global system annotation text (independent of dataset legend)
    system_text: Optional[str] = None
    if args.system is not None:
        system_text = " ".join(str(x) for x in args.system).strip()
        if not system_text:
            system_text = None

    n_cases = max(
        len(freq_paths),
        len(matdyn_paths),
        len(dos_paths),
        len(scf_paths),
        len(kpath_paths) if kpath_paths else 1,
    )

    freq_paths = _broadcast_list("--freq", freq_paths, n_cases)
    matdyn_paths = _broadcast_list("--matdyn-in", matdyn_paths, n_cases)
    dos_paths = _broadcast_list("--dos", dos_paths, n_cases)
    scf_paths = _broadcast_list("--scf-in", scf_paths, n_cases)
    if kpath_paths:
        kpath_paths = _broadcast_list("--kpath", kpath_paths, n_cases)
    else:
        kpath_paths = [None] * n_cases

    norms = _broadcast_list("--dos-norm", norms_in or [1], n_cases)

    if args.legend is not None:
        dataset_labels_raw = _broadcast_list("--legend", legends_in or [""], n_cases)
    else:
        dataset_labels_raw = [Path(p).stem for p in dos_paths]

    dataset_labels = list(dataset_labels_raw)

    # --- Figure layout ---
    if (figsize_bands is None) ^ (figsize_dos is None):
        raise SystemExit("--figsize-bands and --figsize-dos must be provided together")

    if figsize_bands is not None and figsize_dos is not None:
        wb, hb = figsize_bands
        wd, hd = figsize_dos
        if abs(hb - hd) > 1e-8:
            raise SystemExit(
                f"--figsize-bands height ({hb}) must equal --figsize-dos height ({hd}) so the two panels can share the y-axis"
            )
        fig = plt.figure(figsize=(wb + wd, hb))
        width_ratios = [wb, wd]
    else:
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()
        width_ratios = [ratios[0], ratios[1]]

    gs = fig.add_gridspec(1, 2, width_ratios=width_ratios, wspace=0.05)
    ax_band = fig.add_subplot(gs[0, 0])
    ax_dos = fig.add_subplot(gs[0, 1], sharey=ax_band)

    lw = float(args.lw) if args.lw is not None else (0.8 if args.style == "prb" else 1.4)

    # --- Load & plot each dataset ---
    ref_xticks: Optional[List[float]] = None
    ref_xticklabels: Optional[List[str]] = None
    ref_xlim: Optional[Tuple[float, float]] = None
    ref_x_plot: Optional[np.ndarray] = None
    ref_indices: Optional[List[int]] = None
    ref_scheme: Optional[str] = None
    ref_n_data: Optional[int] = None

    # High-contrast dataset colors (bands + total DOS): black, red, blue, ...
    # These are intentionally easy to tell apart in print.
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

    # PDOS is plotted for both single and multi-dataset mode.
    # In multi-dataset mode, element/atom colors are shared while datasets differ by linestyle.
    plot_pdos = True
    if n_cases > 1:
        print(f"Detected {n_cases} datasets. Total DOS and PDOS are plotted on the right panel.")

    dataset_linestyles = ["-", "--", ":", "-."]

    # Data containers for DOS panel
    dos_curves: List[Tuple[np.ndarray, np.ndarray, int, str]] = []  # (dos, freq, dataset_index, label)
    pdos_cases: List[Tuple[np.ndarray, Dict[str, np.ndarray], int, str]] = []  # (freq, {base_label:pdos}, dataset_index, ds_label)

    for ic in range(n_cases):
        freq_path = str(freq_paths[ic])
        matdyn_path = str(matdyn_paths[ic])
        dos_path = str(dos_paths[ic])
        scf_path = str(scf_paths[ic])
        kpath_path = kpath_paths[ic]
        norm = int(norms[ic])

        # --- Bands data ---
        x_path, ymat_cm1 = _read_freq_gp(freq_path)
        ymat = _convert_freq_units(ymat_cm1, args.unit)
        specs = _read_matdyn_in_qpoints(matdyn_path)

        label_entries: Optional[List[Tuple[Tuple[float, float, float], str]]] = None
        if kpath_path:
            label_entries = _read_kpath_labels(str(kpath_path))

        hs_labels: List[str] = []
        for i, sp in enumerate(specs):
            lab = None
            if label_entries is not None:
                lab = _find_label_for_q(sp.q, label_entries)
            if lab is None:
                lab = f"Q{i+1}"
            hs_labels.append(lab)

        indices, scheme = _infer_indices(specs, n_data=len(x_path))
        segments = _build_segments(specs, indices)

        if args.keep_jumps:
            x_plot = x_path
        else:
            x_plot = _compress_x_jumps_by_specs(x_path, specs, indices)

        xticks, xticklabels = _build_ticks_and_labels(x_plot, specs, indices, hs_labels)

        if ref_xticks is None:
            ref_xticks = xticks
            ref_xticklabels = xticklabels
            ref_xlim = (float(x_plot[0]), float(x_plot[-1]))
            ref_x_plot = np.asarray(x_plot, dtype=float)
            ref_indices = list(indices)
            ref_scheme = scheme
            ref_n_data = len(x_path)
        else:
            if len(xticklabels) != len(ref_xticklabels):
                raise SystemExit("Cannot overlay multiple datasets: high-symmetry tick count differs.")
            if any(a != b for a, b in zip(xticklabels, ref_xticklabels)):
                raise SystemExit("Cannot overlay multiple datasets: high-symmetry labels differ.")

        # Map this dataset's x to the reference axis (so different structures can be compared).
        if ref_x_plot is None or ref_indices is None:
            x_mapped = np.asarray(x_plot, dtype=float)
        else:
            x_mapped = _map_x_to_reference(x_plot, indices, ref_x_plot, ref_indices)

        col = case_colors[ic % len(case_colors)]
        alpha = 1.0 if ic == 0 else 0.65

        n_branch = int(ymat.shape[1])
        for j in range(n_branch):
            f = ymat[:, j]
            for (s, t) in segments:
                ax_band.plot(x_mapped[s : t + 1], f[s : t + 1], color=col, lw=lw, alpha=alpha)

        # --- DOS/PDOS data ---
        freq_dos_cm1, dos_tot, pdos_mat = _read_phonon_dos_table(dos_path)
        nat_from_scf = _read_nat_from_scf_in(scf_path)
        atoms = _read_atoms_from_scf_in(scf_path, expected_nat=nat_from_scf)

        n_atoms = nat_from_scf if nat_from_scf is not None else len(atoms)
        if pdos_mat.shape[1] != 0 and pdos_mat.shape[1] != n_atoms:
            raise SystemExit(
                f"PDOS column count mismatch for {dos_path!r}: DOS file has {pdos_mat.shape[1]} per-atom columns, "
                f"but scf.in has {n_atoms} atoms ({scf_path!r})."
            )

        freq_dos = _convert_freq_units(freq_dos_cm1, args.unit)
        dos_tot, pdos_mat = _convert_dos_unit_for_x(
            dos_tot,
            pdos_mat,
            x_unit=str(args.unit),
            disable_jacobian=bool(args.no_jacobian),
        )

        # Per-dataset normalization (e.g. supercell -> per unit cell)
        if norm != 1:
            dos_tot = dos_tot / float(norm)
            pdos_mat = pdos_mat / float(norm)

        # Dataset legend label (formatted)
        fmt_mode = str(args.legend_format)
        ds_lab = _format_system_label(str(dataset_labels[ic]).strip(), fmt_mode)
        # Total DOS label used for DOS legend.
        ds_prefix = ds_lab.strip() if ds_lab.strip() else (f"D{ic+1}" if n_cases > 1 else "")
        tot_lab = f"{ds_prefix}:Total" if ds_prefix else "Total"
        dos_curves.append((dos_tot, freq_dos, ic, tot_lab))

        if plot_pdos:
            elements_filter: Optional[set[str]] = None
            if args.elements:
                elements_filter = {x.strip() for x in args.elements.split(",") if x.strip()}

            series_case: Dict[str, np.ndarray] = {}
            if pdos_mat.shape[1] == 0:
                series_case = {}
            elif args.group == "atom":
                selected_atoms = _parse_atom_selection(args.atoms, n_atoms=n_atoms)
                for ia in selected_atoms:
                    el = atoms[ia - 1].element
                    if elements_filter is not None and el not in elements_filter:
                        continue
                    base_lab = f"{el}{ia}"
                    series_case[base_lab] = pdos_mat[:, ia - 1]
            else:
                by_el: Dict[str, np.ndarray] = {}
                for ia, atom in enumerate(atoms, start=1):
                    el = atom.element
                    if elements_filter is not None and el not in elements_filter:
                        continue
                    if el not in by_el:
                        by_el[el] = np.zeros_like(dos_tot, dtype=float)
                    by_el[el] += pdos_mat[:, ia - 1]
                for el in sorted(by_el.keys()):
                    series_case[el] = by_el[el]

            pdos_cases.append((freq_dos, series_case, ic, ds_lab))

    if ref_xticks is None or ref_xticklabels is None or ref_xlim is None:
        raise SystemExit("No dataset loaded")

    for xpos in ref_xticks:
        ax_band.axvline(xpos, color="black", lw=0.6, alpha=0.6)

    ax_band.set_xticks(ref_xticks)
    ax_band.set_xticklabels(ref_xticklabels)
    # Keep only high-symmetry vertical lines on x-axis; hide tick marks (but keep labels).
    ax_band.tick_params(axis="x", which="both", bottom=False, top=False, length=0)

    ax_band.set_xlim(*ref_xlim)
    if ylim:
        ax_band.set_ylim(*ylim)

    if args.ytick_step is not None:
        step = float(args.ytick_step)
        if step <= 0:
            raise SystemExit("--ytick-step must be > 0")
        ax_band.yaxis.set_major_locator(MultipleLocator(step))

    ylab = "Frequency (THz)" if args.unit == "THz" else "Frequency (cm^-1)"
    ax_band.set_ylabel(ylab)

    if args.label_fontsize is not None:
        fs = float(args.label_fontsize)
        ax_band.xaxis.label.set_size(fs)
        ax_band.yaxis.label.set_size(fs)
        ax_band.tick_params(axis="both", which="both", labelsize=fs)
        ax_dos.tick_params(axis="both", which="both", labelsize=fs)
        try:
            ax_band.xaxis.get_offset_text().set_size(fs)
            ax_band.yaxis.get_offset_text().set_size(fs)
            ax_dos.xaxis.get_offset_text().set_size(fs)
            ax_dos.yaxis.get_offset_text().set_size(fs)
        except Exception:
            pass

    # Dataset legend (left panel). Labels come from --legend (or filename stem). Skip blank labels.
    leg_ds = None
    if any(str(s).strip() for s in dataset_labels):
        handles_ds: List[Line2D] = []
        for ic in range(n_cases):
            raw = str(dataset_labels[ic]).strip()
            if not raw:
                continue
            ds_lab = _format_system_label(raw, str(args.legend_format))
            col = case_colors[ic % len(case_colors)]
            handles_ds.append(Line2D([0], [0], color=col, lw=lw, label=ds_lab))

        fs_ds = args.legend_fontsize
        if fs_ds is None:
            try:
                fs_ds = float(ax_band.yaxis.label.get_size()) * 1.15
            except Exception:
                fs_ds = None

        if handles_ds:
            if legend_bbox is None:
                leg_ds = ax_band.legend(
                    handles=handles_ds,
                    loc=str(args.legend_loc),
                    frameon=bool(args.legend_alpha is not None),
                    fontsize=fs_ds,
                )
            else:
                leg_ds = ax_band.legend(
                    handles=handles_ds,
                    loc=str(args.legend_loc),
                    bbox_to_anchor=legend_bbox,
                    bbox_transform=ax_band.transAxes,
                    frameon=bool(args.legend_alpha is not None),
                    fontsize=fs_ds,
                )
            if args.legend_alpha is not None:
                _apply_legend_frame(leg_ds, alpha=float(args.legend_alpha))
            if want_bold_fonts and leg_ds is not None:
                for t in leg_ds.get_texts():
                    t.set_fontweight("bold")

    # Global system annotation (pure text). Independent of dataset legend.
    if system_text is not None:
        sys_lab = _format_system_label(system_text, str(args.system_format))

        fs_sys = args.system_fontsize
        if fs_sys is None:
            try:
                fs_sys = float(ax_band.yaxis.label.get_size()) * 1.15
            except Exception:
                fs_sys = None

        if leg_ds is not None:
            ax_band.add_artist(leg_ds)

        h = Line2D([0], [0], color="none", lw=0.0, label=sys_lab)
        sys_kwargs = {
            "frameon": bool(args.system_alpha is not None),
            "fontsize": fs_sys,
            "handlelength": 0,
            "handletextpad": 0,
        }
        if system_bbox is None:
            leg_sys = ax_band.legend(
                handles=[h],
                loc=str(args.system_loc),
                **sys_kwargs,
            )
        else:
            leg_sys = ax_band.legend(
                handles=[h],
                loc=str(args.system_loc),
                bbox_to_anchor=system_bbox,
                bbox_transform=ax_band.transAxes,
                **sys_kwargs,
            )
        if args.system_alpha is not None:
            _apply_legend_frame(leg_sys, alpha=float(args.system_alpha))
        if leg_sys is not None:
            for t in leg_sys.get_texts():
                if want_bold_fonts:
                    t.set_fontweight("bold")
                try:
                    t.set_ha("center")
                except Exception:
                    pass
            try:
                leg_sys._legend_box.align = "center"  # noqa: SLF001
            except Exception:
                pass

    # --- Plot rotated DOS/PDOS (x = DOS, y = Frequency) ---
    # Colors: total DOS = black; PDOS cycles red/green/blue.
    total_dos_color = "black"
    pdos_colors = ["tab:red", "tab:green", "tab:blue"]

    # Total DOS curves for each dataset
    for dos_tot, freq_dos, ic, lab in dos_curves:
        ls_i = dataset_linestyles[ic % len(dataset_linestyles)] if n_cases > 1 else "-"
        ax_dos.plot(dos_tot, freq_dos, color=total_dos_color, lw=lw, linestyle=ls_i, label=lab)

    # PDOS: keep colors tied to base label; datasets differ by linestyle.
    all_base_labels: List[str] = []
    for _freq, series_case, _ic, _dslab in pdos_cases:
        for k in series_case.keys():
            if k not in all_base_labels:
                all_base_labels.append(k)
    all_base_labels = sorted(all_base_labels)
    base_color = {lab: pdos_colors[i % len(pdos_colors)] for i, lab in enumerate(all_base_labels)}

    if plot_pdos:
        for freq_dos, series_case, ic, sys_lab in pdos_cases:
            if not series_case:
                continue
            ls_i = dataset_linestyles[ic % len(dataset_linestyles)] if n_cases > 1 else "-"
            ds_prefix = sys_lab.strip() if sys_lab.strip() else (f"D{ic+1}" if n_cases > 1 else "")
            for base_lab in sorted(series_case.keys()):
                col = base_color.get(base_lab, pdos_colors[0])
                lab = f"{ds_prefix}:{base_lab}" if ds_prefix else str(base_lab)
                ax_dos.plot(series_case[base_lab], freq_dos, lw=lw, color=col, linestyle=ls_i, label=lab)

    if dos_xlim:
        ax_dos.set_xlim(*dos_xlim)
    else:
        y0, y1 = ax_dos.get_ylim()
        if y0 > y1:
            y0, y1 = y1, y0
        xmax = 0.0
        for dos_tot, freq_arr, _col, _lab in dos_curves:
            if len(dos_tot) and len(freq_arr) == len(dos_tot):
                m = (freq_arr >= y0) & (freq_arr <= y1)
                if np.any(m):
                    xmax = max(xmax, float(np.nanmax(dos_tot[m])))
        if plot_pdos:
            for freq_arr, series_case, _ic, _sys_lab in pdos_cases:
                if not series_case:
                    continue
                m = (freq_arr >= y0) & (freq_arr <= y1)
                if not np.any(m):
                    continue
                for base_lab, arr in series_case.items():
                    try:
                        xmax = max(xmax, float(np.nanmax(arr[m])))
                    except Exception:
                        continue
        ax_dos.set_xlim(0.0, xmax * 1.05 if xmax > 0 else 1.0)

    if args.xtick_step is not None:
        step = float(args.xtick_step)
        if step <= 0:
            raise SystemExit("--xtick-step must be > 0")
        ax_dos.xaxis.set_major_locator(MultipleLocator(step))

    # Hide duplicate y tick labels on the right
    ax_dos.tick_params(axis="y", which="both", left=False, labelleft=False)

    # Keep DOS panel compact but show x-axis tick values
    if args.unit == "THz":
        ax_dos.set_xlabel("(states/THz/unit cell)")
    else:
        ax_dos.set_xlabel(r"(states/cm$^{-1}$/unit cell)")

    if args.label_fontsize is not None:
        try:
            ax_dos.xaxis.label.set_fontsize(float(args.label_fontsize) * 0.85)
        except Exception:
            pass
        ax_dos.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            labelbottom=True,
            labelsize=float(args.label_fontsize),
        )
    else:
        try:
            ax_dos.xaxis.label.set_fontsize(ax_band.xaxis.label.get_size() * 0.85)
        except Exception:
            pass
        ax_dos.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)

    leg = None
    handles_d, labels_d = ax_dos.get_legend_handles_labels()
    if handles_d and labels_d:
        dos_loc = str(args.dos_legend_loc)
        if (dos_legend_bbox is not None) and (dos_loc.strip().lower() == "best"):
            # When bbox_to_anchor is provided, loc='best' tends to ignore the anchor and
            # auto-place the legend. Use a deterministic anchor-based loc instead.
            dos_loc = "center left"
        dos_fs = args.dos_legend_fontsize
        dos_frame = bool(args.dos_legend_alpha is not None)
        if dos_legend_bbox is None:
            leg = ax_dos.legend(
                loc=dos_loc,
                frameon=dos_frame,
                borderaxespad=0.0,
                fontsize=(float(dos_fs) if dos_fs is not None else None),
            )
        else:
            leg = ax_dos.legend(
                loc=dos_loc,
                bbox_to_anchor=dos_legend_bbox,
                bbox_transform=ax_dos.transAxes,
                frameon=dos_frame,
                borderaxespad=0.0,
                fontsize=(float(dos_fs) if dos_fs is not None else None),
            )

        if args.dos_legend_alpha is not None:
            _apply_legend_frame(leg, alpha=float(args.dos_legend_alpha))

    if args.style == "default":
        ax_band.grid(True, alpha=0.25)
        ax_dos.grid(True, alpha=0.25)
        if not args.no_bold:
            _apply_bold(ax_band, bold=True)
            _apply_bold(ax_dos, bold=True)

    # Make legend bold if requested
    if want_bold_fonts and leg is not None:
        for t in leg.get_texts():
            t.set_fontweight("bold")

    if want_bold_fonts:
        _set_figure_text_weight(fig, "bold")

    fig.tight_layout()
    _fix_dense_xticklabels(fig, ax_band)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)

    print(f"Saved: {args.out}")
    if ref_scheme is not None:
        ndata = ref_n_data if ref_n_data is not None else "?"
        print(f"Q-point indexing convention: {ref_scheme} (data points per branch: {ndata})")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
