#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MultipleLocator

# Allow importing the Perturbo helpers from the sibling folder.
_THIS_DIR = Path(__file__).resolve().parent
_PERTURBO_DIR = _THIS_DIR.parent / "perturbo_plot"
if str(_PERTURBO_DIR) not in sys.path:
    sys.path.insert(0, str(_PERTURBO_DIR))

try:
    from perturbo_meanfp_io import (
        get_energy_by_band,
        get_mu_ev,
        get_velocity_by_band,
        load_meanfp_yaml,
        parse_band_selection,
    )
except Exception:
    # Keep import-time failure non-fatal for phonon-only use.
    get_energy_by_band = None  # type: ignore[assignment]
    get_mu_ev = None  # type: ignore[assignment]
    get_velocity_by_band = None  # type: ignore[assignment]
    load_meanfp_yaml = None  # type: ignore[assignment]
    parse_band_selection = None  # type: ignore[assignment]


def _apply_scienceplots_prb_style() -> None:
    """Apply a PRB-like plotting style via SciencePlots."""

    try:
        import scienceplots  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "SciencePlots is required for --style prb but could not be imported.\n"
            "Install it with: pip install SciencePlots\n"
            f"Original error: {e}"
        )

    # Best-effort: 'prb' is not guaranteed to exist in all SciencePlots versions.
    try:
        plt.style.use(["science", "no-latex", "prb"])
    except Exception:
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


def _set_figure_text_weight(fig, weight: str) -> None:
    for t in fig.findobj(Text):
        try:
            t.set_fontweight(weight)
        except Exception:
            pass


def _apply_legend_frame(leg, *, alpha: Optional[float]) -> None:
    if leg is None or alpha is None:
        return
    a = float(alpha)
    if not (0.0 <= a <= 1.0):
        raise SystemExit("legend alpha must be in [0, 1]")
    try:
        leg.set_frame_on(True)
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_alpha(a)
        frame.set_edgecolor("0.7")
    except Exception:
        pass


def _parse_lim(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = str(s).split(",", 1)
    return float(a), float(b)


def _parse_figsize(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = str(s).split(",", 1)
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


def _format_label(label: str, mode: str) -> str:
    if not label:
        return label
    if (mode or "raw").lower() == "raw":
        return label
    if "$" in label:
        return label

    import re

    return re.sub(r"(?<=[A-Za-z\)])(\d+)", r"$_{\1}$", label)


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
    if n <= 0:
        return []
    if len(xs) == n:
        return list(xs)
    if len(xs) == 1:
        return [str(xs[0])] * n
    raise SystemExit(f"{name} expects 1 value or {n} values, but got {len(xs)}")


def _broadcast_float_list(xs: Sequence[float], n: int, name: str) -> List[float]:
    if n <= 0:
        return []
    if len(xs) == n:
        return [float(x) for x in xs]
    if len(xs) == 1:
        return [float(xs[0])] * n
    raise SystemExit(f"{name} expects 1 value or {n} values, but got {len(xs)}")


def _pad_legend_columns_preserve_order(
    handles: Sequence[object],
    labels: Sequence[str],
    *,
    ncol: int,
    column_real_counts: Optional[Sequence[int]] = None,
) -> Tuple[List[object], List[str]]:
    """Pad legend entries so right columns can have more rows, without reordering."""

    if len(handles) != len(labels):
        raise ValueError("Legend handles/labels length mismatch")

    k = int(ncol)
    if k <= 1:
        return list(handles), list(labels)

    n_items = len(handles)
    if n_items == 0:
        return [], []

    if column_real_counts is None:
        base = n_items // k
        rem = n_items % k
        counts = [base] * k
        for j in range(k - rem, k):
            if 0 <= j < k:
                counts[j] += 1
    else:
        counts = [int(x) for x in column_real_counts]
        if len(counts) != k:
            raise SystemExit(
                f"--legend-column-counts expects {k} integers (same as --legend-ncol), but got {len(counts)}"
            )
        if any(c < 0 for c in counts):
            raise SystemExit("--legend-column-counts values must be >= 0")
        if sum(counts) < n_items:
            counts[-1] += (n_items - sum(counts))

    rows = max(counts) if counts else 0
    if rows <= 0:
        return list(handles), list(labels)

    blank_handle = Line2D([0], [0], color="none", lw=0)
    blank_label = " "

    out_h: List[object] = []
    out_l: List[str] = []
    idx = 0
    for c in counts:
        take = min(int(c), n_items - idx)
        for _ in range(take):
            out_h.append(handles[idx])
            out_l.append(labels[idx])
            idx += 1
        for _ in range(rows - int(c)):
            out_h.append(blank_handle)
            out_l.append(blank_label)

    if idx != n_items:
        for j in range(idx, n_items):
            out_h.append(handles[j])
            out_l.append(labels[j])

    return out_h, out_l


def _detect_source(path: str) -> str:
    p = str(path).lower()
    if p.endswith(".yml") or p.endswith(".yaml"):
        return "perturbo"
    return "shengbte"


def _load_shengbte_v_and_x(
    v_path: str,
    *,
    xmode: str,
    omega_path: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, str]:
    v = np.loadtxt(v_path)
    if v.ndim != 2 or v.shape[1] < 3:
        raise SystemExit(f"Unexpected format in {v_path!r}: expected N x 3 table")
    vmag = np.linalg.norm(v[:, :3], axis=1)

    if xmode == "index":
        x = np.arange(vmag.size, dtype=float)
        return x, vmag, "Mode index"

    # Frequency from BTE.omega: omega [rad/ps] -> THz by dividing 2*pi.
    if omega_path is None:
        omega_path = str(Path(v_path).with_name("BTE.omega"))
    if not os.path.exists(omega_path):
        raise SystemExit(
            f"Cannot find omega file for {v_path!r}. Tried: {omega_path!r}. "
            "Provide --omega explicitly or use --x index."
        )

    omega = np.loadtxt(omega_path)
    omega_flat = np.asarray(omega, dtype=float).reshape(-1)
    x = omega_flat / (2.0 * np.pi)

    if x.size != vmag.size:
        n = int(min(x.size, vmag.size))
        print(
            f"[warn] length mismatch: omega={x.size} vs v={vmag.size}; truncating to {n}"
        )
        x = x[:n]
        vmag = vmag[:n]

    return x, vmag, "Frequency (THz)"


def _load_perturbo_meanfp_v_and_x(
    yml_path: str,
    *,
    xmode: str,
    bands: Optional[str],
    config: int,
    ef_override: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load electron |v| and x from Perturbo *_meanfp.yml.

    - xmode='energy': returns E-E_F (eV)
    - xmode='index': returns state index
    """

    if load_meanfp_yaml is None:
        raise SystemExit(
            "Perturbo helpers could not be imported (perturbo_meanfp_io.py). "
            "Run from the repo root, or keep scripts under workplace/science_plot_script/ in PYTHONPATH."
        )

    data = load_meanfp_yaml(str(yml_path))
    e_by_band = get_energy_by_band(data)  # type: ignore[misc]
    v_by_band = get_velocity_by_band(data)  # type: ignore[misc]

    selected_bands = parse_band_selection(bands, sorted(e_by_band.keys()))  # type: ignore[misc]

    mu = 0.0
    if xmode == "energy":
        if ef_override is not None:
            mu = float(ef_override)
        else:
            mu = float(get_mu_ev(data, int(config)))  # type: ignore[misc]

    xs: List[float] = []
    vs: List[float] = []
    for b in selected_bands:
        e = e_by_band[b]
        v = v_by_band[b]
        if len(e) != len(v):
            raise SystemExit(f"Length mismatch in {yml_path} band {b}: E={len(e)} v={len(v)}")
        if xmode == "index":
            xs.extend(range(len(e)))
        else:
            xs.extend([float(x) - mu for x in e])
        vs.extend([float(x) for x in v])

    if not vs:
        raise SystemExit(f"No band velocity values found in: {yml_path}")

    vmag = np.asarray(vs, dtype=float)
    if xmode == "index":
        x = np.arange(vmag.size, dtype=float)
        return x, vmag, "State index"

    x = np.asarray(xs, dtype=float)
    if x.size != vmag.size:
        n = int(min(x.size, vmag.size))
        x = x[:n]
        vmag = vmag[:n]
    return x, vmag, r"Energy $E-E_F$ (eV)"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot grain-boundary scattering rate derived from group velocity.\n\n"
            "Definition used here: τ = |v| / L (as requested).\n"
            "Units assumed by this script:\n"
            "- ShengBTE: BTE.v is in km/s (numerically equal to nm/ps).\n"
            "- Perturbo: meanfp 'band velocity' is in m/s (converted internally to nm/ps via ×1e-3).\n"
            "- Grain-boundary length L is in nm.\n"
            "Therefore τ = |v|/L is in ps^-1.\n\n"
            "- For ShengBTE inputs: read |v| from BTE.v and x from BTE.omega (THz).\n"
            "- For Perturbo inputs: read |v| and energy from *_meanfp.yml.\n\n"
            "Multiple input files can be provided to overlay and compare datasets."
        )
    )

    p.add_argument(
        "files",
        nargs="+",
        help=(
            "Input file(s): ShengBTE BTE.v (or BTE.v_full) and/or Perturbo *_meanfp.yml. "
            "For ShengBTE, BTE.omega is auto-detected in the same folder unless --omega is given."
        ),
    )

    p.add_argument(
        "--L",
        required=True,
        nargs="+",
        type=float,
        help=(
            "Grain-boundary length L. Provide one per input file, or a single value to broadcast. "
            "This script assumes L is in nm. With ShengBTE v in km/s and Perturbo v in m/s, the plotted τ is in ps^-1."
        ),
    )

    p.add_argument(
        "--omega",
        default=None,
        nargs="+",
        help=(
            "Optional BTE.omega path(s) for ShengBTE inputs. Provide one per input file, or one value to broadcast. "
            "Ignored for Perturbo YAML inputs."
        ),
    )

    # New: separate x-axis choices for phonons vs electrons (for dual-x plots).
    p.add_argument(
        "--x-phonon",
        choices=["freq", "index"],
        default="freq",
        help="ShengBTE x-axis: freq (THz) or index [default: freq].",
    )
    p.add_argument(
        "--x-electron",
        choices=["energy", "index"],
        default="energy",
        help="Perturbo x-axis: energy (eV) or index [default: energy].",
    )

    # Backward-compatible legacy flag: applies to both.
    p.add_argument(
        "--x",
        choices=["auto", "freq", "energy", "index"],
        default="auto",
        help=(
            "[deprecated] Global x-axis choice. Prefer --x-phonon/--x-electron. "
            "auto: ShengBTE->freq, Perturbo->energy."
        ),
    )

    p.add_argument(
        "--ef",
        default=None,
        nargs="+",
        type=float,
        help=(
            "Fermi level(s) E_F in eV for Perturbo inputs (used to plot E-E_F). "
            "Provide one per Perturbo file, or a single value to broadcast. "
            "If omitted, tries to read from YAML chemical potential; if that fails, assumes 0."
        ),
    )
    p.add_argument(
        "--config",
        type=int,
        default=1,
        help="Perturbo configuration index used to read mu (E_F) when plotting E-E_F [default: 1]",
    )
    p.add_argument(
        "--bands",
        default=None,
        help='Perturbo band indices in YAML space, e.g. "1-6" or "1,3,5" [default: all]',
    )

    # Legend: dataset labels
    p.add_argument(
        "--legend",
        default=None,
        nargs="+",
        help=(
            "Legend label(s) for each input file. Provide one per file, or a single value to broadcast. "
            "If omitted, uses the parent directory name (or filename)."
        ),
    )
    p.add_argument(
        "--legend-format",
        choices=["chem", "raw"],
        default="raw",
        help="Render --legend text with subscripts (chem) or raw text (raw) [default: raw].",
    )
    p.add_argument("--legend-fontsize", type=float, default=None, help="Legend font size")
    p.add_argument("--legend-ncol", type=int, default=1, help="Legend columns [default: 1]")
    p.add_argument("--legend-loc", default="best", help="Legend loc [default: best]")
    p.add_argument("--legend-bbox", default=None, help="Legend bbox anchor in axes coords 'x,y'")
    p.add_argument("--legend-alpha", type=float, default=None, help="Legend frame alpha (0..1)")
    p.add_argument("--legend-handlelength", type=float, default=1.6, help="Legend handle length")
    p.add_argument("--legend-handletextpad", type=float, default=0.35, help="Legend handle/text gap")
    p.add_argument("--legend-columnspacing", type=float, default=0.8, help="Legend column spacing")
    p.add_argument(
        "--legend-labelspacing",
        type=float,
        default=0.35,
        help="Legend row spacing (vertical) [default: 0.35]",
    )
    p.add_argument(
        "--legend-column-counts",
        default=None,
        nargs="+",
        type=int,
        help=(
            "Optional custom number of legend entries per column (left to right). "
            "Length must equal --legend-ncol. If sum is smaller than number of entries, remainder goes to last column."
        ),
    )

    # System annotation
    p.add_argument(
        "--system",
        default=None,
        nargs="+",
        help=(
            "Overall system/material label shown as a separate legend entry (e.g. 'Zr2SeC'). "
            "Normally give ONE value. If multiple are given and --legend is omitted, they will be treated as --legend."
        ),
    )
    p.add_argument(
        "--system-format",
        choices=["chem", "raw"],
        default="chem",
        help="Render --system as chemical formula with subscripts (chem) or raw text (raw) [default: chem].",
    )
    p.add_argument("--system-fontsize", type=float, default=None, help="System label font size")
    p.add_argument("--system-loc", default="upper left", help="System legend loc")
    p.add_argument("--system-bbox", default=None, help="System bbox anchor in axes coords 'x,y'")
    p.add_argument("--system-alpha", type=float, default=None, help="System frame alpha (0..1)")

    # Plot style
    p.add_argument("--style", choices=["prb", "default"], default="prb", help="Style preset")
    p.add_argument("--bold-fonts", action="store_true", help="Force bold fonts")
    p.add_argument("--fontsize", type=float, default=None, help="Global font size (rcParams)")
    p.add_argument(
        "--label-fontsize",
        type=float,
        default=None,
        help="Axis label & tick label font size override",
    )

    # Axes and scatter controls
    p.add_argument("--xlog", action="store_true", help="Log x-axis")
    p.add_argument("--ylog", action="store_true", default=True, help="Log y-axis [default: on]")
    p.add_argument("--no-ylog", dest="ylog", action="store_false", help="Disable log y-axis")

    p.add_argument("--ms", type=float, default=10.0, help="Marker size")
    p.add_argument("--alpha", type=float, default=0.35, help="Marker alpha (0..1)")

    p.add_argument(
        "--xlim",
        default=None,
        help=(
            "[legacy] x limits \"xmin,xmax\". If only one dataset type is plotted, applies to that x-axis. "
            "If both phonon+electron are plotted and --xlim-freq/--xlim-energy are omitted, applies to both axes."
        ),
    )
    p.add_argument("--xlim-freq", default=None, help='Phonon x limits (THz) "xmin,xmax"')
    p.add_argument("--xlim-energy", default=None, help='Electron x limits (eV, relative to E_F) "xmin,xmax"')
    p.add_argument("--ylim", default=None, help='y limits "ymin,ymax"')

    p.add_argument(
        "--xtick-step",
        type=float,
        default=None,
        help="[legacy] Major tick step on x-axis (applies to both x-axes if present)",
    )
    p.add_argument(
        "--xtick-step-freq",
        type=float,
        default=None,
        help="Major tick step for phonon x-axis (Frequency, THz)",
    )
    p.add_argument(
        "--xtick-step-energy",
        type=float,
        default=None,
        help="Major tick step for electron x-axis (Energy, eV)",
    )
    p.add_argument("--ytick-step", type=float, default=None, help="Major tick step on y-axis")

    p.add_argument("--xlabel", default=None, help="[legacy] x-axis label override (single-axis plots)")
    p.add_argument("--xlabel-freq", default="Frequency (THz)", help="Phonon x-axis label")
    p.add_argument("--xlabel-energy", default=r"Energy $E-E_F$ (eV)", help="Electron x-axis label")
    p.add_argument(
        "--ylabel",
        default=r"Grain-boundary scattering rate $\tau=|v|/L$ (ps$^{-1}$)",
        help="y-axis label",
    )

    p.add_argument("--figsize", default=None, help='Figure size "w,h" in inches')
    p.add_argument("--grid", action="store_true", help="Show grid")
    p.add_argument("--out", default="gb_scattering_rate.png", help="Output image path")
    p.add_argument("--show", action="store_true", help="Show interactively")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    default_dpi = 150.0

    ylabel_default = r"Grain-boundary scattering rate $\tau=|v|/L$ (ps$^{-1}$)"
    ylabel_default_bold = (
        r"Grain-boundary scattering rate $\mathbf{\tau}=|\mathbf{v}|/\mathbf{L}$ (ps$^{-1}$)"
    )

    if args.legend_ncol <= 0:
        raise SystemExit("--legend-ncol must be >= 1")
    if args.ms <= 0:
        raise SystemExit("--ms must be > 0")
    if not (0.0 <= float(args.alpha) <= 1.0):
        raise SystemExit("--alpha must be within [0, 1]")
    if args.label_fontsize is not None and float(args.label_fontsize) <= 0:
        raise SystemExit("--label-fontsize must be > 0")

    if args.style == "prb":
        _apply_scienceplots_prb_style()

    _apply_global_fontsize(args.fontsize)

    want_bold = bool(args.bold_fonts)
    if want_bold:
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        if args.style == "default":
            plt.rcParams["axes.linewidth"] = 2

        # If user did not override --ylabel, render math symbols in true bold.
        if str(args.ylabel).strip() == ylabel_default:
            args.ylabel = ylabel_default_bold

    files = [str(x) for x in args.files]
    for f in files:
        if not os.path.exists(f):
            raise SystemExit(f"File not found: {f}")

    n = len(files)
    Ls = _broadcast_float_list([float(x) for x in args.L], n, "--L")
    if any(L <= 0 for L in Ls):
        raise SystemExit("--L must be > 0")

    omegas_in = _flatten_tokens(args.omega)
    omega_by_file: List[Optional[str]]
    if omegas_in:
        omega_by_file = _broadcast_list(omegas_in, n, "--omega")
    else:
        omega_by_file = [None] * n

    x_mode = str(args.x).lower()

    # Resolve per-type x modes.
    x_ph = str(args.x_phonon).lower()
    x_el = str(args.x_electron).lower()
    if x_mode != "auto":
        # Legacy override.
        if x_mode in {"freq", "index"}:
            x_ph = x_mode
        if x_mode in {"energy", "index"}:
            x_el = x_mode

    # Legend labels per dataset
    legends_in = _flatten_tokens(args.legend)
    if legends_in:
        legend_labels_raw = _broadcast_list(legends_in, n, "--legend")
    else:
        # Prefer directory name for BTE.v inputs; else filename.
        legend_labels_raw = []
        for f in files:
            p = Path(f)
            if p.name.lower().startswith("bte.v"):
                legend_labels_raw.append(p.parent.name or p.name)
            else:
                legend_labels_raw.append(p.stem)

    legend_labels = [_format_label(str(x), str(args.legend_format)) for x in legend_labels_raw]

    # Global system label
    system_label: Optional[str] = None
    if args.system:
        sys_tokens = [str(x) for x in args.system if str(x).strip()]
        if len(sys_tokens) == 1:
            system_label = sys_tokens[0]
        elif len(sys_tokens) > 1 and not legends_in:
            print("[warn] multiple --system values detected; treating them as --legend labels")
            legend_labels = [_format_label(x, str(args.legend_format)) for x in _broadcast_list(sys_tokens, n, "--system(as-legend)")]
            system_label = None
        elif len(sys_tokens) > 1:
            print("[warn] multiple --system values detected; using the first one as global system label")
            system_label = sys_tokens[0]

    system_bbox = _parse_xy(args.system_bbox)
    legend_bbox = _parse_xy(args.legend_bbox)
    figsize = _parse_figsize(args.figsize)
    xlim_legacy = _parse_lim(args.xlim)
    xlim_freq = _parse_lim(args.xlim_freq)
    xlim_energy = _parse_lim(args.xlim_energy)
    ylim = _parse_lim(args.ylim)

    # Colors
    if n <= 10:
        colors = [plt.get_cmap("tab10")(i) for i in range(n)]
    else:
        colors = [plt.get_cmap("tab20")(i % 20) for i in range(n)]

    # Detect which dataset types are present.
    is_phonon = [(_detect_source(f) == "shengbte") for f in files]
    is_electron = [(_detect_source(f) == "perturbo") for f in files]
    has_phonon = any(is_phonon)
    has_electron = any(is_electron)

    fig = (
        plt.figure(figsize=figsize, dpi=default_dpi)
        if figsize is not None
        else plt.figure(dpi=default_dpi)
    )
    ax_freq = fig.add_subplot(1, 1, 1)
    ax_energy = ax_freq.twiny() if (has_phonon and has_electron) else None

    # Plot
    # Broadcast E_F for electron-only list (in the same order as electron files appear).
    electron_indices = [i for i, ok in enumerate(is_electron) if ok]
    efs_by_electron: List[Optional[float]] = []
    if electron_indices:
        ef_in = _flatten_tokens(args.ef)
        if ef_in:
            efs_by_electron = [float(x) for x in _broadcast_list(ef_in, len(electron_indices), "--ef")]
        else:
            efs_by_electron = [None] * len(electron_indices)

    total_plotted = 0
    for i, f in enumerate(files):
        src = _detect_source(f)

        if src == "shengbte":
            xm = x_ph
            if xm not in {"freq", "index"}:
                raise SystemExit("--x-phonon must be freq|index")
            x, vmag, _ = _load_shengbte_v_and_x(
                f,
                xmode=("index" if xm == "index" else "freq"),
                omega_path=omega_by_file[i],
            )
            target_ax = ax_freq
            # ShengBTE convention: v in km/s, which is numerically equal to nm/ps.
            v_nm_ps = np.asarray(vmag, dtype=float)
        else:
            xm = x_el
            if xm not in {"energy", "index"}:
                raise SystemExit("--x-electron must be energy|index")
            ef_override = None
            try:
                j = electron_indices.index(i)
                ef_override = efs_by_electron[j] if efs_by_electron else None
            except Exception:
                ef_override = None

            x, vmag, _ = _load_perturbo_meanfp_v_and_x(
                f,
                xmode=("index" if xm == "index" else "energy"),
                bands=str(args.bands) if args.bands is not None else None,
                config=int(args.config),
                ef_override=ef_override,
            )

            target_ax = ax_energy if ax_energy is not None else ax_freq

            # Perturbo meanfp 'band velocity' is treated as m/s; convert to nm/ps.
            # 1 m/s = 1e-3 nm/ps
            v_nm_ps = np.asarray(vmag, dtype=float) * 1.0e-3

        # τ = |v| / L
        # With v in nm/ps and L in nm, τ has units ps^-1.
        y = np.asarray(v_nm_ps, dtype=float) / float(Ls[i])

        # Log-scale safety
        mask = np.ones_like(x, dtype=bool)
        if args.xlog:
            mask &= (x > 0)
        if args.ylog:
            mask &= (y > 0)
        dropped = int(np.size(mask) - int(np.count_nonzero(mask)))
        if dropped > 0:
            print(f"[warn] {Path(f).name}: dropped {dropped} non-positive points for log scale")
        x2 = np.asarray(x[mask], dtype=float)
        y2 = np.asarray(y[mask], dtype=float)
        if x2.size == 0 or y2.size == 0:
            print(f"[warn] {Path(f).name}: no points left after filtering; skipped")
            continue

        target_ax.scatter(x2, y2, s=float(args.ms), alpha=float(args.alpha), color=colors[i], edgecolors="none")
        total_plotted += 1

    if total_plotted == 0:
        raise SystemExit("No datasets were plotted (all files empty/filtered).")

    # Labels (dual-x aware)
    if has_phonon and has_electron and ax_energy is not None:
        ax_freq.set_xlabel(str(args.xlabel_freq))
        ax_energy.set_xlabel(str(args.xlabel_energy))
    else:
        # Single axis: keep legacy override.
        if has_phonon:
            ax_freq.set_xlabel(str(args.xlabel) if args.xlabel is not None else str(args.xlabel_freq))
        else:
            ax_freq.set_xlabel(str(args.xlabel) if args.xlabel is not None else str(args.xlabel_energy))

    ax_freq.set_ylabel(str(args.ylabel))

    if args.label_fontsize is not None:
        fs = float(args.label_fontsize)
        for a in (ax_freq, ax_energy):
            if a is None:
                continue
            a.xaxis.label.set_size(fs)
            a.yaxis.label.set_size(fs)
            a.tick_params(axis="both", which="both", labelsize=fs)
            try:
                a.xaxis.get_offset_text().set_size(fs)
                a.yaxis.get_offset_text().set_size(fs)
            except Exception:
                pass
        try:
            pass
        except Exception:
            pass

    if args.xlog:
        ax_freq.set_xscale("log")
        if ax_energy is not None:
            ax_energy.set_xscale("log")
    if args.ylog:
        ax_freq.set_yscale("log")

    # x limits: prefer per-axis args; fall back to legacy.
    if xlim_freq is None and xlim_legacy is not None and (not has_electron or not has_phonon):
        xlim_freq = xlim_legacy
    if xlim_energy is None and xlim_legacy is not None and (not has_electron or not has_phonon):
        xlim_energy = xlim_legacy
    if has_phonon and xlim_freq is not None:
        ax_freq.set_xlim(*xlim_freq)
    if has_electron:
        target = ax_energy if ax_energy is not None else ax_freq
        if xlim_energy is not None:
            target.set_xlim(*xlim_energy)
    if (has_phonon and has_electron) and xlim_legacy is not None and (xlim_freq is None and xlim_energy is None):
        ax_freq.set_xlim(*xlim_legacy)
        if ax_energy is not None:
            ax_energy.set_xlim(*xlim_legacy)
    if ylim:
        ax_freq.set_ylim(*ylim)

    # x tick steps (allow separate control for the two x-axes)
    xstep_freq = args.xtick_step_freq
    xstep_energy = args.xtick_step_energy
    if args.xtick_step is not None:
        # Legacy: apply to both unless a per-axis value is provided.
        if xstep_freq is None:
            xstep_freq = float(args.xtick_step)
        if xstep_energy is None:
            xstep_energy = float(args.xtick_step)

    if not args.xlog:
        if xstep_freq is not None:
            step = float(xstep_freq)
            if step <= 0:
                raise SystemExit("--xtick-step-freq must be > 0")
            ax_freq.xaxis.set_major_locator(MultipleLocator(step))

        if xstep_energy is not None and has_electron:
            step = float(xstep_energy)
            if step <= 0:
                raise SystemExit("--xtick-step-energy must be > 0")
            target = ax_energy if ax_energy is not None else ax_freq
            target.xaxis.set_major_locator(MultipleLocator(step))

    if args.ytick_step is not None:
        step = float(args.ytick_step)
        if step <= 0:
            raise SystemExit("--ytick-step must be > 0")
        if not args.ylog:
            ax_freq.yaxis.set_major_locator(MultipleLocator(step))

    if args.grid:
        ax_freq.grid(True, linestyle="--", alpha=0.3)

    # Build legend entries
    handles_leg: List[Line2D] = []
    for i, lab in enumerate(legend_labels):
        handles_leg.append(
            Line2D(
                [],
                [],
                linestyle="None",
                marker="o",
                markersize=6,
                markerfacecolor=colors[i],
                markeredgecolor="none",
                label=str(lab),
            )
        )

    leg_main = None
    if handles_leg:
        legend_loc = str(args.legend_loc)
        if (legend_bbox is not None) and (legend_loc.strip().lower() == "best"):
            legend_loc = "upper left"

        handles2: List[object] = list(handles_leg)
        labels2: List[str] = [str(h.get_label()) for h in handles_leg]
        if int(args.legend_ncol) > 1:
            handles2, labels2 = _pad_legend_columns_preserve_order(
                handles2,
                labels2,
                ncol=int(args.legend_ncol),
                column_real_counts=(args.legend_column_counts if args.legend_column_counts is not None else None),
            )

        kwargs = dict(
            handles=handles2,
            labels=labels2,
            loc=legend_loc,
            frameon=bool(args.legend_alpha is not None),
            ncols=int(args.legend_ncol),
            borderaxespad=(0.0 if legend_bbox is not None else 0.2),
            handlelength=float(args.legend_handlelength),
            handletextpad=float(args.legend_handletextpad),
            columnspacing=float(args.legend_columnspacing),
            labelspacing=float(args.legend_labelspacing),
        )
        if args.legend_fontsize is not None:
            kwargs["fontsize"] = float(args.legend_fontsize)

        if legend_bbox is None:
            leg_main = ax_freq.legend(**kwargs)
        else:
            leg_main = ax_freq.legend(
                **kwargs,
                bbox_to_anchor=legend_bbox,
                bbox_transform=ax_freq.transAxes,
            )

        _apply_legend_frame(leg_main, alpha=args.legend_alpha)
        try:
            leg_main.set_zorder(1000)
        except Exception:
            pass
        if want_bold and leg_main is not None:
            for t in leg_main.get_texts():
                t.set_fontweight("bold")

    # System annotation
    if system_label is not None and str(system_label).strip():
        sys_lab = _format_label(str(system_label), str(args.system_format))
        h = Line2D([], [], color="none", label=str(sys_lab))

        fs = args.system_fontsize
        if fs is None:
            try:
                fs = float(ax_freq.yaxis.label.get_size()) * 1.15
            except Exception:
                fs = None

        if leg_main is not None:
            ax_freq.add_artist(leg_main)

        sys_kwargs = {
            "frameon": bool(args.system_alpha is not None),
            "handlelength": 0,
            "handletextpad": 0.0,
            "borderaxespad": (0.0 if system_bbox is not None else 0.2),
            "fontsize": fs,
        }

        if system_bbox is None:
            leg_sys = ax_freq.legend(handles=[h], loc=str(args.system_loc), **sys_kwargs)
        else:
            leg_sys = ax_freq.legend(
                handles=[h],
                loc=str(args.system_loc),
                bbox_to_anchor=system_bbox,
                bbox_transform=ax_freq.transAxes,
                **sys_kwargs,
            )
        _apply_legend_frame(leg_sys, alpha=args.system_alpha)
        try:
            if leg_sys is not None:
                leg_sys.set_zorder(1001)
        except Exception:
            pass
        if leg_sys is not None:
            for t in leg_sys.get_texts():
                if want_bold:
                    t.set_fontweight("bold")
                try:
                    t.set_ha("center")
                except Exception:
                    pass
            try:
                leg_sys._legend_box.align = "center"  # noqa: SLF001
            except Exception:
                pass

    if want_bold:
        _set_figure_text_weight(fig, "bold")

    plt.tight_layout()

    out = str(args.out)
    if out:
        fig.savefig(out, dpi=default_dpi)
        print(f"Saved: {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
