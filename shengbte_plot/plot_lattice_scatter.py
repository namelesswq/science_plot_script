#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.text import Text


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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot ShengBTE scattering-rate distributions as scatter points.\n\n"
            "Input: plain text table(s) with at least two columns.\n"
            "Default columns: x=col0 (frequency), y=col1 (scattering rate).\n\n"
            "Multiple files can be provided to overlay and compare different datasets."
        )
    )

    p.add_argument(
        "--file",
        required=True,
        nargs="+",
        help=(
            "One or more input files (e.g. BTE.w_isotopic, T300K/BTE.w_final). "
            "All provided files are overlaid for comparison."
        ),
    )
    p.add_argument(
        "--legend",
        default=None,
        nargs="+",
        help=(
            "Legend label(s) for each --file. Provide one per file, or a single value to broadcast. "
            "If omitted, uses the file basename."
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
        default="best",
        help="Legend location (matplotlib legend loc). Default: best.",
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
        "--legend-alpha",
        type=float,
        default=None,
        help="If set, draw the legend with a white semi-transparent frame (0..1).",
    )

    # Global system annotation (QE-style): a separate legend used to indicate what system/material this figure refers to.
    p.add_argument(
        "--system",
        default=None,
        nargs="+",
        help=(
            "Overall system/material label shown as a separate legend entry (e.g. 'Zr2SC'). "
            "Normally give ONE value. If multiple are given and --legend is omitted, they will be treated as --legend."
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

    p.add_argument(
        "--system-alpha",
        type=float,
        default=None,
        help="If set, draw the system annotation with a white semi-transparent frame (0..1).",
    )

    # Backward-compatible alias:
    p.add_argument(
        "--label",
        default=None,
        nargs="+",
        help="Alias of --legend (kept for backward compatibility).",
    )
    p.add_argument("--xcol", type=int, default=0, help="0-based column index for x (default: 0)")
    p.add_argument("--ycol", type=int, default=1, help="0-based column index for y (default: 1)")

    p.add_argument("--xlog", action="store_true", help="Use log scale on x-axis")
    p.add_argument(
        "--ylog",
        action="store_true",
        default=True,
        help="Use log scale on y-axis (default: on)",
    )
    p.add_argument(
        "--no-ylog",
        dest="ylog",
        action="store_false",
        help="Disable log scale on y-axis",
    )

    p.add_argument("--ms", type=float, default=10.0, help="Marker size (default: 10)")
    p.add_argument("--alpha", type=float, default=0.4, help="Marker alpha (default: 0.4)")

    p.add_argument("--xlim", default=None, help='x limits "xmin,xmax"')
    p.add_argument("--ylim", default=None, help='y limits "ymin,ymax"')

    p.add_argument(
        "--xtick-step",
        type=float,
        default=None,
        help="Major tick step for x-axis (Frequency, THz). Example: --xtick-step 1.",
    )
    p.add_argument(
        "--ytick-step",
        type=float,
        default=None,
        help="Major tick step for y-axis (Scattering rate). Example: --ytick-step 0.1.",
    )

    p.add_argument("--xlabel", default="Frequency (THz)", help="x-axis label")
    p.add_argument("--ylabel", default=r"Scattering Rate (ps$^{-1}$)", help="y-axis label")

    p.add_argument(
        "--label-fontsize",
        type=float,
        default=None,
        help=(
            "Font size for axis labels AND tick numbers. "
            "If omitted, uses matplotlib defaults."
        ),
    )

    p.add_argument(
        "--fontsize",
        type=float,
        default=None,
        help=(
            "Global default font size (rcParams). Does not override explicit per-item sizes like "
            "--label-fontsize/--legend-fontsize/--system-fontsize."
        ),
    )

    p.add_argument(
        "--bold-fonts",
        action="store_true",
        help="Force all text in the figure to bold (including for --style prb).",
    )

    p.add_argument(
        "--style",
        choices=["prb", "default"],
        default="prb",
        help="Plot style preset. 'prb' uses SciencePlots (science,no-latex). Default: prb.",
    )
    p.add_argument("--figsize", default=None, help='Figure size "width,height" in inches (e.g. "3.6,2.8")')

    p.add_argument("--grid", action="store_true", help="Show grid")
    p.add_argument("--out", default="scattering.png", help="Output image path")
    p.add_argument("--show", action="store_true", help="Show interactively")
    return p


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


def _format_system_label(label: str, mode: str) -> str:
    if not label:
        return label
    if mode == "raw":
        return label
    if "$" in label:
        return label
    import re

    return re.sub(r"(?<=[A-Za-z\)])(\d+)", r"$_{\1}$", label)


def _broadcast_list(xs: Sequence[str], n: int, name: str) -> List[str]:
    if len(xs) == n:
        return list(xs)
    if len(xs) == 1:
        return [str(xs[0])] * n
    raise SystemExit(f"{name} expects 1 value or {n} values, but got {len(xs)}")


def _load_two_cols(path: str, xcol: int, ycol: int) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2:
        raise SystemExit(f"Unexpected data format in {path!r}: expected 2D table, got shape={getattr(data, 'shape', None)}")
    if data.shape[1] <= max(xcol, ycol):
        raise SystemExit(
            f"Not enough columns in {path!r}: need col {max(xcol, ycol)} but got shape={data.shape}. "
            "Use --xcol/--ycol to select columns."
        )
    x = np.asarray(data[:, xcol], dtype=float)
    y = np.asarray(data[:, ycol], dtype=float)
    return x, y


def main() -> None:
    args = _build_parser().parse_args()

    if args.ms <= 0:
        raise SystemExit("--ms must be > 0")
    if not (0.0 <= args.alpha <= 1.0):
        raise SystemExit("--alpha must be within [0, 1]")
    if args.xcol < 0 or args.ycol < 0:
        raise SystemExit("--xcol/--ycol must be >= 0")
    if args.label_fontsize is not None and float(args.label_fontsize) <= 0:
        raise SystemExit("--label-fontsize must be > 0")

    if args.style == "prb":
        _apply_scienceplots_prb_style()

    _apply_global_fontsize(args.fontsize)

    want_bold_fonts = bool(args.bold_fonts)
    if want_bold_fonts:
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        if args.style == "default":
            plt.rcParams["axes.linewidth"] = 2

    xlim = _parse_lim(args.xlim)
    ylim = _parse_lim(args.ylim)
    figsize = _parse_figsize(args.figsize)
    system_bbox = _parse_xy(args.system_bbox)
    legend_bbox = _parse_xy(args.legend_bbox)

    files = [str(x) for x in args.file]
    for f in files:
        if not os.path.exists(f):
            raise SystemExit(f"File not found: {f}")

    # Legend labels per dataset
    if args.legend is None and args.label is not None:
        args.legend = args.label

    legend_labels: List[str]
    if args.legend is None:
        legend_labels = [Path(f).name for f in files]
    else:
        legend_labels = _broadcast_list([str(x) for x in args.legend], len(files), "--legend")

    # Global system label (normally single)
    system_label: Optional[str] = None
    if args.system:
        sys_tokens = [str(x) for x in args.system if str(x).strip()]
        if len(sys_tokens) == 1:
            system_label = sys_tokens[0]
        elif len(sys_tokens) > 1 and args.legend is None and args.label is None:
            print("[warn] multiple --system values detected; treating them as --legend labels")
            legend_labels = _broadcast_list(sys_tokens, len(files), "--system(as-legend)")
            system_label = None
        elif len(sys_tokens) > 1:
            print("[warn] multiple --system values detected; using the first one as global system label")
            system_label = sys_tokens[0]

    # Colors
    n = len(files)
    if n <= 10:
        colors = [plt.get_cmap("tab10")(i) for i in range(n)]
    else:
        colors = [plt.get_cmap("tab20")(i % 20) for i in range(n)]

    fig = plt.figure(figsize=figsize) if figsize is not None else plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # ShengBTE w files typically use omega in rad/ps; convert to THz by dividing 2*pi.
    # 1 THz = 2*pi rad/ps
    x_to_thz = 1.0 / (2.0 * np.pi)

    total_plotted = 0
    for i, f in enumerate(files):
        x, y = _load_two_cols(f, xcol=int(args.xcol), ycol=int(args.ycol))

        x = np.asarray(x, dtype=float) * float(x_to_thz)

        # Log-scale safety: drop non-positive values
        mask = np.ones_like(x, dtype=bool)
        if args.xlog:
            mask &= (x > 0)
        if args.ylog:
            mask &= (y > 0)
        dropped = int(np.size(mask) - int(np.count_nonzero(mask)))
        if dropped > 0:
            print(f"[warn] {Path(f).name}: dropped {dropped} non-positive points for log scale")
        x2 = x[mask]
        y2 = y[mask]
        if x2.size == 0 or y2.size == 0:
            print(f"[warn] {Path(f).name}: no points left after filtering; skipped")
            continue

        ax.scatter(x2, y2, s=float(args.ms), alpha=float(args.alpha), color=colors[i], edgecolors="none")
        total_plotted += 1

    if total_plotted == 0:
        raise SystemExit("No datasets were plotted (all files missing/empty/filtered).")

    ax.set_xlabel(str(args.xlabel))
    ax.set_ylabel(str(args.ylabel))

    if args.label_fontsize is not None:
        fs = float(args.label_fontsize)
        ax.xaxis.label.set_size(fs)
        ax.yaxis.label.set_size(fs)
        ax.tick_params(axis="both", which="both", labelsize=fs)
        try:
            ax.xaxis.get_offset_text().set_size(fs)
            ax.yaxis.get_offset_text().set_size(fs)
        except Exception:
            pass

    if args.xlog:
        ax.set_xscale("log")
    if args.ylog:
        ax.set_yscale("log")

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    if args.xtick_step is not None:
        step = float(args.xtick_step)
        if step <= 0:
            raise SystemExit("--xtick-step must be > 0")
        ax.xaxis.set_major_locator(MultipleLocator(step))

    if args.ytick_step is not None:
        step = float(args.ytick_step)
        if step <= 0:
            raise SystemExit("--ytick-step must be > 0")
        if not args.ylog:
            ax.yaxis.set_major_locator(MultipleLocator(step))

    if args.grid:
        ax.grid(True, linestyle="--", alpha=0.3)

    # Dataset legend (colored markers)
    handles_leg: List[Line2D] = []
    for i, lab_raw in enumerate(legend_labels):
        lab = _format_system_label(str(lab_raw), str(args.legend_format))
        handles_leg.append(
            Line2D(
                [],
                [],
                linestyle="None",
                marker="o",
                markersize=6,
                markerfacecolor=colors[i],
                markeredgecolor="none",
                label=lab,
            )
        )

    leg_main = None
    if handles_leg:
        legend_loc = str(args.legend_loc)
        if (legend_bbox is not None) and (legend_loc.strip().lower() == "best"):
            legend_loc = "upper left"
        kwargs = dict(
            handles=handles_leg,
            loc=legend_loc,
            frameon=bool(args.legend_alpha is not None),
            borderaxespad=(0.0 if legend_bbox is not None else 0.2),
            handlelength=0.8,
            handletextpad=0.35,
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

        if args.legend_alpha is not None:
            _apply_legend_frame(leg_main, alpha=float(args.legend_alpha))
        if want_bold_fonts and leg_main is not None:
            for t in leg_main.get_texts():
                t.set_fontweight("bold")

    # Global system annotation legend (pure text)
    if system_label is not None and str(system_label).strip():
        sys_lab = _format_system_label(str(system_label), str(args.system_format))
        h = Line2D([], [], color="none", label=sys_lab)

        fs = args.system_fontsize
        if fs is None:
            try:
                fs = float(ax.yaxis.label.get_size()) * 1.15
            except Exception:
                fs = None

        if leg_main is not None:
            ax.add_artist(leg_main)

        sys_kwargs = {
            "frameon": bool(args.system_alpha is not None),
            "handlelength": 0,
            "handletextpad": 0.0,
            "borderaxespad": 0.2,
            "fontsize": fs,
        }

        if system_bbox is None:
            leg_sys = ax.legend(
                handles=[h],
                loc=str(args.system_loc),
                **sys_kwargs,
            )
        else:
            leg_sys = ax.legend(
                handles=[h],
                loc=str(args.system_loc),
                bbox_to_anchor=system_bbox,
                bbox_transform=ax.transAxes,
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

    if want_bold_fonts:
        _set_figure_text_weight(fig, "bold")

    fig.tight_layout()
    fig.savefig(str(args.out), dpi=300)
    print(f"Saved: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()