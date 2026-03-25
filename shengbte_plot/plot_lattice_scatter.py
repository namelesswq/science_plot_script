#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


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
        kwargs = dict(
            handles=handles_leg,
            loc=str(args.legend_loc),
            frameon=False,
            borderaxespad=0.2,
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

    fig.tight_layout()
    fig.savefig(str(args.out), dpi=300)
    print(f"Saved: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()