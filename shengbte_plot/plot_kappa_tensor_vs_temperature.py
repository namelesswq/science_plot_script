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
            "Plot lattice thermal conductivity tensor components from ShengBTE BTE.KappaTensorVsT_CONV.\n\n"
            "Expected columns (per row):\n"
            "  T, kxx, kxy, kxz, kyx, kyy, kyz, kzx, kzy, kzz, (ignored last column)\n\n"
            "Multiple files can be provided to overlay and compare datasets."
        )
    )

    p.add_argument(
        "--file",
        required=True,
        nargs="+",
        help="One or more BTE.KappaTensorVsT_CONV files to overlay for comparison.",
    )

    # Dataset legend (colored lines)
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

    # Global system annotation (pure text)
    p.add_argument(
        "--system",
        default=None,
        help="Overall system/material label shown as a separate legend entry (e.g. 'Zr2SC').",
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
        "--component",
        default="avg",
        choices=[
            "xx",
            "xy",
            "xz",
            "yx",
            "yy",
            "yz",
            "zx",
            "zy",
            "zz",
            "avg",
            "trace",
        ],
        help=(
            "Which tensor component to plot. 'avg' plots (kxx+kyy+kzz)/3. "
            "'trace' plots (kxx+kyy+kzz). Default: avg."
        ),
    )

    p.add_argument("--xlim", default=None, help='x limits "xmin,xmax" (Temperature in K)')
    p.add_argument("--ylim", default=None, help='y limits "ymin,ymax" (kappa)')
    p.add_argument("--ylog", action="store_true", help="Use log scale on y-axis")

    p.add_argument(
        "--color",
        default=None,
        nargs="+",
        help=(
            "Line/marker color(s) for each dataset. Provide one per file, or a single value to broadcast. "
            "Examples: 'black', 'tab:red', '#1f77b4'."
        ),
    )
    p.add_argument(
        "--marker",
        default=None,
        nargs="+",
        help=(
            "Marker style(s) for each dataset. Provide one per file, or a single value to broadcast. "
            "Examples: 'o', 's', '^', 'D', 'v', 'x', '+', '*'."
        ),
    )
    p.add_argument(
        "--ms",
        type=float,
        default=None,
        nargs="+",
        help=(
            "Marker size(s) for each dataset. Provide one per file, or a single value to broadcast. "
            "Default: 4.5."
        ),
    )

    p.add_argument(
        "--ls",
        default=None,
        nargs="+",
        help=(
            "Line style(s) for each dataset. Provide one per file, or a single value to broadcast. "
            "Examples: '-', ':', '-.', 'dashed'. Note: do not pass a bare '--' token; use 'dashed' or comma-separated like: --ls -,-,--"
        ),
    )

    p.add_argument("--xlabel", default="Temperature (K)", help="x-axis label")
    p.add_argument("--ylabel", default=r"$\kappa_{\mathrm{latt}}$ (W/mK)", help="y-axis label")

    p.add_argument(
        "--style",
        choices=["prb", "default"],
        default="prb",
        help="Plot style preset. 'prb' uses SciencePlots (science,no-latex). Default: prb.",
    )
    p.add_argument("--figsize", default=None, help='Figure size "width,height" in inches (e.g. "3.6,2.8")')
    p.add_argument("--lw", type=float, default=1.8, help="Line width (default: 1.8)")

    p.add_argument("--grid", action="store_true", help="Show grid")
    p.add_argument("--out", default="kappa_vs_T.png", help="Output image path")
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


def _broadcast_float_list(xs: Sequence[float], n: int, name: str) -> List[float]:
    if len(xs) == n:
        return [float(x) for x in xs]
    if len(xs) == 1:
        return [float(xs[0])] * n
    raise SystemExit(f"{name} expects 1 value or {n} values, but got {len(xs)}")


def _normalize_linestyle_token(s: str) -> str:
    """Normalize user-friendly linestyle aliases.

    Note: passing a bare '--' token on the CLI is interpreted by argparse as
    end-of-options. Use 'dashed' instead, or comma-separated input like:
    --ls -,-,--
    """

    t = str(s).strip().lower()
    aliases = {
        "solid": "-",
        "dash": "--",
        "dashed": "--",
        "dashdash": "--",
        "dot": ":",
        "dotted": ":",
        "dashdot": "-.",
        "dash-dot": "-.",
    }
    return aliases.get(t, str(s))


def _read_kappa_tensor_vs_t(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2:
        raise SystemExit(f"Unexpected data format in {path!r}: expected 2D table, got shape={getattr(data, 'shape', None)}")
    if data.shape[1] < 10:
        raise SystemExit(
            f"Unexpected column count in {path!r}: need at least 10 columns (T + 9 tensor comps), got shape={data.shape}."
        )

    t = np.asarray(data[:, 0], dtype=float)
    k9 = np.asarray(data[:, 1:10], dtype=float)
    return t, k9


def _select_component(k9: np.ndarray, comp: str) -> np.ndarray:
    # order: xx,xy,xz,yx,yy,yz,zx,zy,zz
    idx = {
        "xx": 0,
        "xy": 1,
        "xz": 2,
        "yx": 3,
        "yy": 4,
        "yz": 5,
        "zx": 6,
        "zy": 7,
        "zz": 8,
    }
    if comp in idx:
        return np.asarray(k9[:, idx[comp]], dtype=float)

    kxx = np.asarray(k9[:, 0], dtype=float)
    kyy = np.asarray(k9[:, 4], dtype=float)
    kzz = np.asarray(k9[:, 8], dtype=float)
    if comp == "avg":
        return (kxx + kyy + kzz) / 3.0
    if comp == "trace":
        return (kxx + kyy + kzz)

    raise SystemExit(f"Unknown --component {comp!r}")


def main() -> None:
    args = _build_parser().parse_args()

    files = [str(x) for x in args.file]
    for f in files:
        if not os.path.exists(f):
            raise SystemExit(f"File not found: {f}")

    if args.lw <= 0:
        raise SystemExit("--lw must be > 0")

    if args.style == "prb":
        _apply_scienceplots_prb_style()

    xlim = _parse_lim(args.xlim)
    ylim = _parse_lim(args.ylim)
    figsize = _parse_figsize(args.figsize)
    legend_bbox = _parse_xy(args.legend_bbox)
    system_bbox = _parse_xy(args.system_bbox)

    # Legend labels
    if args.legend is None:
        legend_labels = [Path(f).name for f in files]
    else:
        legend_labels = _broadcast_list([str(x) for x in args.legend], len(files), "--legend")

    n = len(files)
    # Default colors/markers
    default_colors: List[object]
    if n <= 10:
        default_colors = [plt.get_cmap("tab10")(i) for i in range(n)]
    else:
        default_colors = [plt.get_cmap("tab20")(i % 20) for i in range(n)]
    default_markers = ["o", "s", "^", "D", "v", ">", "<", "p", "h", "x", "+", "*"]

    if args.color is None:
        colors = list(default_colors)
    else:
        colors = _broadcast_list([str(x) for x in args.color], n, "--color")

    if args.marker is None:
        markers = [default_markers[i % len(default_markers)] for i in range(n)]
    else:
        markers = _broadcast_list([str(x) for x in args.marker], n, "--marker")

    if args.ms is None:
        marker_sizes = [4.5] * n
    else:
        marker_sizes = _broadcast_float_list([float(x) for x in args.ms], n, "--ms")
    if any(m <= 0 for m in marker_sizes):
        raise SystemExit("--ms must be > 0")

    if args.ls is None:
        linestyles = ["-"] * n
    else:
        linestyles_raw = _broadcast_list([str(x) for x in args.ls], n, "--ls")
        linestyles = [_normalize_linestyle_token(x) for x in linestyles_raw]

    fig = plt.figure(figsize=figsize) if figsize is not None else plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Plot
    for i, f in enumerate(files):
        t, k9 = _read_kappa_tensor_vs_t(f)
        y = _select_component(k9, str(args.component))
        ax.plot(
            t,
            y,
            color=colors[i],
            lw=float(args.lw),
            linestyle=str(linestyles[i]),
            marker=markers[i],
            markersize=float(marker_sizes[i]),
            markeredgewidth=0.0,
        )

    ax.set_xlabel(str(args.xlabel))
    ax.set_ylabel(str(args.ylabel))

    if args.ylog:
        ax.set_yscale("log")

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    if args.grid:
        ax.grid(True, linestyle="--", alpha=0.3)

    # Dataset legend (colored lines)
    handles_leg: List[Line2D] = []
    for i, lab_raw in enumerate(legend_labels):
        lab = _format_system_label(str(lab_raw), str(args.legend_format))
        handles_leg.append(
            Line2D(
                [],
                [],
                color=colors[i],
                lw=float(args.lw),
                linestyle=str(linestyles[i]),
                marker=markers[i],
                markersize=float(marker_sizes[i]),
                markeredgewidth=0.0,
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

    fig.tight_layout()
    fig.savefig(str(args.out), dpi=300)
    print(f"Saved: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
