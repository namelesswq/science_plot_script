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
            "Plot cumulative lattice thermal conductivity tensor vs phonon frequency from ShengBTE\n"
            "BTE.cumulative_kappaVsOmega_tensor.\n\n"
            "Expected columns (per row):\n"
            "  omega(rad/ps), kxx, kxy, kxz, kyx, kyy, kyz, kzx, kzy, kzz\n\n"
            "The x-axis is converted from rad/ps to THz via omega/(2*pi).\n"
            "Multiple files can be provided to overlay and compare datasets."
        )
    )

    p.add_argument(
        "--file",
        required=True,
        nargs="+",
        help="One or more BTE.cumulative_kappaVsOmega_tensor files to overlay for comparison.",
    )

    # Dataset legend (colored lines)
    p.add_argument(
        "--legend",
        default=None,
        nargs="+",
        help=(
            "Legend label(s). Provide 1 (broadcast), Nfile (per-file), Ncomp (per-component), or Nfile×Ncomp (per line) values. "
            "If omitted, uses the file basename (and appends component when plotting multiple)."
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
        default=["avg"],
        nargs="+",
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
            "Tensor component(s) to plot. You can pass multiple components to overlay directions on one figure. "
            "'avg' plots (kxx+kyy+kzz)/3. 'trace' plots (kxx+kyy+kzz). Default: avg."
        ),
    )

    p.add_argument(
        "--order",
        choices=["comp-major", "file-major"],
        default="comp-major",
        help=(
            "Order of plotted lines. comp-major: all files for the first component, then next component. "
            "file-major: all components for the first file, then next file. Default: comp-major."
        ),
    )

    p.add_argument("--xlim", default=None, help='x limits "xmin,xmax" (Frequency in THz)')
    p.add_argument("--ylim", default=None, help='y limits "ymin,ymax" (cumulative kappa)')
    p.add_argument("--ylog", action="store_true", help="Use log scale on y-axis")

    p.add_argument(
        "--color",
        default=None,
        nargs="+",
        help=(
            "Line color(s). Provide 1 (broadcast), Nfile (per-file), Ncomp (per-component), or Nfile×Ncomp (per line) values. "
            "Examples: 'black', 'tab:red', '#1f77b4'."
        ),
    )

    p.add_argument(
        "--ls",
        default=None,
        nargs="+",
        help=(
            "Line style(s). Provide 1 (broadcast), Nfile (per-file), Ncomp (per-component), or Nfile×Ncomp (per line) values. "
            "Examples: '-', ':', '-.', 'dashed'. Note: do not pass a bare '--' token; use 'dashed' or comma-separated like: --ls -,-,--"
        ),
    )

    p.add_argument("--xlabel", default="Frequency (THz)", help="x-axis label")
    p.add_argument(
        "--ylabel",
        default=r"Cumulative $\kappa_{\mathrm{latt}}$ (W/mK)",
        help="y-axis label",
    )

    p.add_argument(
        "--style",
        choices=["prb", "default"],
        default="prb",
        help="Plot style preset. 'prb' uses SciencePlots (science,no-latex). Default: prb.",
    )
    p.add_argument("--figsize", default=None, help='Figure size "width,height" in inches (e.g. "3.6,2.8")')
    p.add_argument("--lw", type=float, default=1.8, help="Line width (default: 1.8)")

    p.add_argument("--grid", action="store_true", help="Show grid")
    p.add_argument("--out", default="cumulative_kappa_vs_freq.png", help="Output image path")
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


def _broadcast_str_grid(xs: Sequence[str], *, n_files: int, n_comp: int, name: str, order: str) -> List[str]:
    """Broadcast to per-line list (file×component) in plotting order.

    Accepted lengths: 1, n_files, n_comp, n_files*n_comp.
    """

    n_lines = n_files * n_comp
    if n_lines == 0:
        return []

    if len(xs) == 1:
        return [str(xs[0])] * n_lines

    if len(xs) == n_files:
        if order == "file-major":
            out: List[str] = []
            for v in xs:
                out.extend([str(v)] * n_comp)
            return out
        out = []
        for _ in range(n_comp):
            for v in xs:
                out.append(str(v))
        return out

    if len(xs) == n_comp:
        if order == "file-major":
            out: List[str] = []
            for _ in range(n_files):
                out.extend([str(v) for v in xs])
            return out
        out = []
        for v in xs:
            out.extend([str(v)] * n_files)
        return out

    if len(xs) == n_lines:
        return [str(v) for v in xs]

    raise SystemExit(
        f"{name} expects 1, {n_files} (per-file), {n_comp} (per-component), or {n_lines} (per line; order={order}) values, but got {len(xs)}"
    )


def _normalize_linestyle_token(s: str) -> str:
    """Normalize user-friendly linestyle aliases.

    Note: passing a bare '--' token on the CLI is interpreted by argparse as
    end-of-options. Use 'dashed' instead, or comma-separated input like:
    --ls -,-,-,--,--,--
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


def _broadcast_float_list(xs: Sequence[float], n: int, name: str) -> List[float]:
    if len(xs) == n:
        return [float(x) for x in xs]
    if len(xs) == 1:
        return [float(xs[0])] * n
    raise SystemExit(f"{name} expects 1 value or {n} values, but got {len(xs)}")


def _read_cumulative_kappa_vs_omega_tensor(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2:
        raise SystemExit(
            f"Unexpected data format in {path!r}: expected 2D table, got shape={getattr(data, 'shape', None)}"
        )
    if data.shape[1] < 10:
        raise SystemExit(
            f"Unexpected column count in {path!r}: need at least 10 columns (omega + 9 tensor comps), got shape={data.shape}."
        )

    omega = np.asarray(data[:, 0], dtype=float)
    k9 = np.asarray(data[:, 1:10], dtype=float)
    return omega, k9


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

    components = [str(c).lower() for c in (args.component or [])]
    if not components:
        raise SystemExit("--component cannot be empty")

    order = str(args.order)
    n_files = len(files)
    n_comp = len(components)
    n_lines = n_files * n_comp

    # Linestyles: per component
    component_linestyles = ["-", "--", ":", "-."]

    default_file_labels = [Path(f).name for f in files]

    legends_in = [str(x) for x in args.legend] if args.legend is not None else []
    legend_mode: str
    legends_file: List[str] = []
    legends_comp: List[str] = []
    legends_line: List[str] = []

    if args.legend is None:
        legend_mode = "file"
        legends_file = list(default_file_labels)
    else:
        if len(legends_in) == 1:
            legend_mode = "file"
            legends_file = [legends_in[0]] * n_files
        elif len(legends_in) == n_files:
            legend_mode = "file"
            legends_file = list(legends_in)
        elif len(legends_in) == n_comp:
            legend_mode = "comp"
            legends_comp = list(legends_in)
        elif len(legends_in) == n_lines:
            legend_mode = "line"
            legends_line = _broadcast_str_grid(legends_in, n_files=n_files, n_comp=n_comp, name="--legend", order=order)
        else:
            raise SystemExit(
                f"--legend expects 1, {n_files} (per-file), {n_comp} (per-component), or {n_lines} (per line; order={order}) values, but got {len(legends_in)}"
            )

    # Default colors
    default_colors: List[object]
    if n_files <= 10:
        default_colors = [plt.get_cmap("tab10")(i) for i in range(n_files)]
    else:
        default_colors = [plt.get_cmap("tab20")(i % 20) for i in range(n_files)]

    if args.color is None:
        # Default: color by file
        colors_line: List[object] = []
        if order == "file-major":
            for i in range(n_files):
                colors_line.extend([default_colors[i]] * n_comp)
        else:
            for _ in range(n_comp):
                for i in range(n_files):
                    colors_line.append(default_colors[i])
    else:
        colors_in = [str(x) for x in args.color]
        colors_line = _broadcast_str_grid(colors_in, n_files=n_files, n_comp=n_comp, name="--color", order=order)

    # Linestyles
    if args.ls is None:
        linestyles_line: List[str] = []
        if order == "file-major":
            for _i in range(n_files):
                for ic in range(n_comp):
                    linestyles_line.append(component_linestyles[ic % len(component_linestyles)])
        else:
            for ic in range(n_comp):
                for _i in range(n_files):
                    linestyles_line.append(component_linestyles[ic % len(component_linestyles)])
    else:
        ls_in = [_normalize_linestyle_token(str(x)) for x in args.ls]
        linestyles_line = _broadcast_str_grid(ls_in, n_files=n_files, n_comp=n_comp, name="--ls", order=order)

    fig = plt.figure(figsize=figsize) if figsize is not None else plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    multi_comp = len(components) > 1

    def _line_index(i: int, ic: int) -> int:
        if order == "file-major":
            return i * n_comp + ic
        return ic * n_files + i

    def _line_pairs() -> List[Tuple[int, int]]:
        if order == "file-major":
            return [(i, ic) for i in range(n_files) for ic in range(n_comp)]
        return [(i, ic) for ic in range(n_comp) for i in range(n_files)]

    # Read all datasets once
    omega_all: List[np.ndarray] = []
    k9_all: List[np.ndarray] = []
    for f in files:
        omega_rad_ps, k9 = _read_cumulative_kappa_vs_omega_tensor(f)
        omega_all.append(np.asarray(omega_rad_ps, dtype=float))
        k9_all.append(np.asarray(k9, dtype=float))

    # Plot (color by file, linestyle by component)
    handles_leg: List[Line2D] = []
    for i, ic in _line_pairs():
        omega_rad_ps = omega_all[i]
        k9 = k9_all[i]
        freq_thz = omega_rad_ps / (2.0 * np.pi)

        comp = components[ic]
        y = _select_component(k9, comp)

        idx_line = _line_index(i, ic)

        if legend_mode == "line":
            lab_raw = legends_line[idx_line]
        elif legend_mode == "comp":
            if n_files == 1:
                lab_raw = legends_comp[ic]
            else:
                lab_raw = f"{default_file_labels[i]} {legends_comp[ic]}"
        else:
            lab_raw = legends_file[i]
            if multi_comp:
                lab_raw = f"{lab_raw} {comp}"

        lab = _format_system_label(str(lab_raw), str(args.legend_format))

        (line,) = ax.plot(
            freq_thz,
            y,
            color=colors_line[idx_line],
            lw=float(args.lw),
            linestyle=str(linestyles_line[idx_line]),
            label=lab,
        )
        handles_leg.append(line)

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
