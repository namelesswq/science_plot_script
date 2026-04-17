#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot phonon mean free path (MFP) from ShengBTE outputs. "
            "Uses BTE.omega + BTE.v + one or more BTE.w_* files. "
            "Default units assume: omega [rad/ps], v [km/s], w [ps^-1] => MFP [nm] = |v|/w."
        )
    )

    p.add_argument("--omega", default="BTE.omega", help="Phonon frequency file [default: BTE.omega]")
    p.add_argument("--v", default="BTE.v", help="Phonon group velocity vector file [default: BTE.v]")
    p.add_argument("w_files", nargs="+", help="One or more scattering-rate files (e.g. BTE.w_final, BTE.w_isotopic)")
    p.add_argument("--labels", default=None, help="Comma-separated labels for each w file")

    p.add_argument(
        "--v-unit",
        choices=["km/s", "m/s"],
        default="km/s",
        help="Unit of |v| in BTE.v [default: km/s]",
    )

    p.add_argument("--x", choices=["omega", "wfile"], default="omega", help="x-axis source [default: omega]")
    p.add_argument("--ylog", action="store_true", help="Use log scale for y-axis")
    p.add_argument("--xlog", action="store_true", help="Use log scale for x-axis")

    p.add_argument("--alpha", type=float, default=0.35, help="Scatter alpha [default: 0.35]")
    p.add_argument("--s", type=float, default=8.0, help="Marker size [default: 8]")

    p.add_argument("--marker", default="o", help="Marker style for all series [default: o]")
    p.add_argument(
        "--markers",
        default=None,
        help="Comma-separated marker styles for each series (overrides --marker)",
    )
    p.add_argument("--color", default=None, help="Marker color for all series (overrides default cycle)")
    p.add_argument(
        "--colors",
        default=None,
        help="Comma-separated colors for each series (overrides default cycle)",
    )
    p.add_argument(
        "--edgecolor",
        default="none",
        help="Marker edge color passed to scatter(edgecolors=...) [default: none]",
    )
    p.add_argument(
        "--linewidth",
        type=float,
        default=0.0,
        help="Marker edge line width passed to scatter(linewidths=...) [default: 0]",
    )

    p.add_argument("--xlim", default=None, help='x limits "xmin,xmax" in THz')
    p.add_argument("--ylim", default=None, help='y limits "ymin,ymax" in nm')

    p.add_argument("--title", default="", help="Plot title")

    figopt = p.add_argument_group("figure")
    figopt.add_argument(
        "--figsize",
        default=None,
        help='Figure size "w,h" in inches (e.g. 7,5) [default: 7,5]',
    )
    figopt.add_argument(
        "--dpi",
        type=float,
        default=150.0,
        help="Figure DPI [default: 150]",
    )
    figopt.add_argument(
        "--save-dpi",
        type=float,
        default=300.0,
        help="Output image DPI [default: 300]",
    )

    font = p.add_argument_group("fonts")
    font.add_argument(
        "--fontsize",
        type=float,
        default=None,
        help="Global font size (rcParams: font/axes/ticks/legend) [default: keep style defaults]",
    )
    font.add_argument(
        "--label-fontsize",
        type=float,
        default=None,
        help="Axes label & tick label size override [default: inherit]",
    )
    font.add_argument(
        "--title-fontsize",
        type=float,
        default=None,
        help="Title font size override [default: inherit]",
    )
    bold = font.add_mutually_exclusive_group()
    bold.add_argument("--bold-fonts", dest="bold_fonts", action="store_true", help="Use bold fonts")
    bold.add_argument("--no-bold", dest="bold_fonts", action="store_false", help="Disable bold fonts")
    p.set_defaults(bold_fonts=True)

    ticks = p.add_argument_group("ticks")
    ticks.add_argument("--xtick-step", type=float, default=None, help="Major tick step for x-axis")
    ticks.add_argument("--ytick-step", type=float, default=None, help="Major tick step for y-axis")

    leg = p.add_argument_group("legend")
    leg.add_argument("--no-legend", action="store_true", help="Disable dataset legend")
    leg.add_argument("--legend-loc", default="best", help="Legend location [default: best]")
    leg.add_argument("--legend-bbox", default=None, help='Legend bbox anchor "x,y"')
    leg.add_argument("--legend-ncol", type=int, default=1, help="Legend columns [default: 1]")
    leg.add_argument(
        "--legend-alpha",
        type=float,
        default=0.25,
        help="Legend frame alpha [default: 0.25]",
    )
    leg.add_argument(
        "--legend-edgecolor",
        default="black",
        help="Legend frame edge color [default: black]",
    )
    leg.add_argument(
        "--legend-fontsize",
        type=float,
        default=None,
        help="Legend text size override [default: inherit]",
    )
    leg.add_argument(
        "--legend-handletextpad",
        type=float,
        default=0.4,
        help="Legend handle/text gap [default: 0.4]",
    )
    leg.add_argument(
        "--legend-handlelength",
        type=float,
        default=1.2,
        help="Legend handle length [default: 1.2]",
    )

    sys = p.add_argument_group("system")
    sys.add_argument("--system", default=None, help="System label shown as a small legend box")
    sys.add_argument("--system-loc", default="upper left", help="System box location [default: upper left]")
    sys.add_argument("--system-bbox", default=None, help='System bbox anchor "x,y"')
    sys.add_argument(
        "--system-alpha",
        type=float,
        default=0.0,
        help="System box frame alpha [default: 0.0]",
    )
    sys.add_argument(
        "--system-edgecolor",
        default="black",
        help="System box frame edge color [default: black]",
    )
    sys.add_argument(
        "--system-fontsize",
        type=float,
        default=None,
        help="System text size override [default: inherit]",
    )
    p.add_argument("--out", default="phonon_mfp_scatter.png", help="Output image [default: phonon_mfp_scatter.png]")
    p.add_argument("--show", action="store_true", help="Show interactively (may require display)")
    return p


def _parse_lim(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = s.split(",", 1)
    return float(a), float(b)


def _parse_list(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    items = [x.strip() for x in s.split(",")]
    items = [x for x in items if x]
    return items or None


def _parse_xy(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = s.split(",", 1)
    return float(a), float(b)


def _apply_prb_style() -> None:
    """Apply SciencePlots PRB style if available; otherwise keep Matplotlib defaults."""
    try:
        import scienceplots  # noqa: F401
    except Exception:
        pass

    try:
        plt.style.use(["science", "no-latex", "prb"])
    except Exception:
        try:
            plt.style.use(["science", "no-latex"])
        except Exception:
            return


def _apply_fonts_and_rcparams(args: argparse.Namespace) -> None:
    if args.fontsize is not None:
        fs = float(args.fontsize)
        mpl.rcParams.update(
            {
                "font.size": fs,
                "axes.titlesize": fs,
                "axes.labelsize": fs,
                "xtick.labelsize": fs,
                "ytick.labelsize": fs,
                "legend.fontsize": fs,
            }
        )

    if bool(args.bold_fonts):
        mpl.rcParams.update(
            {
                "font.weight": "bold",
                "axes.labelweight": "bold",
                "axes.titleweight": "bold",
                "axes.linewidth": 2,
            }
        )
    else:
        mpl.rcParams.update(
            {
                "font.weight": "normal",
                "axes.labelweight": "normal",
                "axes.titleweight": "normal",
            }
        )


def _apply_tick_steps(
    ax: plt.Axes,
    *,
    xtick_step: Optional[float],
    ytick_step: Optional[float],
    xlog: bool,
    ylog: bool,
) -> None:
    if xtick_step is not None and not xlog:
        ax.xaxis.set_major_locator(MultipleLocator(float(xtick_step)))
    if ytick_step is not None and not ylog:
        ax.yaxis.set_major_locator(MultipleLocator(float(ytick_step)))


def _legend_common_style(legend: plt.Legend, *, bold: bool) -> None:
    if bold:
        for t in legend.get_texts():
            t.set_fontweight("bold")


def _load_omega_thz(path: str) -> np.ndarray:
    # ShengBTE BTE.omega is typically angular frequency in rad/ps.
    # Convert: THz = (rad/ps) / (2*pi)
    omega_mat = np.loadtxt(path)
    omega_flat = np.asarray(omega_mat, dtype=float).flatten()
    return omega_flat / (2.0 * np.pi)


def _load_v_mag(path: str) -> np.ndarray:
    v_vec = np.loadtxt(path)
    v_vec = np.asarray(v_vec, dtype=float)
    if v_vec.ndim != 2 or v_vec.shape[1] < 3:
        raise RuntimeError(f"Unexpected BTE.v shape: {v_vec.shape} (expected N x 3)")
    return np.linalg.norm(v_vec[:, :3], axis=1)


def _load_w_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(f"Unexpected w-file shape: {data.shape} (expected N x >=2)")
    omega = data[:, 0]
    w = data[:, 1]
    return omega, w


def _mfp_nm(vmag: np.ndarray, w: np.ndarray, v_unit: str) -> np.ndarray:
    # w in ps^-1, tau = 1/w (ps)
    # If v is km/s: 1 km/s = 1 nm/ps => MFP[nm] = v / w
    # If v is m/s:  1 m/s  = 1e-3 nm/ps => MFP[nm] = (v*1e-3) / w
    if v_unit == "km/s":
        factor = 1.0
    else:
        factor = 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        mfp = (vmag * factor) / w
    return mfp


def main() -> None:
    args = _build_parser().parse_args()

    _apply_prb_style()
    _apply_fonts_and_rcparams(args)

    if args.labels is None:
        labels: List[str] = [os.path.basename(f) for f in args.w_files]
    else:
        labels = [x.strip() for x in args.labels.split(",") if x.strip()]
        if len(labels) != len(args.w_files):
            raise SystemExit(f"--labels count {len(labels)} != w_files count {len(args.w_files)}")

    markers = _parse_list(args.markers)
    if markers is not None and len(markers) != len(args.w_files):
        raise SystemExit(f"--markers count {len(markers)} != w_files count {len(args.w_files)}")

    colors = _parse_list(args.colors)
    if colors is not None and len(colors) != len(args.w_files):
        raise SystemExit(f"--colors count {len(colors)} != w_files count {len(args.w_files)}")

    try:
        x_omega_thz = _load_omega_thz(args.omega)
    except Exception as e:
        raise SystemExit(f"[错误] 读取 {args.omega} 失败: {e}")

    try:
        vmag = _load_v_mag(args.v)
    except Exception as e:
        raise SystemExit(f"[错误] 读取 {args.v} 失败: {e}")

    figsize = _parse_xy(args.figsize) if getattr(args, "figsize", None) else None
    if figsize is None:
        figsize = (7.0, 5.0)
    fig, ax = plt.subplots(figsize=figsize, dpi=float(args.dpi))

    if args.ylog:
        ax.set_yscale("log")
    if args.xlog:
        ax.set_xscale("log")

    for i, (w_path, label) in enumerate(zip(args.w_files, labels)):
        try:
            w_omega, w_rate = _load_w_file(w_path)
        except Exception as e:
            print(f"[警告] 读取 {w_path} 失败: {e} (跳过)")
            continue

        # Choose x-axis
        if args.x == "wfile":
            x_thz = np.asarray(w_omega, dtype=float) / (2.0 * np.pi)
        else:
            x_thz = x_omega_thz

        # Align lengths
        n = min(len(x_thz), len(vmag), len(w_rate))
        if n == 0:
            print(f"[警告] {w_path}: 没有可用数据 (跳过)")
            continue
        if len(x_thz) != n or len(vmag) != n or len(w_rate) != n:
            print(
                f"[警告] 长度不匹配，将截断到 {n}: x={len(x_thz)} v={len(vmag)} w={len(w_rate)} (file={w_path})"
            )

        x = np.asarray(x_thz[:n], dtype=float)
        v = np.asarray(vmag[:n], dtype=float)
        w = np.asarray(w_rate[:n], dtype=float)

        # Filter invalid / non-positive rates
        m = np.isfinite(x) & np.isfinite(v) & np.isfinite(w) & (w > 0.0)
        x = x[m]
        v = v[m]
        w = w[m]

        y = _mfp_nm(v, w, args.v_unit)
        m2 = np.isfinite(y) & (y > 0.0)
        x = x[m2]
        y = y[m2]

        marker = markers[i] if markers is not None else args.marker
        if args.color is not None:
            color = args.color
        elif colors is not None:
            color = colors[i]
        else:
            color = None

        ax.scatter(
            x,
            y,
            s=args.s,
            alpha=args.alpha,
            marker=marker,
            color=color,
            edgecolors=args.edgecolor,
            linewidths=args.linewidth,
            label=label,
            zorder=3,
        )

    title_fs = args.title_fontsize
    label_fs = args.label_fontsize
    if title_fs is None:
        title_fs = (float(args.fontsize) + 2.0) if args.fontsize is not None else 16
    if label_fs is None:
        label_fs = float(args.fontsize) if args.fontsize is not None else 14

    ax.set_title(args.title, fontsize=title_fs, fontweight=("bold" if args.bold_fonts else "normal"), pad=15)
    ax.set_xlabel("Frequency (THz)", fontsize=label_fs, fontweight=("bold" if args.bold_fonts else "normal"))
    ax.set_ylabel("Mean free path (nm)", fontsize=label_fs, fontweight=("bold" if args.bold_fonts else "normal"))

    _apply_tick_steps(
        ax,
        xtick_step=args.xtick_step,
        ytick_step=args.ytick_step,
        xlog=bool(args.xlog),
        ylog=bool(args.ylog),
    )

    xlim = _parse_lim(args.xlim)
    ylim = _parse_lim(args.ylim)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)

    if args.label_fontsize is not None:
        ax.tick_params(axis="both", which="major", labelsize=float(args.label_fontsize))
    if args.bold_fonts:
        for t in (ax.get_xticklabels() + ax.get_yticklabels()):
            t.set_fontweight("bold")

    if args.system:
        sys_bbox = _parse_xy(args.system_bbox)
        sys_handle = Line2D([], [], linestyle="none", marker=None, label=str(args.system))
        sys_leg = ax.legend(
            handles=[sys_handle],
            loc=args.system_loc,
            bbox_to_anchor=sys_bbox,
            frameon=True,
            facecolor="white",
            framealpha=float(args.system_alpha),
            edgecolor=args.system_edgecolor,
            handlelength=0,
            handletextpad=0.0,
            borderaxespad=0.5,
            fontsize=(float(args.system_fontsize) if args.system_fontsize is not None else None),
        )
        _legend_common_style(sys_leg, bold=bool(args.bold_fonts))
        sys_leg.set_zorder(1001)
        ax.add_artist(sys_leg)

    if not args.no_legend:
        legend_bbox = _parse_xy(args.legend_bbox)
        leg = ax.legend(
            loc=args.legend_loc,
            bbox_to_anchor=legend_bbox,
            ncol=int(args.legend_ncol),
            frameon=True,
            facecolor="white",
            framealpha=float(args.legend_alpha),
            edgecolor=args.legend_edgecolor,
            handletextpad=float(args.legend_handletextpad),
            handlelength=float(args.legend_handlelength),
            borderaxespad=0.5,
            fontsize=(float(args.legend_fontsize) if args.legend_fontsize is not None else None),
        )
        _legend_common_style(leg, bold=bool(args.bold_fonts))
        leg.set_zorder(1000)

    fig.tight_layout()
    fig.savefig(args.out, dpi=float(args.save_dpi))
    print(f"绘图完成！图片已保存为 {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
