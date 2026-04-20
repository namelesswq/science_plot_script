#!/usr/bin/env python3
"""Scatter plot for EPW inv_tau.fmt (or similar).

Input format (whitespace-separated), with optional comment lines starting with '#':
    itemp  kpt  ibnd  energy[Ry]  inv_tau[Ry]

The EPW header commonly states:
    Multiply the relaxation time by 20670.6944033 to get 1/ps
Despite the wording, this factor converts a quantity in Ry (energy / broadening)
into a scattering rate in ps^-1.

Example:
    python plot_inv_tau_scatter.py inv_tau.fmt --ef 0.25 --out rate.pdf --legend "holes"
    python plot_inv_tau_scatter.py a.fmt b.fmt --ef 0.25 --legend holes electrons --color C0 C3
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np


RY_TO_EV = 13.605693122994  # eV / Ry
RY_TO_PS_INV_DEFAULT = 20670.6944033  # (ps^-1) / Ry, as given in EPW header


def _try_set_style() -> None:
    """Apply scienceplots PRL style; fall back gracefully if LaTeX is unavailable."""
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "需要安装 scienceplots：pip install scienceplots"
        ) from exc

    available = set(plt.style.available)

    latex_ok = shutil.which("latex") is not None

    # scienceplots versions vary. Prefer PRL if available; otherwise fall back.
    base_styles: list[str] = []
    if "science" in available:
        base_styles.append("science")
    if "prl" in available:
        base_styles.append("prl")
    elif "aps" in available:
        base_styles.append("aps")

    if not base_styles:
        # Should not happen if scienceplots imported correctly, but keep robust.
        base_styles = ["default"]

    # scienceplots' "science" style often enables usetex; auto-disable if latex is absent.
    styles_to_use = list(base_styles)
    if not latex_ok and "no-latex" in available:
        styles_to_use.append("no-latex")

    try:
        plt.style.use(styles_to_use)
    except Exception:
        # Last resort: use base styles and force non-TeX rendering.
        plt.style.use(base_styles)
        import matplotlib as mpl

        mpl.rcParams["text.usetex"] = False

    if not latex_ok:
        import matplotlib as mpl

        mpl.rcParams["text.usetex"] = False

    # If no dedicated PRL/APS style exists, apply a minimal PRL-like rcParams set.
    if "prl" not in available and "aps" not in available:
        import matplotlib as mpl

        mpl.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": [
                    "Times New Roman",
                    "Times",
                    "Nimbus Roman",
                    "DejaVu Serif",
                ],
                "mathtext.fontset": "stix",
                "font.size": 10,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 8,
                "axes.linewidth": 0.8,
            }
        )


def read_inv_tau_fmt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return energy_Ry and inv_tau_Ry arrays."""
    energies: list[float] = []
    inv_taus: list[float] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                energy_ry = float(parts[3])
                inv_tau_ry = float(parts[4])
            except ValueError:
                continue
            energies.append(energy_ry)
            inv_taus.append(inv_tau_ry)

    if not energies:
        raise ValueError(f"未能从文件中读到数据: {path}")

    return np.asarray(energies), np.asarray(inv_taus)


def apply_bold_font(ax) -> None:
    import matplotlib as mpl

    mpl.rcParams["font.weight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"
    mpl.rcParams["axes.titleweight"] = "bold"

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    if ax.xaxis.label is not None:
        ax.xaxis.label.set_fontweight("bold")
    if ax.yaxis.label is not None:
        ax.yaxis.label.set_fontweight("bold")

    # Also bold the scientific-notation offset text (e.g., ×10^6).
    try:
        ax.xaxis.get_offset_text().set_fontweight("bold")
    except Exception:
        pass
    try:
        ax.yaxis.get_offset_text().set_fontweight("bold")
    except Exception:
        pass


def _normalize_legend_loc(loc: str) -> str:
    """Accept a few convenient shorthands for legend loc."""
    loc = (loc or "").strip()
    mapping = {
        "upper": "upper right",
        "lower": "lower right",
        "left": "center left",
        "right": "center right",
    }
    return mapping.get(loc, loc)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="将 EPW inv_tau.fmt 绘制为散点图：x=E-Ef(eV)，y=散射率(ps^-1)。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "data",
        type=Path,
        nargs="+",
        help="输入文件路径（可多个，用于对比绘图，如 inv_tau1.fmt inv_tau2.fmt）",
    )
    p.add_argument(
        "--ef",
        nargs="+",
        type=float,
        required=True,
        help="费米能级 E_f（eV）；可多个，分别对应每个输入文件",
    )

    p.add_argument(
        "--ry-to-ev",
        type=float,
        default=RY_TO_EV,
        help="Ry 到 eV 的换算因子",
    )
    p.add_argument(
        "--ry-to-psinv",
        type=float,
        default=RY_TO_PS_INV_DEFAULT,
        help="把 inv_tau[Ry] 换算为散射率[ps^-1] 的因子",
    )

    p.add_argument("--out", type=Path, default=Path("inv_tau_scatter.pdf"), help="输出图片路径")
    p.add_argument("--dpi", type=int, default=300, help="输出分辨率（对 png 等有效）")
    p.add_argument("--show", action="store_true", help="弹窗显示图")

    p.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("W", "H"),
        default=None,
        help="画布大小 figsize，例如：--figsize 7 5",
    )

    p.add_argument("--bold-font", action="store_true", help="所有字体加粗")

    # Scatter appearance
    p.add_argument("--size", type=float, default=8.0, help="点大小 (matplotlib s)")
    p.add_argument(
        "--color",
        nargs="*",
        default=None,
        help="点颜色（可多个，分别对应每个输入文件；不提供则用默认循环色）",
    )
    p.add_argument("--alpha", type=float, default=0.8, help="点透明度")

    p.add_argument("--ylog", action="store_true", help="y 轴使用对数坐标")
    p.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        default=None,
        help="x 轴显示范围，例如：--xlim -1 1",
    )
    p.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        default=None,
        help="y 轴显示范围，例如：--ylim 1e-2 1e2",
    )

    p.add_argument(
        "--xtick",
        nargs="*",
        default=None,
        help='控制 x 轴刻度：给 1 个数表示间距(step)，如 --xtick 5；给多个数表示刻度位置，如 --xtick -1 0 1；或用 --xtick none 关闭刻度',
    )
    p.add_argument(
        "--ytick",
        nargs="*",
        default=None,
        help='控制 y 轴刻度：给 1 个数表示间距(step)，如 --ytick 50；给多个数表示刻度位置，如 --ytick 0 50 100；或用 --ytick none 关闭刻度',
    )

    # Legend controls (only shown when --label is given)
    p.add_argument(
        "--legend",
        nargs="*",
        default=None,
        help="图例文字（可多个，分别对应每个输入文件；不提供则不画 legend）",
    )
    # Backward compatibility: old single-label option.
    p.add_argument(
        "--label",
        type=str,
        default="",
        help="(兼容旧参数) 单个图例标签；等价于 --legend <label>",
    )
    p.add_argument("--legend-loc", type=str, default="best", help="legend 位置")
    p.add_argument(
        "--legend-bbox",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=None,
        help="legend 的 bbox_to_anchor，例如：--legend-bbox 0.02 0.98",
    )
    p.add_argument("--legend-fontsize", type=float, default=8.0, help="legend 字体大小")
    p.add_argument(
        "--legend-framealpha",
        type=float,
        default=0.0,
        help="legend 边框透明度(0=无边框；>0 显示边框)",
    )
    # Backward-compatible toggle; if set and framealpha not explicitly set, treat as 1.0.
    p.add_argument(
        "--legend-frameon",
        action="store_true",
        help="(兼容旧参数) legend 显示边框；等价于 --legend-framealpha 1.0",
    )

    p.add_argument(
        "--label-fontsize",
        type=float,
        default=None,
        help="坐标轴标签字体大小（xlabel/ylabel；同时设置刻度值字体大小）",
    )

    return p


def _broadcast_or_error(values: list[str] | None, n: int, what: str) -> list[str] | None:
    if values is None:
        return None
    if len(values) == 0:
        return None
    if len(values) == 1 and n > 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"{what} 数量({len(values)})与输入文件数量({n})不一致")
    return values


def _broadcast_or_error_float(values: list[float], n: int, what: str) -> list[float]:
    if len(values) == 0:
        raise ValueError(f"{what} 不能为空")
    if len(values) == 1 and n > 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"{what} 数量({len(values)})与输入文件数量({n})不一致")
    return values


def _parse_tick_control(values: list[str] | None) -> tuple[str, list[float] | float | None]:
    """Parse --xtick/--ytick.

    Returns (mode, payload):
      - ("auto", None): not provided
      - ("off", None): disable ticks
      - ("step", step_float): generate ticks by spacing
      - ("explicit", [t1, t2, ...]): explicit tick locations
    """

    if values is None:
        return "auto", None
    if len(values) == 0:
        return "auto", None
    if len(values) == 1 and str(values[0]).strip().lower() in {"none", "off", "false"}:
        return "off", None
    if len(values) == 1:
        return "step", float(values[0])
    return "explicit", [float(v) for v in values]


def _ticks_from_step(lo: float, hi: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("tick 间距必须为正数")
    if hi < lo:
        lo, hi = hi, lo

    # Floating-point robustness:
    # values like -0.3/0.1 may become -2.999999999..., making ceil() jump to -2.
    q_lo = lo / step
    q_hi = hi / step
    q_lo_round = np.round(q_lo)
    q_hi_round = np.round(q_hi)
    if np.isclose(q_lo, q_lo_round, rtol=0.0, atol=1e-10):
        q_lo = float(q_lo_round)
    if np.isclose(q_hi, q_hi_round, rtol=0.0, atol=1e-10):
        q_hi = float(q_hi_round)

    start = np.ceil(q_lo) * step
    end = np.floor(q_hi) * step
    # Handle cases where range is smaller than step
    if end < start:
        ticks = np.asarray([start])
    else:
        ticks = np.arange(start, end + 0.5 * step, step)
    # Avoid -0.0 and floating artifacts
    ticks = np.round(ticks, 12)
    ticks[np.isclose(ticks, 0.0)] = 0.0
    return ticks


def main(argv: list[str]) -> int:
    args = build_argparser().parse_args(argv)

    nfiles = len(args.data)

    for pth in args.data:
        if not pth.exists():
            print(f"输入文件不存在: {pth}", file=sys.stderr)
            return 2

    try:
        efs = _broadcast_or_error_float(list(args.ef), nfiles, "ef")

        legends: list[str] | None
        if args.legend is not None:
            legends = args.legend
        elif args.label:
            legends = [args.label]
        else:
            legends = None

        legends = _broadcast_or_error(legends, nfiles, "legend")

        colors = None
        if args.color is not None:
            colors = _broadcast_or_error(list(args.color), nfiles, "color")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    _try_set_style()
    import matplotlib.pyplot as plt

    figsize = tuple(args.figsize) if args.figsize else None
    fig, ax = plt.subplots(figsize=figsize)

    for idx, pth in enumerate(args.data):
        energy_ry, inv_tau_ry = read_inv_tau_fmt(pth)

        energy_ev = energy_ry * args.ry_to_ev
        x = energy_ev - efs[idx]

        # y: scattering rate in ps^-1
        y = inv_tau_ry * args.ry_to_psinv

        color_i = None if colors is None else colors[idx]
        label_i = None if legends is None else legends[idx]

        ax.scatter(
            x,
            y,
            s=args.size,
            c=color_i,
            alpha=args.alpha,
            label=label_i,
            linewidths=0,
        )

    ax.set_xlabel(r"$E - E_F$ (eV)", fontsize=args.label_fontsize)
    ax.set_ylabel(r"Scattering rate (ps$^{-1}$)", fontsize=args.label_fontsize)

    if args.label_fontsize is not None:
        ax.tick_params(axis="both", which="both", labelsize=args.label_fontsize)
        # scientific-notation offset text (e.g., ×10^6)
        ax.xaxis.get_offset_text().set_fontsize(args.label_fontsize)
        ax.yaxis.get_offset_text().set_fontsize(args.label_fontsize)

    if args.ylog:
        ax.set_yscale("log")

    if args.xlim is not None:
        ax.set_xlim(*args.xlim)
    if args.ylim is not None:
        ax.set_ylim(*args.ylim)

    xtick_mode, xtick_payload = _parse_tick_control(args.xtick)
    ytick_mode, ytick_payload = _parse_tick_control(args.ytick)

    if xtick_mode == "off":
        ax.set_xticks([])
    elif xtick_mode == "explicit":
        ax.set_xticks(xtick_payload)  # type: ignore[arg-type]
    elif xtick_mode == "step":
        lo, hi = ax.get_xlim()
        ticks = _ticks_from_step(lo, hi, float(xtick_payload))
        ax.set_xticks(ticks)

    if ytick_mode == "off":
        ax.set_yticks([])
    elif ytick_mode == "explicit":
        ax.set_yticks(ytick_payload)  # type: ignore[arg-type]
    elif ytick_mode == "step":
        if args.ylog:
            print("y 轴为 log 时不支持用 --ytick <step> 生成线性间距刻度；请改用显式 --ytick v1 v2 ...", file=sys.stderr)
            return 2
        lo, hi = ax.get_ylim()
        ticks = _ticks_from_step(lo, hi, float(ytick_payload))
        ax.set_yticks(ticks)

    if args.bold_font:
        apply_bold_font(ax)

    if legends is not None:
        legend_framealpha = args.legend_framealpha
        if args.legend_frameon and args.legend_framealpha == 0.0:
            legend_framealpha = 1.0

        legend_kwargs = {
            "loc": _normalize_legend_loc(args.legend_loc),
            "fontsize": args.legend_fontsize,
        }
        if args.legend_bbox is not None:
            # Interpret as axes-fraction coordinates to keep behavior consistent.
            legend_kwargs["bbox_to_anchor"] = tuple(args.legend_bbox)
            legend_kwargs["bbox_transform"] = ax.transAxes
            # loc='best' can ignore bbox_to_anchor; force a deterministic loc.
            if str(legend_kwargs["loc"]).lower() == "best":
                legend_kwargs["loc"] = "upper left"

        frameon = legend_framealpha > 0.0
        if frameon:
            legend_kwargs["frameon"] = True
            legend_kwargs["framealpha"] = legend_framealpha
        else:
            legend_kwargs["frameon"] = False

        ax.legend(
            **legend_kwargs,
            handletextpad=0.4,
            labelspacing=0.3,
            borderpad=0.3,
            handlelength=1.0,
        )

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
