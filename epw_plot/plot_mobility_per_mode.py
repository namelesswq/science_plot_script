#!/usr/bin/env python3
"""Scatter plot for EPW mobility_nk.fmt (state-resolved mobility).

This script is intentionally separate from the inv_tau plotting script.

Input format (whitespace-separated), with optional comment lines starting with '#'.
The file is typically organized as 3 lines per state (3x3 tensor):

  kpt band isym kx ky kz  (Energy-ef)[eV]  mu_xx mu_xy mu_xz
                              mu_yx mu_yy mu_yz
                              mu_zx mu_zy mu_zz

x-axis: the column "Energy - ef (eV)" already in the file.
        If your EPW run used an incorrect Fermi level, you can apply a per-file
        correction via --ef-shift (in eV):

            x_corrected = x_file - ef_shift
y-axis: one tensor component selected by --mu-dir (xx/yy/zz/avg/in-plane/out-of-plane).

Examples:
  python plot_mobility_nk_scatter.py mobility_nk.fmt --mu-dir xx --out mu_xx.png
  python plot_mobility_nk_scatter.py a.fmt b.fmt --mu-dir avg --legend A B --color C0 C3 --out cmp.png
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np


_RE_FORTRAN_MISSING_E = re.compile(
    r"^([+-]?(?:\d+\.?\d*|\d*\.?\d+))([+-]\d{2,4})$"
)


def _parse_float(token: str, *, path: Path, raw_line: str) -> float:
    """Parse a float token robustly.

    EPW outputs are sometimes written with fixed-width Fortran formats.
    When the field is too narrow, the exponent marker 'E' can be omitted,
    producing tokens like '-0.25830362-256' meaning '-0.25830362E-256'.
    """

    s = token.strip().replace("D", "E").replace("d", "E")
    try:
        return float(s)
    except ValueError:
        m = _RE_FORTRAN_MISSING_E.match(s)
        if m:
            try:
                return float(f"{m.group(1)}E{m.group(2)}")
            except ValueError as exc:  # pragma: no cover
                raise ValueError(
                    f"解析失败（浮点数格式异常）: {path} token={token!r} 行: {raw_line.rstrip()}"
                ) from exc
        raise ValueError(
            f"解析失败（无法解析为浮点数）: {path} token={token!r} 行: {raw_line.rstrip()}"
        )


def _try_set_style() -> None:
    """Apply scienceplots PRL-like style; fall back if LaTeX/PRL style unavailable."""
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("需要安装 scienceplots：pip install scienceplots") from exc

    available = set(plt.style.available)
    latex_ok = shutil.which("latex") is not None

    base_styles: list[str] = []
    if "science" in available:
        base_styles.append("science")
    if "prl" in available:
        base_styles.append("prl")
    elif "aps" in available:
        base_styles.append("aps")

    if not base_styles:
        base_styles = ["default"]

    styles_to_use = list(base_styles)
    if not latex_ok and "no-latex" in available:
        styles_to_use.append("no-latex")

    try:
        plt.style.use(styles_to_use)
    except Exception:
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
    loc = (loc or "").strip()
    mapping = {
        "upper": "upper right",
        "lower": "lower right",
        "left": "center left",
        "right": "center right",
    }
    return mapping.get(loc, loc)


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


def _broadcast_or_error_float_optional(values: list[float] | None, n: int, what: str) -> list[float] | None:
    if values is None:
        return None
    if len(values) == 0:
        return None
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
    if end < start:
        ticks = np.asarray([start])
    else:
        ticks = np.arange(start, end + 0.5 * step, step)
    ticks = np.round(ticks, 12)
    ticks[np.isclose(ticks, 0.0)] = 0.0
    return ticks


def _next_data_row(lines_iter, path: Path) -> list[float]:
    for raw_line in lines_iter:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        return [
            _parse_float(parts[0], path=path, raw_line=raw_line),
            _parse_float(parts[1], path=path, raw_line=raw_line),
            _parse_float(parts[2], path=path, raw_line=raw_line),
        ]
    raise ValueError(f"文件意外结束，缺少 mobility 张量行: {path}")


def read_mobility_nk_fmt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return x=(Energy-ef)[eV] and mu_tensor[cm^2/Vs] as shape (N,3,3)."""

    xs: list[float] = []
    tensors: list[np.ndarray] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        it = iter(f)
        for raw_line in it:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # First line of a 3-line block should have at least:
            # kpt band isym kx ky kz energy m11 m12 m13
            if len(parts) < 10:
                # Stray continuation line or malformed line; ignore.
                continue

            try:
                x = _parse_float(parts[6], path=path, raw_line=raw_line)
                row1 = [
                    _parse_float(parts[7], path=path, raw_line=raw_line),
                    _parse_float(parts[8], path=path, raw_line=raw_line),
                    _parse_float(parts[9], path=path, raw_line=raw_line),
                ]
            except ValueError:
                continue

            row2 = _next_data_row(it, path)
            row3 = _next_data_row(it, path)

            xs.append(x)
            tensors.append(np.asarray([row1, row2, row3], dtype=float))

    if not xs:
        raise ValueError(f"未能从文件中读到 mobility 数据: {path}")

    return np.asarray(xs, dtype=float), np.stack(tensors, axis=0)


def select_mu_component(mu: np.ndarray, mu_dir: str) -> np.ndarray:
    """Select y-values from mu tensor.

    mu: shape (N,3,3)
    """

    mu_dir = mu_dir.lower().strip()
    if mu_dir == "xx":
        return mu[:, 0, 0]
    if mu_dir == "yy":
        return mu[:, 1, 1]
    if mu_dir == "zz":
        return mu[:, 2, 2]
    if mu_dir in {"in-plane", "in_plane", "inplane"}:
        return (mu[:, 0, 0] + mu[:, 1, 1]) / 2.0
    if mu_dir in {"out-of-plane", "out_of_plane", "outofplane"}:
        return mu[:, 2, 2]
    if mu_dir == "avg":
        return (mu[:, 0, 0] + mu[:, 1, 1] + mu[:, 2, 2]) / 3.0
    raise ValueError(f"未知 mu-dir: {mu_dir}（支持 xx/yy/zz/avg/in-plane/out-of-plane）")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "将 EPW mobility_nk.fmt 绘制为散点图：x=Energy-ef(eV)（文件自带），"
            "y=mu_nk 的指定方向 (cm^2/Vs)。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "data",
        type=Path,
        nargs="+",
        help="输入文件路径（可多个，用于对比绘图，如 mobility1.fmt mobility2.fmt）",
    )

    p.add_argument(
        "--mu-dir",
        type=str,
        default="avg",
        choices=["xx", "yy", "zz", "avg", "in-plane", "out-of-plane"],
        help="选择 mobility 张量方向",
    )

    p.add_argument(
        "--ef-shift",
        nargs="*",
        type=float,
        default=None,
        help=(
            "对 x 轴 (Energy-ef) 做费米能修正（单位 eV）：x_new = x_old - ef_shift。"
            "可多个，分别对应每个输入文件；给 1 个会自动广播到所有文件"
        ),
    )

    p.add_argument("--out", type=Path, default=Path("mobility_scatter.pdf"), help="输出图片路径")
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
        help="y 轴显示范围，例如：--ylim 0 2000",
    )

    p.add_argument(
        "--xtick",
        nargs="*",
        default=None,
        help=(
            "控制 x 轴刻度：给 1 个数表示间距(step)，如 --xtick 0.1；"
            "给多个数表示刻度位置，如 --xtick -0.2 0 0.2；"
            "或用 --xtick none 关闭刻度"
        ),
    )
    p.add_argument(
        "--ytick",
        nargs="*",
        default=None,
        help=(
            "控制 y 轴刻度：给 1 个数表示间距(step)，如 --ytick 50；"
            "给多个数表示刻度位置，如 --ytick 0 50 100；"
            "或用 --ytick none 关闭刻度"
        ),
    )

    # Legend controls (only shown when --legend is given)
    p.add_argument(
        "--legend",
        nargs="*",
        default=None,
        help="图例文字（可多个，分别对应每个输入文件；不提供则不画 legend）",
    )
    p.add_argument("--legend-loc", type=str, default="best", help="legend 位置")
    p.add_argument(
        "--legend-bbox",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=None,
        help="legend 的 bbox_to_anchor（轴坐标 0~1），例如：--legend-bbox 0.02 0.98",
    )
    p.add_argument("--legend-fontsize", type=float, default=8.0, help="legend 字体大小")
    p.add_argument(
        "--legend-framealpha",
        type=float,
        default=0.0,
        help="legend 边框透明度(0=无边框；>0 显示边框)",
    )

    p.add_argument(
        "--label-fontsize",
        type=float,
        default=None,
        help="坐标轴标签字体大小（同时设置刻度值字体大小）",
    )

    return p


def main(argv: list[str]) -> int:
    args = build_argparser().parse_args(argv)

    nfiles = len(args.data)

    for pth in args.data:
        if not pth.exists():
            print(f"输入文件不存在: {pth}", file=sys.stderr)
            return 2

    try:
        legends = _broadcast_or_error(args.legend, nfiles, "legend")
        colors = None
        if args.color is not None:
            colors = _broadcast_or_error(list(args.color), nfiles, "color")
        ef_shifts = _broadcast_or_error_float_optional(
            list(args.ef_shift) if args.ef_shift is not None else None, nfiles, "ef-shift"
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if ef_shifts is None:
        ef_shifts = [0.0] * nfiles

    _try_set_style()
    import matplotlib.pyplot as plt

    figsize = tuple(args.figsize) if args.figsize else None
    fig, ax = plt.subplots(figsize=figsize)

    for idx, pth in enumerate(args.data):
        x, mu = read_mobility_nk_fmt(pth)
        x = x - float(ef_shifts[idx])
        y = select_mu_component(mu, args.mu_dir)

        if args.ylog:
            mask = y > 0
            if not np.all(mask):
                dropped = int(np.size(y) - np.count_nonzero(mask))
                print(f"提示: {pth} 有 {dropped} 个 y<=0 点，ylog 下已忽略", file=sys.stderr)
                x = x[mask]
                y = y[mask]

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
    ax.set_ylabel(r"$\mu_{nk}$ (cm$^2$/Vs)", fontsize=args.label_fontsize)

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

    try:
        xtick_mode, xtick_payload = _parse_tick_control(args.xtick)
        ytick_mode, ytick_payload = _parse_tick_control(args.ytick)

        if xtick_mode == "off":
            ax.set_xticks([])
        elif xtick_mode == "explicit":
            ax.set_xticks(xtick_payload)  # type: ignore[arg-type]
        elif xtick_mode == "step":
            lo, hi = ax.get_xlim()
            ax.set_xticks(_ticks_from_step(lo, hi, float(xtick_payload)))

        if ytick_mode == "off":
            ax.set_yticks([])
        elif ytick_mode == "explicit":
            ax.set_yticks(ytick_payload)  # type: ignore[arg-type]
        elif ytick_mode == "step":
            if args.ylog:
                print(
                    "y 轴为 log 时不支持用 --ytick <step> 生成线性间距刻度；请改用显式 --ytick v1 v2 ...",
                    file=sys.stderr,
                )
                return 2
            lo, hi = ax.get_ylim()
            ax.set_yticks(_ticks_from_step(lo, hi, float(ytick_payload)))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.bold_font:
        apply_bold_font(ax)

    if legends is not None:
        legend_framealpha = args.legend_framealpha
        frameon = legend_framealpha > 0.0

        legend_kwargs = {
            "loc": _normalize_legend_loc(args.legend_loc),
            "fontsize": args.legend_fontsize,
        }
        if args.legend_bbox is not None:
            legend_kwargs["bbox_to_anchor"] = tuple(args.legend_bbox)
            legend_kwargs["bbox_transform"] = ax.transAxes
            if str(legend_kwargs["loc"]).lower() == "best":
                legend_kwargs["loc"] = "upper left"

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
