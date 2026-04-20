#!/usr/bin/env python3
"""Scatter plot for EPW IBTEvel_sup.fmt (state-resolved velocities).

Input format (whitespace-separated), with optional comment lines starting with '#':

  # Number of elements in hole and electrons
    ind_tot  ind_totcb
  # itemp    ef0    efcb
    itemp  ef0[Ry]  efcb[Ry]
    ... (nstemp lines)
  # ik  ibnd      velocity (x,y,z)              eig     weight
    ik  ibnd  vx  vy  vz  eig[Ry]  weight

x-axis: E - E_F (eV), computed from eig[Ry] and ef0/efcb[Ry] (from file header).
    Use --carrier hole/electron to choose ef0 or efcb.

y-axis: one velocity component selected by --v-dir (x/y/z/avg).
  - avg is the magnitude |v|.

Units:
  EPW comments indicate v is stored in units of (Ry * bohr).
  Physical velocity is v_SI = v * (Ry_to_J * bohr_to_m / hbar) in m/s.
  This script defaults to plotting m/s (set --vel-unit rybohr to plot raw).

Examples:
    python plot_ibtevel_scatter.py IBTEvel_sup.fmt --v-dir avg --out vmag.png
  python plot_ibtevel_scatter.py a.fmt b.fmt --v-dir x --legend A B --color C0 C3 --out vx_cmp.pdf
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np


RY_TO_EV = 13.605693122994  # eV / Ry
_E_CHARGE = 1.602176634e-19  # C
RY_TO_J = RY_TO_EV * _E_CHARGE  # J / Ry
BOHR_TO_M = 5.29177210903e-11  # m / bohr
HBAR_SI = 1.054571817e-34  # J*s

# (m/s) per (Ry * bohr)
RY_BOHR_TO_M_S = (RY_TO_J * BOHR_TO_M) / HBAR_SI

_RE_FORTRAN_MISSING_E = re.compile(r"^([+-]?(?:\d+\.?\d*|\d*\.?\d+))([+-]\d{2,4})$")


def _parse_float(token: str, *, path: Path, raw_line: str) -> float:
    """Parse a float token robustly.

    Some fixed-width Fortran outputs can omit the exponent marker 'E', producing
    tokens like '-0.25830362-256' meaning '-0.25830362E-256'.
    """

    s = token.strip().replace("D", "E").replace("d", "E")
    try:
        return float(s)
    except ValueError:
        m = _RE_FORTRAN_MISSING_E.match(s)
        if m:
            return float(f"{m.group(1)}E{m.group(2)}")
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


def _broadcast_or_error_float_optional(
    values: list[float] | None, n: int, what: str
) -> list[float] | None:
    if values is None:
        return None
    if len(values) == 0:
        return None
    if len(values) == 1 and n > 1:
        return values * n
    if len(values) != n:
        raise ValueError(f"{what} 数量({len(values)})与输入文件数量({n})不一致")
    return values


def _require_one_per_file(values: list[float] | None, n: int, what: str) -> list[float] | None:
    """Stricter broadcast rule: if provided, must match number of files."""

    if values is None:
        return None
    if len(values) != n:
        raise ValueError(f"{what} 必须为每个输入文件分别提供（需要 {n} 个，但给了 {len(values)} 个）")
    return values


def _parse_tick_control(values: list[str] | None) -> tuple[str, list[float] | float | None]:
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


def _parse_v_dir(v_dir: str) -> str:
    s = (v_dir or "").strip().lower()
    mapping = {
        "vx": "x",
        "vy": "y",
        "vz": "z",
        "x": "x",
        "y": "y",
        "z": "z",
        "avg": "avg",
        "abs": "avg",
        "mag": "avg",
        "norm": "avg",
        "|v|": "avg",
    }
    if s in mapping:
        return mapping[s]
    raise ValueError("未知 v-dir（支持 x/y/z/avg）")


def read_ibtevel_sup_fmt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (eig_Ry, vx, vy, vz) arrays from the data block."""

    eigs: list[float] = []
    vxs: list[float] = []
    vys: list[float] = []
    vzs: list[float] = []

    mode = "seek_header"

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                if line.lower().startswith("# ik"):
                    mode = "read_data"
                continue

            if mode == "read_data":
                parts = line.split()
                # Expected: ik ibnd vx vy vz eig weight
                if len(parts) < 7:
                    continue
                try:
                    vx = _parse_float(parts[2], path=path, raw_line=raw_line)
                    vy = _parse_float(parts[3], path=path, raw_line=raw_line)
                    vz = _parse_float(parts[4], path=path, raw_line=raw_line)
                    eig = _parse_float(parts[5], path=path, raw_line=raw_line)
                except ValueError:
                    continue

                vxs.append(vx)
                vys.append(vy)
                vzs.append(vz)
                eigs.append(eig)

    if not eigs:
        raise ValueError(f"未能从文件中读到 velocity 数据: {path}")

    return (
        np.asarray(eigs, dtype=float),
        np.asarray(vxs, dtype=float),
        np.asarray(vys, dtype=float),
        np.asarray(vzs, dtype=float),
    )


def read_fermi_from_header(path: Path) -> dict[int, tuple[float, float]]:
    """Return {itemp: (ef0_Ry, efcb_Ry)} from the header."""

    fermi_by_itemp: dict[int, tuple[float, float]] = {}
    mode = "seek"
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "itemp" in line.lower() and "ef0" in line.lower():
                    mode = "read"
                elif line.lower().startswith("# ik"):
                    break
                continue
            if mode == "read":
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        itemp = int(float(parts[0]))
                        ef0 = _parse_float(parts[1], path=path, raw_line=raw_line)
                        efcb = _parse_float(parts[2], path=path, raw_line=raw_line)
                        fermi_by_itemp[itemp] = (ef0, efcb)
                    except Exception:
                        pass
    return fermi_by_itemp


def _fermi_ry_for_itemp(path: Path, itemp: int, carrier: str) -> float:
    carrier = (carrier or "").strip().lower()
    if carrier not in {"hole", "electron"}:
        raise ValueError("carrier 必须是 hole 或 electron")

    fermi_by_itemp = read_fermi_from_header(path)
    if itemp in fermi_by_itemp:
        ef0, efcb = fermi_by_itemp[itemp]
    elif fermi_by_itemp:
        fallback = sorted(fermi_by_itemp.keys())[0]
        print(f"提示: {path} 中未找到 itemp={itemp}，已改用 itemp={fallback}", file=sys.stderr)
        ef0, efcb = fermi_by_itemp[fallback]
    else:
        print(f"提示: {path} header 中未读到 ef0/efcb，默认 Ef=0", file=sys.stderr)
        ef0, efcb = 0.0, 0.0

    ef = ef0 if carrier == "hole" else efcb
    if carrier == "electron" and np.isclose(ef, 0.0):
        print(f"提示: {path} 的 efcb=0；若你在算电子侧，请确认 EPW 是否输出了 efcb", file=sys.stderr)
    return float(ef)


def select_velocity_component(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, v_dir: str) -> np.ndarray:
    v_dir = _parse_v_dir(v_dir)
    if v_dir == "x":
        return vx
    if v_dir == "y":
        return vy
    if v_dir == "z":
        return vz
    # magnitude
    return np.sqrt(vx * vx + vy * vy + vz * vz)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "将 EPW IBTEvel_sup.fmt 绘制为散点图：x=E-Ef(eV)，y=群速度分量。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "data",
        type=Path,
        nargs="+",
        help="输入文件路径（可多个，用于对比绘图，如 IBTEvel1.fmt IBTEvel2.fmt）",
    )

    p.add_argument(
        "--v-dir",
        type=str,
        default="avg",
        help="选择速度分量：x/y/z/avg（avg=|v|）",
    )

    p.add_argument(
        "--carrier",
        type=str,
        default="hole",
        choices=["hole", "electron"],
        help="选择载流子类型：hole 使用 ef0；electron 使用 efcb（都来自文件 header）",
    )

    p.add_argument(
        "--itemp",
        type=int,
        default=1,
        help="选择 header 中哪一个 itemp 的 ef0/efcb 用于计算 E-Ef（仅在未提供 --ef 时生效）",
    )

    p.add_argument(
        "--ef",
        nargs="+",
        type=float,
        default=None,
        help="手动指定费米能级 E_f（eV）；可多个，分别对应每个输入文件；若不提供则使用文件 header 的 ef0",
    )

    p.add_argument(
        "--ry-to-ev",
        type=float,
        default=RY_TO_EV,
        help="Ry 到 eV 的换算因子",
    )

    p.add_argument(
        "--vel-unit",
        type=str,
        default="ms",
        choices=["ms", "rybohr"],
        help=(
            "速度单位：ms=换算到 m/s；rybohr=使用 EPW 输出的原始单位 (Ry*bohr)"
        ),
    )

    p.add_argument("--out", type=Path, default=Path("velocity_scatter.pdf"), help="输出图片路径")
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
        help="y 轴显示范围，例如：--ylim 0 2e6",
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
            "控制 y 轴刻度：给 1 个数表示间距(step)，如 --ytick 1e5；"
            "给多个数表示刻度位置，如 --ytick 0 5e5 1e6；"
            "或用 --ytick none 关闭刻度"
        ),
    )

    # Legend controls
    p.add_argument(
        "--legend",
        nargs="*",
        default=None,
        help="图例文字（可多个，分别对应每个输入文件；不提供则不画 legend）",
    )
    # Backward compatibility with old scripts.
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
        "--legend-frameon",
        action="store_true",
        help="(兼容旧参数) legend 显示边框；等价于 --legend-framealpha 1.0",
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
        # Legend texts
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

        # Important: if --ef is provided, user must give one ef per file.
        ef_overrides = _require_one_per_file(
            list(args.ef) if args.ef is not None else None, nfiles, "ef"
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    _try_set_style()
    import matplotlib.pyplot as plt

    figsize = tuple(args.figsize) if args.figsize else None
    fig, ax = plt.subplots(figsize=figsize)

    for idx, pth in enumerate(args.data):
        if ef_overrides is None:
            ef0_ry = _fermi_ry_for_itemp(pth, args.itemp, args.carrier)

        eig_ry, vx, vy, vz = read_ibtevel_sup_fmt(pth)

        # x: E - Ef (eV)
        energy_ev = eig_ry * args.ry_to_ev
        if ef_overrides is None:
            x = (eig_ry - ef0_ry) * args.ry_to_ev
        else:
            x = energy_ev - float(ef_overrides[idx])

        # y: velocities
        if args.vel_unit == "ms":
            vx = vx * RY_BOHR_TO_M_S
            vy = vy * RY_BOHR_TO_M_S
            vz = vz * RY_BOHR_TO_M_S

        y = select_velocity_component(vx, vy, vz, args.v_dir)

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

    v_dir = _parse_v_dir(args.v_dir)
    if args.vel_unit == "ms":
        if v_dir == "avg":
            ax.set_ylabel(r"$|v_{nk}|$ (m/s)", fontsize=args.label_fontsize)
        else:
            ax.set_ylabel(rf"$v_{{nk,{v_dir}}}$ (m/s)", fontsize=args.label_fontsize)
    else:
        if v_dir == "avg":
            ax.set_ylabel(r"$|v_{nk}|$ (Ry·bohr)", fontsize=args.label_fontsize)
        else:
            ax.set_ylabel(rf"$v_{{nk,{v_dir}}}$ (Ry·bohr)", fontsize=args.label_fontsize)

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
        if args.legend_frameon and args.legend_framealpha == 0.0:
            legend_framealpha = 1.0

        legend_kwargs = {
            "loc": _normalize_legend_loc(args.legend_loc),
            "fontsize": args.legend_fontsize,
        }
        if args.legend_bbox is not None:
            legend_kwargs["bbox_to_anchor"] = tuple(args.legend_bbox)
            legend_kwargs["bbox_transform"] = ax.transAxes
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
