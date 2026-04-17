#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import matplotlib.patheffects as patheffects


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


def _format_chem_label(label: str, mode: str) -> str:
    if not label:
        return label
    if mode == "raw":
        return label
    if "$" in label:
        return label
    import re

    return re.sub(r"(?<=[A-Za-z\)])(\d+)", r"$_{\1}$", label)


def _normalize_math_label(label: str, *, bold: bool) -> str:
    """Normalize legend/system labels that may contain raw math tokens.

    Users often pass labels like: "Pristine \\kappa_{latt,avg}" without surrounding '$'.
    Matplotlib will treat that as plain text unless we wrap it into mathtext.

    Rules:
    - If label already contains '$', keep it as-is.
    - If label contains LaTeX-like markers (\\, _{, ^{), wrap with '$...$'.
    - If bold=True, apply best-effort bolding for text prefix and '\\kappa'.
    """

    s = str(label)
    if not s:
        return s

    if "$" in s:
        # Already mathtext-mixed string like: "Pristine $\\kappa_{latt,avg}$".
        # When bold is requested, we must explicitly bold math symbols.
        if not bold:
            return s
        return _boldify_existing_mathtext(s)

    looks_math = ("\\" in s) or ("_{" in s) or ("^{" in s)
    if not looks_math:
        return s

    # Split into a plain-text prefix and a math tail at the first math marker.
    markers = []
    for ch in ("\\", "_", "^"):
        i = s.find(ch)
        if i >= 0:
            markers.append(i)
    cut = min(markers) if markers else 0

    prefix = s[:cut].strip()
    tail = s[cut:].strip()

    def _tex_escape_spaces(t: str) -> str:
        return t.replace(" ", r"\ ")

    tex_parts = []
    if prefix:
        p = _tex_escape_spaces(prefix)
        tex_parts.append((r"\mathbf{" + p + "}") if bold else (r"\mathrm{" + p + "}"))
    if tail:
        if prefix:
            tex_parts.append(r"\ ")
        tex_parts.append(_tex_escape_spaces(tail))

    tex = "".join(tex_parts)

    # Romanize (and optionally bold) simple subscripts like _{latt,avg}.
    # Skip if the subscript already contains a command (e.g. \mathrm).
    import re

    def _wrap_sub(m: re.Match) -> str:
        inner = m.group(1)
        if "\\" in inner:
            return m.group(0)
        inner2 = inner
        if bold:
            inner2 = r"\mathbf{" + inner2 + "}"
        else:
            inner2 = r"\mathrm{" + inner2 + "}"
        return "_{" + inner2 + "}"

    tex = re.sub(r"_\{([^}]*)\}", _wrap_sub, tex)

    if bold:
        # Greek letters in mathtext are not reliably bolded by text weight.
        tex = tex.replace(r"\kappa", r"\boldsymbol{\kappa}")

    return "$" + tex + "$"


def _boldify_existing_mathtext(label: str) -> str:
    """Best-effort bolding for strings containing one or more $...$ mathtext segments."""

    import re

    s = str(label)

    def _boldify_math(expr: str) -> str:
        t = str(expr)

        # Bold greek kappa.
        if r"\boldsymbol" not in t:
            t = t.replace(r"\kappa", r"\boldsymbol{\kappa}")

        # If someone already wrote \mathrm{latt}, make it bold.
        t = t.replace(r"\mathrm{latt}", r"\mathbf{latt}")
        t = t.replace(r"\mathrm{el}", r"\mathbf{el}")
        t = t.replace(r"\mathrm{avg}", r"\mathbf{avg}")

        # Wrap plain subscripts/superscripts in \mathbf{...} if they don't contain commands.
        def _wrap_sub(m: re.Match) -> str:
            inner = m.group(1)
            if "\\" in inner:
                return m.group(0)
            return "_{" + r"\mathbf{" + inner + "}" + "}"

        def _wrap_sup(m: re.Match) -> str:
            inner = m.group(1)
            if "\\" in inner:
                return m.group(0)
            return "^{" + r"\mathbf{" + inner + "}" + "}"

        t = re.sub(r"_\{([^}]*)\}", _wrap_sub, t)
        t = re.sub(r"\^\{([^}]*)\}", _wrap_sup, t)
        return t

    # Replace each math segment.
    def _repl(m: re.Match) -> str:
        expr = m.group(1)
        return "$" + _boldify_math(expr) + "$"

    return re.sub(r"\$(.+?)\$", _repl, s)


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


def _broadcast_str_list(xs: list[str], n: int, name: str) -> list[str]:
    if n <= 0:
        return []
    if len(xs) == n:
        return list(xs)
    if len(xs) == 1:
        return [str(xs[0])] * n
    raise SystemExit(f"{name} expects 1 value or {n} values, but got {len(xs)}")


def _normalize_linestyle_token(s: str) -> str:
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


def _extend_to_xlim(x: np.ndarray, y: np.ndarray, *, xlim: Optional[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    if x.size < 1:
        return x, y
    x_end = None
    if xlim is not None:
        x_end = float(xlim[1])
    else:
        x_end = float(x[-1]) * 1.2

    if np.isfinite(x_end) and x_end > float(x[-1]):
        x2 = np.concatenate([np.asarray(x, float), np.array([x_end], float)])
        y2 = np.concatenate([np.asarray(y, float), np.array([float(y[-1])], float)])
        return x2, y2
    return x, y


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Overlay cumulative lattice and electronic thermal conductivity vs mean free path on one figure.\n\n"
            "- Lattice (ShengBTE): reads BTE.cumulative_kappa_tensor (mfp_nm + 9 tensor comps).\n"
            "- Electronic (Perturbo): computes cumulative kappa from meanfp.yml + tdf.h5 + tet.h5 (mfd+tetr path)."
        )
    )

    p.add_argument(
        "--style",
        choices=["prb", "default"],
        default="prb",
        help="Plot style preset. 'prb' uses SciencePlots (science,no-latex). Default: prb.",
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
        "--label-fontsize",
        type=float,
        default=None,
        help="Font size for axis labels and tick labels. If omitted, keep defaults/style behavior.",
    )

    p.add_argument(
        "--bold-fonts",
        action="store_true",
        help="Force all text in the figure to bold (including for --style prb).",
    )

    # ShengBTE (lattice)
    p.add_argument(
        "--latt-file",
        required=True,
        nargs="+",
        help="One or more ShengBTE BTE.cumulative_kappa_tensor files to overlay.",
    )
    p.add_argument(
        "--latt-component",
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
        help="Lattice tensor component to plot. Default: avg.",
    )
    p.add_argument(
        "--latt-label",
        default=["latt"],
        nargs="+",
        help="Legend label(s) for lattice curve(s). Provide 1 (broadcast) or Nlatt values.",
    )

    # Perturbo (electronic)
    p.add_argument("--el-meanfp", required=True, help="Path to Perturbo *_meanfp.yml")
    p.add_argument("--el-tdf", required=True, help="Path to Perturbo *_tdf.h5")
    p.add_argument(
        "--el-tet",
        default=None,
        help="Path to Perturbo *_tet.h5. If omitted, inferred from tdf by replacing 'tdf' -> 'tet'.",
    )
    p.add_argument(
        "--el-config",
        type=int,
        default=1,
        help="Perturbo config index (temperature point). Default: 1.",
    )
    p.add_argument(
        "--el-dir",
        choices=["xx", "yy", "zz", "avg"],
        default="avg",
        help="Electronic transport direction. Default: avg.",
    )
    p.add_argument("--el-label", default="el", help="Legend label for electronic curve.")

    p.add_argument(
        "--el-kappa-calib",
        choices=["trans_coef", "trans_ita", "none"],
        default="none",
        help="Optional calibration for electronic kappa (same as perturbo_plot script). Default: none.",
    )
    p.add_argument("--el-trans-coef", default=None, help="Path to reference *.trans_coef (optional).")
    p.add_argument("--el-trans-ita", default=None, help="Path to reference *_trans-ita.yml (optional).")
    p.add_argument(
        "--spin-deg",
        type=float,
        default=2.0,
        help="Spin degeneracy factor for electronic part. Default: 2.",
    )

    # Plot controls (minimal but consistent)
    p.add_argument("--xlog", action=argparse.BooleanOptionalAction, default=True, help="Use log scale on x axis.")
    p.add_argument("--ylog", action="store_true", help="Use log scale on y-axis")
    p.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"), help="Set x limits.")
    p.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"), help="Set y limits.")

    p.add_argument("--xlabel", default="Mean Free Path (nm)", help="x-axis label")
    p.add_argument("--ylabel", default=r"Cumulative $\kappa$ (W/mK)", help="y-axis label")

    p.add_argument("--figsize", default=None, help='Figure size "width,height" in inches (e.g. "7.5,5")')
    p.add_argument("--lw", type=float, default=2.2, help="Line width")
    p.add_argument(
        "--ls-latt",
        default=["solid"],
        nargs="+",
        help=(
            "Line style(s) for lattice curve(s). Provide 1 (broadcast) or Nlatt values. "
            "Recommended: 'solid', 'dashed', 'dotted', 'dashdot'. "
            "If you really want raw tokens like '-' or '-.', pass them with '=' to avoid argparse confusion: "
            "--ls-latt=- --ls-latt=-."
        ),
    )
    p.add_argument("--ls-el", default="--", help="Line style for electronic curve")
    p.add_argument(
        "--color-latt",
        default=None,
        nargs="+",
        help="Color(s) for lattice curve(s). Provide 1 (broadcast) or Nlatt values.",
    )
    p.add_argument("--color-el", default=None, help="Color for electronic curve")

    p.add_argument(
        "--legend-loc",
        default=["best"],
        nargs="+",
        help="Legend location (matplotlib). Example: --legend-loc upper left. Default: best.",
    )
    p.add_argument(
        "--legend-fontsize",
        type=float,
        default=None,
        help="Font size for legend text. If omitted, uses matplotlib default.",
    )
    p.add_argument(
        "--legend-format",
        choices=["chem", "raw"],
        default="raw",
        help="Render legend labels with subscripts (chem) or raw text (raw). Default: raw.",
    )

    p.add_argument(
        "--legend-handlelength",
        type=float,
        default=1.4,
        help="Legend line length (handlelength). Default: 1.4.",
    )
    p.add_argument(
        "--legend-handletextpad",
        type=float,
        default=0.4,
        help="Spacing between legend line and text (handletextpad). Default: 0.4.",
    )
    p.add_argument(
        "--legend-labelspacing",
        type=float,
        default=0.25,
        help="Vertical spacing between legend entries (labelspacing). Default: 0.25.",
    )
    p.add_argument(
        "--legend-borderpad",
        type=float,
        default=0.3,
        help="Legend internal padding (borderpad). Default: 0.3.",
    )
    p.add_argument(
        "--legend-borderaxespad",
        type=float,
        default=0.2,
        help="Legend padding from axes (borderaxespad). Default: 0.2.",
    )
    p.add_argument("--legend-bbox", default=None, help="Optional legend bbox_to_anchor in axes coords 'x,y'.")
    p.add_argument("--legend-alpha", type=float, default=None, help="Legend frame alpha (0..1).")

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
        default=["upper", "left"],
        nargs="+",
        help="Legend location for --system (matplotlib legend loc). Default: upper left.",
    )
    p.add_argument(
        "--system-bbox",
        default=None,
        help="Optional system bbox_to_anchor in axes coords 'x,y' (e.g. '1.02,1.0' for outside right).",
    )
    p.add_argument(
        "--system-alpha",
        type=float,
        default=None,
        help="If set, draw the system annotation with a white semi-transparent frame (0..1).",
    )

    # Optional grain-boundary (GB) marker line + percentage annotations
    p.add_argument(
        "--gb-mfp",
        type=float,
        default=None,
        metavar="L_NM",
        help=(
            "If set, draw a vertical dashed line at this MFP (nm) to represent e.g. grain-boundary length, "
            "and annotate the cumulative percentage on each side for every plotted line."
        ),
    )
    p.add_argument(
        "--gb-text-y",
        type=float,
        nargs="+",
        default=[0.92],
        metavar="YFRAC",
        help="Y position for GB percentage text in axes fraction (0..1). Provide 1 (broadcast) or Nline values. Default: 0.92.",
    )
    p.add_argument(
        "--gb-text-xpad",
        type=float,
        nargs="+",
        default=[1.15],
        metavar="FACTOR",
        help=(
            "Horizontal padding factor from the GB dashed line. Right text uses x*FACTOR, left uses x/FACTOR. "
            "Provide 1 (broadcast) or Nline values. Default: 1.15."
        ),
    )
    p.add_argument(
        "--gb-text-color",
        nargs="+",
        default=["line"],
        metavar="COLOR",
        help=(
            "GB percentage text color. Use 'line' to match each curve color (default), or any matplotlib color string. "
            "Provide 1 (broadcast) or Nline values."
        ),
    )
    p.add_argument(
        "--gb-text-outline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw a white outline around GB percentage text for readability (default: on).",
    )
    p.add_argument(
        "--gb-text-outline-width",
        type=float,
        default=3.0,
        help="Outline width (points) for GB percentage text. Default: 3.0.",
    )
    p.add_argument(
        "--gb-text-outline-color",
        default="white",
        help="Outline color for GB percentage text. Default: white.",
    )
    p.add_argument(
        "--gb-xlabel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate the GB MFP value near the x-axis.",
    )
    p.add_argument(
        "--gb-xlabel-format",
        default="{x:g}",
        help="Format string for GB x-axis value annotation. Supports '{x}'. Default: '{x:g}'.",
    )
    p.add_argument(
        "--gb-xlabel-dy",
        type=float,
        default=-14.0,
        metavar="POINTS",
        help="Vertical offset (points) for GB x-axis value annotation (negative goes below axis). Default: -14.",
    )

    p.add_argument("--grid", action="store_true", help="Show grid")

    p.add_argument("--out", default="cumulative_kappa_total_vs_mfp.png", help="Output image path")
    p.add_argument("--show", action="store_true", help="Show interactively")

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.style == "prb":
        _apply_scienceplots_prb_style()

    _apply_global_fontsize(args.fontsize)

    want_bold_fonts = bool(args.bold_fonts)
    if want_bold_fonts:
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    # Ensure we can import the two existing plotting scripts as modules, including
    # their sibling/helper modules that are imported without package prefixes.
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "perturbo_plot"))
    sys.path.insert(0, str(repo_root / "shengbte_plot"))

    # Lazy imports from the existing scripts to keep logic consistent.
    from shengbte_plot.plot_cumulative_kappa_vs_mfp_tensor import (
        _read_cumulative_kappa_vs_mfp_tensor,
        _select_component,
    )

    from perturbo_plot.plot_cumulative_transport import (
        _cumulative_from_mfd_tetra,
        _get_ref_total,
        _infer_tet_path,
        load_meanfp_yaml,
    )

    xlim = tuple(args.xlim) if args.xlim is not None else None
    ylim = tuple(args.ylim) if args.ylim is not None else None

    # --- lattice (ShengBTE)
    latt_files = [str(x) for x in (args.latt_file or [])]
    if not latt_files:
        raise SystemExit("--latt-file cannot be empty")

    latt_labels_in = [str(x) for x in (args.latt_label or [])]
    latt_labels_in = _broadcast_str_list(latt_labels_in, len(latt_files), "--latt-label")

    latt_colors_in: Optional[list[str]] = None
    if args.color_latt is not None:
        latt_colors_in = _broadcast_str_list([str(x) for x in args.color_latt], len(latt_files), "--color-latt")

    latt_ls_in = [_normalize_linestyle_token(str(x)) for x in (args.ls_latt or [])]
    latt_ls_in = _broadcast_str_list(latt_ls_in, len(latt_files), "--ls-latt")

    x_latt_list: list[np.ndarray] = []
    y_latt_list: list[np.ndarray] = []
    for f in latt_files:
        mfp_latt, k9 = _read_cumulative_kappa_vs_mfp_tensor(f)
        y_latt = _select_component(k9, str(args.latt_component))
        x_latt, y_latt = _extend_to_xlim(np.asarray(mfp_latt, float), np.asarray(y_latt, float), xlim=xlim)
        x_latt_list.append(x_latt)
        y_latt_list.append(y_latt)

    # --- electronic (Perturbo)
    meanfp_data = load_meanfp_yaml(str(args.el_meanfp))
    tet_path = Path(args.el_tet) if args.el_tet else _infer_tet_path(str(args.el_tdf))

    mfp_el, _cum_sigma, cum_kappa, T_K = _cumulative_from_mfd_tetra(
        meanfp_data=meanfp_data,
        tdf_h5_path=str(args.el_tdf),
        tet_path=tet_path,
        config=int(args.el_config),
        direction=str(args.el_dir),
        spin_deg=float(args.spin_deg),
        h5py_module=__import__("h5py"),
    )

    y_el = np.asarray(cum_kappa, float)

    if str(args.el_kappa_calib) != "none":
        ref_val = _get_ref_total(
            method=str(args.el_kappa_calib),
            qty="kappa",
            direction=str(args.el_dir),
            T_K=float(T_K),
            tdf_path=str(args.el_tdf),
            trans_coef_path=(Path(args.el_trans_coef) if args.el_trans_coef else None),
            trans_ita_path=(Path(args.el_trans_ita) if args.el_trans_ita else None),
        )
        if ref_val is not None and np.isfinite(ref_val) and abs(float(y_el[-1])) > 0:
            scale = float(ref_val / float(y_el[-1]))
            y_el = y_el * scale

    x_el, y_el = _extend_to_xlim(np.asarray(mfp_el, float), np.asarray(y_el, float), xlim=xlim)

    # --- plot
    figsize = _parse_figsize(args.figsize)
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150) if figsize is None else plt.subplots(figsize=figsize, dpi=150)

    lab_el = _format_chem_label(str(args.el_label), str(args.legend_format))
    lab_el = _normalize_math_label(lab_el, bold=want_bold_fonts)

    kw_el = dict(label=lab_el, lw=float(args.lw), linestyle=str(args.ls_el))
    if args.color_el:
        kw_el["color"] = str(args.color_el)

    handles = []
    x_line_list = []
    y_line_list = []

    for i, (x_latt, y_latt) in enumerate(zip(x_latt_list, y_latt_list)):
        lab_latt_i = _format_chem_label(str(latt_labels_in[i]), str(args.legend_format))
        lab_latt_i = _normalize_math_label(lab_latt_i, bold=want_bold_fonts)
        kw_latt_i = dict(label=lab_latt_i, lw=float(args.lw), linestyle=str(latt_ls_in[i]))
        if latt_colors_in is not None:
            kw_latt_i["color"] = str(latt_colors_in[i])
        (line_latt_i,) = ax.plot(x_latt, y_latt, **kw_latt_i)
        handles.append(line_latt_i)
        x_line_list.append(x_latt)
        y_line_list.append(y_latt)

    (line_el,) = ax.plot(x_el, y_el, **kw_el)
    handles.append(line_el)
    x_line_list.append(x_el)
    y_line_list.append(y_el)

    ax.set_xlabel(str(args.xlabel))
    ax.set_ylabel(str(args.ylabel))

    if args.xlog:
        ax.set_xscale("log")
    if args.ylog:
        ax.set_yscale("log")

    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))

    if args.grid:
        ax.grid(True, linestyle="--", alpha=0.3)

    # GB dashed line + x-axis annotation
    if args.gb_mfp is not None:
        gb_x = float(args.gb_mfp)
        if not (np.isfinite(gb_x) and gb_x > 0):
            raise SystemExit("--gb-mfp must be a finite positive number")

        ax.axvline(x=gb_x, color="grey", linestyle="--", linewidth=1.5, zorder=0)

        if args.gb_xlabel:
            try:
                txt = str(args.gb_xlabel_format).format(x=gb_x)
            except Exception:
                txt = f"{gb_x:g}"
            ax.annotate(
                txt,
                xy=(gb_x, 0.0),
                xycoords=("data", "axes fraction"),
                xytext=(0.0, float(args.gb_xlabel_dy)),
                textcoords="offset points",
                ha="center",
                va="top",
                color="grey",
            )

    # Apply label/tick font sizes
    if args.label_fontsize is not None:
        fs = float(args.label_fontsize)
        if fs <= 0:
            raise SystemExit("--label-fontsize must be > 0")
        ax.xaxis.label.set_size(fs)
        ax.yaxis.label.set_size(fs)
        ax.tick_params(axis="both", which="both", labelsize=fs)
        try:
            ax.xaxis.get_offset_text().set_size(fs)
        except Exception:
            pass
        try:
            ax.yaxis.get_offset_text().set_size(fs)
        except Exception:
            pass

    legend_bbox = _parse_xy(args.legend_bbox)
    legend_loc = " ".join(args.legend_loc) if isinstance(args.legend_loc, list) else str(args.legend_loc)

    # Matplotlib's loc='best' can effectively ignore/defeat bbox_to_anchor intent.
    # If user provides --legend-bbox, switch to a deterministic loc.
    if legend_bbox is not None and legend_loc.strip().lower() == "best":
        legend_loc = "upper left"

    leg_kw = dict(
        loc=legend_loc,
        frameon=bool(args.legend_alpha is not None),
        handlelength=float(args.legend_handlelength),
        handletextpad=float(args.legend_handletextpad),
        labelspacing=float(args.legend_labelspacing),
        borderpad=float(args.legend_borderpad),
        borderaxespad=float(args.legend_borderaxespad),
    )
    if args.legend_fontsize is not None:
        fs = float(args.legend_fontsize)
        if fs <= 0:
            raise SystemExit("--legend-fontsize must be > 0")
        leg_kw["fontsize"] = fs
    if legend_bbox is None:
        leg = ax.legend(**leg_kw)
    else:
        leg = ax.legend(**leg_kw, bbox_to_anchor=legend_bbox, bbox_transform=ax.transAxes)

    if args.legend_alpha is not None:
        _apply_legend_frame(leg, alpha=float(args.legend_alpha))

    # GB percentage annotations (after limits are finalized)
    if args.gb_mfp is not None and handles:
        gb_x = float(args.gb_mfp)
        n_lines = len(handles)

        gb_y_in = [float(x) for x in (args.gb_text_y or [])]
        gb_xpad_in = [float(x) for x in (args.gb_text_xpad or [])]
        gb_color_in = [str(x) for x in (args.gb_text_color or [])]

        if len(gb_y_in) not in (1, n_lines):
            raise SystemExit(f"--gb-text-y expects 1 or {n_lines} values, got {len(gb_y_in)}")
        if len(gb_xpad_in) not in (1, n_lines):
            raise SystemExit(f"--gb-text-xpad expects 1 or {n_lines} values, got {len(gb_xpad_in)}")
        if len(gb_color_in) not in (1, n_lines):
            raise SystemExit(f"--gb-text-color expects 1 or {n_lines} values, got {len(gb_color_in)}")

        if n_lines > 1 and len(gb_y_in) == 1:
            y0 = float(gb_y_in[0])
            step = 0.06
            gb_y_in = [max(0.05, min(0.95, y0 - i * step)) for i in range(n_lines)]

        gb_y_line = gb_y_in if len(gb_y_in) == n_lines else [float(gb_y_in[0])] * n_lines
        gb_xpad_line = gb_xpad_in if len(gb_xpad_in) == n_lines else [float(gb_xpad_in[0])] * n_lines
        gb_color_line = gb_color_in if len(gb_color_in) == n_lines else [str(gb_color_in[0])] * n_lines

        outline_on = bool(args.gb_text_outline)
        outline_width = float(args.gb_text_outline_width)
        outline_color = str(args.gb_text_outline_color)
        if outline_width < 0:
            raise SystemExit("--gb-text-outline-width must be >= 0")

        xmin, xmax = ax.get_xlim()

        for j in range(n_lines):
            x = np.asarray(x_line_list[j], float)
            y = np.asarray(y_line_list[j], float)
            if y.size < 2:
                continue
            y_final = float(y[-1])
            if not (np.isfinite(y_final) and y_final != 0):
                continue

            y_at = float(np.interp(gb_x, x, y, left=0.0, right=float(y[-1])))
            pct_left = max(0.0, min(100.0, 100.0 * y_at / y_final))
            pct_right = 100.0 - pct_left

            y_frac = float(gb_y_line[j])
            if not (0.0 <= y_frac <= 1.0):
                raise SystemExit("--gb-text-y must be within [0, 1]")

            xpad = float(gb_xpad_line[j])
            if not np.isfinite(xpad) or xpad < 1.0:
                raise SystemExit("--gb-text-xpad must be a finite number >= 1")

            xpad_eff = xpad if xpad > 1.0 else 1.0001
            x_left = max(float(xmin), gb_x / xpad_eff)
            x_right = min(float(xmax), gb_x * xpad_eff)

            copt = str(gb_color_line[j]).strip()
            if copt.lower() == "line":
                try:
                    txt_color = str(handles[j].get_color())
                except Exception:
                    txt_color = None
            else:
                txt_color = copt

            kw = {"transform": ax.get_xaxis_transform()}
            if args.legend_fontsize is not None:
                kw["fontsize"] = float(args.legend_fontsize)
            if txt_color is not None:
                kw["color"] = txt_color

            t_left = ax.text(x_left, y_frac, f"{pct_left:.1f}%", ha="right", va="top", **kw)
            t_right = ax.text(x_right, y_frac, f"{pct_right:.1f}%", ha="left", va="top", **kw)

            if outline_on and outline_width > 0:
                pe = [
                    patheffects.Stroke(linewidth=outline_width, foreground=outline_color),
                    patheffects.Normal(),
                ]
                t_left.set_path_effects(pe)
                t_right.set_path_effects(pe)

            if want_bold_fonts:
                try:
                    t_left.set_fontweight("bold")
                    t_right.set_fontweight("bold")
                except Exception:
                    pass

    # System annotation legend
    if args.system is not None and str(args.system).strip():
        sys_lab = _format_chem_label(str(args.system), str(args.system_format))
        sys_lab = _normalize_math_label(sys_lab, bold=want_bold_fonts)
        h = Line2D([], [], color="none", label=sys_lab)

        sys_fs = args.system_fontsize
        if sys_fs is None:
            try:
                sys_fs = float(ax.yaxis.label.get_size()) * 1.15
            except Exception:
                sys_fs = None

        if leg is not None:
            ax.add_artist(leg)

        system_bbox = _parse_xy(args.system_bbox)
        system_loc = " ".join(args.system_loc) if isinstance(args.system_loc, list) else str(args.system_loc)
        sys_kwargs = {
            "frameon": bool(args.system_alpha is not None),
            "handlelength": 0,
            "handletextpad": 0.0,
            "borderaxespad": 0.2,
            "fontsize": sys_fs,
        }

        if system_bbox is None:
            leg_sys = ax.legend(handles=[h], loc=system_loc, **sys_kwargs)
        else:
            leg_sys = ax.legend(
                handles=[h],
                loc=system_loc,
                bbox_to_anchor=system_bbox,
                bbox_transform=ax.transAxes,
                **sys_kwargs,
            )
        if args.system_alpha is not None:
            _apply_legend_frame(leg_sys, alpha=float(args.system_alpha))
        if want_bold_fonts and leg_sys is not None:
            for t in leg_sys.get_texts():
                t.set_fontweight("bold")

    if want_bold_fonts:
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontweight("bold")
        _set_figure_text_weight(fig, "bold")

    fig.tight_layout()
    fig.savefig(str(args.out), dpi=300)
    print(f"Saved: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
