#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from perturbo_meanfp_io import (
    apply_default_bold_rcparams,
    apply_plot_style,
    apply_scienceplots_prb_style,
    broadcast_list,
    default_label,
    flatten_tokens,
    format_label,
    parse_figsize,
    parse_xy,
)


@dataclass(frozen=True)
class Series:
    temperatures: List[float]
    values: List[float]


def _parse_lim(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = str(s).split(",", 1)
    return float(a), float(b)


def _broadcast_float_pairs(xs: Sequence[float], n_files: int, name: str) -> List[float]:
    n_lines = 2 * n_files
    if len(xs) == n_lines:
        return [float(x) for x in xs]
    if len(xs) == 1:
        return [float(xs[0])] * n_lines
    if len(xs) == 2:
        return [float(xs[0]), float(xs[1])] * n_files
    raise SystemExit(f"{name} expects 1, 2, or {n_lines} values, but got {len(xs)}")


def _broadcast_str_pairs(xs: Sequence[str], n_files: int, name: str) -> List[str]:
    n_lines = 2 * n_files
    if len(xs) == n_lines:
        return [str(x) for x in xs]
    if len(xs) == 1:
        return [str(xs[0])] * n_lines
    if len(xs) == 2:
        return [str(xs[0]), str(xs[1])] * n_files
    raise SystemExit(f"{name} expects 1, 2, or {n_lines} values, but got {len(xs)}")


def _assert_same_temperature_grid(
    t_a: Sequence[float],
    t_b: Sequence[float],
    *,
    tol: float = 1e-6,
    a_label: str = "A",
    b_label: str = "B",
) -> None:
    if len(t_a) != len(t_b):
        raise SystemExit(
            f"Temperature grid mismatch: {a_label} has {len(t_a)} points, {b_label} has {len(t_b)} points."
        )

    mism: List[Tuple[int, float, float]] = []
    for i, (a, b) in enumerate(zip(t_a, t_b)):
        if abs(float(a) - float(b)) > tol:
            mism.append((i, float(a), float(b)))

    if mism:
        preview = mism[:8]
        details = ", ".join([f"i={i}: {ta} vs {tb}" for i, ta, tb in preview])
        more = "" if len(mism) <= 8 else f" (and {len(mism) - 8} more)"
        raise SystemExit(f"Temperature grid mismatch: {details}{more}")


def _get_component_series(
    series_by_comp: Dict[str, Series],
    component: str,
    *,
    series_label: str,
    tol: float = 1e-6,
) -> Series:
    comp = component.lower()
    if comp != "avg":
        if comp not in series_by_comp:
            raise SystemExit(
                f"Component {comp} not found in {series_label}. Available: {sorted(series_by_comp.keys())}"
            )
        return series_by_comp[comp]

    for c in ("xx", "yy", "zz"):
        if c not in series_by_comp:
            raise SystemExit(
                f"Component {c} required for avg in {series_label}, but not found. Available: {sorted(series_by_comp.keys())}"
            )

    sxx = series_by_comp["xx"]
    syy = series_by_comp["yy"]
    szz = series_by_comp["zz"]

    _assert_same_temperature_grid(
        sxx.temperatures,
        syy.temperatures,
        tol=tol,
        a_label=f"{series_label} xx",
        b_label=f"{series_label} yy",
    )
    _assert_same_temperature_grid(
        sxx.temperatures,
        szz.temperatures,
        tol=tol,
        a_label=f"{series_label} xx",
        b_label=f"{series_label} zz",
    )

    temps = list(sxx.temperatures)
    vals = [(a + b + c) / 3.0 for a, b, c in zip(sxx.values, syy.values, szz.values)]
    return Series(temps, vals)


def read_perturbo_trans_ita_tensor_vs_t(path: str, key: str) -> Dict[str, Series]:
    """Read a tensor-like quantity vs temperature from Perturbo trans-ita YAML."""

    try:
        import yaml  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Missing dependency PyYAML. Install with: pip install pyyaml") from exc

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "trans" not in data:
        raise RuntimeError(f"Not a Perturbo trans YAML: {path}")

    trans = data["trans"]
    cfgs = trans.get("configuration index")
    if not isinstance(cfgs, dict):
        raise RuntimeError(f"Missing trans.configuration index in: {path}")

    comp_map: Dict[str, List[Tuple[float, float]]] = {}

    for _, cfg in cfgs.items():
        if not isinstance(cfg, dict):
            continue
        t = cfg.get("temperature")
        if t is None:
            continue
        try:
            temp = float(t)
        except Exception:
            continue

        block = cfg.get(key)
        if not isinstance(block, dict):
            continue
        comps = block.get("components")
        if not isinstance(comps, dict):
            continue

        for comp, val in comps.items():
            c = str(comp).lower()
            if c not in {"xx", "yy", "zz", "xy", "xz", "yz", "yx", "zx", "zy"}:
                continue
            try:
                comp_map.setdefault(c, []).append((temp, float(val)))
            except Exception:
                continue

    if not comp_map:
        raise RuntimeError(f"No '{key}' components found in: {path}")

    out: Dict[str, Series] = {}
    for c, pairs in comp_map.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        out[c] = Series([p[0] for p in pairs_sorted], [p[1] for p in pairs_sorted])

    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot conductivity σ(T) and electronic thermal conductivity κ(T) from Perturbo trans-ita.yml on one figure.\n\n"
            "Style lists for --ls/--color/--marker/--ms/--lw are interpreted as pairs per file:\n"
            "  (σ for file1, κ for file1, σ for file2, κ for file2, ...)\n"
            "You can provide 2*N values, or provide 2 values to broadcast (σ, κ), or 1 value to broadcast all.\n"
        )
    )

    p.add_argument(
        "--style",
        choices=["prb", "default"],
        default="prb",
        help="Plot style preset. 'prb' uses SciencePlots if available [default: prb].",
    )

    p.add_argument("files", nargs="+", help="One or more Perturbo trans-ita YAML files")

    p.add_argument(
        "--legend",
        default=None,
        nargs="+",
        help=(
            "Legend label(s) for each input file. Provide one per file, or a single value to broadcast. "
            "If omitted, uses the filename stem."
        ),
    )
    p.add_argument(
        "--legend-format",
        choices=["chem", "raw"],
        default="raw",
        help="Render --legend text with subscripts (chem) or raw text (raw) [default: raw].",
    )

    p.add_argument(
        "--component",
        default="avg",
        choices=["xx", "yy", "zz", "xy", "xz", "yz", "yx", "zx", "zy", "avg"],
        help="Tensor component to plot [default: avg]. Use 'avg' for (xx+yy+zz)/3.",
    )

    p.add_argument(
        "--ls",
        default=None,
        nargs="+",
        help=(
            "Linestyle(s) for σ/κ lines. Provide 2*N values (σ1 κ1 σ2 κ2 ...), or 2 values to broadcast (σ κ), or 1 value to broadcast all."
        ),
    )
    p.add_argument(
        "--color",
        default=None,
        nargs="+",
        help=(
            "Color(s) for σ/κ lines. Provide 2*N values (σ1 κ1 σ2 κ2 ...), or 2 values to broadcast (σ κ), or 1 value to broadcast all."
        ),
    )
    p.add_argument(
        "--marker",
        default=None,
        nargs="+",
        help=(
            "Marker(s) for σ/κ lines. Provide 2*N values (σ1 κ1 σ2 κ2 ...), or 2 values to broadcast (σ κ), or 1 value to broadcast all. Use 'none' to disable."
        ),
    )
    p.add_argument(
        "--ms",
        default=None,
        nargs="+",
        help=(
            "Marker size(s) for σ/κ lines. Provide 2*N values (σ1 κ1 σ2 κ2 ...), or 2 values to broadcast (σ κ), or 1 value to broadcast all. Default: 4.5."
        ),
    )
    p.add_argument(
        "--lw",
        default=None,
        nargs="+",
        help=(
            "Line width(s) for σ/κ lines. Provide 2*N values (σ1 κ1 σ2 κ2 ...), or 2 values to broadcast (σ κ), or 1 value to broadcast all. Default: 2.0."
        ),
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
        help="Legend location (matplotlib legend loc) [default: best].",
    )
    p.add_argument(
        "--legend-bbox",
        default=None,
        help="Optional legend anchor (bbox_to_anchor) in axes coordinates 'x,y'.",
    )

    p.add_argument(
        "--system",
        default=None,
        help="Overall system/material label shown as a separate legend entry (pure text).",
    )
    p.add_argument(
        "--system-format",
        choices=["chem", "raw"],
        default="chem",
        help="Render --system as chemical formula with subscripts (chem) or raw text (raw) [default: chem].",
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
        help="Legend location for --system (matplotlib legend loc) [default: upper left].",
    )
    p.add_argument(
        "--system-bbox",
        default=None,
        help="Optional system anchor (bbox_to_anchor) in axes coordinates 'x,y'.",
    )

    p.add_argument("--xlim", default=None, help='x limits "xmin,xmax" in K')
    p.add_argument(
        "--ylim-sigma",
        default=None,
        help='y limits for σ axis "ymin,ymax" (left y-axis).',
    )
    p.add_argument(
        "--ylim-kappa",
        default=None,
        help='y limits for κ axis "ymin,ymax" (right y-axis).',
    )
    p.add_argument(
        "--figsize",
        default=None,
        help='Figure size "width,height" in inches (e.g. "7.2,4.6"). If omitted, uses the default size.',
    )
    p.add_argument("--title", default=None, help="Plot title")
    p.add_argument("--out", default=None, help="Output image path (png/pdf/svg). If omitted, show interactively.")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.style == "prb":
        apply_scienceplots_prb_style()
    else:
        apply_default_bold_rcparams()

    n_files = len(args.files)
    n_lines = 2 * n_files

    legends_in = flatten_tokens(args.legend)
    if legends_in:
        legends_raw = broadcast_list(legends_in, n_files, "--legend")
    else:
        legends_raw = [default_label(f) for f in args.files]
    legends = [format_label(str(x), str(args.legend_format)) for x in legends_raw]

    # Defaults: per-file colors, same color for (σ, κ), different linestyles.
    if n_files <= 10:
        base_colors: List[object] = [plt.get_cmap("tab10")(i) for i in range(n_files)]
    else:
        base_colors = [plt.get_cmap("tab20")(i % 20) for i in range(n_files)]

    default_ls = ["-", "--"] * n_files
    default_color: List[object] = []
    for c in base_colors:
        default_color.extend([c, c])

    default_markers = ["o", "o"] * n_files
    default_ms = [4.5] * n_lines
    default_lw = [2.0] * n_lines

    ls_in = flatten_tokens(args.ls)
    linestyles = _broadcast_str_pairs(ls_in, n_files, "--ls") if ls_in else list(default_ls)

    colors_in = flatten_tokens(args.color)
    colors: List[object] = _broadcast_str_pairs(colors_in, n_files, "--color") if colors_in else list(default_color)

    markers_in = flatten_tokens(args.marker)
    markers = _broadcast_str_pairs(markers_in, n_files, "--marker") if markers_in else list(default_markers)

    ms_in = flatten_tokens(args.ms)
    if ms_in:
        try:
            marker_sizes = _broadcast_float_pairs([float(x) for x in ms_in], n_files, "--ms")
        except ValueError as exc:
            raise SystemExit(f"Invalid --ms values: {ms_in!r}") from exc
    else:
        marker_sizes = list(default_ms)

    lw_in = flatten_tokens(args.lw)
    if lw_in:
        try:
            line_widths = _broadcast_float_pairs([float(x) for x in lw_in], n_files, "--lw")
        except ValueError as exc:
            raise SystemExit(f"Invalid --lw values: {lw_in!r}") from exc
    else:
        line_widths = list(default_lw)

    if any(m <= 0 for m in marker_sizes):
        raise SystemExit("--ms must be > 0")
    if any(w <= 0 for w in line_widths):
        raise SystemExit("--lw must be > 0")

    xlim = _parse_lim(args.xlim)
    ylim_sigma = _parse_lim(args.ylim_sigma)
    ylim_kappa = _parse_lim(args.ylim_kappa)
    figsize_override = parse_figsize(args.figsize)
    legend_bbox = parse_xy(args.legend_bbox)
    system_bbox = parse_xy(args.system_bbox)

    if figsize_override is None:
        fig, ax_sigma = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    else:
        fig, ax_sigma = plt.subplots(figsize=figsize_override, dpi=150)

    ax_kappa = ax_sigma.twinx()

    comp = str(args.component).lower()

    for i, (path, ds_label) in enumerate(zip(args.files, legends)):
        sigma_by_comp = read_perturbo_trans_ita_tensor_vs_t(path, "conductivity")
        kappa_by_comp = read_perturbo_trans_ita_tensor_vs_t(path, "thermal conductivity")

        sigma_s = _get_component_series(sigma_by_comp, comp, series_label=f"{ds_label} conductivity")
        kappa_s = _get_component_series(kappa_by_comp, comp, series_label=f"{ds_label} thermal conductivity")

        _assert_same_temperature_grid(
            sigma_s.temperatures,
            kappa_s.temperatures,
            tol=1e-6,
            a_label=f"{ds_label} sigma",
            b_label=f"{ds_label} kappa",
        )

        t = sigma_s.temperatures
        sigma = sigma_s.values
        kappa = kappa_s.values

        # Pair order: (σ for file i, κ for file i)
        j_sigma = 2 * i
        j_kappa = 2 * i + 1

        m_sigma = markers[j_sigma]
        m_kappa = markers[j_kappa]
        if m_sigma is not None and str(m_sigma).lower() in {"none", "null", ""}:
            m_sigma = None
        if m_kappa is not None and str(m_kappa).lower() in {"none", "null", ""}:
            m_kappa = None

        ax_sigma.plot(
            t,
            sigma,
            color=colors[j_sigma],
            lw=float(line_widths[j_sigma]),
            linestyle=linestyles[j_sigma],
            marker=m_sigma,
            markersize=float(marker_sizes[j_sigma]),
            markeredgewidth=0.0,
            label=f"{ds_label} $\\sigma$",
        )
        ax_kappa.plot(
            t,
            kappa,
            color=colors[j_kappa],
            lw=float(line_widths[j_kappa]),
            linestyle=linestyles[j_kappa],
            marker=m_kappa,
            markersize=float(marker_sizes[j_kappa]),
            markeredgewidth=0.0,
            label=f"{ds_label} $\\kappa_{{el}}$",
        )

    ax_sigma.set_xlabel("Temperature (K)")
    ax_sigma.set_ylabel(r"Electrical conductivity $\sigma$ (S/m)")
    ax_kappa.set_ylabel(r"Thermal conductivity $\kappa_{el}$ (W/mK)")

    if xlim:
        ax_sigma.set_xlim(*xlim)

    if ylim_sigma:
        ax_sigma.set_ylim(*ylim_sigma)
    if ylim_kappa:
        ax_kappa.set_ylim(*ylim_kappa)

    if args.title:
        ax_sigma.set_title(args.title)

    if args.style != "prb":
        ax_sigma.grid(True, alpha=0.25)

    handles: List[object] = []
    labels: List[str] = []
    for a in (ax_sigma, ax_kappa):
        h, l = a.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    if legend_bbox is None:
        leg = ax_sigma.legend(
            handles,
            labels,
            loc=str(args.legend_loc),
            frameon=False,
            ncols=1,
            fontsize=args.legend_fontsize,
        )
    else:
        leg = ax_sigma.legend(
            handles,
            labels,
            loc=str(args.legend_loc),
            bbox_to_anchor=legend_bbox,
            bbox_transform=ax_sigma.transAxes,
            frameon=False,
            ncols=1,
            fontsize=args.legend_fontsize,
        )

    if args.system is not None and str(args.system).strip():
        ax_sigma.add_artist(leg)
        sys_lab = format_label(str(args.system), str(args.system_format))
        handle = Line2D([0], [0], color="none", lw=0, label=sys_lab)
        fs = args.system_fontsize
        if fs is None:
            try:
                fs = float(ax_sigma.yaxis.label.get_size()) * 1.15
            except Exception:
                fs = None

        if system_bbox is None:
            leg_sys = ax_sigma.legend(
                handles=[handle],
                loc=str(args.system_loc),
                frameon=False,
                fontsize=fs,
                handlelength=0,
            )
        else:
            leg_sys = ax_sigma.legend(
                handles=[handle],
                loc=str(args.system_loc),
                bbox_to_anchor=system_bbox,
                bbox_transform=ax_sigma.transAxes,
                frameon=False,
                fontsize=fs,
                handlelength=0,
            )
        if leg_sys is not None:
            for t in leg_sys.get_texts():
                t.set_fontweight("bold")

    apply_plot_style(ax_sigma, legend=leg, bold=(args.style != "prb"), sci_y="auto", ylog=False)
    apply_plot_style(ax_kappa, legend=None, bold=(args.style != "prb"), sci_y="auto", ylog=False)

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
