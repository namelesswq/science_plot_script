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


def _normalize_linestyle_token(s: str) -> str:
    """Normalize user-friendly linestyle aliases.

    Note: passing a bare '--' token on the CLI is interpreted by argparse as
    end-of-options. Use 'dashed' instead, or comma-separated input like:
    --ls-kappa dashed
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
    """Read a tensor-like quantity vs temperature from Perturbo trans-ita YAML.

    Parameters
    ----------
    key:
        The YAML key inside each configuration, e.g. 'conductivity' or 'thermal conductivity'.

    Returns
    -------
    Mapping from component name ('xx', ...) to series (T list, value list).
    """

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


def _parse_lim(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = str(s).split(",", 1)
    return float(a), float(b)


def _require_lim_order(lim: Optional[Tuple[float, float]], name: str) -> None:
    if lim is None:
        return
    lo, hi = lim
    if not (lo < hi):
        raise SystemExit(f"{name} expects ymin<ymax, but got {lo},{hi}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot conductivity σ(T), electronic thermal conductivity κ(T), and Lorenz number L(T) "
            "from Perturbo trans-ita.yml on a single figure.\n\n"
            "- Left y-axis: σ\n"
            "- Right y-axis: κ\n"
            "- Extra right y-axis (offset): L = κ/(σ·T)\n"
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
        "--color",
        default=None,
        nargs="+",
        help=(
            "Line/marker color(s) for each input file. Provide one per file, or a single value to broadcast. "
            "Examples: 'black', 'tab:red', '#1f77b4'."
        ),
    )

    p.add_argument(
        "--color-sigma",
        default="tab:blue",
        help="Color for σ(T) curves [default: tab:blue].",
    )
    p.add_argument(
        "--color-kappa",
        default="tab:red",
        help="Color for κ(T) curves [default: tab:red].",
    )
    p.add_argument(
        "--color-lorenz",
        default="tab:green",
        help="Color for L(T) curves [default: tab:green].",
    )

    p.add_argument(
        "--ls-sigma",
        default="-",
        help="Linestyle for σ(T) curves [default: -].",
    )
    p.add_argument(
        "--ls-kappa",
        default="dashed",
        help="Linestyle for κ(T) curves. Use 'dashed' for '--' [default: dashed].",
    )
    p.add_argument(
        "--ls-lorenz",
        default=":",
        help="Linestyle for L(T) curves [default: :].",
    )
    p.add_argument(
        "--marker",
        default=None,
        nargs="+",
        help=(
            "Marker style(s) for each input file. Provide one per file, or a single value to broadcast. "
            "Examples: 'o', 's', '^', 'D', 'v', 'x', '+', '*'."
        ),
    )
    p.add_argument(
        "--ms",
        default=None,
        nargs="+",
        help=(
            "Marker size(s) for each input file. Provide one per file, or a single value to broadcast. "
            "Default: 4.5."
        ),
    )

    p.add_argument("--lw", type=float, default=2.0, help="Line width (default: 2.0)")

    p.add_argument(
        "--label-fontsize",
        type=float,
        default=None,
        help="Font size for x/y axis labels (applies to all axes). If omitted, keep the default.",
    )
    p.add_argument(
        "--right-axis-spacing",
        type=float,
        default=0.06,
        help=(
            "Spacing between the two right y-axes (kappa vs Lorenz). "
            "Lorenz axis spine is placed at x = 1 + spacing in axes coordinates [default: 0.06]."
        ),
    )

    p.add_argument(
        "--legend-fontsize",
        type=float,
        default=None,
        help="Font size for legend text. If omitted, uses matplotlib default.",
    )
    p.add_argument(
        "--legend-ncol",
        type=int,
        default=1,
        help="Number of columns in the main legend [default: 1].",
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
        help='y limits for σ axis "ymin,ymax" (optional)',
    )
    p.add_argument(
        "--ylim-kappa",
        default=None,
        help='y limits for κ axis "ymin,ymax" (optional)',
    )
    p.add_argument(
        "--ylim-lorenz",
        default=None,
        help='y limits for Lorenz axis "ymin,ymax" (optional)',
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

    if args.lw <= 0:
        raise SystemExit("--lw must be > 0")

    if args.right_axis_spacing <= 0:
        raise SystemExit("--right-axis-spacing must be > 0")

    if args.style == "prb":
        apply_scienceplots_prb_style()
    else:
        apply_default_bold_rcparams()

    n_files = len(args.files)

    if args.legend_ncol <= 0:
        raise SystemExit("--legend-ncol must be >= 1")

    legends_in = flatten_tokens(args.legend)
    if legends_in:
        legends_raw = broadcast_list(legends_in, n_files, "--legend")
    else:
        legends_raw = [default_label(f) for f in args.files]
    legends = [format_label(str(x), str(args.legend_format)) for x in legends_raw]

    if args.color is not None:
        raise SystemExit(
            "--color (per-file colors) is deprecated for this script now. "
            "Use --color-sigma/--color-kappa/--color-lorenz to set colors per variable."
        )

    default_markers = ["o", "s", "^", "D", "v", ">", "<", "p", "h", "x", "+", "*"]

    markers_in = flatten_tokens(args.marker)
    if markers_in:
        markers = broadcast_list(markers_in, n_files, "--marker")
    else:
        markers = [default_markers[i % len(default_markers)] for i in range(n_files)]

    ms_in = flatten_tokens(args.ms)
    if ms_in:
        try:
            marker_sizes = _broadcast_float_list([float(x) for x in ms_in], n_files, "--ms")
        except ValueError as exc:
            raise SystemExit(f"Invalid --ms values: {ms_in!r}") from exc
    else:
        marker_sizes = [4.5] * n_files
    if any(m <= 0 for m in marker_sizes):
        raise SystemExit("--ms must be > 0")

    xlim = _parse_lim(args.xlim)
    ylim_sigma = _parse_lim(args.ylim_sigma)
    ylim_kappa = _parse_lim(args.ylim_kappa)
    ylim_lorenz = _parse_lim(args.ylim_lorenz)
    _require_lim_order(ylim_sigma, "--ylim-sigma")
    _require_lim_order(ylim_kappa, "--ylim-kappa")
    _require_lim_order(ylim_lorenz, "--ylim-lorenz")
    figsize_override = parse_figsize(args.figsize)

    legend_bbox = parse_xy(args.legend_bbox)
    system_bbox = parse_xy(args.system_bbox)

    if figsize_override is None:
        fig, ax_sigma = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    else:
        fig, ax_sigma = plt.subplots(figsize=figsize_override, dpi=150)

    ax_kappa = ax_sigma.twinx()
    ax_lorenz = ax_sigma.twinx()
    lorenz_spine_x = 1.0 + float(args.right_axis_spacing)
    ax_lorenz.spines["right"].set_position(("axes", lorenz_spine_x))
    ax_lorenz.set_frame_on(True)
    ax_lorenz.patch.set_visible(False)

    comp = str(args.component).lower()

    # Use different linestyles for different quantities.
    # Keep per-file color/marker consistent.
    ls_sigma = _normalize_linestyle_token(str(args.ls_sigma))
    ls_kappa = _normalize_linestyle_token(str(args.ls_kappa))
    ls_lorenz = _normalize_linestyle_token(str(args.ls_lorenz))

    c_sigma = str(args.color_sigma)
    c_kappa = str(args.color_kappa)
    c_lorenz = str(args.color_lorenz)

    for i, (path, ds_label) in enumerate(zip(args.files, legends)):
        marker = markers[i]
        if marker is not None and str(marker).lower() in {"none", "null", ""}:
            marker = None

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

        lorenz: List[float] = []
        for tt, ss, kk in zip(t, sigma, kappa):
            if float(ss) == 0.0:
                lorenz.append(float("nan"))
            else:
                lorenz.append(float(kk) / (float(ss) * float(tt)))

        ax_sigma.plot(
            t,
            sigma,
            color=c_sigma,
            lw=float(args.lw),
            linestyle=ls_sigma,
            marker=marker,
            markersize=float(marker_sizes[i]),
            markeredgewidth=0.0,
            label=f"{ds_label} $\\sigma$",
        )
        ax_kappa.plot(
            t,
            kappa,
            color=c_kappa,
            lw=float(args.lw),
            linestyle=ls_kappa,
            marker=marker,
            markersize=float(marker_sizes[i]),
            markeredgewidth=0.0,
            label=f"{ds_label} $\\kappa_{{el}}$",
        )
        ax_lorenz.plot(
            t,
            lorenz,
            color=c_lorenz,
            lw=float(args.lw),
            linestyle=ls_lorenz,
            marker=marker,
            markersize=float(marker_sizes[i]),
            markeredgewidth=0.0,
            label=f"{ds_label} $L$",
        )

    # Theoretical (Sommerfeld) Lorenz number reference line
    # Use the same color as L(T) curves to avoid introducing new colors.
    L0 = 2.44e-8  # WΩ/K^2
    ax_lorenz.axhline(
        L0,
        color="gray",
        linestyle="--",
        lw=float(args.lw) * 0.9,
        alpha=0.8,
        label=r"Theory $L_0$",
    )

    ax_sigma.set_xlabel("Temperature (K)")
    ax_sigma.set_ylabel(r"Electrical conductivity $\sigma$ (S/m)")
    ax_kappa.set_ylabel(r"Thermal conductivity $\kappa_{el}$ (W/mK)")
    ax_lorenz.set_ylabel(r"Lorenz number $L=\kappa/(\sigma T)$ (W$\Omega$/K$^2$)")

    if xlim:
        ax_sigma.set_xlim(*xlim)

    if ylim_sigma:
        ax_sigma.set_ylim(*ylim_sigma)
    if ylim_kappa:
        ax_kappa.set_ylim(*ylim_kappa)
    if ylim_lorenz:
        ax_lorenz.set_ylim(*ylim_lorenz)

    if args.title:
        ax_sigma.set_title(args.title)

    if args.style != "prb":
        ax_sigma.grid(True, alpha=0.25)

    # One combined legend from all axes.
    handles: List[object] = []
    labels: List[str] = []
    for a in (ax_sigma, ax_kappa, ax_lorenz):
        h, l = a.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    if legend_bbox is None:
        leg = ax_sigma.legend(
            handles,
            labels,
            loc=str(args.legend_loc),
            frameon=False,
            ncols=int(args.legend_ncol),
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
            ncols=int(args.legend_ncol),
            fontsize=args.legend_fontsize,
        )

    # Global system annotation (pure text)
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
    apply_plot_style(ax_lorenz, legend=None, bold=(args.style != "prb"), sci_y="auto", ylog=False)

    if args.label_fontsize is not None:
        fs = float(args.label_fontsize)
        if fs <= 0:
            raise SystemExit("--label-fontsize must be > 0")
        for a in (ax_sigma, ax_kappa, ax_lorenz):
            a.xaxis.label.set_size(fs)
            a.yaxis.label.set_size(fs)
            a.tick_params(axis="both", which="both", labelsize=fs)

            # Keep scientific-notation offset text consistent when visible.
            try:
                a.xaxis.get_offset_text().set_size(fs)
            except Exception:
                pass
            try:
                a.yaxis.get_offset_text().set_size(fs)
            except Exception:
                pass

    ax_lorenz.yaxis.get_offset_text().set_visible(False)

    ax_lorenz.text(
        lorenz_spine_x,
        1.01,
        r"$\times 10^{-8}$",
        transform=ax_lorenz.transAxes,
        ha="left",
        va="bottom",
        fontsize=(float(args.label_fontsize) if args.label_fontsize is not None else ax_lorenz.yaxis.label.get_size()),
    )

    # Reserve extra right margin for the 3rd (offset) y-axis.
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    if args.out:
        fig.savefig(args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
