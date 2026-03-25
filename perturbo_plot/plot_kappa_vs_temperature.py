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
class KappaSeries:
    temperatures: List[float]
    values: List[float]


def _broadcast_float_list(xs: Sequence[float], n: int, name: str) -> List[float]:
    if len(xs) == n:
        return [float(x) for x in xs]
    if len(xs) == 1:
        return [float(xs[0])] * n
    raise SystemExit(f"{name} expects 1 value or {n} values, but got {len(xs)}")

def read_perturbo_trans_ita_e_kappa(path: str) -> Dict[str, KappaSeries]:
    """Read electronic thermal conductivity from Perturbo trans-ita YAML."""

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
        t = float(cfg.get("temperature"))
        tc = cfg.get("thermal conductivity")
        if not isinstance(tc, dict):
            continue
        comps = tc.get("components")
        if not isinstance(comps, dict):
            continue
        for comp, val in comps.items():
            c = str(comp).lower()
            if c not in {"xx", "yy", "zz", "xy", "xz", "yz", "yx", "zx", "zy"}:
                continue
            comp_map.setdefault(c, []).append((t, float(val)))

    if not comp_map:
        raise RuntimeError(f"No thermal conductivity components found in: {path}")

    out: Dict[str, KappaSeries] = {}
    for c, pairs in comp_map.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        out[c] = KappaSeries([p[0] for p in pairs_sorted], [p[1] for p in pairs_sorted])

    return out


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
        raise SystemExit(
            "Temperature grid mismatch between tensor components. "
            f"Mismatches: {details}{more}"
        )


def _get_component_series(
    series_by_comp: Dict[str, KappaSeries],
    component: str,
    *,
    series_label: str,
    tol: float = 1e-6,
) -> KappaSeries:
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
    return KappaSeries(temps, vals)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot electronic thermal conductivity vs temperature from Perturbo trans-ita.yml "
            "(supports multiple files for comparison)."
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
        default=["avg"],
        nargs="+",
        help=(
            "Tensor component(s) to plot. Provide one or more values from: "
            "xx yy zz xy xz yz yx zx zy avg. "
            "Use 'avg' for (xx+yy+zz)/3. Default: avg. "
            "Examples: --component avg | --component xx yy zz"
        ),
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

    p.add_argument("--lw", type=float, default=2.2, help="Line width (default: 2.2)")

    # Legend placement (main curve legend)
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

    # Global system annotation (pure text)
    p.add_argument("--system", default=None, help="Overall system/material label shown as a separate legend entry (pure text).")
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

    p.add_argument("--ylog", action="store_true", help="Use log scale on y-axis")

    p.add_argument("--xlim", default=None, help='x limits "xmin,xmax" in K')
    p.add_argument("--ylim", default=None, help='y limits "ymin,ymax" in W/m/K')
    p.add_argument(
        "--figsize",
        default=None,
        help='Figure size "width,height" in inches (e.g. "7.2,4.6"). If omitted, uses the default size.',
    )
    p.add_argument(
        "--label-fontsize",
        type=float,
        default=None,
        help="Font size for axis labels AND tick numbers. If omitted, uses matplotlib defaults.",
    )
    p.add_argument("--title", default=None, help="Plot title")
    p.add_argument("--out", default=None, help="Output image path (png/pdf/svg). If omitted, show interactively.")
    return p


def _parse_lim(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = s.split(",", 1)
    return float(a), float(b)


def main() -> None:
    args = _build_parser().parse_args()

    if args.label_fontsize is not None and float(args.label_fontsize) <= 0:
        raise SystemExit("--label-fontsize must be > 0")

    if args.lw <= 0:
        raise SystemExit("--lw must be > 0")

    if args.style == "prb":
        apply_scienceplots_prb_style()
    else:
        apply_default_bold_rcparams()

    n_files = len(args.files)

    components_in = flatten_tokens(args.component)
    if not components_in:
        components_in = ["avg"]
    components = [str(c).lower() for c in components_in]
    allowed_components = {"xx", "yy", "zz", "xy", "xz", "yz", "yx", "zx", "zy", "avg"}
    bad = [c for c in components if c not in allowed_components]
    if bad:
        raise SystemExit(f"Invalid --component {bad}. Allowed: {sorted(allowed_components)}")

    legends_in = flatten_tokens(args.legend)
    if legends_in:
        legends_raw = broadcast_list(legends_in, n_files, "--legend")
    else:
        legends_raw = [default_label(f) for f in args.files]
    legends = [format_label(str(x), str(args.legend_format)) for x in legends_raw]

    # Per-dataset styles (broadcast: 1 value -> all files)
    default_markers = ["o", "s", "^", "D", "v", ">", "<", "p", "h", "x", "+", "*"]
    component_linestyles = ["-", "--", ":", "-."]

    if n_files <= 10:
        default_colors: List[object] = [plt.get_cmap("tab10")(i) for i in range(n_files)]
    else:
        default_colors = [plt.get_cmap("tab20")(i % 20) for i in range(n_files)]

    colors_in = flatten_tokens(args.color)
    if colors_in:
        colors: List[object] = broadcast_list(colors_in, n_files, "--color")
    else:
        colors = list(default_colors)

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
    ylim = _parse_lim(args.ylim)
    figsize_override = parse_figsize(args.figsize)

    legend_bbox = parse_xy(args.legend_bbox)
    system_bbox = parse_xy(args.system_bbox)

    if figsize_override is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    else:
        fig, ax = plt.subplots(figsize=figsize_override, dpi=150)

    multi_comp = len(components) > 1
    for i, (path, dataset_label) in enumerate(zip(args.files, legends)):
        ele = read_perturbo_trans_ita_e_kappa(path)

        marker = markers[i]
        if marker is not None and str(marker).lower() in {"none", "null", ""}:
            marker = None

        for j, comp in enumerate(components):
            ele_s = _get_component_series(ele, comp, series_label=f"{dataset_label} electron")
            tE, kE = ele_s.temperatures, ele_s.values
            ls = component_linestyles[j % len(component_linestyles)]
            curve_label = f"{dataset_label} {comp}" if multi_comp else dataset_label
            ax.plot(
                tE,
                kE,
                lw=float(args.lw),
                linestyle=ls,
                color=colors[i],
                marker=marker,
                markersize=float(marker_sizes[i]),
                markeredgewidth=0.0,
                label=curve_label,
            )

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"$\kappa_{\mathrm{el}}$ (W/mK)")

    if args.ylog:
        ax.set_yscale("log")

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    if args.title:
        ax.set_title(args.title)

    if args.style != "prb":
        ax.grid(True, alpha=0.25)

    if legend_bbox is None:
        leg = ax.legend(loc=str(args.legend_loc), frameon=False, ncols=1, fontsize=args.legend_fontsize)
    else:
        leg = ax.legend(
            loc=str(args.legend_loc),
            bbox_to_anchor=legend_bbox,
            bbox_transform=ax.transAxes,
            frameon=False,
            ncols=1,
            fontsize=args.legend_fontsize,
        )

    if args.system is not None and str(args.system).strip():
        ax.add_artist(leg)
        sys_lab = format_label(str(args.system), str(args.system_format))
        handle = Line2D([0], [0], color="none", lw=0, label=sys_lab)
        fs = args.system_fontsize
        if fs is None:
            try:
                fs = float(ax.yaxis.label.get_size()) * 1.15
            except Exception:
                fs = None

        if system_bbox is None:
            leg_sys = ax.legend(handles=[handle], loc=str(args.system_loc), frameon=False, fontsize=fs, handlelength=0)
        else:
            leg_sys = ax.legend(
                handles=[handle],
                loc=str(args.system_loc),
                bbox_to_anchor=system_bbox,
                bbox_transform=ax.transAxes,
                frameon=False,
                fontsize=fs,
                handlelength=0,
            )
        if leg_sys is not None:
            for t in leg_sys.get_texts():
                t.set_fontweight("bold")

    apply_plot_style(
        ax,
        legend=leg,
        bold=(args.style != "prb"),
        label_fontsize=args.label_fontsize,
        sci_y="auto",
        ylog=args.ylog,
    )

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
