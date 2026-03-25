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
    end-of-options. Use 'dashed' (recommended) or comma-separated input
    like: --ls -,-,-,--,--,--
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


def _broadcast_str_grid(
    xs: Sequence[str],
    *,
    n_files: int,
    n_comp: int,
    name: str,
    order: str,
) -> List[str]:
    """Broadcast string styles to per-line list.

    Final length is n_files * n_comp, in the plotting order.
    Accepted lengths:
    - 1: broadcast to all lines
    - n_files: per-file, repeated for each component
    - n_comp: per-component, repeated for each file
    - n_files * n_comp: fully specified per line
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
        f"{name} expects 1, {n_files} (per-file), {n_comp} (per-component), or {n_lines} values, but got {len(xs)}"
    )


def _broadcast_float_grid(
    xs: Sequence[float],
    *,
    n_files: int,
    n_comp: int,
    name: str,
    order: str,
) -> List[float]:
    """Broadcast float styles to per-line list. See _broadcast_str_grid."""

    n_lines = n_files * n_comp
    if n_lines == 0:
        return []

    if len(xs) == 1:
        return [float(xs[0])] * n_lines
    if len(xs) == n_files:
        if order == "file-major":
            out: List[float] = []
            for v in xs:
                out.extend([float(v)] * n_comp)
            return out
        out = []
        for _ in range(n_comp):
            for v in xs:
                out.append(float(v))
        return out
    if len(xs) == n_comp:
        if order == "file-major":
            out: List[float] = []
            for _ in range(n_files):
                out.extend([float(v) for v in xs])
            return out
        out = []
        for v in xs:
            out.extend([float(v)] * n_files)
        return out
    if len(xs) == n_lines:
        return [float(v) for v in xs]

    raise SystemExit(
        f"{name} expects 1, {n_files} (per-file), {n_comp} (per-component), or {n_lines} values, but got {len(xs)}"
    )


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


def _parse_lim(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = str(s).split(",", 1)
    return float(a), float(b)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot Lorenz number L(T)=κ/(σT) from Perturbo trans-ita.yml on one figure, "
            "and draw the theoretical constant line (2.44×10^-8 WΩ/K^2)."
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
            "Legend label(s). You may provide: 1 value (broadcast), Nfile values (per-file), or Nfile×Ncomp values (per line, file×component). "
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
        choices=["xx", "yy", "zz", "xy", "xz", "yz", "yx", "zx", "zy", "avg"],
        help=(
            "Tensor component(s) to plot. You can pass multiple components to plot different directions on one figure. "
            "Use 'avg' for (xx+yy+zz)/3. [default: avg]."
        ),
    )

    p.add_argument(
        "--order",
        choices=["comp-major", "file-major"],
        default="comp-major",
        help=(
            "Order of plotted lines (and per-line legend/styles). "
            "comp-major: all files for the first component, then next component. "
            "file-major: all components for the first file, then next file. [default: comp-major]"
        ),
    )

    p.add_argument(
        "--ls",
        default=None,
        nargs="+",
        help=(
            "Linestyle(s) per plotted line (file × component). Provide 1 (broadcast), Nfile (per-file), "
            "Ncomp (per-component), or Nfile×Ncomp (fully specified) values."
            " Note: a bare '--' token cannot be passed as an argument (argparse treats it as end-of-options). "
            "Use 'dashed' instead, or pass comma-separated like: --ls -,-,-,--,--,--"
        ),
    )
    p.add_argument(
        "--color",
        default=None,
        nargs="+",
        help=(
            "Color(s) per plotted line (file × component). Provide 1 (broadcast), Nfile (per-file), "
            "Ncomp (per-component), or Nfile×Ncomp (fully specified) values."
        ),
    )
    p.add_argument(
        "--marker",
        default=None,
        nargs="+",
        help=(
            "Marker(s) per plotted line (file × component). Provide 1 (broadcast), Nfile (per-file), "
            "Ncomp (per-component), or Nfile×Ncomp (fully specified) values. Use 'none' to disable."
        ),
    )
    p.add_argument(
        "--ms",
        default=None,
        nargs="+",
        help=(
            "Marker size(s) per plotted line (file × component). Provide 1 (broadcast), Nfile (per-file), "
            "Ncomp (per-component), or Nfile×Ncomp (fully specified) values. Default: 4.5."
        ),
    )
    p.add_argument(
        "--lw",
        default=None,
        nargs="+",
        help=(
            "Line width(s) per plotted line (file × component). Provide 1 (broadcast), Nfile (per-file), "
            "Ncomp (per-component), or Nfile×Ncomp (fully specified) values. Default: 2.0."
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


def main() -> None:
    args = _build_parser().parse_args()

    if args.label_fontsize is not None and float(args.label_fontsize) <= 0:
        raise SystemExit("--label-fontsize must be > 0")

    if args.style == "prb":
        apply_scienceplots_prb_style()
    else:
        apply_default_bold_rcparams()

    components = [c.lower() for c in flatten_tokens(args.component)]
    if not components:
        raise SystemExit("--component cannot be empty")
    n_files = len(args.files)
    n_comp = len(components)
    n_lines = n_files * n_comp
    order = str(args.order)

    legends_in = flatten_tokens(args.legend)
    legend_mode: str
    if legends_in:
        if len(legends_in) == 1:
            legends_file = [format_label(str(legends_in[0]), str(args.legend_format))] * n_files
            legends_line: List[str] = []
            legend_mode = "file"
        elif len(legends_in) == n_files:
            legends_file = [format_label(str(x), str(args.legend_format)) for x in legends_in]
            legends_line = []
            legend_mode = "file"
        elif len(legends_in) == n_lines:
            legends_file = []
            legends_line = [format_label(str(x), str(args.legend_format)) for x in legends_in]
            legend_mode = "line"
        else:
            raise SystemExit(
                f"--legend expects 1, {n_files} (per-file), or {n_lines} (per line; order={order}) values, but got {len(legends_in)}"
            )
    else:
        legends_file = [format_label(default_label(f), str(args.legend_format)) for f in args.files]
        legends_line = []
        legend_mode = "file"

    default_markers = ["o", "s", "^", "D", "v", ">", "<", "p", "h", "x", "+", "*"]
    if n_files <= 10:
        default_colors: List[object] = [plt.get_cmap("tab10")(i) for i in range(n_files)]
    else:
        default_colors = [plt.get_cmap("tab20")(i % 20) for i in range(n_files)]

    component_linestyles = ["-", "--", ":", "-."]

    default_ls: List[str] = []
    default_color: List[object] = []
    default_marker: List[str] = []

    def _iter_lines() -> List[Tuple[int, int]]:
        if order == "file-major":
            return [(i, ic) for i in range(n_files) for ic in range(n_comp)]
        return [(i, ic) for ic in range(n_comp) for i in range(n_files)]

    for i, ic in _iter_lines():
        default_ls.append(component_linestyles[ic % len(component_linestyles)])
        default_color.append(default_colors[i])
        default_marker.append(default_markers[i % len(default_markers)])

    ls_in = flatten_tokens(args.ls)
    linestyles = (
        _broadcast_str_grid(ls_in, n_files=n_files, n_comp=n_comp, name="--ls", order=order)
        if ls_in
        else list(default_ls)
    )
    linestyles = [_normalize_linestyle_token(x) for x in linestyles]
    colors_in = flatten_tokens(args.color)
    colors: List[object] = (
        _broadcast_str_grid(colors_in, n_files=n_files, n_comp=n_comp, name="--color", order=order)
        if colors_in
        else list(default_color)
    )
    markers_in = flatten_tokens(args.marker)
    markers = (
        _broadcast_str_grid(markers_in, n_files=n_files, n_comp=n_comp, name="--marker", order=order)
        if markers_in
        else list(default_marker)
    )

    ms_in = flatten_tokens(args.ms)
    if ms_in:
        try:
            marker_sizes = _broadcast_float_grid(
                [float(x) for x in ms_in],
                n_files=n_files,
                n_comp=n_comp,
                name="--ms",
                order=order,
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid --ms values: {ms_in!r}") from exc
    else:
        marker_sizes = [4.5] * n_lines

    lw_in = flatten_tokens(args.lw)
    if lw_in:
        try:
            line_widths = _broadcast_float_grid(
                [float(x) for x in lw_in],
                n_files=n_files,
                n_comp=n_comp,
                name="--lw",
                order=order,
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid --lw values: {lw_in!r}") from exc
    else:
        line_widths = [2.0] * n_lines

    if any(m <= 0 for m in marker_sizes):
        raise SystemExit("--ms must be > 0")
    if any(w <= 0 for w in line_widths):
        raise SystemExit("--lw must be > 0")

    xlim = _parse_lim(args.xlim)
    figsize_override = parse_figsize(args.figsize)
    legend_bbox = parse_xy(args.legend_bbox)
    system_bbox = parse_xy(args.system_bbox)

    if figsize_override is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    else:
        fig, ax = plt.subplots(figsize=figsize_override, dpi=150)

    multi_comp = len(components) > 1

    def _line_index(i: int, ic: int) -> int:
        return i * n_comp + ic if order == "file-major" else ic * n_files + i

    sigma_all: List[Dict[str, Series]] = []
    kappa_all: List[Dict[str, Series]] = []
    for path in args.files:
        sigma_all.append(read_perturbo_trans_ita_tensor_vs_t(path, "conductivity"))
        kappa_all.append(read_perturbo_trans_ita_tensor_vs_t(path, "thermal conductivity"))

    if order == "file-major":
        line_pairs = [(i, ic) for i in range(n_files) for ic in range(n_comp)]
    else:
        line_pairs = [(i, ic) for ic in range(n_comp) for i in range(n_files)]

    for i, ic in line_pairs:
        path = args.files[i]
        ds_label = legends_file[i] if legend_mode == "file" else ""
        sigma_by_comp = sigma_all[i]
        kappa_by_comp = kappa_all[i]

        comp = components[ic]
        idx = _line_index(i, ic)

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

        marker = markers[idx]
        if marker is not None and str(marker).lower() in {"none", "null", ""}:
            marker = None

        if legend_mode == "line":
            lab = legends_line[idx]
        else:
            lab = ds_label if not multi_comp else f"{ds_label} {comp}"

        ax.plot(
            t,
            lorenz,
            color=colors[idx],
            lw=float(line_widths[idx]),
            linestyle=str(linestyles[idx]),
            marker=marker,
            markersize=float(marker_sizes[idx]),
            markeredgewidth=0.0,
            label=lab,
        )

    # Theoretical line (Wiedemann–Franz)
    l0 = 2.44e-8
    ax.axhline(
        l0,
        color="k",
        lw=1.2,
        linestyle="--",
        label=r"$L_0=2.44\times 10^{-8}$",
    )

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Lorenz number $L=\kappa/(\sigma T)$ (W$\Omega$/K$^2$)")

    if xlim:
        ax.set_xlim(*xlim)

    if args.title:
        ax.set_title(args.title)

    if args.style != "prb":
        ax.grid(True, alpha=0.25)

    if legend_bbox is None:
        leg = ax.legend(
            loc=str(args.legend_loc),
            frameon=False,
            ncols=1,
            fontsize=args.legend_fontsize,
        )
    else:
        leg = ax.legend(
            loc=str(args.legend_loc),
            bbox_to_anchor=legend_bbox,
            bbox_transform=ax.transAxes,
            frameon=False,
            ncols=1,
            fontsize=args.legend_fontsize,
        )

    # Global system annotation (pure text)
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
            leg_sys = ax.legend(
                handles=[handle],
                loc=str(args.system_loc),
                frameon=False,
                fontsize=fs,
                handlelength=0,
            )
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
        ylog=False,
    )

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
