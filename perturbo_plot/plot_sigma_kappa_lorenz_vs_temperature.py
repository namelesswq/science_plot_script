#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from perturbo_meanfp_io import (
    apply_default_bold_rcparams,
    apply_global_fontsize,
    apply_tick_steps,
    apply_legend_frame,
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

def _component_subscript_tex(comp: str, *, bold: bool) -> str:
    """Return mathtext fragment for tensor-component subscript.

    - in-plane  -> \\parallel
    - out-of-plane -> \\bot
    - avg -> avg
    - xx/yy/zz/... -> xx/yy/zz/... (typeset via \\mathrm or \\mathbf)
    """

    c = str(comp).strip().lower()
    if c in {"in-plane", "in_plane", "inplane", "ip"}:
        return r"\boldsymbol{\parallel}" if bold else r"\parallel"
    if c in {"out-of-plane", "out_of_plane", "outofplane", "oop"}:
        return r"\boldsymbol{\bot}" if bold else r"\bot"

    tag = "avg" if c == "avg" else c
    if bold:
        return rf"\mathbf{{{tag}}}"
    return rf"\mathrm{{{tag}}}"


def _legend_sym_with_comp(sym: str, comp: str, *, bold: bool) -> str:
    sub = _component_subscript_tex(comp, bold=bold)
    if bold:
        return rf"$\boldsymbol{{{sym}}}_{{{sub}}}$"
    return rf"${sym}_{{{sub}}}$"


def _legend_kappa_el_with_comp(comp: str, *, bold: bool) -> str:
    sub = _component_subscript_tex(comp, bold=bold)
    # Keep 'el' upright; don't try to bold it.
    if bold:
        return rf"$\boldsymbol{{\kappa}}_{{\mathbf{{el}},{sub}}}$"
    return rf"$\kappa_{{\mathrm{{el}},{sub}}}$"


def _linestyle_rank(ls: str) -> int:
    """Z-order rank by linestyle: solid < dashed < dotted."""

    norm = _normalize_linestyle_token(str(ls))
    if norm == "-":
        return 1
    if norm == "--":
        return 2
    if norm == ":":
        return 3
    # Default: treat other styles similar to dashed.
    return 2


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

    # Aliases / derived components
    if comp in {"out-of-plane", "out_of_plane", "outofplane", "oop"}:
        comp = "zz"

    if comp in {"in-plane", "in_plane", "inplane", "ip"}:
        for c in ("xx", "yy"):
            if c not in series_by_comp:
                raise SystemExit(
                    f"Component {c} required for in-plane in {series_label}, but not found. Available: {sorted(series_by_comp.keys())}"
                )
        sxx = series_by_comp["xx"]
        syy = series_by_comp["yy"]
        _assert_same_temperature_grid(
            sxx.temperatures,
            syy.temperatures,
            tol=tol,
            a_label=f"{series_label} xx",
            b_label=f"{series_label} yy",
        )
        temps = list(sxx.temperatures)
        vals = [(a + b) / 2.0 for a, b in zip(sxx.values, syy.values)]
        return Series(temps, vals)
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


def _pad_legend_columns_preserve_order(
    handles: Sequence[object],
    labels: Sequence[str],
    *,
    ncol: int,
    column_real_counts: Optional[Sequence[int]] = None,
) -> Tuple[List[object], List[str]]:
    """Pad legend entries so right columns can have more rows, without reordering.

    Matplotlib assigns legend entries to columns in order (column-major):
    first chunk -> 1st column, second chunk -> 2nd column, ...

    To make the *right* column(s) have more real entries while keeping the
    original entry order, we insert invisible placeholders at the end of the
    earlier (shorter) columns.
    """

    if len(handles) != len(labels):
        raise ValueError("Legend handles/labels length mismatch")

    k = int(ncol)
    if k <= 1:
        return list(handles), list(labels)

    n_items = len(handles)
    if n_items == 0:
        return [], []

    if column_real_counts is None:
        base = n_items // k
        rem = n_items % k
        # Put the remainder into the RIGHTMOST columns.
        counts = [base] * k
        for j in range(k - rem, k):
            if 0 <= j < k:
                counts[j] += 1
    else:
        counts = [int(x) for x in column_real_counts]
        if len(counts) != k:
            raise SystemExit(
                f"--legend-column-counts expects {k} integers (same as --legend-ncol), but got {len(counts)}"
            )
        if any(c < 0 for c in counts):
            raise SystemExit("--legend-column-counts values must be >= 0")
        if sum(counts) < n_items:
            counts[-1] += (n_items - sum(counts))

    rows = max(counts) if counts else 0
    if rows <= 0:
        return list(handles), list(labels)

    blank_handle = Line2D([0], [0], color="none", lw=0)
    blank_label = " "

    out_handles: List[object] = []
    out_labels: List[str] = []
    idx = 0
    for c in counts:
        take = min(int(c), n_items - idx)
        for _ in range(take):
            out_handles.append(handles[idx])
            out_labels.append(labels[idx])
            idx += 1
        # Fill the rest of this column to a uniform height.
        for _ in range(rows - int(c)):
            out_handles.append(blank_handle)
            out_labels.append(blank_label)

    # If the user passed very large counts (sum > n_items), the remainder are already blanks.
    # If counts were computed automatically, idx must consume all items.
    if idx != n_items:
        # Shouldn't happen, but avoid dropping items silently.
        for j in range(idx, n_items):
            out_handles.append(handles[j])
            out_labels.append(labels[j])

    return out_handles, out_labels


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
        choices=[
            "xx",
            "yy",
            "zz",
            "xy",
            "xz",
            "yz",
            "yx",
            "zx",
            "zy",
            "avg",
            "in-plane",
            "out-of-plane",
        ],
        help=(
            "Tensor component to plot [default: avg]. "
            "Use 'avg' for (xx+yy+zz)/3, 'in-plane' for (xx+yy)/2, 'out-of-plane' for zz."
        ),
    )

    p.add_argument(
        "--components",
        default=None,
        nargs="+",
        choices=[
            "xx",
            "yy",
            "zz",
            "xy",
            "xz",
            "yz",
            "yx",
            "zx",
            "zy",
            "avg",
            "in-plane",
            "out-of-plane",
        ],
        help=(
            "Plot multiple tensor components on the same figure (overrides --component). "
            "Examples: --components xx yy avg  |  --components in-plane out-of-plane avg."
        ),
    )

    p.add_argument(
        "--plot",
        nargs="+",
        default=None,
        choices=["sigma", "kappa", "lorenz"],
        help=(
            "Select which quantities to plot. Any subset of: sigma kappa lorenz. "
            "Examples: --plot sigma  |  --plot kappa lorenz  |  --plot sigma kappa lorenz. "
            "If omitted, plots all three."
        ),
    )

    p.add_argument(
        "--color",
        default=None,
        nargs="+",
        help=(
            "Per-file color(s). Provide one per file, or a single value to broadcast. "
            "Note: only allowed when plotting a single quantity via --plot (e.g. --plot sigma), "
            "otherwise colors are controlled by --color-sigma/--color-kappa/--color-lorenz."
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

    p.add_argument(
        "--ls",
        default=None,
        nargs="+",
        help=(
            "Per-file linestyle(s). Provide one per file, or a single value to broadcast. "
            "Examples: '-', ':', '-.', 'dashed'. Note: do not pass a bare '--' token; use 'dashed'."
        ),
    )

    p.add_argument("--lw", type=float, default=2.0, help="Line width (default: 2.0)")

    p.add_argument(
        "--label-fontsize",
        type=float,
        default=None,
        help="Font size for x/y axis labels (applies to all axes). If omitted, keep the default.",
    )

    p.add_argument("--no-bold", action="store_true", help="Disable bold text in the figure")
    p.add_argument("--bold-fonts", action="store_true", help="Force bold text across the whole figure")
    p.add_argument(
        "--fontsize",
        type=float,
        default=None,
        help="Global/default font size (rcParams). Does not override explicit per-item sizes.",
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
        "--legend-alpha",
        type=float,
        default=None,
        help="Optional alpha (0..1) for the main legend background frame.",
    )
    p.add_argument(
        "--legend-handlelength",
        type=float,
        default=2.6,
        help="Legend handle length (longer helps distinguish dashed vs dotted) [default: 2.6].",
    )
    p.add_argument(
        "--legend-handletextpad",
        type=float,
        default=0.4,
        help="Legend handle/text gap [default: 0.4].",
    )
    p.add_argument(
        "--legend-columnspacing",
        type=float,
        default=0.8,
        help="Legend column spacing (only relevant when --legend-ncol > 1) [default: 0.8].",
    )

    p.add_argument(
        "--legend-labelspacing",
        type=float,
        default=None,
        help=(
            "Legend row spacing (vertical spacing between entries). "
            "If omitted, uses matplotlib default. Typical tighter values: 0.2~0.4."
        ),
    )

    p.add_argument(
        "--legend-column-counts",
        default=None,
        nargs="+",
        type=int,
        help=(
            "Optional custom number of legend entries per column (left to right). "
            "Length must equal --legend-ncol. If the sum is smaller than the number of legend entries, "
            "the remainder is added to the last column. Shorter columns are padded with invisible blanks. "
            "Example (2 columns): --legend-ncol 2 --legend-column-counts 4 6"
        ),
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
    p.add_argument(
        "--system-alpha",
        type=float,
        default=None,
        help="Optional alpha (0..1) for the system label background frame.",
    )

    p.add_argument("--xtick-step", type=float, default=None, help="Major tick step on x-axis")
    p.add_argument(
        "--ytick-step",
        type=float,
        default=None,
        help="Major tick step on y-axis (applies to all three y-axes unless overridden).",
    )
    p.add_argument("--ytick-step-sigma", type=float, default=None, help="Major tick step on σ y-axis")
    p.add_argument("--ytick-step-kappa", type=float, default=None, help="Major tick step on κ y-axis")
    p.add_argument("--ytick-step-lorenz", type=float, default=None, help="Major tick step on Lorenz y-axis")

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

    apply_global_fontsize(args.fontsize)

    want_bold = bool(args.bold_fonts) or ((args.style != "prb") and (not args.no_bold))
    if want_bold:
        try:
            import matplotlib as mpl

            mpl.rcParams.update({"font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold"})
        except Exception:
            pass
        if args.style != "prb":
            apply_default_bold_rcparams()

    n_files = len(args.files)

    plot_set = {"sigma", "kappa", "lorenz"} if args.plot is None else {str(x).strip().lower() for x in args.plot}
    plot_set.discard("")
    unknown = sorted([x for x in plot_set if x not in {"sigma", "kappa", "lorenz"}])
    if unknown:
        raise SystemExit(f"Unknown --plot values: {unknown}. Choices: sigma kappa lorenz")
    if not plot_set:
        raise SystemExit("--plot expects at least one of: sigma kappa lorenz")
    want_sigma = "sigma" in plot_set
    want_kappa = "kappa" in plot_set
    want_lorenz = "lorenz" in plot_set
    n_qty = int(want_sigma) + int(want_kappa) + int(want_lorenz)

    if args.legend_ncol <= 0:
        raise SystemExit("--legend-ncol must be >= 1")

    legends_in = flatten_tokens(args.legend)
    if legends_in:
        legends_raw = broadcast_list(legends_in, n_files, "--legend")
    else:
        legends_raw = [default_label(f) for f in args.files]
    legends = [format_label(str(x), str(args.legend_format)) for x in legends_raw]

    # Component selection (single or multiple)
    comps_in = flatten_tokens(args.components)
    if comps_in:
        components = [str(c).strip().lower() for c in comps_in if str(c).strip()]
    else:
        components = [str(args.component).strip().lower()]
    for c in components:
        if c not in {
            "xx",
            "yy",
            "zz",
            "xy",
            "xz",
            "yz",
            "yx",
            "zx",
            "zy",
            "avg",
            "in-plane",
            "out-of-plane",
        }:
            raise SystemExit(f"Unknown component {c!r} in --components")

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

    ls_in = flatten_tokens(args.ls)
    if ls_in:
        ls_by_file_raw = broadcast_list(ls_in, n_files, "--ls")
        ls_by_file = [_normalize_linestyle_token(str(x)) for x in ls_by_file_raw]
    else:
        ls_by_file = None

    colors_in = flatten_tokens(args.color)
    if colors_in:
        colors_by_file = broadcast_list(colors_in, n_files, "--color")
    else:
        colors_by_file = None

    if colors_by_file is not None:
        if n_qty != 1:
            raise SystemExit(
                "--color (per-file colors) is only supported when plotting a single quantity. "
                "Use --plot sigma (or kappa/lorenz) together with --color, or use --color-sigma/--color-kappa/--color-lorenz."
            )

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
        fig, ax_left = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    else:
        fig, ax_left = plt.subplots(figsize=figsize_override, dpi=150)

    # Decide which variable lives on which axis.
    # For 1 variable: use single left axis.
    # For 2 variables: use left + one right (twinx).
    # For 3 variables: use left + right + offset right (as before).
    ax_sigma = None
    ax_kappa = None
    ax_lorenz = None

    requested = [q for q in ("sigma", "kappa", "lorenz") if q in plot_set]
    left_var = requested[0]
    right_var = requested[1] if len(requested) >= 2 else None
    third_var = requested[2] if len(requested) >= 3 else None

    var_to_ax = {left_var: ax_left}
    ax_right = None
    ax_third = None
    if right_var is not None:
        ax_right = ax_left.twinx()
        var_to_ax[right_var] = ax_right
    lorenz_spine_x = 1.0
    if third_var is not None:
        ax_third = ax_left.twinx()
        lorenz_spine_x = 1.0 + float(args.right_axis_spacing)
        ax_third.spines["right"].set_position(("axes", lorenz_spine_x))
        ax_third.set_frame_on(True)
        ax_third.patch.set_visible(False)
        var_to_ax[third_var] = ax_third

    ax_sigma = var_to_ax.get("sigma")
    ax_kappa = var_to_ax.get("kappa")
    ax_lorenz = var_to_ax.get("lorenz")

    # When plotting multiple components, use different linestyles per component
    # to keep them visually distinguishable.
    comp_ls_cycle = ["-", "--", ":", "-."]
    comp_to_ls = {c: comp_ls_cycle[i % len(comp_ls_cycle)] for i, c in enumerate(components)}

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

        per_file_color = None if colors_by_file is None else str(colors_by_file[i])

        for comp in components:
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

            # Legend: encode tensor component as a math subscript, e.g. σ_xx, κ_el,xx, L_xx;
            # in-plane/out-of-plane are shown as ∥/⊥.
            sigma_sym = _legend_sym_with_comp("\\sigma", comp, bold=want_bold)
            kappa_sym = _legend_kappa_el_with_comp(comp, bold=want_bold)
            lorenz_sym = _legend_sym_with_comp("L", comp, bold=want_bold)

            sigma_leg = f"{ds_label} {sigma_sym}"
            kappa_leg = f"{ds_label} {kappa_sym}"
            lorenz_leg = f"{ds_label} {lorenz_sym}"

            comp_ls = comp_to_ls.get(comp, "-")
            per_file_ls = None if ls_by_file is None else str(ls_by_file[i])

            if want_sigma and ax_sigma is not None:
                use_ls = comp_ls if len(components) > 1 else (per_file_ls if per_file_ls is not None else ls_sigma)
                use_color = per_file_color if per_file_color is not None else c_sigma
                z = 100 * _linestyle_rank(use_ls) + i
                ax_sigma.plot(
                    t,
                    sigma,
                    color=use_color,
                    lw=float(args.lw),
                    linestyle=use_ls,
                    marker=marker,
                    markersize=float(marker_sizes[i]),
                    markeredgewidth=0.0,
                    label=sigma_leg,
                    zorder=z,
                )
            if want_kappa and ax_kappa is not None:
                use_ls = comp_ls if len(components) > 1 else (per_file_ls if per_file_ls is not None else ls_kappa)
                use_color = per_file_color if per_file_color is not None else c_kappa
                z = 100 * _linestyle_rank(use_ls) + i
                ax_kappa.plot(
                    t,
                    kappa,
                    color=use_color,
                    lw=float(args.lw),
                    linestyle=use_ls,
                    marker=marker,
                    markersize=float(marker_sizes[i]),
                    markeredgewidth=0.0,
                    label=kappa_leg,
                    zorder=z,
                )
            if want_lorenz and ax_lorenz is not None:
                use_ls = comp_ls if len(components) > 1 else (per_file_ls if per_file_ls is not None else ls_lorenz)
                use_color = per_file_color if per_file_color is not None else c_lorenz
                z = 100 * _linestyle_rank(use_ls) + i
                ax_lorenz.plot(
                    t,
                    lorenz,
                    color=use_color,
                    lw=float(args.lw),
                    linestyle=use_ls,
                    marker=marker,
                    markersize=float(marker_sizes[i]),
                    markeredgewidth=0.0,
                    label=lorenz_leg,
                    zorder=z,
                )

    # Theoretical (Sommerfeld) Lorenz number reference line
    # Use the same color as L(T) curves to avoid introducing new colors.
    if want_lorenz and ax_lorenz is not None:
        L0 = 2.44e-8  # WΩ/K^2
        theory_leg = r"Theory $\boldsymbol{L}_{0}$" if want_bold else r"Theory $L_0$"
        ax_lorenz.axhline(
            L0,
            color="gray",
            linestyle="--",
            lw=float(args.lw) * 0.9,
            alpha=0.8,
            label=theory_leg,
        )

    ax_left.set_xlabel("Temperature (K)")
    if want_sigma and ax_sigma is not None:
        ax_sigma.set_ylabel(
            r"Electrical conductivity $\boldsymbol{\sigma}$ (S/m)" if want_bold else r"Electrical conductivity $\sigma$ (S/m)"
        )
    if want_kappa and ax_kappa is not None:
        ax_kappa.set_ylabel(
            r"Thermal conductivity $\boldsymbol{\kappa}_{\mathbf{el}}$ (W/mK)"
            if want_bold
            else r"Thermal conductivity $\kappa_{\mathrm{el}}$ (W/mK)"
        )
    if want_lorenz and ax_lorenz is not None:
        # Keep the formula consistent regardless of axis layout.
        # In bold mode, mathtext does not automatically bold plain letters like 'T' or unit symbols.
        # Use \mathbf / \boldsymbol explicitly.
        ax_lorenz.set_ylabel(
            (
                r"Lorenz number $L=\boldsymbol{\kappa}_{\mathbf{el}}/(\boldsymbol{\sigma}\,\mathbf{T})$ "
                r"($\mathbf{W}\,\boldsymbol{\Omega}/\mathbf{K}^{2}$)"
            )
            if want_bold
            else r"Lorenz number $L=\kappa_{\mathrm{el}}/(\sigma T)$ (W$\Omega$/K$^2$)"
        )

    ystep_sigma = args.ytick_step_sigma if args.ytick_step_sigma is not None else args.ytick_step
    ystep_kappa = args.ytick_step_kappa if args.ytick_step_kappa is not None else args.ytick_step
    ystep_lorenz = args.ytick_step_lorenz if args.ytick_step_lorenz is not None else args.ytick_step

    if ax_sigma is not None and want_sigma:
        apply_tick_steps(ax_sigma, xtick_step=args.xtick_step, ytick_step=ystep_sigma, ylog=False)
    if ax_kappa is not None and want_kappa:
        apply_tick_steps(ax_kappa, xtick_step=args.xtick_step, ytick_step=ystep_kappa, ylog=False)
    if ax_lorenz is not None and want_lorenz:
        apply_tick_steps(ax_lorenz, xtick_step=args.xtick_step, ytick_step=ystep_lorenz, ylog=False)

    if xlim:
        ax_left.set_xlim(*xlim)

    if ylim_sigma:
        if ax_sigma is not None and want_sigma:
            ax_sigma.set_ylim(*ylim_sigma)
    if ylim_kappa:
        if ax_kappa is not None and want_kappa:
            ax_kappa.set_ylim(*ylim_kappa)
    if ylim_lorenz:
        if ax_lorenz is not None and want_lorenz:
            ax_lorenz.set_ylim(*ylim_lorenz)

    if args.title:
        ax_left.set_title(args.title)

    if args.style != "prb":
        ax_left.grid(True, alpha=0.25)

    # One combined legend from all axes.
    handles: List[object] = []
    labels: List[str] = []
    for a in (ax_sigma, ax_kappa, ax_lorenz):
        if a is None:
            continue
        h, l = a.get_legend_handles_labels()
        if h and l:
            handles.extend(h)
            labels.extend(l)

    if int(args.legend_ncol) > 1 and handles:
        # Default behavior: when the number of items does not divide evenly,
        # push the extra items into the RIGHTMOST columns (without changing order).
        # If --legend-column-counts is provided, it controls per-column real counts.
        handles, labels = _pad_legend_columns_preserve_order(
            handles,
            labels,
            ncol=int(args.legend_ncol),
            column_real_counts=(args.legend_column_counts if args.legend_column_counts is not None else None),
        )

    legend_loc = str(args.legend_loc)
    if legend_bbox is not None and legend_loc.strip().lower() == "best":
        legend_loc = "upper left"

    legend_frameon = args.legend_alpha is not None
    # Attach to the topmost axis (drawn last) so it won't be covered by curves.
    legend_ax = ax_lorenz if ax_lorenz is not None else (ax_kappa if ax_kappa is not None else ax_left)

    if legend_bbox is None:
        leg = legend_ax.legend(
            handles,
            labels,
            loc=legend_loc,
            frameon=legend_frameon,
            ncols=int(args.legend_ncol),
            fontsize=args.legend_fontsize,
            handletextpad=float(args.legend_handletextpad),
            handlelength=float(args.legend_handlelength),
            columnspacing=float(args.legend_columnspacing),
            labelspacing=args.legend_labelspacing,
        )
    else:
        leg = legend_ax.legend(
            handles,
            labels,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            bbox_transform=legend_ax.transAxes,
            frameon=legend_frameon,
            ncols=int(args.legend_ncol),
            fontsize=args.legend_fontsize,
            handletextpad=float(args.legend_handletextpad),
            handlelength=float(args.legend_handlelength),
            columnspacing=float(args.legend_columnspacing),
            labelspacing=args.legend_labelspacing,
            borderaxespad=0.0,
        )
    apply_legend_frame(leg, alpha=args.legend_alpha)
    try:
        leg.set_zorder(1000)
    except Exception:
        pass

    # Global system annotation (pure text)
    if args.system is not None and str(args.system).strip():
        legend_ax.add_artist(leg)
        sys_lab = format_label(str(args.system), str(args.system_format))
        handle = Line2D([0], [0], color="none", lw=0, label=sys_lab)
        fs = args.system_fontsize
        if fs is None:
            try:
                fs = float(ax_left.yaxis.label.get_size()) * 1.15
            except Exception:
                fs = None

        system_loc = str(args.system_loc)
        if system_bbox is not None and system_loc.strip().lower() == "best":
            system_loc = "upper left"

        system_frameon = args.system_alpha is not None
        if system_bbox is None:
            leg_sys = legend_ax.legend(
                handles=[handle],
                loc=system_loc,
                frameon=system_frameon,
                fontsize=fs,
                handlelength=0,
                handletextpad=0.0,
            )
        else:
            leg_sys = legend_ax.legend(
                handles=[handle],
                loc=system_loc,
                bbox_to_anchor=system_bbox,
                bbox_transform=legend_ax.transAxes,
                frameon=system_frameon,
                fontsize=fs,
                handlelength=0,
                handletextpad=0.0,
                borderaxespad=0.0,
            )
        apply_legend_frame(leg_sys, alpha=args.system_alpha)
        try:
            if leg_sys is not None:
                leg_sys.set_zorder(1001)
        except Exception:
            pass
        if leg_sys is not None:
            for t in leg_sys.get_texts():
                t.set_fontweight("bold")

    for a in (ax_left, ax_right, ax_third):
        if a is None:
            continue
        apply_plot_style(a, legend=(leg if a is legend_ax else None), bold=want_bold, sci_y="auto", ylog=False)

    if args.label_fontsize is not None:
        fs = float(args.label_fontsize)
        if fs <= 0:
            raise SystemExit("--label-fontsize must be > 0")
        for a in (ax_left, ax_right, ax_third):
            if a is None:
                continue
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

    if want_lorenz and ax_lorenz is not None:
        # Only add the manual scale annotation when Lorenz is plotted on an offset/twin axis.
        # For Lorenz-only plots (single left axis), keep Matplotlib's automatic offset text.
        if n_qty != 1:
            try:
                ax_lorenz.yaxis.get_offset_text().set_visible(False)
            except Exception:
                pass

            ax_lorenz.text(
                lorenz_spine_x,
                1.01,
                (r"$\boldsymbol{\times}\ 10^{\mathbf{-8}}$" if want_bold else r"$\times 10^{-8}$"),
                transform=ax_lorenz.transAxes,
                ha="left",
                va="bottom",
                fontsize=(
                    float(args.label_fontsize)
                    if args.label_fontsize is not None
                    else ax_lorenz.yaxis.label.get_size()
                ),
                fontweight=("bold" if want_bold else "normal"),
            )

    # Reserve extra right margin only when we have the 3rd (offset) y-axis.
    if ax_third is not None:
        fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    else:
        fig.tight_layout()
    if args.out:
        fig.savefig(args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
