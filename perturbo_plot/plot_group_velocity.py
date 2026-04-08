#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from perturbo_meanfp_io import (
    apply_plot_style,
    apply_scienceplots_prb_style,
    apply_default_bold_rcparams,
    apply_global_fontsize,
    apply_tick_steps,
    apply_legend_frame,
    broadcast_list,
    bin_statistics,
    default_label,
    flatten_tokens,
    format_label,
    get_energy_by_band,
    get_mu_ev,
    get_velocity_by_band,
    load_meanfp_yaml,
    parse_figsize,
    parse_xy,
    parse_band_selection,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot electron group velocity (|v|) from Perturbo *_meanfp.yml (supports multiple files for comparison)."
    )
    p.add_argument("files", nargs="+", help="One or more Perturbo meanfp YAML files")

    p.add_argument(
        "--style",
        choices=["prb", "default"],
        default="prb",
        help="Plot style preset. 'prb' uses SciencePlots if available [default: prb].",
    )

    # Dataset legend (colored handles)
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
        "--labels",
        default=None,
        help="(Deprecated) Comma-separated labels for each file. Use --legend instead.",
    )
    p.add_argument(
        "--legend-format",
        choices=["chem", "raw"],
        default="raw",
        help="Render --legend text with subscripts (chem) or raw text (raw) [default: raw].",
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
        "--legend-alpha",
        type=float,
        default=None,
        help="Optional alpha (0..1) for the dataset legend background frame.",
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
    p.add_argument(
        "--system-alpha",
        type=float,
        default=None,
        help="Optional alpha (0..1) for the system label background frame.",
    )

    p.add_argument("--config", type=int, default=1, help="Configuration index (used only for mu shift in x-axis) [default: 1]")
    p.add_argument(
        "--x",
        choices=["energy", "e_minus_mu"],
        default="e_minus_mu",
        help="x-axis: energy E (eV) or E-mu (eV) [default: e_minus_mu]",
    )
    p.add_argument("--bands", default=None, help='Band indices in YAML space, e.g. "1-6" or "1,3,5" [default: all]')
    p.add_argument("--mode", choices=["binned", "scatter"], default="scatter", help="Plot mode [default: binned]")
    p.add_argument("--bin-width", type=float, default=0.01, help="Bin width in eV for binned mode [default: 0.01]")
    p.add_argument("--reducer", choices=["median", "mean"], default="median", help="Reducer for binned mode")
    p.add_argument("--alpha", type=float, default=0.15, help="Alpha for scatter mode [default: 0.15]")
    p.add_argument("--s", type=float, default=4.0, help="Marker size for scatter mode [default: 4]")
    p.add_argument("--ylog", action="store_true", help="Use log scale on y-axis")
    p.add_argument("--no-bold", action="store_true", help="Disable bold text in the figure")
    p.add_argument("--bold-fonts", action="store_true", help="Force bold text across the whole figure")
    p.add_argument(
        "--fontsize",
        type=float,
        default=None,
        help="Global/default font size (rcParams). Does not override explicit per-item sizes.",
    )
    p.add_argument(
        "--sci-y",
        choices=["auto", "on", "off"],
        default="auto",
        help="Y-axis scientific notation factor (\u00d710^n) [default: auto]",
    )
    p.add_argument("--xlim", default=None, help='x limits "xmin,xmax" in eV')
    p.add_argument("--ylim", default=None, help='y limits "ymin,ymax"')
    p.add_argument("--xtick-step", type=float, default=None, help="Major tick step on x-axis")
    p.add_argument("--ytick-step", type=float, default=None, help="Major tick step on y-axis (ignored with --ylog)")
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


def _parse_lim(s: Optional[str]):
    if not s:
        return None
    a, b = s.split(",", 1)
    return float(a), float(b)


def main() -> None:
    args = _build_parser().parse_args()

    if args.label_fontsize is not None and float(args.label_fontsize) <= 0:
        raise SystemExit("--label-fontsize must be > 0")

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

    legend_bbox = parse_xy(args.legend_bbox)
    system_bbox = parse_xy(args.system_bbox)

    # dataset legend labels
    if args.legend is not None:
        legends_in = flatten_tokens(args.legend)
    elif args.labels is not None:
        legends_in = [x.strip() for x in str(args.labels).split(",") if x.strip()]
    else:
        legends_in = []

    if legends_in:
        legends_raw = broadcast_list(legends_in, len(args.files), "--legend")
    else:
        legends_raw = [default_label(f) for f in args.files]
    legends = [format_label(str(x), str(args.legend_format)) for x in legends_raw]

    xlim = _parse_lim(args.xlim)
    ylim = _parse_lim(args.ylim)
    figsize_override = parse_figsize(args.figsize)

    if figsize_override is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    else:
        fig, ax = plt.subplots(figsize=figsize_override, dpi=150)

    for path, label in zip(args.files, legends):
        data = load_meanfp_yaml(path)
        e_by_band = get_energy_by_band(data)
        v_by_band = get_velocity_by_band(data)

        bands = parse_band_selection(args.bands, sorted(e_by_band.keys()))
        mu = get_mu_ev(data, args.config) if args.x == "e_minus_mu" else 0.0

        xs = []
        ys = []
        for b in bands:
            e = e_by_band[b]
            v = v_by_band[b]
            if len(e) != len(v):
                raise SystemExit(f"Length mismatch in {path} band {b}: E={len(e)} v={len(v)}")
            xs.extend([x - mu for x in e])
            ys.extend(v)

        if args.mode == "scatter":
            ax.scatter(xs, ys, s=args.s, alpha=args.alpha, label=label)
        else:
            cx, cy = bin_statistics(xs, ys, bin_width=args.bin_width, reducer=args.reducer)
            ax.plot(cx, cy, lw=2.0, label=label)

    if args.x == "e_minus_mu":
        ax.set_xlabel(r"$\mathbf{E-E_{f}}\ (\mathbf{eV})$" if want_bold else r"$E-E_{\mathrm{f}}$ (eV)")
    else:
        ax.set_xlabel(r"$\mathbf{E}\ (\mathbf{eV})$" if want_bold else r"$E$ (eV)")

    ax.set_ylabel(r"Electron Group Velocity $|\boldsymbol{v}|$ (m/s)" if want_bold else r"Electron Group Velocity $|v|$ (m/s)")

    if args.ylog:
        ax.set_yscale("log")

    apply_tick_steps(ax, xtick_step=args.xtick_step, ytick_step=args.ytick_step, ylog=args.ylog)

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    if args.title:
        ax.set_title(args.title)

    if args.style != "prb":
        ax.grid(True, alpha=0.25)

    # main legend (datasets)
    legend_loc = str(args.legend_loc)
    if legend_bbox is not None and legend_loc.strip().lower() == "best":
        legend_loc = "upper left"

    legend_frameon = args.legend_alpha is not None
    if legend_bbox is None:
        leg = ax.legend(
            loc=legend_loc,
            frameon=legend_frameon,
            fontsize=args.legend_fontsize,
            handletextpad=0.4,
            handlelength=1.2,
        )
    else:
        leg = ax.legend(
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            bbox_transform=ax.transAxes,
            frameon=legend_frameon,
            fontsize=args.legend_fontsize,
            handletextpad=0.4,
            handlelength=1.2,
            borderaxespad=0.0,
        )
    apply_legend_frame(leg, alpha=args.legend_alpha)

    # global system annotation (separate pure-text legend)
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

        system_loc = str(args.system_loc)
        if system_bbox is not None and system_loc.strip().lower() == "best":
            system_loc = "upper left"

        system_frameon = args.system_alpha is not None
        if system_bbox is None:
            leg_sys = ax.legend(
                handles=[handle],
                loc=system_loc,
                frameon=system_frameon,
                fontsize=fs,
                handlelength=0,
                handletextpad=0.0,
            )
        else:
            leg_sys = ax.legend(
                handles=[handle],
                loc=system_loc,
                bbox_to_anchor=system_bbox,
                bbox_transform=ax.transAxes,
                frameon=system_frameon,
                fontsize=fs,
                handlelength=0,
                handletextpad=0.0,
                borderaxespad=0.0,
            )
        apply_legend_frame(leg_sys, alpha=args.system_alpha)
        if leg_sys is not None:
            for t in leg_sys.get_texts():
                t.set_fontweight("bold")

    apply_plot_style(
        ax,
        legend=leg,
        bold=want_bold,
        label_fontsize=args.label_fontsize,
        sci_y=args.sci_y,
        ylog=args.ylog,
    )
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
