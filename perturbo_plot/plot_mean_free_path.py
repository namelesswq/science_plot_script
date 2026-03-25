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
    broadcast_list,
    bin_statistics,
    default_label,
    flatten_tokens,
    format_label,
    get_config_band_series,
    get_energy_by_band,
    get_mu_ev,
    load_meanfp_yaml,
    parse_figsize,
    parse_xy,
    parse_band_selection,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot electron mean free path (MFP) from Perturbo *_meanfp.yml (supports multiple files for comparison)."
    )
    p.add_argument("files", nargs="+", help="One or more Perturbo meanfp YAML files")

    p.add_argument(
        "--style",
        choices=["prb", "default"],
        default="prb",
        help="Plot style preset. 'prb' uses SciencePlots if available [default: prb].",
    )

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

    p.add_argument("--config", type=int, default=1, help="Configuration index [default: 1]")
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
    p.add_argument(
        "--sci-y",
        choices=["auto", "on", "off"],
        default="auto",
        help="Y-axis scientific notation factor (\u00d710^n) [default: auto]",
    )
    p.add_argument("--xlim", default=None, help='x limits "xmin,xmax" in eV')
    p.add_argument("--ylim", default=None, help='y limits "ymin,ymax"')
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
    else:
        if not args.no_bold:
            apply_default_bold_rcparams()

    legend_bbox = parse_xy(args.legend_bbox)
    system_bbox = parse_xy(args.system_bbox)

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
        bands = parse_band_selection(args.bands, sorted(e_by_band.keys()))

        mu = get_mu_ev(data, args.config) if args.x == "e_minus_mu" else 0.0

        xs = []
        ys = []

        for b in bands:
            mfp_nm = get_config_band_series(data, config_index=args.config, band=b, key="MFP")
            e = e_by_band[b]
            if len(e) != len(mfp_nm):
                raise SystemExit(
                    f"Length mismatch in {path} config {args.config} band {b}: E={len(e)} MFP={len(mfp_nm)}"
                )
            xs.extend([ei - mu for ei in e])
            ys.extend(mfp_nm)

        if args.mode == "scatter":
            ax.scatter(xs, ys, s=args.s, alpha=args.alpha, label=label)
        else:
            cx, cy = bin_statistics(xs, ys, bin_width=args.bin_width, reducer=args.reducer)
            ax.plot(cx, cy, lw=2.0, label=label)

    ax.set_xlabel(r"$E-E_{\mathrm{f}}$ (eV)" if args.x == "e_minus_mu" else r"$E$ (eV)")
    ax.set_ylabel("Mean Free Path (nm)")

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
        leg = ax.legend(loc=str(args.legend_loc), frameon=False, fontsize=args.legend_fontsize)
    else:
        leg = ax.legend(
            loc=str(args.legend_loc),
            bbox_to_anchor=legend_bbox,
            bbox_transform=ax.transAxes,
            frameon=False,
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
        bold=(not args.no_bold) and (args.style != "prb"),
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
