#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import List, Optional

import matplotlib.pyplot as plt

from perturbo_meanfp_io import (
    apply_plot_style,
    bin_statistics,
    default_label,
    get_config_band_series,
    get_energy_by_band,
    get_mu_ev,
    load_meanfp_yaml,
    parse_band_selection,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot electron scattering rate (1/tau) from Perturbo *_meanfp.yml (supports multiple files for comparison)."
    )
    p.add_argument("files", nargs="+", help="One or more Perturbo meanfp YAML files")
    p.add_argument("--labels", default=None, help="Comma-separated labels for each file")
    p.add_argument("--config", type=int, default=1, help="Configuration index [default: 1]")
    p.add_argument(
        "--x",
        choices=["energy", "e_minus_mu"],
        default="e_minus_mu",
        help="x-axis: energy E (eV) or E-mu (eV) [default: e_minus_mu]",
    )
    p.add_argument("--bands", default=None, help='Band indices in YAML space, e.g. "1-6" or "1,3,5" [default: all]')
    p.add_argument(
        "--unit",
        choices=["ps^-1", "s^-1"],
        default="ps^-1",
        help="Rate unit [default: ps^-1] (note: ps^-1 == THz)",
    )
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

    labels: List[str]
    if args.labels is None:
        labels = [default_label(f) for f in args.files]
    else:
        labels = [x.strip() for x in args.labels.split(",")]
        if len(labels) != len(args.files):
            raise SystemExit(f"--labels count {len(labels)} != files count {len(args.files)}")

    xlim = _parse_lim(args.xlim)
    ylim = _parse_lim(args.ylim)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)

    for path, label in zip(args.files, labels):
        data = load_meanfp_yaml(path)
        e_by_band = get_energy_by_band(data)
        bands = parse_band_selection(args.bands, sorted(e_by_band.keys()))

        mu = get_mu_ev(data, args.config) if args.x == "e_minus_mu" else 0.0

        xs = []
        ys = []

        for b in bands:
            tau_fs = get_config_band_series(data, config_index=args.config, band=b, key="relaxation time")
            e = e_by_band[b]
            if len(e) != len(tau_fs):
                raise SystemExit(
                    f"Length mismatch in {path} config {args.config} band {b}: E={len(e)} tau={len(tau_fs)}"
                )

            # scattering rate = 1/tau
            # tau_fs: fs
            # ps^-1: 1/ps = 1000/tau_fs
            for ei, tf in zip(e, tau_fs):
                if tf <= 0.0:
                    continue
                x = ei - mu
                if args.unit == "ps^-1":
                    rate = 1000.0 / tf
                else:
                    rate = 1.0 / (tf * 1e-15)
                xs.append(x)
                ys.append(rate)

        if args.mode == "scatter":
            ax.scatter(xs, ys, s=args.s, alpha=args.alpha, label=label)
        else:
            cx, cy = bin_statistics(xs, ys, bin_width=args.bin_width, reducer=args.reducer)
            ax.plot(cx, cy, lw=2.0, label=label)

    ax.set_xlabel("E - μ (eV)" if args.x == "e_minus_mu" else "Energy E (eV)")
    ax.set_ylabel(r"Scattering rate $1/\tau$ (ps$^{-1}$)" if args.unit == "ps^-1" else r"Scattering rate $1/\tau$ (s$^{-1}$)")

    if args.ylog:
        ax.set_yscale("log")

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title(f"Electron scattering rate (config {args.config})")

    ax.grid(True, alpha=0.25)
    leg = ax.legend(frameon=False)
    apply_plot_style(ax, legend=leg, bold=not args.no_bold, sci_y=args.sci_y, ylog=args.ylog)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
