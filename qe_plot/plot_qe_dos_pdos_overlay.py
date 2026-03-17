#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Overlay total DOS and element/orbital-resolved PDOS from Quantum ESPRESSO projwfc outputs.\n"
            "- Total DOS file:  <prefix>.pdos.pdos_tot\n"
            "- PDOS files:      <prefix>.pdos.pdos_atm#N(El)_wfc#M(orb)\n\n"
            "By default, the script sums PDOS over atoms for the same (element, wfc-index, orbital) and plots them on one figure.\n"
            "Use --merge-wfc to merge different wfc indices (e.g. wfc#1(s) + wfc#4(s)) into a single (element, orbital) curve."
        )
    )

    p.add_argument(
        "--tot",
        required=True,
        help="Total DOS file, e.g. zr2sc.pdos.pdos_tot",
    )
    p.add_argument(
        "--pdos",
        nargs="*",
        default=None,
        help="Explicit list of pdos_atm#... files. If omitted, auto-glob from --tot prefix.",
    )
    p.add_argument(
        "--pdos-glob",
        default=None,
        help="Glob pattern for PDOS files (overrides auto). Example: 'zr2sc.pdos.pdos_atm#*'",
    )

    p.add_argument(
        "--elements",
        default=None,
        help="Comma-separated element filter, e.g. 'Zr,S,C'. Default: all found.",
    )
    p.add_argument(
        "--orbitals",
        default=None,
        help=(
            "Comma-separated orbital filter, e.g. 's,p,d,f'. Default: all found. "
            "Special tokens: 'no-tot' disables plotting total DOS; 'tot' forces plotting total DOS."
        ),
    )

    p.add_argument(
        "--style",
        choices=["prb", "default"],
        default="prb",
        help="Plot style preset. 'prb' is suitable for PRB manuscripts. Default: prb.",
    )

    p.add_argument(
        "--figsize",
        default=None,
        help='Figure size "width,height" in inches. Overrides the preset size from --style (e.g. "3.4,2.6").',
    )

    p.add_argument(
        "--merge-wfc",
        action="store_true",
        help="Merge PDOS with same element+orbital across different wfc indices (default: keep wfc# separated)",
    )

    p.add_argument(
        "--n0",
        default=None,
        help=(
            "Per-element starting principal quantum number for relabeling wfc# projectors. "
            "Format: 'Zr=4,S=3,C=2'. When provided (and not using --merge-wfc), labels become e.g. Zr-4s, Zr-5s, ... "
            "by ranking wfc# within each (element, orbital) in increasing wfc# order."
        ),
    )

    p.add_argument(
        "--xlim",
        default=None,
        help='x limits "xmin,xmax" in eV (energy axis is whatever is in the files; often E-Ef).',
    )
    p.add_argument("--ylim", default=None, help='y limits "ymin,ymax"')

    p.add_argument(
        "--fermi-line",
        action="store_true",
        help="Draw a vertical line at E=0 (useful when energies are E-Ef).",
    )

    p.add_argument(
        "--fermi",
        type=float,
        default=None,
        help=(
            "Fermi energy in eV. If provided, the script shifts the energy axis as E -> E - Ef so that Ef is at 0 eV. "
            "(Applied consistently to both total DOS and PDOS.)"
        ),
    )

    p.add_argument("--ylog", action="store_true", help="Use log scale on y-axis")
    p.add_argument("--no-bold", action="store_true", help="Disable bold text in the figure")
    p.add_argument(
        "--sci-y",
        choices=["auto", "on", "off"],
        default="auto",
        help="Y-axis scientific notation factor (×10^n) [default: auto]",
    )

    p.add_argument(
        "--tot-col",
        type=int,
        default=1,
        help="Which column to use for total DOS (0-based). Typical tot file: E(0), dos(E)(1), pdos(E)(2). Default 1.",
    )
    p.add_argument(
        "--pdos-col",
        type=int,
        default=1,
        help=(
            "Which column to use for orbital-resolved DOS in pdos_atm#..._wfc#...(orb) files (0-based). "
            "Typical wfc file: E(0), ldos(E)(1), pdos(E)(2). Default 1 (ldos(E))."
        ),
    )

    p.add_argument("--title", default=None, help="Plot title")
    p.add_argument("--out", default="dos_pdos_overlay.png", help="Output image path")
    p.add_argument("--show", action="store_true", help="Show interactively")

    return p


def _parse_lim(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = s.split(",", 1)
    return float(a), float(b)


def _parse_figsize(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = s.split(",", 1)
    w = float(a)
    h = float(b)
    if w <= 0 or h <= 0:
        raise SystemExit(f"Invalid --figsize {s!r}: width and height must be > 0")
    return w, h


def _parse_n0_map(s: Optional[str]) -> Dict[str, int]:
    if not s:
        return {}
    out: Dict[str, int] = {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for part in parts:
        if "=" not in part:
            raise SystemExit(f"Invalid --n0 entry {part!r}. Expected like 'Zr=4'.")
        el, val = part.split("=", 1)
        el = el.strip()
        val = val.strip()
        if not el or not val:
            raise SystemExit(f"Invalid --n0 entry {part!r}. Expected like 'Zr=4'.")
        try:
            out[el] = int(val)
        except ValueError as e:
            raise SystemExit(f"Invalid --n0 value for element {el!r}: {val!r} (must be integer)") from e
    return out


def _build_n_label_map(
    keys: Sequence[Tuple[str, int, str]],
    n0_map: Dict[str, int],
) -> Dict[Tuple[str, int, str], str]:
    """Map (el,wfc,orb) -> label like 'Zr-4s' based on per-element n0.

    For each element and each orbital type separately, wfc indices are sorted,
    and assigned n = n0 + rank (rank starting at 0).
    """
    if not n0_map:
        return {}

    label_map: Dict[Tuple[str, int, str], str] = {}
    # Group wfc indices by (element, orbital)
    by_el_orb: Dict[Tuple[str, str], List[int]] = {}
    for el, wfc_idx, orb in keys:
        by_el_orb.setdefault((el, orb), []).append(int(wfc_idx))

    for (el, orb), wfc_list in by_el_orb.items():
        if el not in n0_map:
            continue
        wfc_sorted = sorted(set(wfc_list))
        n0 = int(n0_map[el])
        for rank, wfc_idx in enumerate(wfc_sorted):
            n = n0 + rank
            label_map[(el, wfc_idx, orb)] = f"{el}-{n}{orb}"

    return label_map


def _load_two_cols(path: str, xcol: int, ycol: int) -> Tuple[np.ndarray, np.ndarray]:
    # QE files often start with comment lines beginning with '#'
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] <= max(xcol, ycol):
        raise RuntimeError(f"Unexpected table in {path}: shape={data.shape}")
    x = np.asarray(data[:, xcol], dtype=float)
    y = np.asarray(data[:, ycol], dtype=float)
    return x, y


_RE_EL = re.compile(r"atm#\d+\(([^)]+)\)")
_RE_WFC = re.compile(r"wfc#(\d+)\(([^)]+)\)")


def _parse_element_wfc_orbital_from_name(path: str) -> Tuple[str, int, str]:
    base = os.path.basename(path)
    m1 = _RE_EL.search(base)
    m2 = _RE_WFC.search(base)
    if not m1 or not m2:
        raise RuntimeError(f"Cannot parse element/orbital from filename: {base}")
    el = m1.group(1).strip()
    wfc_idx = int(m2.group(1))
    orb = m2.group(2).strip()
    return el, wfc_idx, orb


def _apply_style(ax, *, legend=None, bold: bool = True, sci_y: str = "auto", ylog: bool = False) -> None:
    if bold:
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            t.set_fontweight("bold")
        if legend is not None:
            for t in legend.get_texts():
                t.set_fontweight("bold")

    sci_y = (sci_y or "auto").lower()
    if ylog:
        sci_y = "off"

    if sci_y not in {"auto", "on", "off"}:
        raise ValueError("--sci-y must be auto|on|off")

    def _should() -> bool:
        if sci_y == "on":
            return True
        if sci_y == "off":
            return False
        lo, hi = ax.get_ylim()
        m = max(abs(float(lo)), abs(float(hi)))
        if m == 0.0:
            return False
        import math

        exp = math.floor(math.log10(m))
        return exp >= 3 or exp <= -3

    if _should():
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        if bold:
            ax.yaxis.get_offset_text().set_fontweight("bold")


def _apply_scienceplots_prb_style() -> None:
    """Apply a SciencePlots-based PRB-friendly style (non-LaTeX)"""
    try:
        import scienceplots  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "SciencePlots is required for --style prb but could not be imported.\n"
            "Install it with: pip install SciencePlots\n"
            f"Original error: {e}"
        )

    # Register and apply styles
    plt.style.use(["science", "no-latex"])


def main() -> None:
    args = _build_parser().parse_args()

    xlim = _parse_lim(args.xlim)
    ylim = _parse_lim(args.ylim)
    n0_map = _parse_n0_map(args.n0)
    figsize_override = _parse_figsize(args.figsize)

    # total DOS
    e_tot, dos_tot = _load_two_cols(args.tot, xcol=0, ycol=args.tot_col)

    # find PDOS files
    pdos_files: List[str]
    if args.pdos is not None and len(args.pdos) > 0:
        pdos_files = list(args.pdos)
    else:
        if args.pdos_glob:
            pattern = args.pdos_glob
        else:
            # auto: use the part before '.pdos.pdos_tot'
            base = os.path.basename(args.tot)
            if base.endswith(".pdos.pdos_tot"):
                prefix = base[: -len(".pdos.pdos_tot")]
            else:
                prefix = os.path.splitext(base)[0]
            # Search in the same directory as the tot file
            pattern = os.path.join(os.path.dirname(args.tot) or ".", f"{prefix}.pdos.pdos_atm#*")
        pdos_files = sorted(glob.glob(pattern))

    if not pdos_files:
        msg = [
            "No PDOS files found.",
            "- If you used --pdos-glob, please check the pattern.",
        ]
        if args.pdos_glob:
            msg.append(f"- Your --pdos-glob was: {args.pdos_glob!r}")
            # Common typo: pdos_atom#* vs pdos_atm#*
            if "pdos_atom#" in args.pdos_glob:
                alt = args.pdos_glob.replace("pdos_atom#", "pdos_atm#")
                alt_hits = sorted(glob.glob(alt))
                if alt_hits:
                    msg.append(f"- Did you mean: {alt!r} ? (found {len(alt_hits)} files)")

            # Also try a generic search in the same directory as --tot
            tot_dir = os.path.dirname(args.tot) or "."
            generic = os.path.join(tot_dir, "*.pdos.pdos_atm#*")
            generic_hits = sorted(glob.glob(generic))
            if generic_hits:
                msg.append(f"- In this directory, files matching {generic!r}: {len(generic_hits)}")

        msg.append("Provide --pdos (explicit file list) or a correct --pdos-glob.")
        raise SystemExit("\n".join(msg))

    elements_filter = None
    if args.elements:
        elements_filter = {x.strip() for x in args.elements.split(",") if x.strip()}

    orbitals_filter = None
    plot_total = True
    if args.orbitals:
        raw = [x.strip() for x in args.orbitals.split(",") if x.strip()]
        norm = [x.lower().replace("_", "-") for x in raw]
        if any(t in {"no-tot", "no-total", "notot", "nototal"} for t in norm):
            plot_total = False
        if any(t in {"tot", "total"} for t in norm):
            plot_total = True

        # keep only real orbital tokens
        keep: List[str] = []
        for t_raw, t_norm in zip(raw, norm):
            if t_norm in {"no-tot", "no-total", "notot", "nototal", "tot", "total"}:
                continue
            keep.append(t_raw)
        if keep:
            orbitals_filter = set(keep)

    # Sum PDOS by (element, wfc_index, orbital) unless --merge-wfc
    groups_wfc: Dict[Tuple[str, int, str], np.ndarray] = {}
    groups_merged: Dict[Tuple[str, str], np.ndarray] = {}
    energy_ref: Optional[np.ndarray] = None

    for f in pdos_files:
        el, wfc_idx, orb = _parse_element_wfc_orbital_from_name(f)
        if elements_filter is not None and el not in elements_filter:
            continue
        if orbitals_filter is not None and orb not in orbitals_filter:
            continue

        e, y = _load_two_cols(f, xcol=0, ycol=args.pdos_col)

        if energy_ref is None:
            energy_ref = e
        else:
            if len(e) != len(energy_ref) or np.max(np.abs(e - energy_ref)) > 1e-8:
                raise SystemExit(
                    f"Energy grid mismatch among PDOS files. Offending file: {f}. "
                    "Please regenerate PDOS with consistent energy grid."
                )

        if args.merge_wfc:
            key2 = (el, orb)
            if key2 not in groups_merged:
                groups_merged[key2] = np.zeros_like(y, dtype=float)
            groups_merged[key2] += y
        else:
            key3 = (el, wfc_idx, orb)
            if key3 not in groups_wfc:
                groups_wfc[key3] = np.zeros_like(y, dtype=float)
            groups_wfc[key3] += y

    if args.merge_wfc:
        if not groups_merged:
            raise SystemExit("No PDOS series selected after applying filters.")
    else:
        if not groups_wfc:
            raise SystemExit("No PDOS series selected after applying filters.")

    # Use PDOS energy grid if available; otherwise total
    e_pdos = energy_ref if energy_ref is not None else e_tot

    # Optional Fermi shift: E -> E - Ef
    if args.fermi is not None:
        e_tot = e_tot - float(args.fermi)
        e_pdos = e_pdos - float(args.fermi)

    # Also require total grid to match pdos grid (common for QE)
    if len(e_tot) != len(e_pdos) or np.max(np.abs(e_tot - e_pdos)) > 1e-8:
        # not fatal; we can still plot with different x arrays
        pass

    # Plot
    if args.style == "prb":
        # One-click PRB-ish formatting from SciencePlots.
        _apply_scienceplots_prb_style()
        if figsize_override is not None:
            fig, ax = plt.subplots(figsize=figsize_override)
        else:
            fig, ax = plt.subplots()
    else:
        if figsize_override is not None:
            fig, ax = plt.subplots(figsize=figsize_override)
        else:
            fig, ax = plt.subplots()

    # optional bold text (legacy behavior)
    if (args.style != "prb") and (not args.no_bold):
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["axes.linewidth"] = 2

    # total DOS: black (optional)
    if plot_total:
        ax.plot(e_tot, dos_tot, color="black", lw=1.6 if args.style == "prb" else 2.4, label="Total DOS")

    # PDOS curves: solid lines, different colors
    if args.merge_wfc:
        keys_sorted = sorted(groups_merged.keys(), key=lambda k: (k[0], k[1]))
    else:
        keys_sorted = sorted(groups_wfc.keys(), key=lambda k: (k[0], k[2], k[1]))
    # Use a high-contrast, manuscript-friendly color list (red/blue/green/...).
    color_cycle = [
        "tab:red",
        "tab:blue",
        "tab:green",
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:cyan",
        "tab:olive",
        "tab:gray",
    ]

    label_map = {}
    if (not args.merge_wfc) and n0_map:
        label_map = _build_n_label_map(keys_sorted, n0_map)

    for i, key in enumerate(keys_sorted):
        if args.merge_wfc:
            el, orb = key  # type: ignore[misc]
            y = groups_merged[(el, orb)]
            label = f"{el}-{orb}"
        else:
            el, wfc_idx, orb = key  # type: ignore[misc]
            y = groups_wfc[(el, wfc_idx, orb)]
            label = label_map.get((el, int(wfc_idx), orb), f"{el}-{wfc_idx}{orb}")
        ax.plot(e_pdos, y, lw=1.2 if args.style == "prb" else 2.0, color=color_cycle[i % len(color_cycle)], label=label)

    if args.fermi_line:
        ax.axvline(0.0, color="gray", linestyle="--", lw=1.2, alpha=0.8)

    if args.fermi is not None:
        ax.set_xlabel(r"$E-E_{f}$ (eV)")
    else:
        ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS (states/eV/unit cell)")

    if args.title:
        ax.set_title(args.title)
    else:
        if args.style != "prb":
            ax.set_title("DOS and PDOS")

    if args.ylog:
        ax.set_yscale("log")

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    if args.style == "prb":
        leg = ax.legend(frameon=False, ncols=2)
    else:
        ax.grid(True, alpha=0.25)
        leg = ax.legend(frameon=False, fontsize=10, ncols=2)
    _apply_style(ax, legend=leg, bold=(not args.no_bold) and (args.style != "prb"), sci_y=args.sci_y, ylog=args.ylog)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"Saved: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
