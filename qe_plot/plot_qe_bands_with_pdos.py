#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class KPointSpec:
    k: Tuple[float, float, float]
    n: int  # number of points from this k-point to the next one; n==1 means a jump/break


_RE_EL = re.compile(r"atm#\d+\(([^)]+)\)")
_RE_WFC = re.compile(r"wfc#(\d+)\(([^)]+)\)")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot band structure (bands.out.gnu) together with PDOS (projwfc) on one figure.\n\n"
            "Layout:\n"
            "- Left panel: band structure vs k-path\n"
            "- Right panel: PDOS rotated by 90 degrees (DOS on x, energy on shared y)\n\n"
            "High-symmetry labels are taken from KPATH.in (VASPKIT format).\n"
            "Jump points: if a k-point line in band.in has N=1, it is treated as a discontinuity;\n"
            "the tick label at that position is merged as 'A|L' (end|start), and bands are not connected across it."
        )
    )

    # Bands
    p.add_argument("--bands", required=True, help="Path to bands.out.gnu")
    p.add_argument("--band-in", required=True, help="Path to band.in (QE input with K_POINTS crystal_b)")
    p.add_argument("--kpath", default=None, help="Path to KPATH.in (VASPKIT). If omitted, labels are guessed.")

    # PDOS
    p.add_argument("--tot", required=True, help="Total DOS file, e.g. zr2sc.pdos.pdos_tot")
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
    p.add_argument("--elements", default=None, help="Comma-separated element filter, e.g. 'Zr,S,C'.")
    p.add_argument(
        "--orbitals",
        default=None,
        help=(
            "Comma-separated orbital filter, e.g. 's,p,d,f'. "
            "Special tokens: 'no-tot' disables plotting total DOS; 'tot' forces plotting total DOS."
        ),
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
            "Format: 'Zr=4,S=3,C=2'. When provided (and not using --merge-wfc), labels become e.g. Zr-4s, Zr-5s, ..."
        ),
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

    # Shared / plot
    p.add_argument(
        "--fermi",
        type=float,
        default=None,
        help=(
            "Fermi energy in eV. If provided, shift energies as E -> E - Ef so Ef is at 0 eV. "
            "Applied to both bands and DOS/PDOS."
        ),
    )
    p.add_argument("--fermi-line", action="store_true", help="Draw a horizontal line at E=0")

    p.add_argument("--ylim", default=None, help='Shared energy limits "ymin,ymax" in eV (after Fermi shift if used)')

    p.add_argument(
        "--style",
        choices=["prb", "default"],
        default="prb",
        help="Plot style preset. 'prb' uses SciencePlots (science,no-latex). Default: prb.",
    )
    p.add_argument(
        "--lw",
        type=float,
        default=None,
        help=(
            "Line width for bands and (P)DOS curves. If omitted, keep style defaults "
            "(currently ~0.8 for prb and larger for default)."
        ),
    )
    p.add_argument(
        "--figsize",
        default=None,
        help='Figure size "width,height" in inches.',
    )
    p.add_argument(
        "--figsize-bands",
        default=None,
        help=(
            'Bands panel size "width,height" in inches (e.g. "7,3"). '
            "When used, you must also provide --figsize-dos."
        ),
    )
    p.add_argument(
        "--figsize-dos",
        default=None,
        help=(
            'DOS panel size "width,height" in inches (e.g. "2,3"). '
            "When used, you must also provide --figsize-bands."
        ),
    )
    p.add_argument(
        "--ratios",
        default="3,1",
        help='Panel width ratios "bands,pdos" (default: 3,1).',
    )

    p.add_argument(
        "--dos-xlim",
        default=None,
        help='PDOS panel x limits (DOS axis) "xmin,xmax". If omitted, auto from data.',
    )
    p.add_argument(
        "--legend-loc",
        default="best",
        help=(
            "Legend location inside the DOS panel. Passed to matplotlib legend(loc=...). "
            "Examples: 'best', 'upper right', 'upper left', 'lower right', 'lower left'."
        ),
    )
    p.add_argument("--legend-fontsize", type=float, default=None, help="Legend fontsize for PDOS panel")

    p.add_argument("--out", default="bands_pdos.png", help="Output image path")
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


def _parse_ratios(s: str) -> Tuple[float, float]:
    a, b = s.split(",", 1)
    ra = float(a)
    rb = float(b)
    if ra <= 0 or rb <= 0:
        raise SystemExit(f"Invalid --ratios {s!r}: both must be > 0")
    return ra, rb


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


def _read_bands_out_gnu(path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    bands_xy: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = []
    prev_x: Optional[float] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if cur:
                    bands_xy.append(cur)
                    cur = []
                prev_x = None
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue

            if prev_x is not None and x < prev_x - 1e-6 and len(cur) > 5:
                bands_xy.append(cur)
                cur = []
            cur.append((x, y))
            prev_x = x

    if cur:
        bands_xy.append(cur)

    if not bands_xy:
        raise SystemExit(f"No band data found in {path!r}")

    x0 = np.asarray([t[0] for t in bands_xy[0]], dtype=float)
    energies: List[np.ndarray] = []
    for i, band in enumerate(bands_xy):
        x = np.asarray([t[0] for t in band], dtype=float)
        e = np.asarray([t[1] for t in band], dtype=float)
        if len(x) != len(x0) or np.max(np.abs(x - x0)) > 1e-7:
            raise SystemExit(
                "bands.out.gnu does not seem to have a common x-grid across bands. "
                f"Band 0 length={len(x0)}, band {i} length={len(x)}."
            )
        energies.append(e)

    return x0, energies


def _read_band_in_kpoints(path: str) -> List[KPointSpec]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()

    kline = None
    for i, line in enumerate(text):
        if re.match(r"^\s*K_POINTS\b", line, flags=re.IGNORECASE) and "crystal_b" in line.lower():
            kline = i
            break
    if kline is None:
        raise SystemExit(f"Cannot find 'K_POINTS crystal_b' in {path}")

    j = kline + 1
    while j < len(text) and not text[j].strip():
        j += 1
    if j >= len(text):
        raise SystemExit(f"Unexpected end of file after K_POINTS in {path}")

    try:
        nk = int(text[j].strip().split()[0])
    except ValueError as e:
        raise SystemExit(f"Cannot parse number of K points after K_POINTS in {path}: {text[j]!r}") from e

    specs: List[KPointSpec] = []
    j += 1
    while j < len(text) and len(specs) < nk:
        line = text[j].strip()
        j += 1
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            kx, ky, kz = float(parts[0]), float(parts[1]), float(parts[2])
            n = int(float(parts[3]))
        except ValueError:
            continue
        specs.append(KPointSpec(k=(kx, ky, kz), n=n))

    if len(specs) != nk:
        raise SystemExit(f"Parsed {len(specs)} k-points but expected {nk} from {path}")

    return specs


def _read_kpath_labels(path: str) -> List[Tuple[Tuple[float, float, float], str]]:
    entries: List[Tuple[Tuple[float, float, float], str]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if any(line.lower().startswith(s) for s in ["k-path", "line-mode", "reciprocal"]):
                continue
            if re.fullmatch(r"\d+", line):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                kx, ky, kz = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            label = parts[3].strip()
            if label:
                entries.append(((kx, ky, kz), label))

    if not entries:
        raise SystemExit(f"No label entries parsed from {path}")

    return entries


def _normalize_label(label: str) -> str:
    u = label.strip()
    if not u:
        return u
    up = u.upper()
    if up in {"GAMMA", "Γ", "G"}:
        return "Γ"
    return u


def _find_label_for_k(
    k: Tuple[float, float, float],
    entries: Sequence[Tuple[Tuple[float, float, float], str]],
) -> Optional[str]:
    best: Optional[Tuple[float, str]] = None
    for kk, lab in entries:
        dx = k[0] - kk[0]
        dy = k[1] - kk[1]
        dz = k[2] - kk[2]
        d2 = dx * dx + dy * dy + dz * dz
        if best is None or d2 < best[0]:
            best = (d2, lab)

    if best is None:
        return None

    if best[0] <= (1e-3) ** 2:
        return _normalize_label(best[1])
    return None


def _infer_indices(specs: Sequence[KPointSpec], n_data: int) -> Tuple[List[int], str]:
    def build(overlap: bool) -> List[int]:
        idx = 0
        out = [0]
        for i in range(len(specs) - 1):
            n = int(specs[i].n)
            if n <= 1:
                out.append(idx)
                continue
            idx += (n - 1) if overlap else n
            out.append(idx)
        return out

    ind1 = build(overlap=True)
    ind2 = build(overlap=False)

    len1 = ind1[-1] + 1
    len2 = ind2[-1] + 1

    if len1 == n_data and len2 != n_data:
        return ind1, "overlap"
    if len2 == n_data and len1 != n_data:
        return ind2, "no-overlap"
    if len1 == n_data and len2 == n_data:
        return ind1, "overlap"

    if abs(len1 - n_data) <= abs(len2 - n_data):
        return ind1, "overlap*"
    return ind2, "no-overlap*"


def _build_segments(specs: Sequence[KPointSpec], indices: Sequence[int]) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    for i in range(len(specs) - 1):
        n = int(specs[i].n)
        if n <= 1:
            continue
        s = int(indices[i])
        e = int(indices[i + 1])
        if e <= s:
            continue
        segs.append((s, e))
    if not segs:
        raise SystemExit("No continuous segments found (did you set all N=1?)")
    return segs


def _build_ticks_and_labels(
    x: np.ndarray,
    specs: Sequence[KPointSpec],
    indices: Sequence[int],
    labels: Sequence[str],
) -> Tuple[List[float], List[str]]:
    xticks: List[float] = []
    xlabels: List[str] = []

    i = 0
    while i < len(specs):
        pos = float(x[int(indices[i])])
        lab = labels[i]

        if i < len(specs) - 1 and int(specs[i].n) == 1:
            lab2 = labels[i + 1]
            lab = f"{lab}|{lab2}"
            i += 2
        else:
            i += 1

        xticks.append(pos)
        xlabels.append(lab)

    xticks2: List[float] = []
    xlabels2: List[str] = []
    for pos, lab in zip(xticks, xlabels):
        if xticks2 and abs(pos - xticks2[-1]) < 1e-10:
            if lab != xlabels2[-1] and lab not in xlabels2[-1]:
                xlabels2[-1] = f"{xlabels2[-1]}|{lab}"
            continue
        xticks2.append(pos)
        xlabels2.append(lab)

    return xticks2, xlabels2


def _load_two_cols(path: str, xcol: int, ycol: int) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] <= max(xcol, ycol):
        raise RuntimeError(f"Unexpected table in {path}: shape={data.shape}")
    x = np.asarray(data[:, xcol], dtype=float)
    y = np.asarray(data[:, ycol], dtype=float)
    return x, y


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


def _build_n_label_map(keys: Sequence[Tuple[str, int, str]], n0_map: Dict[str, int]) -> Dict[Tuple[str, int, str], str]:
    if not n0_map:
        return {}

    label_map: Dict[Tuple[str, int, str], str] = {}
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


def _load_pdos_groups(
    *,
    tot_path: str,
    pdos_files: Sequence[str],
    elements_filter: Optional[set[str]],
    orbitals_filter: Optional[set[str]],
    merge_wfc: bool,
    pdos_col: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """Return energy grid, series dict (label->dos), and label list in plotting order."""

    groups_wfc: Dict[Tuple[str, int, str], np.ndarray] = {}
    groups_merged: Dict[Tuple[str, str], np.ndarray] = {}
    energy_ref: Optional[np.ndarray] = None

    for f in pdos_files:
        el, wfc_idx, orb = _parse_element_wfc_orbital_from_name(f)
        if elements_filter is not None and el not in elements_filter:
            continue
        if orbitals_filter is not None and orb not in orbitals_filter:
            continue

        e, y = _load_two_cols(f, xcol=0, ycol=pdos_col)

        if energy_ref is None:
            energy_ref = e
        else:
            if len(e) != len(energy_ref) or np.max(np.abs(e - energy_ref)) > 1e-8:
                raise SystemExit(
                    f"Energy grid mismatch among PDOS files. Offending file: {f}. "
                    "Please regenerate PDOS with consistent energy grid."
                )

        if merge_wfc:
            key2 = (el, orb)
            if key2 not in groups_merged:
                groups_merged[key2] = np.zeros_like(y, dtype=float)
            groups_merged[key2] += y
        else:
            key3 = (el, wfc_idx, orb)
            if key3 not in groups_wfc:
                groups_wfc[key3] = np.zeros_like(y, dtype=float)
            groups_wfc[key3] += y

    if merge_wfc:
        if not groups_merged:
            raise SystemExit("No PDOS series selected after applying filters.")
        keys_sorted = sorted(groups_merged.keys(), key=lambda k: (k[0], k[1]))
        labels = [f"{el}-{orb}" for (el, orb) in keys_sorted]
        series = {lab: groups_merged[key] for lab, key in zip(labels, keys_sorted)}
    else:
        if not groups_wfc:
            raise SystemExit("No PDOS series selected after applying filters.")
        keys_sorted = sorted(groups_wfc.keys(), key=lambda k: (k[0], k[2], k[1]))
        labels = [f"{el}-{wfc}{orb}" for (el, wfc, orb) in keys_sorted]
        series = {lab: groups_wfc[key] for lab, key in zip(labels, keys_sorted)}

    if energy_ref is None:
        raise SystemExit("No PDOS energy grid could be determined.")

    return energy_ref, series, labels


def main() -> None:
    args = _build_parser().parse_args()

    ylim = _parse_lim(args.ylim)
    dos_xlim = _parse_lim(args.dos_xlim)
    figsize = _parse_figsize(args.figsize)
    figsize_bands = _parse_figsize(args.figsize_bands)
    figsize_dos = _parse_figsize(args.figsize_dos)
    ratios = _parse_ratios(args.ratios)
    n0_map = _parse_n0_map(args.n0)

    # Style
    if args.style == "prb":
        _apply_scienceplots_prb_style()

    # --- Bands data ---
    xk, bands = _read_bands_out_gnu(args.bands)
    specs = _read_band_in_kpoints(args.band_in)

    label_entries: Optional[List[Tuple[Tuple[float, float, float], str]]] = None
    if args.kpath:
        label_entries = _read_kpath_labels(args.kpath)

    hs_labels: List[str] = []
    for i, sp in enumerate(specs):
        lab = None
        if label_entries is not None:
            lab = _find_label_for_k(sp.k, label_entries)
        if lab is None:
            lab = f"K{i+1}"
        hs_labels.append(lab)

    indices, scheme = _infer_indices(specs, n_data=len(xk))
    segments = _build_segments(specs, indices)
    xticks, xticklabels = _build_ticks_and_labels(xk, specs, indices, hs_labels)

    # Apply Fermi shift to bands if requested
    y_arrays = bands
    if args.fermi is not None:
        y_arrays = [b - float(args.fermi) for b in bands]

    # --- DOS/PDOS data ---
    # total DOS
    e_tot, dos_tot = _load_two_cols(args.tot, xcol=0, ycol=args.tot_col)

    # PDOS file list
    if args.pdos is not None and len(args.pdos) > 0:
        pdos_files = list(args.pdos)
    else:
        if args.pdos_glob:
            pattern = args.pdos_glob
        else:
            base = os.path.basename(args.tot)
            if base.endswith(".pdos.pdos_tot"):
                prefix = base[: -len(".pdos.pdos_tot")]
            else:
                prefix = os.path.splitext(base)[0]
            pattern = os.path.join(os.path.dirname(args.tot) or ".", f"{prefix}.pdos.pdos_atm#*")
        pdos_files = sorted(glob.glob(pattern))

    if not pdos_files:
        raise SystemExit("No PDOS files found. Provide --pdos or a correct --pdos-glob.")

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

        keep: List[str] = []
        for t_raw, t_norm in zip(raw, norm):
            if t_norm in {"no-tot", "no-total", "notot", "nototal", "tot", "total"}:
                continue
            keep.append(t_raw)
        if keep:
            orbitals_filter = set(keep)

    e_pdos, series, labels = _load_pdos_groups(
        tot_path=args.tot,
        pdos_files=pdos_files,
        elements_filter=elements_filter,
        orbitals_filter=orbitals_filter,
        merge_wfc=args.merge_wfc,
        pdos_col=args.pdos_col,
    )

    # Optional relabeling with n0 (only when not merging wfc)
    if (not args.merge_wfc) and n0_map:
        keys: List[Tuple[str, int, str]] = []
        for lab in labels:
            # lab format: El-<wfc><orb>
            m = re.match(r"^([A-Za-z]+)-(\d+)([A-Za-z]+)$", lab)
            if not m:
                continue
            keys.append((m.group(1), int(m.group(2)), m.group(3)))
        label_map = _build_n_label_map(keys, n0_map)
        new_series: Dict[str, np.ndarray] = {}
        new_labels: List[str] = []
        for lab in labels:
            m = re.match(r"^([A-Za-z]+)-(\d+)([A-Za-z]+)$", lab)
            if m:
                key = (m.group(1), int(m.group(2)), m.group(3))
                new_lab = label_map.get(key, lab)
            else:
                new_lab = lab
            new_series[new_lab] = series[lab]
            new_labels.append(new_lab)
        series = new_series
        labels = new_labels

    # Apply Fermi shift to DOS energies if requested
    if args.fermi is not None:
        e_tot = e_tot - float(args.fermi)
        e_pdos = e_pdos - float(args.fermi)

    # --- Figure layout ---
    # Option A: absolute per-panel sizes (recommended when preparing figures for manuscripts)
    if (figsize_bands is None) ^ (figsize_dos is None):
        raise SystemExit("--figsize-bands and --figsize-dos must be provided together")

    if figsize_bands is not None and figsize_dos is not None:
        wb, hb = figsize_bands
        wd, hd = figsize_dos
        if abs(hb - hd) > 1e-8:
            raise SystemExit(
                f"--figsize-bands height ({hb}) must equal --figsize-dos height ({hd}) so the two panels can share the y-axis"
            )
        fig = plt.figure(figsize=(wb + wd, hb))
        width_ratios = [wb, wd]
    else:
        # Option B: overall figure size + width ratios
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()
        width_ratios = [ratios[0], ratios[1]]

    gs = fig.add_gridspec(1, 2, width_ratios=width_ratios, wspace=0.05)
    ax_band = fig.add_subplot(gs[0, 0])
    ax_dos = fig.add_subplot(gs[0, 1], sharey=ax_band)

    # Line widths
    if args.lw is not None and float(args.lw) <= 0:
        raise SystemExit("--lw must be > 0")
    lw_band = float(args.lw) if args.lw is not None else (0.8 if args.style == "prb" else 1.4)
    lw_tot = float(args.lw) if args.lw is not None else (0.8 if args.style == "prb" else 1.8)
    lw_pdos = float(args.lw) if args.lw is not None else (0.8 if args.style == "prb" else 1.6)

    # --- Plot bands ---
    for e in y_arrays:
        for (s, t) in segments:
            ax_band.plot(xk[s : t + 1], e[s : t + 1], color="black", lw=lw_band)

    for xpos in xticks:
        ax_band.axvline(xpos, color="black", lw=0.6, alpha=0.6)

    ax_band.set_xticks(xticks)
    ax_band.set_xticklabels(xticklabels)

    # --- Plot rotated DOS/PDOS (x = DOS, y = Energy) ---
    # Total DOS (optional)
    if plot_total:
        ax_dos.plot(dos_tot, e_tot, color="black", lw=lw_tot, label="Total DOS")

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

    for i, lab in enumerate(labels):
        y = series[lab]
        ax_dos.plot(y, e_pdos, lw=lw_pdos, color=color_cycle[i % len(color_cycle)], label=lab)

    # Shared y decorations
    if args.fermi_line:
        ax_band.axhline(0.0, color="gray", linestyle="--", lw=1.0, alpha=0.8)
        ax_dos.axhline(0.0, color="gray", linestyle="--", lw=1.0, alpha=0.8)

    if ylim:
        ax_band.set_ylim(*ylim)

    ax_band.set_xlim(float(xk[0]), float(xk[-1]))

    # DOS xlim auto unless specified
    if dos_xlim:
        ax_dos.set_xlim(*dos_xlim)
    else:
        # estimate from visible range (respect ylim if provided)
        if ylim:
            mask_tot = (e_tot >= ylim[0]) & (e_tot <= ylim[1])
            mask_p = (e_pdos >= ylim[0]) & (e_pdos <= ylim[1])
        else:
            mask_tot = slice(None)
            mask_p = slice(None)

        xmax = 0.0
        if plot_total:
            xmax = max(xmax, float(np.nanmax(dos_tot[mask_tot])))
        for lab in labels:
            xmax = max(xmax, float(np.nanmax(series[lab][mask_p])))
        ax_dos.set_xlim(0.0, xmax * 1.05 if xmax > 0 else 1.0)

    # Labels: only left y-label; PDOS panel has no x ticks/label as requested
    ax_band.set_ylabel(r"$E - E_{f}$ (eV)" if args.fermi is not None else "Energy (eV)")

    ax_dos.set_xlabel("")
    ax_dos.set_xticks([])
    ax_dos.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    # Hide duplicate y tick labels on the right
    ax_dos.tick_params(axis="y", which="both", left=False, labelleft=False)

    # Legend on DOS panel (keeps figure interpretable even without x-axis label)
    leg_loc = str(args.legend_loc)
    leg_fs = args.legend_fontsize
    if leg_fs is None:
        leg = ax_dos.legend(loc=leg_loc, frameon=False)
    else:
        leg = ax_dos.legend(loc=leg_loc, frameon=False, fontsize=float(leg_fs))

    # Tight layout and save
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)

    print(f"Saved: {args.out}")
    print(f"K-point indexing convention: {scheme} (data points per band: {len(xk)})")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
