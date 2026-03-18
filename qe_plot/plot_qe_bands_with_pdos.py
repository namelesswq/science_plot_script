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
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb


@dataclass(frozen=True)
class KPointSpec:
    k: Tuple[float, float, float]
    n: int  # number of points from this k-point to the next one; n==1 means a jump/break


_RE_EL = re.compile(r"atm#\d+\(([^)]+)\)")
_RE_WFC = re.compile(r"wfc#(\d+)\(([^)]+)\)")


def _format_system_label(label: str, mode: str) -> str:
    if not label:
        return label
    if mode == "raw":
        return label
    if "$" in label:
        return label
    return re.sub(r"(?<=[A-Za-z\)])(\d+)", r"$_{\1}$", label)


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
    p.add_argument(
        "--bands",
        required=True,
        nargs="+",
        help="One or more bands.out.gnu files. If multiple are given, they are overlaid for comparison.",
    )
    p.add_argument(
        "--band-in",
        required=True,
        nargs="+",
        help="One or more band.in files. If a single file is given, it is reused for all datasets.",
    )
    p.add_argument(
        "--kpath",
        default=None,
        nargs="+",
        help=(
            "One or more KPATH.in files (VASPKIT). If omitted, labels are guessed. "
            "If a single file is given, it is reused."
        ),
    )

    # PDOS
    p.add_argument(
        "--tot",
        required=True,
        nargs="+",
        help="One or more total DOS files, e.g. zr2sc.pdos.pdos_tot. If multiple are given, they are overlaid.",
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
        nargs="+",
        help=(
            "Glob pattern(s) for PDOS files (overrides auto). Example: 'zr2sc.pdos.pdos_atm#*'. "
            "Provide one per dataset, or a single value to broadcast. Comma-separated tokens are accepted."
        ),
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
        default=None,
        nargs="+",
        help=(
            "Fermi energy in eV. If provided, shift energies as E -> E - Ef so Ef is at 0 eV. "
            "Applied to both bands and DOS/PDOS. Provide one per dataset, or a single value to broadcast. "
            "Comma-separated tokens are accepted."
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

    p.add_argument(
        "--system",
        default=None,
        nargs="+",
        help=(
            "System label(s) used to identify datasets. Provide one per dataset, e.g. '--system Zr2SC Zr15S8C8'. "
            "If only one label is given, it is broadcast. Comma-separated tokens are also accepted."
        ),
    )
    p.add_argument(
        "--system-format",
        choices=["chem", "raw"],
        default="chem",
        help="Render --system as chemical formula with subscripts (chem) or raw text (raw). Default: chem.",
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
        help="Legend location for --system (matplotlib legend loc). Default: upper left.",
    )

    p.add_argument(
        "--system-bbox",
        default=None,
        help=(
            "Optional legend anchor (bbox_to_anchor) in axes coordinates 'x,y' (e.g. '1.02,1.0' for outside right). "
            "If provided, legend placement uses both --system-loc and this anchor."
        ),
    )

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


def _parse_xy(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = str(s).split(",", 1)
    return float(a), float(b)


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


def _flatten_tokens(tokens: Optional[Sequence[str]]) -> List[str]:
    if not tokens:
        return []
    out: List[str] = []
    for t in tokens:
        if t is None:
            continue
        for s in str(t).split(","):
            s2 = s.strip()
            if s2:
                out.append(s2)
    return out


def _broadcast_list(xs: Sequence[Optional[str]], n: int, name: str) -> List[Optional[str]]:
    if len(xs) == n:
        return list(xs)
    if len(xs) == 1:
        return [xs[0]] * n
    raise SystemExit(f"{name} expects 1 value or {n} values, but got {len(xs)}")


def _parse_float_list(tokens: Optional[Sequence[str]], *, n: int, name: str) -> List[Optional[float]]:
    flat = _flatten_tokens(tokens)
    if not flat:
        return [None] * n
    flat2 = _broadcast_list([str(x) for x in flat], n, name)
    out: List[Optional[float]] = []
    for t in flat2:
        if t is None or str(t).strip() == "":
            out.append(None)
            continue
        try:
            out.append(float(t))
        except ValueError as e:
            raise SystemExit(f"{name} contains non-float token {t!r}") from e
    return out


def _map_x_to_reference(
    x_plot: np.ndarray,
    indices: Sequence[int],
    ref_x_plot: np.ndarray,
    ref_indices: Sequence[int],
) -> np.ndarray:
    """Piecewise linear mapping of this dataset's x axis onto the reference x axis."""

    x = np.asarray(x_plot, dtype=float)
    x_ref = np.asarray(ref_x_plot, dtype=float)
    if len(indices) != len(ref_indices):
        raise SystemExit("Cannot overlay: high-symmetry point count differs.")

    x2 = x.copy()
    for i in range(len(indices) - 1):
        a = int(indices[i])
        b = int(indices[i + 1])
        ra = int(ref_indices[i])
        rb = int(ref_indices[i + 1])
        if b <= a or rb <= ra:
            continue
        if a < 0 or b >= len(x2) or ra < 0 or rb >= len(x_ref):
            continue

        x0 = float(x[a])
        x1 = float(x[b])
        rx0 = float(x_ref[ra])
        rx1 = float(x_ref[rb])

        denom = (x1 - x0)
        if abs(denom) < 1e-14:
            x2[a : b + 1] = rx0
            continue
        scale = (rx1 - rx0) / denom
        x2[a : b + 1] = rx0 + (x[a : b + 1] - x0) * scale
    return x2


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
    def build(*, overlap: bool, break_advances: bool) -> List[int]:
        idx = 0
        out = [0]
        for i in range(len(specs) - 1):
            n = int(specs[i].n)
            if n <= 1:
                if break_advances:
                    idx += 1
                out.append(idx)
                continue
            idx += (n - 1) if overlap else n
            out.append(idx)
        return out

    # Try 4 conventions:
    # - overlap vs no-overlap for N
    # - whether a discontinuity (N=1) advances by one data point (QE often outputs an extra point)
    candidates: List[Tuple[str, List[int]]] = []
    candidates.append(("overlap+break", build(overlap=True, break_advances=True)))
    candidates.append(("no-overlap+break", build(overlap=False, break_advances=True)))
    candidates.append(("overlap", build(overlap=True, break_advances=False)))
    candidates.append(("no-overlap", build(overlap=False, break_advances=False)))

    exact = [(name, ind) for (name, ind) in candidates if (ind[-1] + 1) == n_data]
    if exact:
        return exact[0][1], exact[0][0]

    best_name, best_ind = candidates[0]
    best_err = abs((best_ind[-1] + 1) - n_data)
    for name, ind in candidates[1:]:
        err = abs((ind[-1] + 1) - n_data)
        if err < best_err:
            best_name, best_ind, best_err = name, ind, err
    return best_ind, best_name + "*"


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


def _find_overlapping_xticklabels(fig: plt.Figure, ax: plt.Axes) -> List[int]:
    """Return indices (in ax.get_xticklabels() order) that overlap with neighbors."""

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    ticks = ax.get_xticklabels()
    bboxes: List[Optional[object]] = []
    for t in ticks:
        if not t.get_visible() or not t.get_text():
            bboxes.append(None)
            continue
        bboxes.append(t.get_window_extent(renderer=renderer))

    bad: set[int] = set()
    for i in range(len(ticks) - 1):
        a = bboxes[i]
        b = bboxes[i + 1]
        if a is None or b is None:
            continue
        if a.overlaps(b):
            bad.add(i)
            bad.add(i + 1)
    return sorted(bad)


def _fix_dense_xticklabels(fig: plt.Figure, ax: plt.Axes) -> None:
    """If x tick labels overlap, rotate/shrink them to avoid collisions."""
    ticks = ax.get_xticklabels()
    bad = _find_overlapping_xticklabels(fig, ax)
    if not bad:
        return

    # Keep non-overlapping labels unchanged.
    bad_ticks = [ticks[i] for i in bad if 0 <= i < len(ticks)]
    bad_ticks = [t for t in bad_ticks if t.get_visible() and t.get_text()]
    if not bad_ticks:
        return

    # Remember original texts so we can try newline staggering later.
    orig_text: Dict[int, str] = {}
    for i in bad:
        if 0 <= i < len(ticks):
            orig_text[i] = ticks[i].get_text()

    base_fs = float(bad_ticks[0].get_fontsize())

    # 1) Try shrinking only the overlapping labels (no rotation).
    for scale in (0.95, 0.9, 0.85, 0.8):
        for t in bad_ticks:
            t.set_fontsize(base_fs * scale)
        fig.tight_layout()
        if not _find_overlapping_xticklabels(fig, ax):
            return

    # 2) Rotate only the overlapping labels (max 45°; do NOT rotate to 90°).
    for t in bad_ticks:
        t.set_rotation(45)
        t.set_ha("right")
        t.set_rotation_mode("anchor")
    fig.tight_layout()
    if not _find_overlapping_xticklabels(fig, ax):
        return

    # 3) Still overlapping: shrink the overlapping labels further (keep 45°).
    for scale in (0.75, 0.7, 0.65):
        for t in bad_ticks:
            t.set_fontsize(base_fs * scale)
        fig.tight_layout()
        if not _find_overlapping_xticklabels(fig, ax):
            return

    # 4) Last resort: stagger only the overlapping labels into two rows via leading newlines.
    #    This keeps non-overlapping labels untouched.
    #    Try with 0° first (often looks cleaner), then 45° if needed.
    for t in bad_ticks:
        t.set_rotation(0)
        t.set_ha("center")
        t.set_rotation_mode("default")
        t.set_fontsize(base_fs * 0.8)

    for j, i in enumerate(bad):
        if i not in orig_text:
            continue
        txt = orig_text[i]
        if j % 2 == 1:
            ticks[i].set_text("\n" + txt)
        else:
            ticks[i].set_text(txt)

    fig.tight_layout()
    if not _find_overlapping_xticklabels(fig, ax):
        return

    # If still overlapping, combine staggering with 45° for the overlapping labels only.
    for t in bad_ticks:
        t.set_rotation(45)
        t.set_ha("right")
        t.set_rotation_mode("anchor")
    fig.tight_layout()


def main() -> None:
    args = _build_parser().parse_args()

    ylim = _parse_lim(args.ylim)
    dos_xlim = _parse_lim(args.dos_xlim)
    figsize = _parse_figsize(args.figsize)
    figsize_bands = _parse_figsize(args.figsize_bands)
    figsize_dos = _parse_figsize(args.figsize_dos)
    ratios = _parse_ratios(args.ratios)
    n0_map = _parse_n0_map(args.n0)
    system_bbox = _parse_xy(args.system_bbox)

    # Style
    if args.style == "prb":
        _apply_scienceplots_prb_style()

    bands_paths = _flatten_tokens(args.bands)
    band_in_paths = _flatten_tokens(args.band_in)
    tot_paths = _flatten_tokens(args.tot)
    if not bands_paths or not band_in_paths or not tot_paths:
        raise SystemExit("--bands/--band-in/--tot must not be empty")

    n_dataset = max(len(bands_paths), len(band_in_paths), len(tot_paths))
    bands_paths = _broadcast_list(bands_paths, n_dataset, "--bands")
    band_in_paths = _broadcast_list(band_in_paths, n_dataset, "--band-in")
    tot_paths = _broadcast_list(tot_paths, n_dataset, "--tot")

    kpath_paths: List[Optional[str]]
    if args.kpath is None:
        kpath_paths = [None] * n_dataset
    else:
        kpath_flat = _flatten_tokens(args.kpath)
        if not kpath_flat:
            kpath_paths = [None] * n_dataset
        else:
            kpath_paths = _broadcast_list([str(x) for x in kpath_flat], n_dataset, "--kpath")

    fermi_list = _parse_float_list(args.fermi, n=n_dataset, name="--fermi")

    systems: List[Optional[str]]
    sys_flat = _flatten_tokens(args.system)
    if not sys_flat:
        systems = [None] * n_dataset
    else:
        systems = _broadcast_list([str(x) for x in sys_flat], n_dataset, "--system")

    pdos_globs: List[Optional[str]]
    if args.pdos_glob is None:
        pdos_globs = [None] * n_dataset
    else:
        pdos_glob_flat = _flatten_tokens(args.pdos_glob)
        if not pdos_glob_flat:
            pdos_globs = [None] * n_dataset
        else:
            pdos_globs = _broadcast_list([str(x) for x in pdos_glob_flat], n_dataset, "--pdos-glob")

    if n_dataset > 1 and args.pdos is not None and len(args.pdos) > 0:
        raise SystemExit("When comparing multiple datasets, --pdos (explicit file list) is not supported. Use per-dataset --pdos-glob or auto discovery from each --tot.")

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

    dataset_colors = [
        "black",
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

    # --- Load all datasets (bands + dos/pdos) ---
    ref_xk: Optional[np.ndarray] = None
    ref_indices: Optional[List[int]] = None
    ref_xticks: Optional[List[float]] = None
    ref_xticklabels: Optional[List[str]] = None
    ref_segments: Optional[List[Tuple[int, int]]] = None
    scheme_ref: Optional[str] = None

    bands_x_plot: List[np.ndarray] = []
    bands_y_arrays: List[List[np.ndarray]] = []
    bands_segments: List[List[Tuple[int, int]]] = []

    dos_e_tot: List[np.ndarray] = []
    dos_tot: List[np.ndarray] = []
    pdos_e: List[np.ndarray] = []
    pdos_series: List[Dict[str, np.ndarray]] = []
    pdos_labels: List[List[str]] = []

    for i in range(n_dataset):
        # --- Bands ---
        xk_i, bands_i = _read_bands_out_gnu(str(bands_paths[i]))
        specs_i = _read_band_in_kpoints(str(band_in_paths[i]))

        label_entries_i: Optional[List[Tuple[Tuple[float, float, float], str]]] = None
        if kpath_paths[i]:
            label_entries_i = _read_kpath_labels(str(kpath_paths[i]))

        hs_labels_i: List[str] = []
        for j, sp in enumerate(specs_i):
            lab = None
            if label_entries_i is not None:
                lab = _find_label_for_k(sp.k, label_entries_i)
            if lab is None:
                lab = f"K{j+1}"
            hs_labels_i.append(lab)

        indices_i, scheme_i = _infer_indices(specs_i, n_data=len(xk_i))
        segments_i = _build_segments(specs_i, indices_i)
        xticks_i, xticklabels_i = _build_ticks_and_labels(xk_i, specs_i, indices_i, hs_labels_i)

        if i == 0:
            ref_xk = xk_i
            ref_indices = list(indices_i)
            ref_xticks = list(xticks_i)
            ref_xticklabels = list(xticklabels_i)
            ref_segments = list(segments_i)
            scheme_ref = scheme_i
        else:
            if ref_xticklabels is not None and list(xticklabels_i) != list(ref_xticklabels):
                raise SystemExit(
                    "Cannot overlay datasets: high-symmetry tick labels differ. "
                    f"Reference={ref_xticklabels}, dataset#{i+1}={list(xticklabels_i)}"
                )

        # Fermi shift for bands
        fermi_i = fermi_list[i]
        y_arrays_i = bands_i
        if fermi_i is not None:
            y_arrays_i = [b - float(fermi_i) for b in bands_i]

        # map x to reference (except reference itself)
        if i == 0:
            x_plot_i = xk_i
        else:
            x_plot_i = _map_x_to_reference(xk_i, indices_i, np.asarray(ref_xk, dtype=float), list(ref_indices or []))

        bands_x_plot.append(x_plot_i)
        bands_y_arrays.append(y_arrays_i)
        bands_segments.append(list(segments_i))

        # --- DOS/PDOS ---
        e_tot_i, dos_tot_i = _load_two_cols(str(tot_paths[i]), xcol=0, ycol=args.tot_col)

        # PDOS file list (per dataset)
        if n_dataset == 1 and args.pdos is not None and len(args.pdos) > 0:
            pdos_files_i = list(args.pdos)
        else:
            if pdos_globs[i]:
                pattern = str(pdos_globs[i])
            else:
                base = os.path.basename(str(tot_paths[i]))
                if base.endswith(".pdos.pdos_tot"):
                    prefix = base[: -len(".pdos.pdos_tot")]
                else:
                    prefix = os.path.splitext(base)[0]
                pattern = os.path.join(os.path.dirname(str(tot_paths[i])) or ".", f"{prefix}.pdos.pdos_atm#*")
            pdos_files_i = sorted(glob.glob(pattern))

        if not pdos_files_i:
            raise SystemExit(
                "No PDOS files found for dataset#{idx}. Provide a correct --pdos-glob (one per dataset) or ensure auto-discovery works for --tot.".format(
                    idx=i + 1
                )
            )

        e_pdos_i, series_i, labels_i = _load_pdos_groups(
            tot_path=str(tot_paths[i]),
            pdos_files=pdos_files_i,
            elements_filter=elements_filter,
            orbitals_filter=orbitals_filter,
            merge_wfc=args.merge_wfc,
            pdos_col=args.pdos_col,
        )

        # Optional relabeling with n0 (only when not merging wfc)
        if (not args.merge_wfc) and n0_map:
            keys: List[Tuple[str, int, str]] = []
            for lab in labels_i:
                m = re.match(r"^([A-Za-z]+)-(\d+)([A-Za-z]+)$", lab)
                if not m:
                    continue
                keys.append((m.group(1), int(m.group(2)), m.group(3)))
            label_map = _build_n_label_map(keys, n0_map)
            new_series: Dict[str, np.ndarray] = {}
            new_labels: List[str] = []
            for lab in labels_i:
                m = re.match(r"^([A-Za-z]+)-(\d+)([A-Za-z]+)$", lab)
                if m:
                    key = (m.group(1), int(m.group(2)), m.group(3))
                    new_lab = label_map.get(key, lab)
                else:
                    new_lab = lab
                new_series[new_lab] = series_i[lab]
                new_labels.append(new_lab)
            series_i = new_series
            labels_i = new_labels

        # Apply Fermi shift to DOS energies
        if fermi_i is not None:
            e_tot_i = e_tot_i - float(fermi_i)
            e_pdos_i = e_pdos_i - float(fermi_i)

        dos_e_tot.append(e_tot_i)
        dos_tot.append(dos_tot_i)
        pdos_e.append(e_pdos_i)
        pdos_series.append(series_i)
        pdos_labels.append(list(labels_i))

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
    for i in range(n_dataset):
        color_i = dataset_colors[i % len(dataset_colors)]
        alpha_i = 1.0 if i == 0 else 0.9
        for e in bands_y_arrays[i]:
            for (s, t) in bands_segments[i]:
                ax_band.plot(bands_x_plot[i][s : t + 1], e[s : t + 1], color=color_i, lw=lw_band, alpha=alpha_i)

    for xpos in (ref_xticks or []):
        ax_band.axvline(xpos, color="black", lw=0.6, alpha=0.6)

    ax_band.set_xticks(list(ref_xticks or []))
    ax_band.set_xticklabels(list(ref_xticklabels or []))
    # Keep only high-symmetry vertical lines on x-axis; hide tick marks (but keep labels).
    ax_band.tick_params(axis="x", which="both", bottom=False, top=False, length=0)

    if any(s is not None and str(s).strip() for s in systems):
        handles_sys: List[Line2D] = []
        for i in range(n_dataset):
            if systems[i] is None or str(systems[i]).strip() == "":
                continue
            sys_lab = _format_system_label(str(systems[i]), str(args.system_format))
            color_i = dataset_colors[i % len(dataset_colors)]
            if n_dataset == 1:
                handles_sys.append(Line2D([], [], color="none", label=sys_lab))
            else:
                handles_sys.append(Line2D([], [], color=color_i, lw=lw_band, label=sys_lab))

        if handles_sys:
            fs = args.system_fontsize
            if fs is None:
                try:
                    fs = float(ax_band.yaxis.label.get_size()) * 1.15
                except Exception:
                    fs = None
            if system_bbox is None:
                leg_sys = ax_band.legend(
                    handles=handles_sys,
                    loc=str(args.system_loc),
                    frameon=False,
                    handlelength=(0 if n_dataset == 1 else 1.8),
                    handletextpad=(0.0 if n_dataset == 1 else 0.6),
                    borderaxespad=0.2,
                    fontsize=fs,
                )
            else:
                leg_sys = ax_band.legend(
                    handles=handles_sys,
                    loc=str(args.system_loc),
                    bbox_to_anchor=system_bbox,
                    bbox_transform=ax_band.transAxes,
                    frameon=False,
                    handlelength=(0 if n_dataset == 1 else 1.8),
                    handletextpad=(0.0 if n_dataset == 1 else 0.6),
                    borderaxespad=0.2,
                    fontsize=fs,
                )
            if leg_sys is not None:
                for t in leg_sys.get_texts():
                    t.set_fontweight("bold")

    # --- Plot rotated DOS/PDOS (x = DOS, y = Energy) ---
    # Total DOS (optional)
    total_rgbs: List[Tuple[float, float, float]] = []
    for i in range(min(n_dataset, len(dataset_colors))):
        try:
            total_rgbs.append(tuple(float(x) for x in to_rgb(dataset_colors[i])))
        except Exception:
            pass

    pdos_color_pool: List[Tuple[float, float, float]] = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        try:
            cols = list(plt.get_cmap(cmap_name).colors)
        except Exception:
            continue
        for c in cols:
            rgb = tuple(float(x) for x in c)
            # Filter out colors too close to any total-DOS (dataset) color.
            too_close = False
            for tr in total_rgbs:
                d = ((rgb[0] - tr[0]) ** 2 + (rgb[1] - tr[1]) ** 2 + (rgb[2] - tr[2]) ** 2) ** 0.5
                if d < 0.12:
                    too_close = True
                    break
            if not too_close:
                pdos_color_pool.append(rgb)

    if not pdos_color_pool:
        pdos_color_pool = [tuple(float(x) for x in c) for c in plt.get_cmap("tab20").colors]

    def pick_pdos_color(j: int, taken: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        for k in range(len(pdos_color_pool)):
            cand = pdos_color_pool[(j + k) % len(pdos_color_pool)]
            if cand in taken:
                continue
            return cand
        return pdos_color_pool[j % len(pdos_color_pool)]

    taken_pdos: List[Tuple[float, float, float]] = []

    for i in range(n_dataset):
        color_i = dataset_colors[i % len(dataset_colors)]
        alpha_i = 1.0 if i == 0 else 0.9
        sys_i = systems[i]
        sys_prefix = None
        if n_dataset > 1:
            if sys_i is not None and str(sys_i).strip():
                sys_prefix = str(sys_i)
            else:
                sys_prefix = f"D{i+1}"

        if plot_total:
            if sys_prefix is None:
                lab_tot = "Total"
            else:
                lab_tot = f"{sys_prefix}:Total"
            ax_dos.plot(dos_tot[i], dos_e_tot[i], color=color_i, lw=lw_tot, alpha=alpha_i, label=lab_tot)

        # PDOS
        for lab in pdos_labels[i]:
            y = pdos_series[i][lab]
            c = pick_pdos_color(len(taken_pdos), taken_pdos)
            taken_pdos.append(c)
            lab2 = lab if sys_prefix is None else f"{sys_prefix}:{lab}"
            ax_dos.plot(y, pdos_e[i], lw=lw_pdos, color=c, alpha=alpha_i, label=lab2)

    # Shared y decorations
    if args.fermi_line:
        ax_band.axhline(0.0, color="gray", linestyle="--", lw=1.0, alpha=0.8)
        ax_dos.axhline(0.0, color="gray", linestyle="--", lw=1.0, alpha=0.8)

    if ylim:
        ax_band.set_ylim(*ylim)

    if ref_xk is not None:
        ax_band.set_xlim(float(ref_xk[0]), float(ref_xk[-1]))

    # DOS xlim auto unless specified
    if dos_xlim:
        ax_dos.set_xlim(*dos_xlim)
    else:
        # estimate from visible range (respect ylim if provided)
        xmax = 0.0
        for i in range(n_dataset):
            if ylim:
                mask_tot = (dos_e_tot[i] >= ylim[0]) & (dos_e_tot[i] <= ylim[1])
                mask_p = (pdos_e[i] >= ylim[0]) & (pdos_e[i] <= ylim[1])
            else:
                mask_tot = slice(None)
                mask_p = slice(None)

            if plot_total:
                xmax = max(xmax, float(np.nanmax(dos_tot[i][mask_tot])))
            for lab in pdos_labels[i]:
                xmax = max(xmax, float(np.nanmax(pdos_series[i][lab][mask_p])))

        ax_dos.set_xlim(0.0, xmax * 1.05 if xmax > 0 else 1.0)

    # Labels: keep compact xlabel on PDOS panel and show x-axis tick values
    any_fermi = any(x is not None for x in fermi_list)
    ax_band.set_ylabel(r"$E - E_{f}$ (eV)" if any_fermi else "Energy (eV)")

    ax_dos.set_xlabel("Electron DOS\n(states/eV/unit cell)")
    try:
        base = ax_band.yaxis.label.get_size()
        ax_dos.xaxis.label.set_fontsize(base * 0.85)
    except Exception:
        pass
    ax_dos.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)

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
    # If high-symmetry ticks are dense, auto-fix label overlap.
    _fix_dense_xticklabels(fig, ax_band)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)

    print(f"Saved: {args.out}")
    if scheme_ref is not None and ref_xk is not None:
        print(f"K-point indexing convention (reference): {scheme_ref} (data points per band: {len(ref_xk)})")
    else:
        print("K-point indexing convention: (unknown)")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
