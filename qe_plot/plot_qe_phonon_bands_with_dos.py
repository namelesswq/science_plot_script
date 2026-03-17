#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


_CM1_PER_THz = 33.35641  # THz = (cm^-1)/33.35641


@dataclass(frozen=True)
class KPointSpec:
    q: Tuple[float, float, float]
    n: int  # number of points from this q-point to the next; n==1 means a jump/break


@dataclass(frozen=True)
class AtomSpec:
    index: int  # 1-based
    element: str


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot phonon dispersion together with phonon DOS/PDOS on one figure.\n\n"
            "Layout:\n"
            "- Left panel: phonon dispersion vs q-path (from matdyn *.freq.gp)\n"
            "- Right panel: phonon DOS/PDOS rotated by 90 degrees (DOS on x, frequency on shared y)\n\n"
            "Inputs (bands):\n"
            "- *.freq.gp: first column is path coordinate (x), each following column is one phonon branch\n"
            "- matdyn.in: contains q-point path in band form with per-segment point counts (N=1 means a break)\n"
            "- KPATH.in (optional): VASPKIT-style labels for high-symmetry points\n\n"
            "Inputs (DOS/PDOS):\n"
            "- *.dos: col0 frequency (cm^-1), col1 total DOS, col2.. per-atom PDOS\n"
            "- scf.in: used to map per-atom PDOS columns to atoms via ATOMIC_POSITIONS\n\n"
            "Unit handling:\n"
            "- If --unit THz, converts frequency as THz=(cm^-1)/33.35641.\n"
            "- By default, DOS/PDOS are scaled by Jacobian (×33.35641) so y-unit becomes states/THz/unit cell.\n"
            "  Disable with --no-jacobian."
        )
    )

    # Bands
    p.add_argument("--freq", required=True, help="Path to *.freq.gp (matdyn band output)")
    p.add_argument("--matdyn-in", required=True, help="Path to matdyn.in")
    p.add_argument("--kpath", default=None, help="Path to KPATH.in (VASPKIT). If omitted, labels are guessed.")

    p.add_argument(
        "--keep-jumps",
        action="store_true",
        help="Keep the original x-axis jumps at discontinuities (default: compress jumps so segments meet at one x).",
    )

    # DOS/PDOS
    p.add_argument("--dos", required=True, help="Path to phonon DOS file (e.g. zr2sc.dos)")
    p.add_argument("--scf-in", required=True, help="Path to QE scf input (used to map PDOS columns to atoms)")

    p.add_argument(
        "--group",
        choices=["atom", "element"],
        default="element",
        help="Plot PDOS grouped by atom or summed by element [default: element]",
    )
    p.add_argument(
        "--elements",
        default=None,
        help="Comma-separated element filter (applies to PDOS only), e.g. 'Zr,S,C'. Default: all.",
    )
    p.add_argument(
        "--atoms",
        default=None,
        help="Comma-separated atom indices (1-based) to plot when --group atom, e.g. '1,2,5-8'. Default: all.",
    )

    # Shared
    p.add_argument(
        "--unit",
        choices=["cm^-1", "THz"],
        default="THz",
        help=(
            "Frequency unit for plotting. matdyn *.freq.gp and fldos *.dos are commonly in cm^-1. "
            "When using THz, the script converts as THz = (cm^-1)/33.35641. [default: THz]"
        ),
    )
    p.add_argument(
        "--no-jacobian",
        action="store_true",
        help=(
            "Disable DOS Jacobian scaling when converting cm^-1 -> THz. "
            "By default, if --unit THz, DOS/PDOS are multiplied by 33.35641 so the y-unit becomes states/THz/unit cell."
        ),
    )

    p.add_argument("--ylim", default=None, help='Shared frequency limits "ymin,ymax" (in selected unit)')

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
            "Line width for phonon branches and DOS/PDOS curves. If omitted, keep style defaults "
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
        help='Panel width ratios "bands,dos" (default: 3,1).',
    )

    p.add_argument(
        "--dos-xlim",
        default=None,
        help='DOS panel x limits (DOS axis) "xmin,xmax". If omitted, auto from data.',
    )

    p.add_argument(
        "--legend-loc",
        default="best",
        help="Legend location inside the DOS panel (matplotlib loc=...) [default: best]",
    )
    p.add_argument("--legend-fontsize", type=float, default=None, help="Legend fontsize")

    p.add_argument("--out", default="phonon_bands_dos.png", help="Output image path")
    p.add_argument("--show", action="store_true", help="Show interactively")
    p.add_argument("--no-bold", action="store_true", help="Disable bold text in default style")

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


def _apply_bold(ax, *, bold: bool) -> None:
    if not bold:
        return
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        t.set_fontweight("bold")


def _read_freq_gp(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise SystemExit(f"Unexpected *.freq.gp table: shape={data.shape}")
    x = np.asarray(data[:, 0], dtype=float)
    y = np.asarray(data[:, 1:], dtype=float)  # (nq, nbranch)
    return x, y


def _convert_freq_units(freq_cm1: np.ndarray, unit: str) -> np.ndarray:
    freq = np.asarray(freq_cm1, dtype=float)
    if unit == "cm^-1":
        return freq
    return freq / _CM1_PER_THz


def _read_matdyn_in_qpoints(path: str) -> List[KPointSpec]:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()

    end_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*/\s*$", line):
            end_idx = i
            break
    if end_idx is None:
        end_idx = -1

    j = end_idx + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j >= len(lines):
        raise SystemExit(f"Cannot find q-point count after namelist in {path}")

    try:
        nq = int(lines[j].strip().split()[0])
    except ValueError as e:
        raise SystemExit(f"Cannot parse q-point count in {path}: {lines[j]!r}") from e

    specs: List[KPointSpec] = []
    j += 1
    while j < len(lines) and len(specs) < nq:
        line = lines[j].strip()
        j += 1
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            qx, qy, qz = float(parts[0]), float(parts[1]), float(parts[2])
            n = int(float(parts[3]))
        except ValueError:
            continue
        specs.append(KPointSpec(q=(qx, qy, qz), n=n))

    if len(specs) != nq:
        raise SystemExit(f"Parsed {len(specs)} q-points but expected {nq} from {path}")

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


def _find_label_for_q(
    q: Tuple[float, float, float],
    entries: Sequence[Tuple[Tuple[float, float, float], str]],
) -> Optional[str]:
    best: Optional[Tuple[float, str]] = None
    for qq, lab in entries:
        dx = q[0] - qq[0]
        dy = q[1] - qq[1]
        dz = q[2] - qq[2]
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
                idx += 1
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


def _compress_x_jumps_by_specs(x: np.ndarray, specs: Sequence[KPointSpec], indices: Sequence[int]) -> np.ndarray:
    x2 = np.asarray(x, dtype=float).copy()
    if len(indices) != len(specs):
        return x2

    for i in range(len(specs) - 1):
        if int(specs[i].n) != 1:
            continue
        a = int(indices[i])
        b = int(indices[i + 1])
        if a < 0 or b < 0 or a >= len(x2) or b >= len(x2):
            continue
        if b <= a:
            continue
        gap = float(x2[b] - x2[a])
        if gap == 0.0:
            continue
        x2[b:] -= gap
    return x2


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


def _find_overlapping_xticklabels(fig: plt.Figure, ax: plt.Axes) -> List[int]:
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
    ticks = ax.get_xticklabels()
    bad = _find_overlapping_xticklabels(fig, ax)
    if not bad:
        return

    bad_ticks = [ticks[i] for i in bad if 0 <= i < len(ticks)]
    bad_ticks = [t for t in bad_ticks if t.get_visible() and t.get_text()]
    if not bad_ticks:
        return

    orig_text: Dict[int, str] = {}
    for i in bad:
        if 0 <= i < len(ticks):
            orig_text[i] = ticks[i].get_text()

    base_fs = float(bad_ticks[0].get_fontsize())

    for scale in (0.95, 0.9, 0.85, 0.8):
        for t in bad_ticks:
            t.set_fontsize(base_fs * scale)
        fig.tight_layout()
        if not _find_overlapping_xticklabels(fig, ax):
            return

    for t in bad_ticks:
        t.set_rotation(45)
        t.set_ha("right")
        t.set_rotation_mode("anchor")
    fig.tight_layout()
    if not _find_overlapping_xticklabels(fig, ax):
        return

    for scale in (0.75, 0.7, 0.65):
        for t in bad_ticks:
            t.set_fontsize(base_fs * scale)
        fig.tight_layout()
        if not _find_overlapping_xticklabels(fig, ax):
            return

    for t in bad_ticks:
        t.set_rotation(0)
        t.set_ha("center")
        t.set_rotation_mode("default")
        t.set_fontsize(base_fs * 0.8)

    for j, i in enumerate(bad):
        if i not in orig_text:
            continue
        txt = orig_text[i]
        ticks[i].set_text(txt if (j % 2 == 0) else ("\n" + txt))

    fig.tight_layout()
    if not _find_overlapping_xticklabels(fig, ax):
        return

    for t in bad_ticks:
        t.set_rotation(45)
        t.set_ha("right")
        t.set_rotation_mode("anchor")
    fig.tight_layout()


def _read_phonon_dos_table(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise SystemExit(f"Unexpected DOS table in {path!r}: shape={data.shape}")

    freq_cm1 = np.asarray(data[:, 0], dtype=float)
    dos_tot = np.asarray(data[:, 1], dtype=float)
    pdos = np.asarray(data[:, 2:], dtype=float) if data.shape[1] > 2 else np.zeros((len(freq_cm1), 0), dtype=float)
    return freq_cm1, dos_tot, pdos


def _read_nat_from_scf_in(path: str) -> Optional[int]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"\bnat\s*=\s*([0-9]+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    nat = int(m.group(1))
    if nat <= 0:
        return None
    return nat


def _read_atoms_from_scf_in(path: str, *, expected_nat: Optional[int] = None) -> List[AtomSpec]:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()

    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*ATOMIC_POSITIONS\b", line, flags=re.IGNORECASE):
            start = i + 1
            break
    if start is None:
        raise SystemExit(f"Cannot find ATOMIC_POSITIONS in {path}")

    atoms: List[AtomSpec] = []
    idx = 1
    for j in range(start, len(lines)):
        line = lines[j].strip()
        if not line:
            continue
        if re.match(r"^(K_POINTS|CELL_PARAMETERS|ATOMIC_SPECIES|CONSTRAINTS|OCCUPATIONS)\b", line, flags=re.IGNORECASE):
            break
        parts = line.split()
        if len(parts) < 4:
            continue
        el = parts[0]
        el2 = re.sub(r"[^A-Za-z]", "", el)
        if not el2:
            continue
        atoms.append(AtomSpec(index=idx, element=el2))
        idx += 1

        if expected_nat is not None and len(atoms) >= expected_nat:
            break

    if not atoms:
        raise SystemExit(f"No atoms parsed from ATOMIC_POSITIONS in {path}")

    if expected_nat is not None and len(atoms) != expected_nat:
        raise SystemExit(
            f"Parsed {len(atoms)} atoms from ATOMIC_POSITIONS, but nat={expected_nat} in {path}. "
            "Please check the scf.in format."
        )

    return atoms


def _parse_atom_selection(s: Optional[str], n_atoms: int) -> List[int]:
    if not s:
        return list(range(1, n_atoms + 1))

    out: List[int] = []
    parts = [x.strip() for x in s.split(",") if x.strip()]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            ia = int(a)
            ib = int(b)
            if ia <= 0 or ib <= 0:
                raise SystemExit(f"Invalid --atoms range: {part!r}")
            if ia > ib:
                ia, ib = ib, ia
            out.extend(range(ia, ib + 1))
        else:
            out.append(int(part))

    out2: List[int] = []
    for i in out:
        if i < 1 or i > n_atoms:
            raise SystemExit(f"Atom index out of range in --atoms: {i} (1..{n_atoms})")
        if i not in out2:
            out2.append(i)
    return out2


def _convert_dos_unit_for_x(
    dos: np.ndarray,
    pdos: np.ndarray,
    *,
    x_unit: str,
    disable_jacobian: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if x_unit != "THz" or disable_jacobian:
        return dos, pdos
    return dos * _CM1_PER_THz, pdos * _CM1_PER_THz


def main() -> None:
    args = _build_parser().parse_args()

    ylim = _parse_lim(args.ylim)
    dos_xlim = _parse_lim(args.dos_xlim)
    figsize = _parse_figsize(args.figsize)
    figsize_bands = _parse_figsize(args.figsize_bands)
    figsize_dos = _parse_figsize(args.figsize_dos)
    ratios = _parse_ratios(args.ratios)

    if args.lw is not None and float(args.lw) <= 0:
        raise SystemExit("--lw must be > 0")

    if args.style == "prb":
        _apply_scienceplots_prb_style()

    # --- Bands data ---
    x_path, ymat_cm1 = _read_freq_gp(args.freq)
    ymat = _convert_freq_units(ymat_cm1, args.unit)
    specs = _read_matdyn_in_qpoints(args.matdyn_in)

    label_entries: Optional[List[Tuple[Tuple[float, float, float], str]]] = None
    if args.kpath:
        label_entries = _read_kpath_labels(args.kpath)

    hs_labels: List[str] = []
    for i, sp in enumerate(specs):
        lab = None
        if label_entries is not None:
            lab = _find_label_for_q(sp.q, label_entries)
        if lab is None:
            lab = f"Q{i+1}"
        hs_labels.append(lab)

    indices, scheme = _infer_indices(specs, n_data=len(x_path))
    segments = _build_segments(specs, indices)

    if args.keep_jumps:
        x_plot = x_path
    else:
        x_plot = _compress_x_jumps_by_specs(x_path, specs, indices)

    xticks, xticklabels = _build_ticks_and_labels(x_plot, specs, indices, hs_labels)

    # --- DOS/PDOS data ---
    freq_dos_cm1, dos_tot, pdos_mat = _read_phonon_dos_table(args.dos)
    nat_from_scf = _read_nat_from_scf_in(args.scf_in)
    atoms = _read_atoms_from_scf_in(args.scf_in, expected_nat=nat_from_scf)

    n_atoms = nat_from_scf if nat_from_scf is not None else len(atoms)
    if pdos_mat.shape[1] != 0 and pdos_mat.shape[1] != n_atoms:
        raise SystemExit(
            f"PDOS column count mismatch: DOS file has {pdos_mat.shape[1]} per-atom columns, "
            f"but scf.in has {n_atoms} atoms. Please confirm the DOS file format/order."
        )

    freq_dos = _convert_freq_units(freq_dos_cm1, args.unit)

    # Keep DOS/PDOS unit consistent with selected x unit (THz)
    dos_tot, pdos_mat = _convert_dos_unit_for_x(
        dos_tot,
        pdos_mat,
        x_unit=str(args.unit),
        disable_jacobian=bool(args.no_jacobian),
    )

    elements_filter: Optional[set[str]] = None
    if args.elements:
        elements_filter = {x.strip() for x in args.elements.split(",") if x.strip()}

    series: Dict[str, np.ndarray] = {}
    labels: List[str] = []

    if pdos_mat.shape[1] == 0:
        pass
    elif args.group == "atom":
        selected_atoms = _parse_atom_selection(args.atoms, n_atoms=n_atoms)
        for ia in selected_atoms:
            el = atoms[ia - 1].element
            if elements_filter is not None and el not in elements_filter:
                continue
            lab = f"{el}{ia}"
            series[lab] = pdos_mat[:, ia - 1]
            labels.append(lab)
    else:
        by_el: Dict[str, np.ndarray] = {}
        for ia, atom in enumerate(atoms, start=1):
            el = atom.element
            if elements_filter is not None and el not in elements_filter:
                continue
            if el not in by_el:
                by_el[el] = np.zeros_like(dos_tot, dtype=float)
            by_el[el] += pdos_mat[:, ia - 1]
        for el in sorted(by_el.keys()):
            series[el] = by_el[el]
            labels.append(el)

    # --- Figure layout ---
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
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()
        width_ratios = [ratios[0], ratios[1]]

    gs = fig.add_gridspec(1, 2, width_ratios=width_ratios, wspace=0.05)
    ax_band = fig.add_subplot(gs[0, 0])
    ax_dos = fig.add_subplot(gs[0, 1], sharey=ax_band)

    lw = float(args.lw) if args.lw is not None else (0.8 if args.style == "prb" else 1.4)

    # --- Plot phonon bands ---
    n_branch = int(ymat.shape[1])
    for j in range(n_branch):
        f = ymat[:, j]
        for (s, t) in segments:
            ax_band.plot(x_plot[s : t + 1], f[s : t + 1], color="black", lw=lw)

    for xpos in xticks:
        ax_band.axvline(xpos, color="black", lw=0.6, alpha=0.6)

    ax_band.set_xticks(xticks)
    ax_band.set_xticklabels(xticklabels)

    ax_band.set_xlim(float(x_plot[0]), float(x_plot[-1]))
    if ylim:
        ax_band.set_ylim(*ylim)

    ylab = "Frequency (THz)" if args.unit == "THz" else "Frequency (cm^-1)"
    ax_band.set_ylabel(ylab)

    # --- Plot rotated DOS/PDOS (x = DOS, y = Frequency) ---
    ax_dos.plot(dos_tot, freq_dos, color="black", lw=lw, label="Total DOS")

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
        ax_dos.plot(series[lab], freq_dos, lw=lw, color=color_cycle[i % len(color_cycle)], label=lab)

    if dos_xlim:
        ax_dos.set_xlim(*dos_xlim)
    else:
        if ylim:
            mask = (freq_dos >= ylim[0]) & (freq_dos <= ylim[1])
        else:
            mask = slice(None)
        xmax = float(np.nanmax(dos_tot[mask])) if len(dos_tot) else 0.0
        for lab in labels:
            xmax = max(xmax, float(np.nanmax(series[lab][mask])))
        ax_dos.set_xlim(0.0, xmax * 1.05 if xmax > 0 else 1.0)

    # Hide duplicate y tick labels on the right
    ax_dos.tick_params(axis="y", which="both", left=False, labelleft=False)

    # Keep DOS panel compact but show x-axis tick values
    if args.unit == "THz":
        ax_dos.set_xlabel("Phonon DOS\n(states/THz/unit cell)")
    else:
        ax_dos.set_xlabel(r"Phonon DOS\n(states/cm$^{-1}$/unit cell)")
    try:
        ax_dos.xaxis.label.set_fontsize(ax_band.xaxis.label.get_size() * 0.85)
    except Exception:
        pass
    ax_dos.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)

    leg_fs = args.legend_fontsize
    if leg_fs is None:
        leg = ax_dos.legend(loc=str(args.legend_loc), frameon=False)
    else:
        leg = ax_dos.legend(loc=str(args.legend_loc), frameon=False, fontsize=float(leg_fs))

    if args.style == "default":
        ax_band.grid(True, alpha=0.25)
        ax_dos.grid(True, alpha=0.25)
        if not args.no_bold:
            _apply_bold(ax_band, bold=True)
            _apply_bold(ax_dos, bold=True)

        # Make legend bold if requested
        if (not args.no_bold) and leg is not None:
            for t in leg.get_texts():
                t.set_fontweight("bold")

    fig.tight_layout()
    _fix_dense_xticklabels(fig, ax_band)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)

    print(f"Saved: {args.out}")
    print(f"Q-point indexing convention: {scheme} (data points per branch: {len(x_path)})")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
