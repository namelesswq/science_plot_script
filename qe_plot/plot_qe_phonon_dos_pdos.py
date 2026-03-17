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
class AtomSpec:
    index: int  # 1-based
    element: str


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot phonon DOS + PDOS from Quantum ESPRESSO matdyn.x DOS output (fldos).\n\n"
            "Expected DOS file format (columns):\n"
            "  col0: frequency (cm^-1, will be converted to THz by default)\n"
            "  col1: total DOS\n"
            "  col2..: per-atom PDOS (one column per atom)\n\n"
            "Atom order is determined from scf.in ATOMIC_POSITIONS, so PDOS columns can be labeled and/or summed by element."
        )
    )

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

    p.add_argument(
        "--unit",
        choices=["THz", "cm^-1"],
        default="THz",
        help=f"Frequency unit for x-axis. If THz, converts as THz=(cm^-1)/{_CM1_PER_THz}. [default: THz]",
    )

    p.add_argument(
        "--no-jacobian",
        action="store_true",
        help=(
            "Disable DOS Jacobian scaling when converting cm^-1 -> THz. "
            "By default, if --unit THz, DOS/PDOS are multiplied by 33.35641 so the y-unit becomes states/THz."
        ),
    )

    p.add_argument(
        "--style",
        choices=["prb", "default"],
        default="prb",
        help="Plot style preset. 'prb' uses SciencePlots (science,no-latex). Default: prb.",
    )
    p.add_argument(
        "--figsize",
        default=None,
        help=(
            'Figure size in inches. Use "width,height" (e.g. "3.4,2.6"). '
            'You may also pass a single number "width" (e.g. "3.4"), '
            'in which case height is set automatically (height=0.75*width).'
        ),
    )
    p.add_argument("--lw", type=float, default=None, help="Line width for curves")

    p.add_argument("--xlim", default=None, help='x limits "xmin,xmax" (in selected unit)')
    p.add_argument("--ylim", default=None, help='y limits "ymin,ymax"')

    p.add_argument("--ylog", action="store_true", help="Use log scale on y-axis")
    p.add_argument("--no-bold", action="store_true", help="Disable bold text")
    p.add_argument(
        "--sci-y",
        choices=["auto", "on", "off"],
        default="auto",
        help="Y-axis scientific notation factor (×10^n) [default: auto]",
    )

    p.add_argument("--title", default=None, help="Plot title")

    p.add_argument(
        "--legend-loc",
        default="best",
        help="Legend location inside the axes (matplotlib loc=...) [default: best]",
    )
    p.add_argument("--legend-fontsize", type=float, default=None, help="Legend fontsize")

    p.add_argument("--out", default="phonon_dos_pdos.png", help="Output image path")
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
    s2 = s.strip()
    if "," in s2:
        a, b = s2.split(",", 1)
        w = float(a)
        h = float(b)
    else:
        w = float(s2)
        h = 0.75 * w
    if w <= 0 or h <= 0:
        raise SystemExit(f"Invalid --figsize {s!r}: width and height must be > 0")
    return w, h


def _read_nat_from_scf_in(path: str) -> Optional[int]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    # nat may share a line with other &system parameters, so don't anchor to end-of-line.
    m = re.search(r"\bnat\s*=\s*([0-9]+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    nat = int(m.group(1))
    if nat <= 0:
        return None
    return nat


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


def _apply_plot_style(ax: plt.Axes, *, bold: bool, sci_y: str, ylog: bool, legend=None) -> None:
    if bold:
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            t.set_fontweight("bold")

    if sci_y != "auto":
        if sci_y == "on":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        else:
            ax.ticklabel_format(axis="y", style="plain", useMathText=True)

    if legend is not None and bold:
        for t in legend.get_texts():
            t.set_fontweight("bold")

    # Avoid weird formatting for log-scale
    if ylog:
        try:
            ax.yaxis.get_major_formatter().set_useOffset(False)
        except Exception:
            pass


def _read_phonon_dos_table(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise SystemExit(f"Unexpected DOS table in {path!r}: shape={data.shape}")

    x_cm1 = np.asarray(data[:, 0], dtype=float)
    dos_tot = np.asarray(data[:, 1], dtype=float)
    pdos = np.asarray(data[:, 2:], dtype=float) if data.shape[1] > 2 else np.zeros((len(x_cm1), 0), dtype=float)
    return x_cm1, dos_tot, pdos


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

    out2 = []
    for i in out:
        if i < 1 or i > n_atoms:
            raise SystemExit(f"Atom index out of range in --atoms: {i} (1..{n_atoms})")
        if i not in out2:
            out2.append(i)
    return out2


def _read_atoms_from_scf_in(path: str, *, expected_nat: Optional[int] = None) -> List[AtomSpec]:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()

    # find ATOMIC_POSITIONS
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
        # stop at next section keyword
        if re.match(r"^(K_POINTS|CELL_PARAMETERS|ATOMIC_SPECIES|CONSTRAINTS|OCCUPATIONS)\b", line, flags=re.IGNORECASE):
            break
        # a typical line starts with element symbol
        parts = line.split()
        if len(parts) < 4:
            continue
        el = parts[0]
        # element can be like 'Zr' or 'zr' or with weird chars; keep letters only
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


def _convert_x_unit(x_cm1: np.ndarray, unit: str) -> np.ndarray:
    if unit == "cm^-1":
        return x_cm1
    return x_cm1 / _CM1_PER_THz


def _convert_dos_unit_for_x(
    dos: np.ndarray,
    pdos: np.ndarray,
    *,
    x_unit: str,
    disable_jacobian: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert DOS/PDOS units consistently with selected x unit.

    Input DOS is assumed to be per (cm^-1). If plotting x in THz,
    g_THz = g_cm1 * (cm^-1 per THz) so that the total states integral is preserved.
    """
    if x_unit != "THz" or disable_jacobian:
        return dos, pdos
    return dos * _CM1_PER_THz, pdos * _CM1_PER_THz


def main() -> None:
    args = _build_parser().parse_args()

    xlim = _parse_lim(args.xlim)
    ylim = _parse_lim(args.ylim)
    figsize = _parse_figsize(args.figsize)

    if args.lw is not None and float(args.lw) <= 0:
        raise SystemExit("--lw must be > 0")

    if args.style == "prb":
        _apply_scienceplots_prb_style()

    x_cm1, dos_tot, pdos_mat = _read_phonon_dos_table(args.dos)
    nat_from_scf = _read_nat_from_scf_in(args.scf_in)
    atoms = _read_atoms_from_scf_in(args.scf_in, expected_nat=nat_from_scf)

    n_atoms = nat_from_scf if nat_from_scf is not None else len(atoms)
    if pdos_mat.shape[1] != 0 and pdos_mat.shape[1] != n_atoms:
        raise SystemExit(
            f"PDOS column count mismatch: DOS file has {pdos_mat.shape[1]} per-atom columns, "
            f"but scf.in has {n_atoms} atoms. Please confirm the DOS file format/order."
        )

    x = _convert_x_unit(x_cm1, args.unit)

    # Keep DOS/PDOS unit consistent with selected x unit
    dos_tot, pdos_mat = _convert_dos_unit_for_x(
        dos_tot,
        pdos_mat,
        x_unit=str(args.unit),
        disable_jacobian=bool(args.no_jacobian),
    )

    elements_filter: Optional[set[str]] = None
    if args.elements:
        elements_filter = {x.strip() for x in args.elements.split(",") if x.strip()}

    # Build PDOS series
    series: Dict[str, np.ndarray] = {}

    if pdos_mat.shape[1] == 0:
        # no PDOS columns available
        pass
    elif args.group == "atom":
        selected_atoms = _parse_atom_selection(args.atoms, n_atoms=n_atoms)
        for ia in selected_atoms:
            el = atoms[ia - 1].element
            if elements_filter is not None and el not in elements_filter:
                continue
            lab = f"{el}{ia}"
            series[lab] = pdos_mat[:, ia - 1]
    else:
        # element-summed
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

    # --- Plot ---
    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
    else:
        fig, ax = plt.subplots(dpi=150)

    lw = float(args.lw) if args.lw is not None else (0.8 if args.style == "prb" else 1.6)

    # total DOS
    ax.plot(x, dos_tot, color="black", lw=lw, label="Total DOS")

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
    for i, (lab, y) in enumerate(series.items()):
        ax.plot(x, y, lw=lw, color=color_cycle[i % len(color_cycle)], label=lab)

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    ax.set_xlabel("Frequency (THz)" if args.unit == "THz" else r"Frequency (cm$^{-1}$)")
    if args.unit == "THz":
        ax.set_ylabel("DOS (states/THz/unit cell)")
    else:
        ax.set_ylabel(r"DOS (states/cm$^{-1}$/unit cell)")

    if args.ylog:
        ax.set_yscale("log")

    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title("Phonon DOS")

    ax.grid(True, alpha=0.25)

    leg_fs = args.legend_fontsize
    if leg_fs is None:
        leg = ax.legend(loc=str(args.legend_loc), frameon=False)
    else:
        leg = ax.legend(loc=str(args.legend_loc), frameon=False, fontsize=float(leg_fs))

    _apply_plot_style(ax, bold=not args.no_bold, sci_y=args.sci_y, ylog=args.ylog, legend=leg)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"Saved: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
