#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


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

    p.add_argument(
        "--dos",
        required=True,
        nargs="+",
        help=(
            "One or more phonon DOS files (e.g. zr2sc.dos). "
            "If multiple are given, the script overlays them for comparison (total DOS only)."
        ),
    )
    p.add_argument(
        "--scf-in",
        required=True,
        nargs="+",
        help=(
            "One or more QE scf inputs (used to map PDOS columns to atoms). "
            "If a single scf.in is given, it is reused for all --dos inputs."
        ),
    )
    p.add_argument(
        "--labels",
        default=None,
        help=(
            "Optional comma-separated labels for each dataset when using multiple --dos. "
            "If omitted, labels are derived from DOS filenames."
        ),
    )
    p.add_argument(
        "--dos-norm",
        default=None,
        help=(
            "Per-dataset integer normalization factors (comma-separated). "
            "Each DOS/PDOS curve is divided by its factor after reading. "
            "Typical use: supercell DOS -> per unit cell (e.g. 2x2x2: factor=8)."
        ),
    )

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
        "--system",
        default=None,
        nargs="+",
        help=(
            "Optional system label(s) shown as a separate legend text. "
            "Provide one per dataset when using multiple --dos; a single value is broadcast. "
            "Comma-separated tokens are accepted."
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


def _parse_xy(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = str(s).split(",", 1)
    return float(a), float(b)


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


def _format_system_label(label: str, mode: str) -> str:
    if not label:
        return label
    if mode == "raw":
        return label
    if "$" in label:
        return label
    return re.sub(r"(?<=[A-Za-z\)])(\d+)", r"$_{\1}$", label)


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


def _parse_csv_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    parts = [x.strip() for x in str(s).split(",") if x.strip()]
    return parts if parts else None


def _parse_csv_ints(s: Optional[str]) -> Optional[List[int]]:
    parts = _parse_csv_list(s)
    if parts is None:
        return None
    out: List[int] = []
    for p in parts:
        try:
            v = int(p)
        except ValueError as e:
            raise SystemExit(f"Invalid --dos-norm entry {p!r}: must be integer") from e
        if v <= 0:
            raise SystemExit(f"Invalid --dos-norm entry {p!r}: must be > 0")
        out.append(v)
    return out


def _broadcast_list(name: str, values: Sequence, n: int) -> List:
    if len(values) == n:
        return list(values)
    if len(values) == 1 and n > 1:
        return [values[0]] * n
    raise SystemExit(f"Length mismatch for {name}: got {len(values)}, expected 1 or {n}")


def main() -> None:
    args = _build_parser().parse_args()

    xlim = _parse_lim(args.xlim)
    ylim = _parse_lim(args.ylim)
    figsize = _parse_figsize(args.figsize)
    system_bbox = _parse_xy(args.system_bbox)

    if args.lw is not None and float(args.lw) <= 0:
        raise SystemExit("--lw must be > 0")

    if args.style == "prb":
        _apply_scienceplots_prb_style()

    dos_paths = list(args.dos)
    scf_paths = list(args.scf_in)
    labels_in = _parse_csv_list(args.labels)
    norms_in = _parse_csv_ints(args.dos_norm)

    systems_in = _flatten_tokens(args.system)

    n_cases = max(len(dos_paths), len(scf_paths), len(labels_in or ["x"]), len(norms_in or [1]))
    dos_paths = _broadcast_list("--dos", dos_paths, n_cases)
    scf_paths = _broadcast_list("--scf-in", scf_paths, n_cases)

    if labels_in is None:
        labels = [Path(p).stem for p in dos_paths]
    else:
        labels = _broadcast_list("--labels", labels_in, n_cases)

    if systems_in:
        systems = _broadcast_list("--system", systems_in, n_cases)
    else:
        systems = [""] * n_cases

    if norms_in is None:
        norms = [1] * n_cases
    else:
        norms = _broadcast_list("--dos-norm", norms_in, n_cases)

    # --- Plot ---
    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
    else:
        fig, ax = plt.subplots(dpi=150)

    lw = float(args.lw) if args.lw is not None else (0.8 if args.style == "prb" else 1.6)

    # Multi-dataset compare mode: overlay total DOS only (keeps the figure readable)
    if n_cases > 1:
        print(
            f"Detected {n_cases} datasets. For comparison mode, only total DOS curves are plotted (PDOS is skipped)."
        )

    show_curve_legend = True
    if n_cases > 1 and any(systems):
        # When --system is provided, use the system legend to identify datasets to avoid duplicate legends.
        show_curve_legend = False

    case_colors = [
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

    x_ref: Optional[np.ndarray] = None
    for ic in range(n_cases):
        dos_path = str(dos_paths[ic])
        scf_path = str(scf_paths[ic])
        lab_case = str(labels[ic])
        norm = int(norms[ic])

        x_cm1, dos_tot, pdos_mat = _read_phonon_dos_table(dos_path)
        nat_from_scf = _read_nat_from_scf_in(scf_path)
        atoms = _read_atoms_from_scf_in(scf_path, expected_nat=nat_from_scf)

        n_atoms = nat_from_scf if nat_from_scf is not None else len(atoms)
        if pdos_mat.shape[1] != 0 and pdos_mat.shape[1] != n_atoms:
            raise SystemExit(
                f"PDOS column count mismatch for {dos_path!r}: DOS file has {pdos_mat.shape[1]} per-atom columns, "
                f"but scf.in has {n_atoms} atoms ({scf_path!r})."
            )

        x = _convert_x_unit(x_cm1, args.unit)
        dos_tot, pdos_mat = _convert_dos_unit_for_x(
            dos_tot,
            pdos_mat,
            x_unit=str(args.unit),
            disable_jacobian=bool(args.no_jacobian),
        )

        # Per-dataset normalization (e.g. supercell -> per unit cell)
        if norm != 1:
            dos_tot = dos_tot / float(norm)
            pdos_mat = pdos_mat / float(norm)

        if x_ref is None:
            x_ref = x
        else:
            if len(x) != len(x_ref) or float(np.max(np.abs(x - x_ref))) > 1e-8:
                raise SystemExit(
                    "Multiple datasets must share the same x-grid to be overlaid. "
                    f"Mismatch detected for {dos_path!r}."
                )

        col = case_colors[ic % len(case_colors)]
        ax.plot(x, dos_tot, color=col, lw=lw, label=(lab_case if show_curve_legend else "_nolegend_"))

        # PDOS is only plotted in single-dataset mode.
        if n_cases == 1:
            elements_filter: Optional[set[str]] = None
            if args.elements:
                elements_filter = {x.strip() for x in args.elements.split(",") if x.strip()}

            series: Dict[str, np.ndarray] = {}
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
        ax.set_title("Phonon DOS" if n_cases == 1 else "Phonon DOS (comparison)")

    ax.grid(True, alpha=0.25)

    leg = None
    if show_curve_legend:
        leg_fs = args.legend_fontsize
        if leg_fs is None:
            leg = ax.legend(loc=str(args.legend_loc), frameon=False)
        else:
            leg = ax.legend(loc=str(args.legend_loc), frameon=False, fontsize=float(leg_fs))

    # System legend (positionable)
    if any(systems):
        handles: List[Line2D] = []
        for ic in range(n_cases):
            if not systems[ic]:
                continue
            sys_lab = _format_system_label(str(systems[ic]), str(args.system_format))
            col = case_colors[ic % len(case_colors)]
            handles.append(Line2D([0], [0], color=col, lw=lw, label=sys_lab))

        fs = args.system_fontsize
        if fs is None:
            try:
                fs = float(ax.yaxis.label.get_size()) * 1.15
            except Exception:
                fs = None

        if leg is not None:
            ax.add_artist(leg)

        if system_bbox is None:
            leg_sys = ax.legend(
                handles=handles,
                loc=str(args.system_loc),
                frameon=False,
                fontsize=fs,
            )
        else:
            leg_sys = ax.legend(
                handles=handles,
                loc=str(args.system_loc),
                bbox_to_anchor=system_bbox,
                bbox_transform=ax.transAxes,
                frameon=False,
                fontsize=fs,
            )
        if leg_sys is not None:
            for t in leg_sys.get_texts():
                t.set_fontweight("bold")

    _apply_plot_style(ax, bold=not args.no_bold, sci_y=args.sci_y, ylog=args.ylog, legend=leg)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"Saved: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
