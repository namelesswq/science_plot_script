#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.text import Text


def _apply_global_fontsize(fontsize: Optional[float]) -> None:
    """Apply global default font sizes via rcParams.

    This intentionally does NOT override explicit per-component sizes set later
    (e.g. --label-fontsize, --legend-fontsize, --system-fontsize).
    """

    if fontsize is None:
        return
    fs = float(fontsize)
    if fs <= 0:
        raise SystemExit("--fontsize must be > 0")

    plt.rcParams.update(
        {
            "font.size": fs,
            "axes.titlesize": fs,
            "axes.labelsize": fs,
            "xtick.labelsize": fs,
            "ytick.labelsize": fs,
            "legend.fontsize": fs,
        }
    )


def _set_figure_text_weight(fig, weight: str) -> None:
    for t in fig.findobj(Text):
        try:
            t.set_fontweight(weight)
        except Exception:
            pass


def _apply_legend_frame(leg, *, alpha: float) -> None:
    if leg is None:
        return
    a = float(alpha)
    if not (0.0 <= a <= 1.0):
        raise SystemExit("legend alpha must be in [0, 1]")
    leg.set_frame_on(True)
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(a)
    try:
        frame.set_edgecolor("0.6")
    except Exception:
        pass


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
        nargs="+",
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
        nargs="+",
        help="Glob pattern for PDOS files (overrides auto). Example: 'zr2sc.pdos.pdos_atm#*'",
    )

    
    # Dataset legend (colored lines)
    p.add_argument(
        "--legend",
        default=None,
        nargs="+",
        help=(
            "Legend label(s) for each dataset (each --tot). Provide one per dataset, or a single value to broadcast. "
            "If omitted, uses the total DOS filename stem."
        ),
    
    )
    p.add_argument(
        "--legend-format",
        choices=["chem", "raw"],
        default="raw",
        help="Render --legend text with subscripts (chem) or raw text (raw). Default: raw.",
    )
    p.add_argument(
        "--legend-fontsize",
        type=float,
        default=None,
        help="Font size for legend text. If omitted, uses matplotlib default.",
    )
    p.add_argument(
        "--legend-loc",
        default="upper left",
        help="Legend location (matplotlib legend loc). Default: upper left.",
    )

    p.add_argument(
        "--legend-alpha",
        type=float,
        default=None,
        help=(
            "If set, draw the curve legend with a white semi-transparent background (0..1). "
            "Helps avoid lines obscuring legend text."
        ),
    )
    
    p.add_argument(
        "--legend-bbox",
        default=None,
        help=(
            "Optional legend anchor (bbox_to_anchor) in axes coordinates 'x,y'. "
            "If provided, legend placement uses both --legend-loc and this anchor."
        ),
    
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
            "Special tokens: 'no-tot' disables plotting total DOS; 'tot' forces plotting total DOS. "
            "If provided but empty/blank, the script plots per-element summed PDOS (no orbital-resolved curves)."
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
        "--fontsize",
        type=float,
        default=None,
        help=(
            "Global default font size (rcParams). Does not override explicit per-item sizes like "
            "--label-fontsize/--legend-fontsize/--system-fontsize."
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
        "--xtick-step",
        type=float,
        default=None,
        help="Major tick step for x-axis (energy) in eV. Example: --xtick-step 2.",
    )
    p.add_argument(
        "--ytick-step",
        type=float,
        default=None,
        help="Major tick step for y-axis (DOS). Example: --ytick-step 1.",
    )

    p.add_argument(
        "--fermi-line",
        action="store_true",
        help="Draw a vertical line at E=0 (useful when energies are E-Ef).",
    )

    p.add_argument(
        "--fermi",
        default=None,
        nargs="+",
        help=(
            "Fermi energy in eV. If provided, the script shifts the energy axis as E -> E - Ef so that Ef is at 0 eV. "
            "(Applied consistently to both total DOS and PDOS.)"
        ),
    )

    p.add_argument(
        "--norm",
        default=None,
        nargs="+",
        help=(
            "Optional per-dataset normalization factor(s) applied to DOS/PDOS values: y_plot = y/norm. "
            "Provide one per dataset, or a single value to broadcast."
        ),
    )

    p.add_argument("--ylog", action="store_true", help="Use log scale on y-axis")
    p.add_argument("--no-bold", action="store_true", help="Disable bold text in the figure")
    p.add_argument(
        "--bold-fonts",
        action="store_true",
        help="Force all text in the figure to bold (including for --style prb).",
    )
    p.add_argument(
        "--label-fontsize",
        type=float,
        default=None,
        help="Font size for axis labels (x/y). If omitted, uses matplotlib/style default.",
    )
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

    p.add_argument(
        "--pdos-components",
        default=None,
        help=(
            "Optional m-component selection for specific orbitals (from projwfc PDOS files). "
            "QE pdos_atm files are assumed to have columns: E, ldos, then 2l+1 component columns. "
            "If enabled for an orbital, the script will plot the selected components (and by default also the orbital total/ldos unless --pdos-components-only is set). "
            "Syntax: 'all' or semicolon-separated assignments like 'p=all;d=1,3,5'. "
            "Component indices are 1-based within the component columns."
        ),
    )

    p.add_argument(
        "--pdos-components-only",
        action="store_true",
        help=(
            "When using --pdos-components, plot only the selected m-components and do NOT plot the orbital total (ldos). "
            "This is useful if you want only component curves (e.g. only d_{xz}, d_{yz}, ...)."
        ),
    )

    p.add_argument(
        "--linestyle",
        default=None,
        help=(
            "Optional per-curve line style rules to help distinguish overlapping curves. "
            "Provide semicolon-separated assignments 'key=value'. You may also provide a single value (e.g. '--') to broadcast to all curves. Supported keys:\n"
            "  - tot: total DOS curve\n"
            "  - pdos: any PDOS curve (fallback)\n"
            "  - all: any curve (fallback)\n"
            "  - <el> / <el>-total: element-summed PDOS (when --orbitals is blank), e.g. 'zr=--'\n"
            "  - orb-total: PDOS orbital totals (comp_idx=0)\n"
            "  - orb-comp: PDOS orbital components (comp_idx>0)\n"
            "  - orb-comp-<k>: PDOS component #k across orbitals (k is 1-based within that orbital)\n"
            "  - <orb>: any PDOS for that orbital (p/d/f), e.g. 'd=--'\n"
            "  - <orb>-total / <orb>-comp: e.g. 'd-total=-', 'd-comp=:'\n"
            "  - <orb>-comp-<k>: component #k for a given orbital (e.g. 'd-comp-3=:' )\n"
            "  - <el>-<orb>: element+orbital PDOS (e.g. 'zr-d=--')\n"
            "  - <el>-<orb>-total / <el>-<orb>-comp: (e.g. 'zr-d-total=-', 'zr-d-comp=:' )\n"
            "  - <el>-<orb>-comp-<k>: component #k for an element+orbital (e.g. 'zr-d-comp-5=-.' )\n"
            "  - Per-dataset overrides: prefix the key with 'N:' where N is 1-based dataset index.\n"
            "    Example: '1:tot=-;2:tot=--;2:zr-d-total=--'\n"
            "Values are matplotlib linestyles like '-', '--', '-.', ':', or names like 'solid', 'dashed'. "
            "Example: --linestyle 'tot=-;orb-total=--;orb-comp=:;d-comp=-.'"
        ),
    )

    p.add_argument(
        "--linewidth",
        default=None,
        help=(
            "Optional per-curve line width rules. Syntax and keys are the same as --linestyle, but values are floats. "
            "You may also provide a single number to broadcast to all curves (e.g. '--linewidth 3'). "
            "Example: --linewidth 'tot=2.5;zr=1.8;orb-total=1.8;d-comp-1=1.2;2:tot=3.0'"
        ),
    )

    p.add_argument("--title", default=None, help="Plot title")

    p.add_argument(
        "--system",
        default=None,
        help="Overall system/material label shown as a separate legend entry (pure text).",
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
        "--system-alpha",
        type=float,
        default=None,
        help=(
            "If set, draw the --system legend with a white semi-transparent background (0..1). "
            "Helps avoid lines obscuring the system label."
        ),
    )
    
    p.add_argument(
        "--system-bbox",
        default=None,
        help=(
            "Optional legend anchor (bbox_to_anchor) in axes coordinates 'x,y' (e.g. '1.02,1.0' for outside right). "
            "If provided, legend placement uses both --system-loc and this anchor."
        ),
    
    )
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


def _parse_xy(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = str(s).split(",", 1)
    return float(a), float(b)


def _broadcast_list(xs: Sequence[str], n: int, name: str) -> List[str]:
    if len(xs) == n:
        return list(xs)
    if len(xs) == 1:
        return [str(xs[0])] * n
    raise SystemExit(f"{name} expects 1 value or {n} values, but got {len(xs)}")


def _parse_float_list(tokens: Optional[Sequence[str]], *, n: int, name: str) -> List[Optional[float]]:
    flat = _flatten_tokens(tokens)
    if not flat:
        return [None] * n
    flat = _broadcast_list(flat, n, name)
    out: List[Optional[float]] = []
    for t in flat:
        if t is None or str(t).strip() == "":
            out.append(None)
            continue
        try:
            out.append(float(t))
        except ValueError as e:
            raise SystemExit(f"{name} contains non-float token {t!r}") from e
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


def _flatten_tokens_allow_blank(tokens: Optional[Sequence[str]]) -> List[str]:
    """Flatten comma-separated tokens but keep blanks.

    This is useful for arguments like --legend where an explicitly blank token
    means "no prefix".
    """

    if not tokens:
        return []
    out: List[str] = []
    for t in tokens:
        if t is None:
            continue
        for s in str(t).split(","):
            out.append(s.strip())
    return out


def _format_system_label(label: str, mode: str) -> str:
    if not label:
        return label
    if mode == "raw":
        return label
    if "$" in label:
        return label
    return re.sub(r"(?<=[A-Za-z\)])(\d+)", r"$_{\1}$", label)


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


def _load_table(path: str) -> np.ndarray:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(f"Unexpected table in {path}: shape={data.shape}")
    return np.asarray(data, dtype=float)


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


def _orbital_component_count(orb: str) -> Optional[int]:
    o = str(orb).strip().lower()
    # For this script, we treat s as having no meaningful m-components.
    # QE may output one column after ldos for s, but plotting it would just
    # duplicate the s total (ldos), so we disable component expansion for s.
    if o == "s":
        return None
    if o == "p":
        return 3
    if o == "d":
        return 5
    if o == "f":
        return 7
    return None


def _orbital_component_label(orb: str, comp_idx: int, *, bold: bool = False) -> str:
    """Human-friendly label for an orbital component (1-based).

    For d, use conventional real-orbital names in the user-preferred order.
    """

    o = str(orb).strip().lower()
    k = int(comp_idx)
    if o == "p":
        # Conventional real-orbital labeling.
        if bold:
            names = [r"$\mathbf{p_{x}}$", r"$\mathbf{p_{y}}$", r"$\mathbf{p_{z}}$"]
        else:
            names = [r"$p_{x}$", r"$p_{y}$", r"$p_{z}$"]
        if 1 <= k <= len(names):
            return names[k - 1]
    if o == "d":
        if bold:
            names = [
                r"$\mathbf{d_{xz}}$",
                r"$\mathbf{d_{yz}}$",
                r"$\mathbf{d_{xy}}$",
                r"$\mathbf{d_{x^2-y^2}}$",
                r"$\mathbf{d_{z^2}}$",
            ]
        else:
            names = [
                r"$d_{xz}$",
                r"$d_{yz}$",
                r"$d_{xy}$",
                r"$d_{x^2-y^2}$",
                r"$d_{z^2}$",
            ]
        if 1 <= k <= len(names):
            return names[k - 1]
    return f"{o}[{k}]"


def _parse_pdos_components(spec: Optional[str]) -> Tuple[bool, Dict[str, Optional[List[int]]]]:
    """Parse --pdos-components.

    Returns:
      - all_orbitals: if True, apply to any orbital with known l (s/p/d/f)
      - map: orb -> None (meaning all components) or explicit 1-based indices
    """

    if spec is None:
        return False, {}

    s = str(spec).strip()
    if not s:
        return False, {}

    if s.lower() == "all":
        return True, {}

    out: Dict[str, Optional[List[int]]] = {}
    parts = [p.strip() for p in s.split(";") if p.strip()]
    if not parts:
        return False, {}

    for part in parts:
        if "=" not in part:
            raise SystemExit(
                f"Invalid --pdos-components entry {part!r}. Expected like 'p=all' or 'd=1,3,5' (semicolon-separated)."
            )
        orb_raw, rhs = part.split("=", 1)
        orb = orb_raw.strip().lower()
        if not orb:
            raise SystemExit(f"Invalid --pdos-components entry {part!r}: missing orbital")
        if orb == "s":
            raise SystemExit("--pdos-components does not support orbital 's' (s has no meaningful components).")
        if _orbital_component_count(orb) is None:
            raise SystemExit(f"Invalid --pdos-components orbital {orb!r}: expected one of p,d,f")

        rhs2 = rhs.strip().lower()
        if rhs2 in {"all", "*"}:
            out[orb] = None
            continue
        if not rhs2:
            raise SystemExit(f"Invalid --pdos-components entry {part!r}: missing component list")

        idxs: List[int] = []
        for tok in [t.strip() for t in rhs.split(",") if t.strip()]:
            if "-" in tok:
                a, b = tok.split("-", 1)
                try:
                    ia = int(a)
                    ib = int(b)
                except ValueError as e:
                    raise SystemExit(f"Invalid component range token {tok!r} in --pdos-components") from e
                if ia <= 0 or ib <= 0:
                    raise SystemExit(f"Component indices must be positive in --pdos-components (got {tok!r})")
                lo, hi = (ia, ib) if ia <= ib else (ib, ia)
                idxs.extend(list(range(lo, hi + 1)))
            else:
                try:
                    idxs.append(int(tok))
                except ValueError as e:
                    raise SystemExit(f"Invalid component index token {tok!r} in --pdos-components") from e

        if not idxs:
            raise SystemExit(f"Invalid --pdos-components entry {part!r}: empty component list")

        seen: set[int] = set()
        idxs2: List[int] = []
        for x in idxs:
            if x <= 0:
                raise SystemExit(f"Component indices must be >= 1 in --pdos-components (got {x})")
            if x in seen:
                continue
            seen.add(x)
            idxs2.append(x)
        out[orb] = idxs2

    return False, out


def _parse_linestyle_rules(spec: Optional[str]) -> Dict[str, str]:
    """Parse --linestyle rules.

    Syntax: semicolon-separated assignments 'key=value'.
    Keys are normalized to lowercase.
    """

    if spec is None:
        return {}
    s = str(spec).strip()
    if not s:
        return {}

    # Be tolerant to common punctuation and separators in shell usage.
    # Users may type Chinese punctuation or use comma instead of semicolon.
    s = s.replace("；", ";").replace("，", ",")

    # Broadcast: a single value applies to all curves.
    # Example: --linestyle '--'
    if "=" not in s:
        v = s.strip()
        if not v:
            return {}
        return {"all": v}

    out: Dict[str, str] = {}
    parts = [p.strip() for p in re.split(r"[;,]+", s) if p.strip()]
    for part in parts:
        if "=" not in part:
            raise SystemExit(f"Invalid --linestyle token {part!r}: expected 'key=value' (semicolon-separated)")
        k, v = part.split("=", 1)
        key = k.strip().lower()
        val = v.strip()
        if not key:
            raise SystemExit(f"Invalid --linestyle token {part!r}: empty key")
        if not val:
            raise SystemExit(f"Invalid --linestyle token {part!r}: empty value")
        out[key] = val
    return out


def _parse_linewidth_rules(spec: Optional[str]) -> Dict[str, float]:
    """Parse --linewidth rules.

    Syntax is the same as --linestyle, but values are positive floats.
    """

    if spec is None:
        return {}
    s = str(spec).strip()
    if not s:
        return {}

    s = s.replace("；", ";").replace("，", ",")

    # Broadcast: a single float applies to all curves.
    # Example: --linewidth 3
    if "=" not in s:
        try:
            w = float(s)
        except ValueError as e:
            raise SystemExit(
                f"Invalid --linewidth {s!r}: expected a single float (broadcast) or 'key=value' rules"
            ) from e
        if not (w > 0.0):
            raise SystemExit(f"Invalid --linewidth {w}: must be > 0")
        return {"all": w}

    out: Dict[str, float] = {}
    parts = [p.strip() for p in re.split(r"[;,]+", s) if p.strip()]
    for part in parts:
        if "=" not in part:
            raise SystemExit(f"Invalid --linewidth token {part!r}: expected 'key=value' (semicolon-separated)")
        k, v = part.split("=", 1)
        key = k.strip().lower()
        val = v.strip()
        if not key:
            raise SystemExit(f"Invalid --linewidth token {part!r}: empty key")
        if not val:
            raise SystemExit(f"Invalid --linewidth token {part!r}: empty value")
        try:
            w = float(val)
        except ValueError as e:
            raise SystemExit(f"Invalid --linewidth value for {key!r}: {val!r} (expected float)") from e
        if not (w > 0.0):
            raise SystemExit(f"Invalid --linewidth value for {key!r}: {w} (must be > 0)")
        out[key] = w
    return out


def _choose_linewidth(
    rules: Dict[str, float],
    *,
    kind: str,
    ds: Optional[int] = None,
    el: Optional[str] = None,
    orb: Optional[str] = None,
    comp_idx: Optional[int] = None,
) -> Optional[float]:
    """Pick linewidth based on the same key/precedence system as _choose_linestyle."""

    if not rules:
        return None

    def _get(key: str) -> Optional[float]:
        kk = str(key).strip().lower()
        if ds is not None:
            v = rules.get(f"{int(ds)}:{kk}")
            if v is not None:
                return float(v)
        v2 = rules.get(kk)
        return float(v2) if v2 is not None else None

    k = str(kind).strip().lower()
    if k in {"tot", "total"}:
        return _get("tot") or _get("total") or _get("all")

    if k != "pdos":
        return _get(k) or _get("all")

    el_norm = str(el).strip().lower() if el is not None else ""
    orb_norm = str(orb).strip().lower() if orb is not None else ""
    is_comp = (comp_idx is not None) and (int(comp_idx) > 0)
    suffix = "comp" if is_comp else "total"
    k_comp = int(comp_idx) if (comp_idx is not None) else 0

    # Element-only PDOS (no orbital key available): allow rules like 'zr=1.8' or 'zr-total=2.0'.
    if el_norm and not orb_norm:
        v = _get(f"{el_norm}-{suffix}") or _get(f"{el_norm}_{suffix}")
        if v is not None:
            return v
        v = _get(el_norm)
        if v is not None:
            return v

    if el_norm and orb_norm:
        if is_comp:
            v = _get(f"{el_norm}-{orb_norm}-comp-{k_comp}") or _get(f"{el_norm}_{orb_norm}_comp_{k_comp}")
            if v is not None:
                return v
        v = _get(f"{el_norm}-{orb_norm}-{suffix}") or _get(f"{el_norm}_{orb_norm}_{suffix}")
        if v is not None:
            return v
        v = _get(f"{el_norm}-{orb_norm}") or _get(f"{el_norm}_{orb_norm}")
        if v is not None:
            return v

    if orb_norm:
        if is_comp:
            v = _get(f"{orb_norm}-comp-{k_comp}") or _get(f"{orb_norm}_comp_{k_comp}")
            if v is not None:
                return v
        v = _get(f"{orb_norm}-{suffix}") or _get(f"{orb_norm}_{suffix}")
        if v is not None:
            return v
        v = _get(orb_norm)
        if v is not None:
            return v

    if is_comp:
        v = _get(f"orb-comp-{k_comp}") or _get(f"orb_comp_{k_comp}")
        if v is not None:
            return v

    v = _get("orb-comp" if is_comp else "orb-total")
    if v is not None:
        return v

    return _get("pdos") or _get("all")


def _choose_linestyle(
    rules: Dict[str, str],
    *,
    kind: str,
    ds: Optional[int] = None,
    el: Optional[str] = None,
    orb: Optional[str] = None,
    comp_idx: Optional[int] = None,
) -> Optional[str]:
    """Pick linestyle based on curve kind/orbital/component.

    kind:
      - 'tot' for total DOS
      - 'pdos' for any PDOS curve

        For PDOS, comp_idx==0 means orbital total, comp_idx>0 means component.
        Rules support per-dataset overrides by prefixing keys with 'N:' where N is
        1-based dataset index (order of --tot). For example: '2:tot=--'.

        Precedence (most specific first), each with optional 'N:' dataset override:
            1) '<el>-<orb>-comp-<k>'
            2) '<el>-<orb>-total' / '<el>-<orb>-comp'
            3) '<el>-<orb>'
            4) '<orb>-comp-<k>'
            5) '<orb>-total' / '<orb>-comp'
            6) '<orb>'
            7) 'orb-comp-<k>'
            8) 'orb-total' / 'orb-comp'
            9) 'pdos'
            10) None (matplotlib default)
    """

    if not rules:
        return None

    def _get(key: str) -> Optional[str]:
        kk = str(key).strip().lower()
        if ds is not None:
            v = rules.get(f"{int(ds)}:{kk}")
            if v is not None:
                return v
        return rules.get(kk)

    k = str(kind).strip().lower()
    if k in {"tot", "total"}:
        return _get("tot") or _get("total") or _get("all")

    if k != "pdos":
        return _get(k) or _get("all")

    el_norm = str(el).strip().lower() if el is not None else ""
    orb_norm = str(orb).strip().lower() if orb is not None else ""
    is_comp = (comp_idx is not None) and (int(comp_idx) > 0)
    suffix = "comp" if is_comp else "total"
    k_comp = int(comp_idx) if (comp_idx is not None) else 0

    # Element-only PDOS (no orbital key available): allow rules like 'zr=--' or 'zr-total=-'.
    if el_norm and not orb_norm:
        v = _get(f"{el_norm}-{suffix}") or _get(f"{el_norm}_{suffix}")
        if v is not None:
            return v
        v = _get(el_norm)
        if v is not None:
            return v

    if el_norm and orb_norm:
        if is_comp:
            v = _get(f"{el_norm}-{orb_norm}-comp-{k_comp}") or _get(f"{el_norm}_{orb_norm}_comp_{k_comp}")
            if v is not None:
                return v
        v = _get(f"{el_norm}-{orb_norm}-{suffix}") or _get(f"{el_norm}_{orb_norm}_{suffix}")
        if v is not None:
            return v
        v = _get(f"{el_norm}-{orb_norm}") or _get(f"{el_norm}_{orb_norm}")
        if v is not None:
            return v

    if orb_norm:
        if is_comp:
            v = _get(f"{orb_norm}-comp-{k_comp}") or _get(f"{orb_norm}_comp_{k_comp}")
            if v is not None:
                return v
        v = _get(f"{orb_norm}-{suffix}") or _get(f"{orb_norm}_{suffix}")
        if v is not None:
            return v
        v = _get(orb_norm)
        if v is not None:
            return v

    if is_comp:
        v = _get(f"orb-comp-{k_comp}") or _get(f"orb_comp_{k_comp}")
        if v is not None:
            return v

    v = _get("orb-comp" if is_comp else "orb-total")
    if v is not None:
        return v

    return _get("pdos") or _get("all")


def main() -> None:
    args = _build_parser().parse_args()

    if args.bold_fonts and args.no_bold:
        raise SystemExit("Do not use --bold-fonts together with --no-bold")

    pdos_components_all, pdos_components_map = _parse_pdos_components(args.pdos_components)
    linestyle_rules = _parse_linestyle_rules(args.linestyle)
    linewidth_rules = _parse_linewidth_rules(args.linewidth)

    xlim = _parse_lim(args.xlim)
    ylim = _parse_lim(args.ylim)
    n0_map = _parse_n0_map(args.n0)
    figsize_override = _parse_figsize(args.figsize)
    legend_bbox = _parse_xy(args.legend_bbox)
    system_bbox = _parse_xy(args.system_bbox)
    tot_paths = [str(x) for x in args.tot]
    n_cases = len(tot_paths)

    pdos_globs_in = _flatten_tokens(args.pdos_glob)
    if pdos_globs_in:
        pdos_globs = _broadcast_list(pdos_globs_in, n_cases, "--pdos-glob")
    else:
        pdos_globs = [""] * n_cases

    fermis = _parse_float_list(args.fermi, n=n_cases, name="--fermi")

    # Dataset legend / prefix: allow explicit blanks via --legend ' '. If --legend is
    # omitted and there is only one dataset, default to no prefix.
    if args.legend is None:
        legends = [""] * n_cases if n_cases == 1 else [Path(p).stem for p in tot_paths]
    else:
        legends_in = _flatten_tokens_allow_blank(args.legend)
        if legends_in:
            legends = _broadcast_list(legends_in, n_cases, "--legend")
        else:
            # Should be rare (nargs='+'), but keep a safe fallback.
            legends = [""] * n_cases if n_cases == 1 else [Path(p).stem for p in tot_paths]

    norms = _parse_float_list(args.norm, n=n_cases, name="--norm")
    for i, nv in enumerate(norms):
        if nv is not None and float(nv) == 0.0:
            raise SystemExit(f"--norm must be non-zero (dataset#{i+1})")

    # Elements filter. Special behavior: if user provided --elements but it parses to empty,
    # interpret it as "disable PDOS; plot only total DOS".
    plot_pdos = True
    elements_filter: Optional[set[str]] = None
    if args.elements is not None:
        elems = {x.strip() for x in str(args.elements).split(",") if x.strip()}
        if not elems:
            plot_pdos = False
            elements_filter = set()
        else:
            elements_filter = elems

    orbitals_filter = None
    plot_total = True
    element_sum_mode = False
    # Special behavior: if user provided --orbitals but it parses to empty/blank,
    # interpret it as "plot per-element summed PDOS (no orbital-resolved curves)".
    if args.orbitals is not None and str(args.orbitals).strip() == "":
        element_sum_mode = True
        orbitals_filter = None
        plot_total = True
    elif args.orbitals:
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

    # If PDOS is disabled (e.g. --elements ''), force plotting total DOS.
    if not plot_pdos:
        plot_total = True
        orbitals_filter = None

    if plot_pdos and (n_cases > 1) and (args.pdos is not None) and (len(args.pdos) > 0):
        raise SystemExit(
            "When providing multiple --tot, do not use --pdos (explicit PDOS list). Use --pdos-glob or auto discovery."
        )

    # Plot
    if n_cases > 1:
        print(f"Detected {n_cases} datasets. Total DOS and PDOS are plotted for comparison.")

    if args.style == "prb":
        # One-click PRB-ish formatting from SciencePlots.
        _apply_scienceplots_prb_style()

    # Apply global defaults after selecting the base style.
    _apply_global_fontsize(args.fontsize)

    # Bold selection: keep legacy behavior for non-prb, but allow forcing bold for prb.
    want_bold_fonts = bool(args.bold_fonts) or ((args.style != "prb") and (not args.no_bold))
    if want_bold_fonts:
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        if args.style != "prb":
            plt.rcParams["axes.linewidth"] = 2

    if figsize_override is not None:
        fig, ax = plt.subplots(figsize=figsize_override)
    else:
        fig, ax = plt.subplots()

    # Colors
    case_colors = [
        "black",
        "tab:red",
        "tab:blue",
        "tab:green",
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:cyan",
        "tab:pink",
        "tab:olive",
        "tab:gray",
    ]

    used_total_colors = {case_colors[i % len(case_colors)] for i in range(n_cases)}

    # PDOS palette: fixed order, easy to distinguish (no cycling). Keep the leading colors' order.
    pdos_base = [
        "tab:red",
        "tab:green",
        "tab:blue",
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:cyan",
        "tab:pink",
        "tab:olive",
        "tab:gray",
    ]
    pdos_colors = [c for c in pdos_base if c not in used_total_colors]
    if not pdos_colors:
        pdos_colors = list(pdos_base)

    extra_pdos_colors: List[object] = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors"):
            extra_pdos_colors.extend(list(getattr(cmap, "colors")))
        else:
            extra_pdos_colors.extend([cmap(k / 19.0) for k in range(20)])
    extra_pdos_colors = [c for c in extra_pdos_colors if c not in used_total_colors]

    pdos_palette: List[object] = list(pdos_colors) + list(extra_pdos_colors)
    next_pdos_color = 0

    # Curve legend entries should always include dataset prefix (from --legend),
    # so show total DOS in the curve legend as well.
    show_total_in_curve_legend = True

    for ic in range(n_cases):
        tot_path = tot_paths[ic]
        ef = fermis[ic]
        nv = norms[ic]

        # total DOS
        e_tot, dos_tot = _load_two_cols(tot_path, xcol=0, ycol=args.tot_col)
        if ef is not None:
            e_tot = e_tot - float(ef)
        if nv is not None:
            dos_tot = dos_tot / float(nv)

        pdos_files: List[str] = []
        if plot_pdos:
            # PDOS file discovery
            if n_cases == 1 and args.pdos is not None and len(args.pdos) > 0:
                pdos_files = list(args.pdos)
            else:
                pat = str(pdos_globs[ic]).strip()
                if pat:
                    pattern = pat
                else:
                    base = os.path.basename(tot_path)
                    if base.endswith(".pdos.pdos_tot"):
                        prefix = base[: -len(".pdos.pdos_tot")]
                    else:
                        prefix = os.path.splitext(base)[0]
                    pattern = os.path.join(os.path.dirname(tot_path) or ".", f"{prefix}.pdos.pdos_atm#*")
                pdos_files = sorted(glob.glob(pattern))

            if not pdos_files:
                msg = [
                    "No PDOS files found.",
                    f"- Total DOS file: {tot_path!r}",
                ]
                pat = str(pdos_globs[ic]).strip()
                if pat:
                    msg.append(f"- Your --pdos-glob for this dataset was: {pat!r}")
                    if "pdos_atom#" in pat:
                        alt = pat.replace("pdos_atom#", "pdos_atm#")
                        alt_hits = sorted(glob.glob(alt))
                        if alt_hits:
                            msg.append(f"- Did you mean: {alt!r} ? (found {len(alt_hits)} files)")
                else:
                    tot_dir = os.path.dirname(tot_path) or "."
                    generic = os.path.join(tot_dir, "*.pdos.pdos_atm#*")
                    generic_hits = sorted(glob.glob(generic))
                    if generic_hits:
                        msg.append(f"- In this directory, files matching {generic!r}: {len(generic_hits)}")
                msg.append("Provide --pdos-glob (one per --tot) or ensure PDOS files exist next to each tot file.")
                raise SystemExit("\n".join(msg))

        # Key layout:
        #   - element_sum_mode: el -> summed PDOS
        #   - non-merged: (el, wfc_idx, orb, comp_idx) where comp_idx==0 means orbital total (ldos or --pdos-col)
        #   - merged:     (el, orb, comp_idx)
        groups_el: Dict[str, np.ndarray] = {}
        groups_wfc: Dict[Tuple[str, int, str, int], np.ndarray] = {}
        groups_merged: Dict[Tuple[str, str, int], np.ndarray] = {}
        energy_ref: Optional[np.ndarray] = None

        if plot_pdos:
            # Sum PDOS by (element, wfc_index, orbital) unless --merge-wfc
            for f in pdos_files:
                el, wfc_idx, orb = _parse_element_wfc_orbital_from_name(f)
                if elements_filter is not None and el not in elements_filter:
                    continue
                if (not element_sum_mode) and (orbitals_filter is not None) and (orb not in orbitals_filter):
                    continue

                # In element-sum mode, we ignore component expansion and always sum the chosen PDOS column.
                if element_sum_mode:
                    e, y = _load_two_cols(f, xcol=0, ycol=args.pdos_col)
                    if ef is not None:
                        e = e - float(ef)
                    if nv is not None:
                        y = y / float(nv)

                    if energy_ref is None:
                        energy_ref = e
                    else:
                        if len(e) != len(energy_ref) or np.max(np.abs(e - energy_ref)) > 1e-8:
                            raise SystemExit(
                                f"Energy grid mismatch among PDOS files for dataset {tot_path!r}. Offending file: {f}. "
                                "Please regenerate PDOS with consistent energy grid."
                            )

                    if el not in groups_el:
                        groups_el[el] = np.zeros_like(y, dtype=float)
                    groups_el[el] += y
                    continue

                orb_norm = str(orb).strip().lower()
                comp_count = _orbital_component_count(orb_norm)
                use_components = False
                comp_sel: Optional[List[int]] = None
                if pdos_components_all and comp_count is not None:
                    use_components = True
                    comp_sel = None
                elif orb_norm in pdos_components_map:
                    use_components = True
                    comp_sel = pdos_components_map.get(orb_norm)

                if use_components:
                    tab = _load_table(f)
                    if tab.shape[1] < 2:
                        raise SystemExit(f"Unexpected PDOS table in {f}: expected at least 2 columns")
                    e = np.asarray(tab[:, 0], dtype=float)
                    if ef is not None:
                        e = e - float(ef)

                    # By default, include orbital total (ldos). If --pdos-components-only is set,
                    # we skip plotting/summing the orbital total and only keep the selected components.
                    if not args.pdos_components_only:
                        y_tot = np.asarray(tab[:, 1], dtype=float)
                        if nv is not None:
                            y_tot = y_tot / float(nv)

                        if args.merge_wfc:
                            key2 = (el, orb, 0)
                            if key2 not in groups_merged:
                                groups_merged[key2] = np.zeros_like(y_tot, dtype=float)
                            groups_merged[key2] += y_tot
                        else:
                            key3 = (el, int(wfc_idx), orb, 0)
                            if key3 not in groups_wfc:
                                groups_wfc[key3] = np.zeros_like(y_tot, dtype=float)
                            groups_wfc[key3] += y_tot

                    n_comp_in_file = int(tab.shape[1] - 2)
                    if n_comp_in_file <= 0:
                        raise SystemExit(
                            f"Requested m components for orbital {orb!r} from file {f!r}, but file has no component columns (shape={tab.shape})."
                        )

                    n_avail = n_comp_in_file
                    if comp_sel is None:
                        idxs = list(range(1, n_avail + 1))
                    else:
                        idxs = list(comp_sel)

                    for k in idxs:
                        if k < 1 or k > n_avail:
                            raise SystemExit(
                                f"--pdos-components selects component {k} for orbital {orb_norm!r}, but only {n_avail} components are available in {f!r}."
                            )
                        yk = np.asarray(tab[:, 1 + k], dtype=float)
                        if nv is not None:
                            yk = yk / float(nv)
                        if args.merge_wfc:
                            key2 = (el, orb, int(k))
                            if key2 not in groups_merged:
                                groups_merged[key2] = np.zeros_like(yk, dtype=float)
                            groups_merged[key2] += yk
                        else:
                            key3 = (el, int(wfc_idx), orb, int(k))
                            if key3 not in groups_wfc:
                                groups_wfc[key3] = np.zeros_like(yk, dtype=float)
                            groups_wfc[key3] += yk
                else:
                    e, y = _load_two_cols(f, xcol=0, ycol=args.pdos_col)
                    if ef is not None:
                        e = e - float(ef)
                    if nv is not None:
                        y = y / float(nv)

                if energy_ref is None:
                    energy_ref = e
                else:
                    if len(e) != len(energy_ref) or np.max(np.abs(e - energy_ref)) > 1e-8:
                        raise SystemExit(
                            f"Energy grid mismatch among PDOS files for dataset {tot_path!r}. Offending file: {f}. "
                            "Please regenerate PDOS with consistent energy grid."
                        )

                if not use_components:
                    if args.merge_wfc:
                        key2 = (el, orb, 0)
                        if key2 not in groups_merged:
                            groups_merged[key2] = np.zeros_like(y, dtype=float)
                        groups_merged[key2] += y
                    else:
                        key3 = (el, int(wfc_idx), orb, 0)
                        if key3 not in groups_wfc:
                            groups_wfc[key3] = np.zeros_like(y, dtype=float)
                        groups_wfc[key3] += y

            if element_sum_mode:
                if not groups_el:
                    raise SystemExit(f"No PDOS series selected after applying filters for {tot_path!r}.")
            else:
                if args.merge_wfc:
                    if not groups_merged:
                        raise SystemExit(f"No PDOS series selected after applying filters for {tot_path!r}.")
                else:
                    if not groups_wfc:
                        raise SystemExit(f"No PDOS series selected after applying filters for {tot_path!r}.")

        e_pdos = energy_ref if energy_ref is not None else e_tot

        # Labeling
        ds_lab = _format_system_label(str(legends[ic]), str(args.legend_format))
        col_case = case_colors[ic % len(case_colors)]

        if plot_total:
            ls_tot = _choose_linestyle(linestyle_rules, kind="tot", ds=ic + 1)
            kw_ls = {} if ls_tot is None else {"linestyle": ls_tot}
            lw_tot_default = 1.6 if args.style == "prb" else 2.4
            lw_tot = _choose_linewidth(linewidth_rules, kind="tot", ds=ic + 1)
            lw_tot_plot = lw_tot_default if lw_tot is None else float(lw_tot)
            ax.plot(
                e_tot,
                dos_tot,
                color=col_case,
                lw=lw_tot_plot,
                label=((f"{ds_lab}:Total" if ds_lab else "Total") if show_total_in_curve_legend else "_nolegend_"),
                **kw_ls,
            )

        if plot_pdos:
            # PDOS curves: solid, unique colors (avoid total DOS colors)
            if element_sum_mode:
                keys_sorted = sorted(groups_el.keys(), key=lambda x: str(x))
            elif args.merge_wfc:
                keys_sorted = sorted(groups_merged.keys(), key=lambda k: (k[0], k[1], int(k[2])))
            else:
                keys_sorted = sorted(groups_wfc.keys(), key=lambda k: (k[0], k[2], int(k[1]), int(k[3])))

            label_map = {}
            if (not element_sum_mode) and (not args.merge_wfc) and n0_map:
                keys_base = sorted({(k[0], int(k[1]), k[2]) for k in keys_sorted})
                label_map = _build_n_label_map(keys_base, n0_map)

            for key in keys_sorted:
                if element_sum_mode:
                    el = str(key)
                    orb = ""
                    comp_idx = 0
                    y = groups_el[el]
                    lab0 = f"{el}"
                elif args.merge_wfc:
                    el, orb, comp_idx = key  # type: ignore[misc]
                    y = groups_merged[(el, orb, int(comp_idx))]
                    if int(comp_idx) == 0:
                        lab0 = f"{el}-{orb}"
                    else:
                        lab0 = f"{el}-{_orbital_component_label(str(orb), int(comp_idx), bold=want_bold_fonts)}"
                else:
                    el, wfc_idx, orb, comp_idx = key  # type: ignore[misc]
                    y = groups_wfc[(el, int(wfc_idx), orb, int(comp_idx))]
                    base = label_map.get((el, int(wfc_idx), orb), f"{el}-{wfc_idx}{orb}")
                    if int(comp_idx) == 0:
                        lab0 = base
                    else:
                        # Preserve any n0-based prefix when available (e.g. 'Zr-4d' -> 'Zr-4$d_{xz}$').
                        if str(base).endswith(str(orb)):
                            prefix = str(base)[: -len(str(orb))]
                            lab0 = f"{prefix}{_orbital_component_label(str(orb), int(comp_idx), bold=want_bold_fonts)}"
                        else:
                            lab0 = f"{el}-{int(wfc_idx)}{_orbital_component_label(str(orb), int(comp_idx), bold=want_bold_fonts)}"

                if next_pdos_color >= len(pdos_palette):
                    raise SystemExit(
                        "Too many PDOS curves to assign distinct colors. "
                        "Reduce --elements/--orbitals/--pdos-components selection, or use --merge-wfc."
                    )
                col = pdos_palette[next_pdos_color]
                next_pdos_color += 1
                ls_p = _choose_linestyle(
                    linestyle_rules,
                    kind="pdos",
                    ds=ic + 1,
                    el=str(el),
                    orb=str(orb),
                    comp_idx=int(comp_idx),
                )
                kw_ls = {} if ls_p is None else {"linestyle": ls_p}
                lw_pdos_default = 1.2 if args.style == "prb" else 2.0
                lw_p = _choose_linewidth(
                    linewidth_rules,
                    kind="pdos",
                    ds=ic + 1,
                    el=str(el),
                    orb=str(orb),
                    comp_idx=int(comp_idx),
                )
                lw_p_plot = lw_pdos_default if lw_p is None else float(lw_p)
                ax.plot(
                    e_pdos,
                    y,
                    lw=lw_p_plot,
                    color=col,
                    label=(f"{ds_lab}:{lab0}" if ds_lab else str(lab0)),
                    **kw_ls,
                )

    if args.fermi_line:
        ax.axvline(0.0, color="gray", linestyle="--", lw=1.2, alpha=0.8)

    if any(f is not None for f in fermis):
        if want_bold_fonts:
            ax.set_xlabel(r"$\mathbf{E-E_{f}}$ (eV)")
        else:
            ax.set_xlabel(r"$E-E_{f}$ (eV)")
    else:
        ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("DOS (states/eV/unit cell)")

    if args.label_fontsize is not None:
        fs = float(args.label_fontsize)
        ax.xaxis.label.set_size(fs)
        ax.yaxis.label.set_size(fs)

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

    if args.xtick_step is not None:
        step = float(args.xtick_step)
        if step <= 0:
            raise SystemExit("--xtick-step must be > 0")
        ax.xaxis.set_major_locator(MultipleLocator(step))

    if args.ytick_step is not None:
        step = float(args.ytick_step)
        if step <= 0:
            raise SystemExit("--ytick-step must be > 0")
        if not args.ylog:
            ax.yaxis.set_major_locator(MultipleLocator(step))

    # Curve legend (orbitals; can be large)
    legend_kwargs = {"frameon": bool(args.legend_alpha is not None), "ncols": 2, "loc": str(args.legend_loc)}
    if args.legend_fontsize is not None:
        legend_kwargs["fontsize"] = float(args.legend_fontsize)

    if legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox
        legend_kwargs["bbox_transform"] = ax.transAxes

    if args.style != "prb":
        ax.grid(True, alpha=0.25)

    leg_curve = ax.legend(**legend_kwargs)
    if args.legend_alpha is not None:
        _apply_legend_frame(leg_curve, alpha=float(args.legend_alpha))

    # Note: we intentionally do NOT add a separate dataset legend here.
    # PDOS/TOT curve legend already uses "<legend>:<series>" labels.
    leg_main = None

    # Global system annotation legend (pure text)
    if args.system is not None and str(args.system).strip():
        sys_lab = _format_system_label(str(args.system), str(args.system_format))
        handle = Line2D([0], [0], color="none", lw=0, label=sys_lab)
        fs = args.system_fontsize
        if fs is None:
            try:
                fs = float(ax.yaxis.label.get_size()) * 1.15
            except Exception:
                fs = None

        if leg_curve is not None:
            ax.add_artist(leg_curve)

        sys_frame = bool(args.system_alpha is not None)
        sys_kwargs = {
            "frameon": sys_frame,
            "fontsize": fs,
            "handlelength": 0,
            "handletextpad": 0,
        }
        if system_bbox is None:
            leg_sys = ax.legend(
                handles=[handle],
                loc=str(args.system_loc),
                **sys_kwargs,
            )
        else:
            leg_sys = ax.legend(
                handles=[handle],
                loc=str(args.system_loc),
                bbox_to_anchor=system_bbox,
                bbox_transform=ax.transAxes,
                **sys_kwargs,
            )
        if args.system_alpha is not None:
            _apply_legend_frame(leg_sys, alpha=float(args.system_alpha))
        if leg_sys is not None:
            for t in leg_sys.get_texts():
                try:
                    t.set_ha("center")
                except Exception:
                    pass
            # Best-effort centering of the legend box contents.
            try:
                leg_sys._legend_box.align = "center"  # noqa: SLF001
            except Exception:
                pass

    _apply_style(
        ax,
        legend=leg_curve,
        bold=want_bold_fonts,
        sci_y=args.sci_y,
        ylog=args.ylog,
    )

    if want_bold_fonts:
        _set_figure_text_weight(fig, "bold")

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"Saved: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
