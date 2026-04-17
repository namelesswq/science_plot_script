#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from matplotlib.lines import Line2D
from matplotlib.text import Text
import matplotlib.patheffects as patheffects

# 复用你的现成接口
from perturbo_meanfp_io import (
    apply_plot_style,
    apply_scienceplots_prb_style,
    apply_default_bold_rcparams,
    apply_global_fontsize,
    apply_tick_steps,
    apply_legend_frame,
    get_config_band_series,
    get_energy_by_band,
    get_mu_ev,
    get_velocity_by_band,
    load_meanfp_yaml,
)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Cumulative electron transport vs MFP using meanfp.yml (MFP/tau/E on irr-k) + "
            "tdf.h5 (per-state mfd on reducible-k) + tet.h5 (kpt mapping + tetra list). "
            "The mfd-based path reproduces Perturbo's tetra integration without using tdf(E)."
        )
    )

    # -------------------------
    # Inputs (explicit, no -p)
    # -------------------------
    p.add_argument(
        "--meanfp",
        action="append",
        default=None,
        metavar="MEANFP_YML",
        help="Path to *_meanfp.yml (can be given multiple times to plot multiple systems).",
    )
    p.add_argument(
        "--tdf",
        action="append",
        default=None,
        metavar="TDF_H5",
        help="Path to *_tdf.h5 matching each --meanfp (can be given multiple times).",
    )
    p.add_argument(
        "--legend",
        action="append",
        default=None,
        metavar="LABEL",
        help="Curve label for each dataset (optional; can be given multiple times).",
    )
    # Global system annotation (pure text)
    p.add_argument(
        "--system",
        default=None,
        help="Overall system/material label shown as a separate legend entry (e.g. 'Zr2SC').",
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
        help="Font size for --system legend text. If omitted, uses an automatic slightly larger size.",
    )
    p.add_argument(
        "--system-loc",
        default=["upper", "left"],
        nargs="+",
        help="Legend location for --system (matplotlib legend loc). Example: --system-loc upper left. Default: upper left.",
    )
    p.add_argument(
        "--system-bbox",
        default=None,
        help="Optional legend anchor (bbox_to_anchor) in axes coordinates 'x,y'.",
    )
    p.add_argument(
        "--system-alpha",
        type=float,
        default=None,
        help="If set, draw the system annotation with a white semi-transparent frame (0..1).",
    )

    # Backward compatible (deprecated): --pair MEANFP TDF LABEL (no -p)
    p.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("MEANFP_YML", "TDF_H5", "LABEL"),
        required=False,
        help="(deprecated) Provide MEANFP_YML TDF_H5 LABEL; prefer --meanfp/--tdf/--legend.",
    )
    
    p.add_argument(
        "--config",
        type=int,
        default=1,
        help=(
            "Configuration index [default: 1]. "
            "Use 0 together with --check-trans-coef to validate all temperature points."
        ),
    )
    p.add_argument("--dir", choices=['xx', 'yy', 'zz', 'avg'], default='xx', help="Transport direction [default: xx]; use 'avg' for (xx+yy+zz)/3")
    p.add_argument("--qty", choices=["kappa", "sigma"], default="kappa", help="Quantity to plot")
    p.add_argument(
        "--tet",
        default=None,
        help=(
            "Path to *_tet.h5 for k-point mapping (kpt2ir). If omitted, tries to infer from the tdf.h5 name by replacing 'tdf' with 'tet'."
        ),
    )

    p.add_argument(
        "--trans-coef",
        default=None,
        help=(
            "Path to reference *.trans_coef (used by --check-trans-coef and *_calib=trans_coef). "
            "If omitted, tries to infer next to each tdf.h5."
        ),
    )
    p.add_argument(
        "--trans-ita",
        default=None,
        help=(
            "Path to reference *_trans-ita.yml (optional; used when *_calib=trans_ita). "
            "If omitted, tries to infer next to each tdf.h5."
        ),
    )

    p.add_argument("--style", choices=["prb", "default"], default="prb", help="Plot style preset.")
    p.add_argument(
        "--spin-deg",
        type=float,
        default=2.0,
        help="Spin degeneracy factor S in transport formulas [default: 2]",
    )

    p.add_argument(
        "--sigma-calib",
        choices=["trans_coef", "trans_ita", "none"],
        default="none",
        help=(
            "How to calibrate absolute sigma to match Perturbo's trans_coef output. "
            "'trans_coef' scales the final cumulative curve to the nearest-T entry in *.trans_coef. "
            "'trans_ita' scales using *_trans-ita.yml (trans-ita output). "
            "'none' disables this scaling (default)."
        ),
    )
    p.add_argument(
        "--kappa-calib",
        choices=["trans_coef", "trans_ita", "none"],
        default="none",
        help=(
            "How to calibrate absolute kappa to match Perturbo's trans_coef output. "
            "'trans_coef' scales the final cumulative curve to the nearest-T entry in *.trans_coef. "
            "'trans_ita' scales using *_trans-ita.yml (trans-ita output). "
            "'none' disables this scaling (default)."
        ),
    )

    # Plot controls (aligned to shengbte_plot scripts)
    p.add_argument(
        "--show-legend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the curve legend box.",
    )
    p.add_argument(
        "--legend-format",
        choices=["chem", "raw"],
        default="raw",
        help="Render legend text with subscripts (chem) or raw text (raw). Default: raw.",
    )
    p.add_argument(
        "--legend-fontsize",
        type=float,
        default=None,
        help="Font size for legend text. If omitted, uses matplotlib default.",
    )
    p.add_argument(
        "--legend-loc",
        default=["upper", "left"],
        nargs="+",
        help="Legend location (matplotlib legend loc). Example: --legend-loc upper left. Default: upper left.",
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
        help="If set, draw the legend with a white semi-transparent frame (0..1).",
    )

    p.add_argument("--xlog", action=argparse.BooleanOptionalAction, default=True, help="Use log scale on x axis.")
    p.add_argument("--ylog", action="store_true", help="Use log scale on y-axis")
    p.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"), help="Set x limits.")
    p.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"), help="Set y limits.")

    p.add_argument(
        "--gb-mfp",
        type=float,
        default=None,
        metavar="L_NM",
        help=(
            "Draw a vertical dashed line at this MFP (nm) to represent e.g. grain-boundary length. "
            "When plotting a single dataset, also annotate the cumulative percentage on each side."
        ),
    )

    p.add_argument(
        "--gb-text-y",
        type=float,
        nargs="+",
        default=[0.92],
        metavar="YFRAC",
        help="Y position for GB percentage text in axes fraction (0..1). Default: 0.92.",
    )
    p.add_argument(
        "--gb-text-xpad",
        type=float,
        nargs="+",
        default=[1.15],
        metavar="FACTOR",
        help=(
            "Horizontal padding factor for GB percentage text from the dashed line. "
            "Right text uses x*FACTOR, left uses x/FACTOR. Default: 1.15."
        ),
    )
    p.add_argument(
        "--gb-text-color",
        nargs="+",
        default=["line"],
        metavar="COLOR",
        help=(
            "GB percentage text color. Use 'line' to match the curve color (default), or any matplotlib color string."
        ),
    )

    p.add_argument(
        "--gb-text-outline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw a white outline around GB percentage text for readability (default: on).",
    )
    p.add_argument(
        "--gb-text-outline-width",
        type=float,
        default=3.0,
        help="Outline width (points) for GB percentage text. Default: 3.0.",
    )
    p.add_argument(
        "--gb-text-outline-color",
        default="white",
        help="Outline color for GB percentage text. Default: white.",
    )
    p.add_argument(
        "--gb-xlabel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate the GB MFP value near the x-axis.",
    )
    p.add_argument(
        "--gb-xlabel-format",
        default="{x:g}",
        help="Format for GB x-axis value annotation. Supports '{x}'. Default: '{x:g}'.",
    )
    p.add_argument(
        "--gb-xlabel-dy",
        type=float,
        default=-14.0,
        metavar="POINTS",
        help="Vertical offset (points) for GB x-axis value annotation (negative goes below axis). Default: -14.",
    )

    p.add_argument("--xlabel", default="Mean Free Path (nm)", help="x-axis label")
    p.add_argument("--ylabel", default=None, help="y-axis label (override default)")

    p.add_argument(
        "--figsize",
        default=None,
        help='Figure size "width,height" in inches (e.g. "7.5,5").',
    )
    p.add_argument(
        "--fontsize",
        type=float,
        default=None,
        help="Global default font size (rcParams).",
    )
    p.add_argument(
        "--label-fontsize",
        type=float,
        default=None,
        help="Font size for axis labels and tick labels.",
    )
    p.add_argument(
        "--bold-fonts",
        action="store_true",
        help="Force all text in the figure to bold.",
    )
    p.add_argument("--grid", action="store_true", help="Show grid")

    p.add_argument("--lw", type=float, default=2.5, help="Line width.")
    p.add_argument(
        "--color",
        action="append",
        default=None,
        help="Line color(s). Can be specified multiple times to match datasets.",
    )
    p.add_argument(
        "--ls",
        default=None,
        nargs="+",
        help="Line style(s). Provide 1 (broadcast) or N datasets values. Examples: '-', ':', '-.', 'dashed'.",
    )

    p.add_argument("--out", default=None, help="Output image path")
    p.add_argument("--show", action="store_true", help="Show interactively")
    p.add_argument(
        "--check-trans-coef",
        action="store_true",
        help=(
            "Check all available temperature points against *.trans_coef and print a table. "
            "Intended for validation; no plot is produced. Requires --config 0."
        ),
    )
    return p


def _parse_xy(s: str | None) -> tuple[float, float] | None:
    if not s:
        return None
    a, b = str(s).split(",", 1)
    return float(a), float(b)


def _parse_figsize(s: str | None) -> tuple[float, float] | None:
    if not s:
        return None
    a, b = str(s).split(",", 1)
    w = float(a)
    h = float(b)
    if w <= 0 or h <= 0:
        raise SystemExit(f"Invalid --figsize {s!r}: width and height must be > 0")
    return w, h


def _format_chem_label(label: str, mode: str) -> str:
    if not label:
        return label
    if mode == "raw":
        return label
    if "$" in label:
        return label
    return re.sub(r"(?<=[A-Za-z\)])(\d+)", r"$_{\1}$", label)


def _normalize_linestyle_token(s: str) -> str:
    t = str(s).strip().lower()
    aliases = {
        "solid": "-",
        "dash": "--",
        "dashed": "--",
        "dot": ":",
        "dotted": ":",
        "dashdot": "-.",
        "dash-dot": "-.",
    }
    return aliases.get(t, str(s))


def _broadcast_list(xs: list[str], n: int, name: str) -> list[str]:
    if n == 0:
        return []
    if len(xs) == n:
        return list(xs)
    if len(xs) == 1:
        return [str(xs[0])] * n
    raise SystemExit(f"{name} expects 1 value or {n} values, but got {len(xs)}")


def _apply_legend_frame_local(leg, *, alpha: float) -> None:
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


def _set_figure_text_weight(fig, weight: str) -> None:
    for t in fig.findobj(Text):
        try:
            t.set_fontweight(weight)
        except Exception:
            pass


def _infer_tet_path(tdf_path: str) -> Path:
    p = Path(tdf_path)
    name = p.name
    candidates = []
    if name.endswith("_tdf.h5"):
        candidates.append(p.with_name(name[:-7] + "_tet.h5"))
    if "tdf" in name:
        candidates.append(p.with_name(name.replace("tdf", "tet")))
    candidates.append(p.with_name(p.stem + "_tet.h5"))
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(
        "Missing tet file for k-point mapping. Provide --tet PATH (e.g. zr2sc_tet.h5), "
        f"or place it next to {tdf_path!r} with a compatible name. Tried: {', '.join(str(x) for x in candidates)}"
    )


def _infer_tet_kpt_path(*, meanfp_path: str, tdf_path: str, tet_h5_path: Path | None) -> Path:
    """Infer *_tet.kpt path for irreducible k-point weights (used by --dir avg)."""

    candidates: list[Path] = []

    if tet_h5_path is not None:
        candidates.append(tet_h5_path.with_suffix(".kpt"))

    p_yml = Path(meanfp_path)
    if p_yml.name.endswith("_meanfp.yml"):
        candidates.append(p_yml.with_name(p_yml.name.replace("_meanfp.yml", "_tet.kpt")))
    # common prefix case: zr2sc_meanfp.yml -> zr2sc_tet.kpt
    candidates.append(p_yml.with_name(p_yml.stem.replace("meanfp", "tet") + ".kpt"))

    p_tdf = Path(tdf_path)
    if p_tdf.name.endswith("_tdf.h5"):
        candidates.append(p_tdf.with_name(p_tdf.name[:-7] + "_tet.kpt"))
    if "tdf" in p_tdf.name:
        candidates.append(p_tdf.with_name(p_tdf.name.replace("tdf", "tet").replace(".h5", ".kpt")))

    # de-dup while preserving order
    seen = set()
    uniq: list[Path] = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)

    for c in uniq:
        if c.exists():
            return c

    raise SystemExit(
        "Missing tet k-point list (*.tet.kpt). Provide --tet-kpt PATH, or place it next to meanfp/tdf with a compatible name. "
        f"Tried: {', '.join(str(x) for x in uniq)}"
    )


def _read_tet_kpt(tet_kpt_path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], int]:
    """Read Perturbo *.tet.kpt.

    Returns:
        kpts_frac: (Nirr,3) crystal fractional coords (as in file)
        weights: (Nirr,) k-point weights
        kgrid_dim: (Nx,Ny,Nz)
        nk_tot: total number of k-points represented by weights (header '#.tot.k')
    """

    lines = tet_kpt_path.read_text(errors="ignore").splitlines()
    if not lines:
        raise SystemExit(f"Empty tet.kpt: {tet_kpt_path}")

    head = lines[0].split()
    if len(head) < 9:
        raise SystemExit(f"Unexpected tet.kpt header format: {lines[0]!r}")

    n_irr = int(head[0])
    Nx, Ny, Nz = int(head[2]), int(head[3]), int(head[4])
    nk_tot = int(head[-1])

    body = lines[1 : 1 + n_irr]
    if len(body) != n_irr:
        raise SystemExit(f"tet.kpt line count mismatch: header says {n_irr}, file has {len(body)} k-points")

    kpts = np.zeros((n_irr, 3), dtype=float)
    w = np.zeros((n_irr,), dtype=float)
    for i, line in enumerate(body):
        cols = line.split()
        if len(cols) < 4:
            raise SystemExit(f"Bad tet.kpt line {i+2}: {line!r}")
        kpts[i, 0] = float(cols[0])
        kpts[i, 1] = float(cols[1])
        kpts[i, 2] = float(cols[2])
        w[i] = float(cols[3])

    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum <= 0:
        raise SystemExit(f"Invalid tet.kpt weights: sum={wsum}")
    # Most Perturbo tet.kpt weights are normalized to sum=1. If not, normalize for safety.
    if abs(wsum - 1.0) > 1e-6:
        w = w / wsum

    return kpts, w, (Nx, Ny, Nz), nk_tot


# NOTE: The tet.kpt-based average path is deprecated/removed.
# We keep the helper functions above only for backward reference; they are no longer used by the CLI.


def _weights_from_tet_h5(*, tet_path: Path, tet_kpt_path: Path, h5py_module) -> tuple[np.ndarray, tuple[int, int, int], int]:
    """Compute irreducible k-point weights from tet.h5 (via kpt2ir multiplicities).

    Note:
        The last column in *_tet.kpt is not a reliable integration weight in general.
        In this dataset it is constant (1/Nirr), i.e. it carries no multiplicity info.

    Returns:
        wk: (Nirr,) weights in the ordering of tet.kpt.
        kgrid_dim: (Nx,Ny,Nz) from tet.h5.
        nk_tot: number of k-points in tet.h5 (len(kpt2ir)).

    Weight convention:
        wk = multiplicity / Nk_full, so sum(wk) = nk_tot / Nk_full.
    """

    kpts_frac, _w_dummy, _kgrid_kpt, _nk_tot_kpt = _read_tet_kpt(tet_kpt_path)

    with h5py_module.File(tet_path, "r") as f:
        Nx, Ny, Nz = [int(x) for x in f["kgrid_dim"][:]]
        kpt2ir_1based = np.asarray(f["kpt2ir"][:], dtype=np.int64)
        kpts_irr_full = np.asarray(f["kpts_irr"][:], dtype=np.int64)

    Nk_full = int(Nx * Ny * Nz)
    nk_tot = int(kpt2ir_1based.shape[0])
    n_irr = int(kpts_irr_full.shape[0])

    if kpts_frac.shape[0] != n_irr:
        raise SystemExit(
            f"Irreducible k-point count mismatch: tet.kpt has {kpts_frac.shape[0]}, tet.h5 has {n_irr}"
        )

    # multiplicity of each irreducible k (in tet.h5's irr indexing)
    if kpt2ir_1based.min() < 1 or kpt2ir_1based.max() > n_irr:
        raise SystemExit("tet.h5 kpt2ir has out-of-range irreducible indices")
    mult = np.bincount(kpt2ir_1based - 1, minlength=n_irr).astype(np.int64)

    def unflatten(full_idx: np.ndarray) -> np.ndarray:
        full_idx = np.asarray(full_idx, dtype=np.int64)
        iz = np.mod(full_idx, Nz)
        tmp = (full_idx - iz) // Nz
        iy = np.mod(tmp, Ny)
        ix = (tmp - iy) // Ny
        return np.stack([ix, iy, iz], axis=1)

    ijk_irr = unflatten(kpts_irr_full)
    key_irr = (
        ijk_irr[:, 0].astype(np.int64) * (Ny * Nz)
        + ijk_irr[:, 1].astype(np.int64) * Nz
        + ijk_irr[:, 2].astype(np.int64)
    )
    if len(np.unique(key_irr)) != key_irr.size:
        raise SystemExit("tet.h5 kpts_irr are not unique; cannot build weight map")
    inv = {int(k): int(i) for i, k in enumerate(key_irr)}

    # tet.kpt coords -> ijk on the same uniform mesh
    wrap = ((kpts_frac + 0.5) % 1.0) - 0.5
    ix = np.mod(np.rint(wrap[:, 0] * Nx).astype(np.int64), Nx)
    iy = np.mod(np.rint(wrap[:, 1] * Ny).astype(np.int64), Ny)
    iz = np.mod(np.rint(wrap[:, 2] * Nz).astype(np.int64), Nz)
    key_kpt = ix * (Ny * Nz) + iy * Nz + iz

    irr_pos = np.full((n_irr,), -1, dtype=np.int64)
    for i, k in enumerate(key_kpt):
        j = inv.get(int(k), -1)
        if j < 0:
            raise SystemExit(f"Failed to map tet.kpt k-point {i} onto tet.h5 kpts_irr")
        irr_pos[i] = j

    wk = (mult[irr_pos].astype(float)) / float(Nk_full)
    return wk, (Nx, Ny, Nz), nk_tot


def _df_fermi_derivative_ev(dE_eV: np.ndarray, kBT_eV: float) -> np.ndarray:
    x = dE_eV / kBT_eV
    valid = np.abs(x) < 20
    out = np.zeros_like(dE_eV, dtype=float)
    out[valid] = 1.0 / (kBT_eV * 4.0 * np.cosh(x[valid] / 2.0) ** 2)
    return out


def _volume_m3_from_meanfp(meanfp_data: dict) -> float:
    """Compute cell volume (m^3) from meanfp.yml.

    Prefer the explicit `basic data: volume` (usually in bohr^3). Fallback to
    lattice vectors (`lattice vectors` + `alat`).
    """

    BOHR_M = 0.529177210903e-10

    try:
        basic = meanfp_data["basic data"]
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"meanfp.yml missing 'basic data': {exc}") from exc

    if "volume" in basic:
        vol = float(basic["volume"])
        units = str(basic.get("volume units", "")).strip().lower()
        if units and ("bohr" not in units):
            raise SystemExit(f"Unsupported volume units in meanfp.yml: {units!r}")
        vol_m3 = vol * (BOHR_M**3)
        if not np.isfinite(vol_m3) or vol_m3 <= 0:
            raise SystemExit(f"Invalid volume from meanfp.yml: {vol} {units}")
        return float(vol_m3)

    if "lattice vectors" in basic and "alat" in basic:
        lv = np.asarray(basic["lattice vectors"], dtype=float)
        if lv.shape != (3, 3):
            raise SystemExit(f"Unexpected lattice vectors shape: {lv.shape}, expected (3,3)")
        alat = float(basic["alat"])  # in bohr
        alat_units = str(basic.get("alat units", "")).strip().lower()
        if alat_units and ("bohr" not in alat_units):
            raise SystemExit(f"Unsupported alat units in meanfp.yml: {alat_units!r}")
        a_m = lv * (alat * BOHR_M)
        vol_m3 = float(abs(np.linalg.det(a_m)))
        if not np.isfinite(vol_m3) or vol_m3 <= 0:
            raise SystemExit(f"Invalid cell volume computed from lattice vectors: {vol_m3}")
        return vol_m3

    raise SystemExit("Cannot determine cell volume from meanfp.yml (missing 'volume' and 'lattice vectors/alat')")


def _meanfp_reorder_to_tet_kpt_list(meanfp_data: dict, tet_kpt_path: Path) -> np.ndarray:
    """Map meanfp irreducible k-point ordering onto tet.kpt ordering.

    In many setups, meanfp uses fklist=*_tet.kpt, so the ordering already matches.
    This function checks that; if not, it builds a robust mapping via k-grid indices.

    Returns:
        pos: array of length Nirr_meanfp, where pos[i] is the index in tet.kpt.
    """

    kpts_tet, _w, (Nx, Ny, Nz), _nk_tot = _read_tet_kpt(tet_kpt_path)

    try:
        k_in = np.asarray(meanfp_data["meanfp"]["k-point coordinates"], dtype=float)
        B = np.asarray(meanfp_data["basic data"]["reciprocal lattice vectors"], dtype=float)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"meanfp.yml missing k-point coordinates / reciprocal lattice vectors: {exc}") from exc

    if k_in.shape[1] != 3 or B.shape != (3, 3):
        raise SystemExit(f"Unexpected shapes: k_in={k_in.shape}, B={B.shape}")

    Binv = np.linalg.inv(B)
    k_frac = k_in @ Binv

    def to_ijk(frac: np.ndarray) -> np.ndarray:
        wrap = ((frac + 0.5) % 1.0) - 0.5
        ix = np.mod(np.rint(wrap[:, 0] * Nx).astype(np.int64), Nx)
        iy = np.mod(np.rint(wrap[:, 1] * Ny).astype(np.int64), Ny)
        iz = np.mod(np.rint(wrap[:, 2] * Nz).astype(np.int64), Nz)
        return np.stack([ix, iy, iz], axis=1)

    ijk_meanfp = to_ijk(k_frac)
    ijk_tet = to_ijk(kpts_tet)

    if ijk_meanfp.shape[0] != ijk_tet.shape[0]:
        raise SystemExit(
            f"Irreducible k-point count mismatch: meanfp has {ijk_meanfp.shape[0]}, tet.kpt has {ijk_tet.shape[0]}"
        )

    # Fast path: already identical ordering.
    if np.array_equal(ijk_meanfp, ijk_tet):
        return np.arange(ijk_tet.shape[0], dtype=np.int64)

    # Build mapping from (ix,iy,iz) -> tet position.
    key = (ijk_tet[:, 0].astype(np.int64) * (Ny * Nz) + ijk_tet[:, 1].astype(np.int64) * Nz + ijk_tet[:, 2].astype(np.int64))
    if len(np.unique(key)) != key.size:
        raise SystemExit("tet.kpt grid indices are not unique; cannot build a one-to-one mapping")
    inv = {int(k): int(i) for i, k in enumerate(key)}

    key_m = (ijk_meanfp[:, 0].astype(np.int64) * (Ny * Nz) + ijk_meanfp[:, 1].astype(np.int64) * Nz + ijk_meanfp[:, 2].astype(np.int64))
    pos = np.full(key_m.shape[0], -1, dtype=np.int64)
    for i, k in enumerate(key_m):
        j = inv.get(int(k), -1)
        if j < 0:
            raise SystemExit(f"Failed to map meanfp k-point {i} to tet.kpt")
        pos[i] = j

    if len(np.unique(pos)) != pos.size:
        raise SystemExit("meanfp->tet.kpt mapping is not one-to-one")

    return pos


def _infer_trans_coef_path(tdf_path: str) -> Path | None:
    """Try to locate the matching *.trans_coef file next to tdf.h5."""

    p = Path(tdf_path)
    candidates: list[Path] = []
    if p.name.endswith("_tdf.h5"):
        candidates.append(p.with_name(p.name[:-7] + ".trans_coef"))
    candidates.append(p.with_suffix(".trans_coef"))
    for c in candidates:
        if c.exists():
            return c
    return None


def _infer_trans_ita_path(tdf_path: str) -> Path | None:
    """Try to locate the matching *_trans-ita.yml file next to tdf.h5."""

    p = Path(tdf_path)
    candidates: list[Path] = []
    if p.name.endswith("_tdf.h5"):
        candidates.append(p.with_name(p.name[:-7] + "_trans-ita.yml"))
    candidates.append(p.with_name(p.stem.replace("_tdf", "") + "_trans-ita.yml"))
    # fallbacks
    candidates.append(p.with_name(p.stem + "_trans-ita.yml"))
    for c in candidates:
        if c.exists():
            return c
    return None


def _parse_trans_ita_sigma_kappa_diag(
    trans_ita_path: Path,
) -> tuple[dict[float, dict[str, float]], dict[float, dict[str, float]]]:
    """Parse conductivity/thermal conductivity diagonal components from perturbo trans-ita YAML.

    Returns:
        (sigma_tbl, kappa_tbl) where mapping is T(K) -> {'xx':..,'yy':..,'zz':..}
    """

    try:
        import yaml  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Missing dependency PyYAML (needed for *_calib=trans_ita). Install with: pip install pyyaml"
        ) from exc

    doc = yaml.safe_load(trans_ita_path.read_text(errors="ignore"))
    if not isinstance(doc, dict):
        raise SystemExit(f"Failed to parse YAML: {trans_ita_path}")

    trans = doc.get("trans", {})
    if not isinstance(trans, dict):
        raise SystemExit(f"Unexpected YAML structure (missing 'trans'): {trans_ita_path}")

    cfg_root = trans.get("configuration index", {})
    if not isinstance(cfg_root, dict):
        raise SystemExit(f"Unexpected YAML structure (missing 'trans/configuration index'): {trans_ita_path}")

    sigma: dict[float, dict[str, float]] = {}
    kappa: dict[float, dict[str, float]] = {}

    for _cfg_id, cfg in cfg_root.items():
        if not isinstance(cfg, dict):
            continue
        try:
            T = float(cfg["temperature"])
        except Exception:
            continue

        def _read_diag(block: dict) -> dict[str, float] | None:
            try:
                comp = block["components"]
                if not isinstance(comp, dict):
                    return None
                return {
                    "xx": float(comp["xx"]),
                    "yy": float(comp["yy"]),
                    "zz": float(comp["zz"]),
                }
            except Exception:
                return None

        cond = cfg.get("conductivity")
        if isinstance(cond, dict):
            d = _read_diag(cond)
            if d is not None:
                sigma[T] = d

        th = cfg.get("thermal conductivity")
        if isinstance(th, dict):
            d = _read_diag(th)
            if d is not None:
                kappa[T] = d

    return sigma, kappa


def _get_ref_total(
    *,
    method: str,
    qty: str,
    direction: str,
    T_K: float,
    tdf_path: str,
    trans_coef_path: Path | None,
    trans_ita_path: Path | None,
) -> float | None:
    """Get reference total for calibration.

    Args:
        method: 'trans_coef' | 'trans_ita'
        qty: 'sigma' | 'kappa'
        direction: 'xx'|'yy'|'zz'|'avg'
    """

    if method == "none":
        return None
    if qty not in {"sigma", "kappa"}:
        raise ValueError(f"Unexpected qty={qty!r}")
    if direction not in {"xx", "yy", "zz", "avg"}:
        raise ValueError(f"Unexpected direction={direction!r}")

    if method == "trans_coef":
        path = trans_coef_path or _infer_trans_coef_path(tdf_path)
        if path is None:
            return None
        tbl = _parse_trans_coef_sigma_diag(path) if qty == "sigma" else _parse_trans_coef_kappa_diag(path)
    elif method == "trans_ita":
        path = trans_ita_path or _infer_trans_ita_path(tdf_path)
        if path is None:
            return None
        sigma_tbl, kappa_tbl = _parse_trans_ita_sigma_kappa_diag(path)
        tbl = sigma_tbl if qty == "sigma" else kappa_tbl
    else:
        raise ValueError(f"Unknown calibration method: {method!r}")

    if not tbl:
        return None
    T_ref = min(tbl.keys(), key=lambda x: abs(x - float(T_K)))

    row = tbl[T_ref]
    if direction == "avg":
        return (float(row["xx"]) + float(row["yy"]) + float(row["zz"])) / 3.0
    return float(row[direction])


def _parse_trans_coef_kappa_diag(trans_coef_path: Path) -> dict[float, dict[str, float]]:
    """Parse electronic thermal conductivity table from Perturbo trans_coef.

    Returns mapping: T(K) -> {'xx':kxx, 'yy':kyy, 'zz':kzz}
    """

    lines = trans_coef_path.read_text(errors="ignore").splitlines()
    in_kappa = False
    out: dict[float, dict[str, float]] = {}
    for line in lines:
        if "kappa_xx" in line:
            in_kappa = True
            continue
        if in_kappa and "Seebeck" in line:
            in_kappa = False
        if not in_kappa:
            continue
        m = re.match(r"^\s*(\d+\.\d+)\s+", line)
        if not m:
            continue
        T = float(m.group(1))
        cols = line.split()
        # columns: T Ef n kxx kxy kyy kxz kyz kzz
        kxx = float(cols[3].replace("E", "e"))
        kyy = float(cols[5].replace("E", "e"))
        kzz = float(cols[8].replace("E", "e"))
        out[T] = {"xx": kxx, "yy": kyy, "zz": kzz}
    return out


def _parse_trans_coef_sigma_diag(trans_coef_path: Path) -> dict[float, dict[str, float]]:
    """Parse electrical conductivity table from Perturbo trans_coef.

    Returns mapping: T(K) -> {'xx':sxx, 'yy':syy, 'zz':szz} in S/m.
    """

    lines = trans_coef_path.read_text(errors="ignore").splitlines()
    in_sigma = False
    out: dict[float, dict[str, float]] = {}
    for line in lines:
        if "sigma_xx" in line:
            in_sigma = True
            continue
        if in_sigma and "Mobility" in line:
            break
        if not in_sigma:
            continue
        s = line.strip()
        if (not s) or s.startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 9:
            continue
        try:
            T = float(cols[0])
            sxx = float(cols[3].replace("E", "e"))
            syy = float(cols[5].replace("E", "e"))
            szz = float(cols[8].replace("E", "e"))
        except Exception:
            continue
        out[T] = {"xx": sxx, "yy": syy, "zz": szz}
    return out


def _expand_irr_to_all(arr_irr: np.ndarray, kpt2ir_1based: np.ndarray) -> np.ndarray:
    idx = np.asarray(kpt2ir_1based, dtype=np.int64) - 1
    if idx.min() < 0:
        raise ValueError("kpt2ir must be 1-based (min >= 1)")
    if idx.max() >= arr_irr.shape[0]:
        raise ValueError(
            f"kpt2ir refers to irr index {idx.max()+1}, but arr_irr length is {arr_irr.shape[0]}"
        )
    return np.asarray(arr_irr, dtype=float)[idx]


def _meanfp_reorder_to_tet_irr(meanfp_data: dict, tet_path: Path, h5py_module) -> np.ndarray:
    """Return tet-irr position for each meanfp irreducible k-point.

    meanfp.yml stores k-point coordinates in the *reciprocal Cartesian basis* used by the
    reciprocal lattice vectors matrix B; tet.h5 uses a uniform FFT k-mesh and stores
    irreducible k-points as flattened full-grid indices (kpts_irr).

    We convert meanfp k to fractional crystal coords via k_frac = k_in @ inv(B), quantize
    to the tet k-grid, compute the flattened full index, and look it up in kpts_irr.
    """

    try:
        k_in = np.asarray(meanfp_data["meanfp"]["k-point coordinates"], dtype=float)
        B = np.asarray(meanfp_data["basic data"]["reciprocal lattice vectors"], dtype=float)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"meanfp.yml missing k-point coordinates / reciprocal lattice vectors: {exc}") from exc

    if B.shape != (3, 3):
        raise SystemExit(f"Unexpected reciprocal lattice vectors shape: {B.shape}, expected (3,3)")

    try:
        Binv = np.linalg.inv(B)
    except np.linalg.LinAlgError as exc:
        raise SystemExit("Reciprocal lattice vectors matrix is singular; cannot invert.") from exc

    k_frac = k_in @ Binv

    with h5py_module.File(tet_path, "r") as f_tet:
        Nx, Ny, Nz = [int(x) for x in f_tet["kgrid_dim"][:]]
        kpts_irr_full = np.asarray(f_tet["kpts_irr"][:], dtype=np.int64)

    Nk_full = Nx * Ny * Nz
    inv_full_to_irrpos = np.full(Nk_full, -1, dtype=np.int32)
    inv_full_to_irrpos[kpts_irr_full] = np.arange(len(kpts_irr_full), dtype=np.int32)

    # Wrap to [-0.5, 0.5) then quantize to the uniform mesh.
    k_wrap = ((k_frac + 0.5) % 1.0) - 0.5
    ix = np.mod(np.rint(k_wrap[:, 0] * Nx).astype(np.int64), Nx)
    iy = np.mod(np.rint(k_wrap[:, 1] * Ny).astype(np.int64), Ny)
    iz = np.mod(np.rint(k_wrap[:, 2] * Nz).astype(np.int64), Nz)

    # Full-grid flattened index ordering in tet.h5: iz fastest, then iy, then ix.
    full_idx = iz + Nz * (iy + Ny * ix)
    irr_pos = inv_full_to_irrpos[full_idx]

    if np.any(irr_pos < 0):
        bad = np.where(irr_pos < 0)[0][:5]
        raise SystemExit(
            "Failed to map some meanfp k-points onto tet kpts_irr. "
            f"Examples indices: {bad.tolist()}; full_idx: {full_idx[bad].tolist()}; "
            f"k_in: {k_in[bad].tolist()}"
        )

    if len(np.unique(irr_pos)) != len(irr_pos):
        raise SystemExit(
            "meanfp k-point mapping to tet irreducible positions is not one-to-one; "
            "this suggests inconsistent k-mesh or coordinate conventions."
        )

    return irr_pos.astype(np.int64)


def _df_fermi_derivative_ry(dE_ry: np.ndarray, kBT_ry: float) -> np.ndarray:
    x = dE_ry / kBT_ry
    valid = np.abs(x) < 20
    out = np.zeros_like(dE_ry, dtype=float)
    out[valid] = 1.0 / (kBT_ry * 4.0 * np.cosh(x[valid] / 2.0) ** 2)
    return out


def _df_fermi_derivative_ry_many(dE_ry: np.ndarray, kBT_ry: np.ndarray) -> np.ndarray:
    """Vectorized (-df/dE) for many temperatures.

    Args:
        dE_ry: (Ncfg, Ne)
        kBT_ry: (Ncfg,)

    Returns:
        df: (Ncfg, Ne)
    """

    kBT = np.asarray(kBT_ry, dtype=float)
    dE = np.asarray(dE_ry, dtype=float)
    if dE.ndim != 2 or kBT.ndim != 1 or dE.shape[0] != kBT.shape[0]:
        raise ValueError(f"Shape mismatch: dE_ry={dE.shape}, kBT_ry={kBT.shape}")
    if np.any(kBT <= 0) or (not np.all(np.isfinite(kBT))):
        raise ValueError("kBT_ry must be finite and > 0")

    x = dE / kBT[:, None]
    valid = np.abs(x) < 20
    denom = kBT[:, None] * 4.0 * (np.cosh(x / 2.0) ** 2)
    out = np.where(valid, 1.0 / denom, 0.0)
    return np.asarray(out, dtype=float)


def _weight_dos_many_sorted(e_sorted: np.ndarray, e_dos: float) -> np.ndarray:
    """Vectorized version of Perturbo's weight_dos (weight_dos.f90).

    Args:
        e_sorted: (Nt,4) corner energies sorted ascending along axis=1.
        e_dos: scalar energy at which to evaluate the delta-weight.

    Returns:
        w_sorted: (Nt,4) weights in the same sorted order as e_sorted.

    Notes:
        - This reproduces the piecewise formulas used in Perturbo.
        - The returned weights do NOT include the global tetra weight V_T/V_G.
    """

    e1 = e_sorted[:, 0]
    e2 = e_sorted[:, 1]
    e3 = e_sorted[:, 2]
    e4 = e_sorted[:, 3]

    w = np.zeros_like(e_sorted, dtype=float)

    # Region masks
    # Special degeneracy clause from Fortran:
    # (e_dos==e2 .and. e2==e3==e4 .and. e2>e1)
    atol = 1e-12
    special = (
        np.isclose(e_dos, e2, atol=atol, rtol=0.0)
        & np.isclose(e2, e3, atol=atol, rtol=0.0)
        & np.isclose(e3, e4, atol=atol, rtol=0.0)
        & (e2 > e1)
    )

    m1 = (e_dos >= e1) & ((e_dos < e2) | special)
    m2 = (e_dos >= e2) & (e_dos < e3) & (~special)
    m3 = (e_dos >= e3) & (e_dos < e4)

    # --- Region 1: e1 <= e < e2
    if np.any(m1):
        et1 = (e_dos - e1[m1])
        et2 = (e2[m1] - e1[m1])
        et3 = (e3[m1] - e1[m1])
        et4 = (e4[m1] - e1[m1])

        den = et2 * et3 * et4
        den = np.where(np.abs(den) > 0, den, 1e-300)
        factor = (et1 * et1) / den
        w2 = factor * et1 / np.where(np.abs(et2) > 0, et2, 1e-300)
        w3 = factor * et1 / np.where(np.abs(et3) > 0, et3, 1e-300)
        w4 = factor * et1 / np.where(np.abs(et4) > 0, et4, 1e-300)
        w1v = 3.0 * factor - w2 - w3 - w4

        w[m1, 0] = w1v
        w[m1, 1] = w2
        w[m1, 2] = w3
        w[m1, 3] = w4

    # --- Region 2: e2 <= e < e3
    if np.any(m2):
        et1 = (e_dos - e1[m2])
        et2 = (e_dos - e2[m2])
        et3 = (e3[m2] - e_dos)
        et4 = (e4[m2] - e_dos)

        e31 = (e3[m2] - e1[m2])
        e32 = (e3[m2] - e2[m2])
        e41 = (e4[m2] - e1[m2])
        e42 = (e4[m2] - e2[m2])

        e31s = np.where(np.abs(e31) > 0, e31, 1e-300)
        e32s = np.where(np.abs(e32) > 0, e32, 1e-300)
        e41s = np.where(np.abs(e41) > 0, e41, 1e-300)
        e42s = np.where(np.abs(e42) > 0, e42, 1e-300)

        dc1 = 0.5 * et1 / (e41s * e31s)
        c1 = 0.5 * et1 * dc1

        c2 = 0.25 / (e41s * e32s * e31s)
        dc2 = c2 * (et2 * et3 + et1 * et3 - et1 * et2)
        c2v = c2 * et1 * et2 * et3

        c3 = 0.25 / (e42s * e32s * e41s)
        dc3 = c3 * (2.0 * et2 * et4 - et2 * et2)
        c3v = c3 * (et2 * et2 * et4)

        w1v = dc1 + (dc1 + dc2) * et3 / e31s + (dc1 + dc2 + dc3) * et4 / e41s
        w1v = w1v - (c1 + c2v) / e31s - (c1 + c2v + c3v) / e41s

        w2v = dc1 + dc2 + dc3 + (dc2 + dc3) * et3 / e32s + dc3 * et4 / e42s
        w2v = w2v - (c2v + c3v) / e32s - c3v / e42s

        w3v = (dc1 + dc2) * et1 / e31s + (dc2 + dc3) * et2 / e32s
        w3v = w3v + (c1 + c2v) / e31s + (c2v + c3v) / e32s

        w4v = (dc1 + dc2 + dc3) * et1 / e41s + dc3 * et2 / e42s
        w4v = w4v + (c1 + c2v + c3v) / e41s + c3v / e42s

        w[m2, 0] = w1v
        w[m2, 1] = w2v
        w[m2, 2] = w3v
        w[m2, 3] = w4v

    # --- Region 3: e3 <= e < e4
    if np.any(m3):
        et1 = (e4[m3] - e1[m3])
        et2 = (e4[m3] - e2[m3])
        et3 = (e4[m3] - e3[m3])
        et4 = (e4[m3] - e_dos)

        den = et1 * et2 * et3
        den = np.where(np.abs(den) > 0, den, 1e-300)
        factor = (et4 * et4) / den
        w1v = factor * et4 / np.where(np.abs(et1) > 0, et1, 1e-300)
        w2v = factor * et4 / np.where(np.abs(et2) > 0, et2, 1e-300)
        w3v = factor * et4 / np.where(np.abs(et3) > 0, et3, 1e-300)
        w4v = 3.0 * factor - w1v - w2v - w3v

        w[m3, 0] = w1v
        w[m3, 1] = w2v
        w[m3, 2] = w3v
        w[m3, 3] = w4v

    return w


def _tetra_weights_W0_W1(
    *,
    e_corner: np.ndarray,
    energy_grid: np.ndarray,
    efermi_ry: float,
    kBT_ry: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute integrated tetra weights for a single band over the energy grid.

    We compute (per tetra vertex):

    - W0 = \\int dE\\, w_delta(E; e_corner) * (-df/dE)
    - W1 = \\int dE\\, w_delta(E; e_corner) * (-df/dE) * (E-\\mu)

    using the same discrete energy grid integration as Perturbo (sum * dE).

    Args:
        e_corner: (Nt,4) vertex energies (Ry) for each tetra.
        energy_grid: (Ne,) energy grid points (Ry), assumed uniform.
        efermi_ry: chemical potential (Ry).
        kBT_ry: kBT (Ry).

    Returns:
        W0, W1: both (Nt,4)
    """

    if energy_grid.ndim != 1 or energy_grid.size < 2:
        raise SystemExit("tdf.h5 energy_grid is missing or too short")
    if kBT_ry <= 0:
        raise SystemExit(f"Invalid temperature (kBT) from tdf.h5: {kBT_ry}. Expected > 0 (units: Ry).")

    de = float((energy_grid[-1] - energy_grid[0]) / (energy_grid.size - 1.0))
    if not np.isfinite(de) or de <= 0:
        raise SystemExit(f"Invalid energy_grid step: dE={de}")

    Nt = int(e_corner.shape[0])
    if e_corner.shape != (Nt, 4):
        raise SystemExit(f"Unexpected e_corner shape {e_corner.shape}; expected (Nt,4)")

    order = np.argsort(e_corner, axis=1)
    e_sorted = np.take_along_axis(e_corner, order, axis=1)
    row = np.arange(Nt, dtype=np.int64)

    W0 = np.zeros((Nt, 4), dtype=float)
    W1 = np.zeros((Nt, 4), dtype=float)

    dE_grid = energy_grid - float(efermi_ry)
    df_grid = _df_fermi_derivative_ry(dE_grid, float(kBT_ry))
    factor0_grid = df_grid * de
    factor1_grid = factor0_grid * dE_grid

    for e_dos, factor0, factor1 in zip(energy_grid, factor0_grid, factor1_grid, strict=True):
        if not np.isfinite(factor0) or factor0 == 0.0:
            continue

        w_sorted = _weight_dos_many_sorted(e_sorted, float(e_dos))  # (Nt,4)
        # scatter-add into original vertex order
        for j in range(4):
            jj = order[:, j]
            W0[row, jj] += w_sorted[:, j] * float(factor0)
            W1[row, jj] += w_sorted[:, j] * float(factor1)

    return W0, W1


def _tetra_weights_W0_W1_multi_cfg(
    *,
    e_corner: np.ndarray,
    energy_grid: np.ndarray,
    efermi_ry: np.ndarray,
    kBT_ry: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (W0,W1) for many configurations at once for a single band.

    Returns:
        W0, W1: (Ncfg, Nt, 4) in original vertex order.
    """

    efermi = np.asarray(efermi_ry, dtype=float)
    kBT = np.asarray(kBT_ry, dtype=float)
    if efermi.ndim != 1 or kBT.ndim != 1 or efermi.shape != kBT.shape:
        raise ValueError(f"efermi_ry and kBT_ry must be 1D and same shape: {efermi.shape} vs {kBT.shape}")
    ncfg = int(kBT.shape[0])

    if energy_grid.ndim != 1 or energy_grid.size < 2:
        raise SystemExit("tdf.h5 energy_grid is missing or too short")
    if np.any(kBT <= 0):
        raise SystemExit("Invalid temperature (kBT) from tdf.h5; expected > 0 (units: Ry).")

    de = float((energy_grid[-1] - energy_grid[0]) / (energy_grid.size - 1.0))
    if not np.isfinite(de) or de <= 0:
        raise SystemExit(f"Invalid energy_grid step: dE={de}")

    Nt = int(e_corner.shape[0])
    if e_corner.shape != (Nt, 4):
        raise SystemExit(f"Unexpected e_corner shape {e_corner.shape}; expected (Nt,4)")

    order = np.argsort(e_corner, axis=1)
    e_sorted = np.take_along_axis(e_corner, order, axis=1)

    dE_grid = energy_grid[None, :] - efermi[:, None]
    df_grid = _df_fermi_derivative_ry_many(dE_grid, kBT)
    factor0_grid = df_grid * de
    factor1_grid = factor0_grid * dE_grid

    # Accumulate in sorted-vertex order, then unsort at end.
    W0s = np.zeros((ncfg, Nt, 4), dtype=float)
    W1s = np.zeros((ncfg, Nt, 4), dtype=float)

    for ie, e_dos in enumerate(energy_grid):
        f0 = factor0_grid[:, ie]
        if not np.any(f0 != 0.0):
            continue
        w_sorted = _weight_dos_many_sorted(e_sorted, float(e_dos))  # (Nt,4)
        W0s += f0[:, None, None] * w_sorted[None, :, :]
        W1s += factor1_grid[:, ie][:, None, None] * w_sorted[None, :, :]

    # Unsort: per tetra, order gives mapping sorted_pos -> original_vertex_pos
    W0 = np.zeros_like(W0s)
    W1 = np.zeros_like(W1s)
    row = np.arange(Nt, dtype=np.int64)
    for j in range(4):
        jj = order[:, j]
        W0[:, row, jj] = W0s[:, :, j]
        W1[:, row, jj] = W1s[:, :, j]

    return W0, W1


def _configs_in_meanfp(meanfp_data: dict) -> list[int]:
    try:
        cfg_root = meanfp_data["meanfp"]["configuration index"]
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"meanfp.yml missing meanfp/configuration index: {exc}") from exc
    return sorted(int(k) for k in cfg_root.keys())


def _configs_in_tdf(tdf_h5_path: str, h5py_module) -> list[int]:
    with h5py_module.File(tdf_h5_path, "r") as f:
        cfg = []
        for k in f.keys():
            if not k.startswith("configuration_"):
                continue
            try:
                cfg.append(int(k.split("_", 1)[1]))
            except Exception:
                continue
    return sorted(set(cfg))


def _check_all_temperatures_against_trans_coef(
    *,
    meanfp_path: str,
    tdf_path: str,
    tet_path: Path,
    meanfp_data: dict,
    direction: str,
    spin_deg: float,
    h5py_module,
    trans_coef_path: Path | None,
) -> None:
    """Validate sigma/kappa totals against trans_coef for all configuration points."""

    trans_path = trans_coef_path or _infer_trans_coef_path(tdf_path)
    if trans_path is None:
        raise SystemExit(f"Cannot find *.trans_coef next to {tdf_path!r}; needed for --check-trans-coef")
    ref_sigma = _parse_trans_coef_sigma_diag(trans_path)
    ref_kappa = _parse_trans_coef_kappa_diag(trans_path)
    if not ref_sigma and not ref_kappa:
        raise SystemExit(f"Failed to parse sigma/kappa tables from {trans_path}")

    # Constants from Perturbo
    unitcharge = 1.60217733e-19
    bohr2ang = 0.52917721092
    ryd2ev = 13.605698066
    kelvin2ev = 8.6173427909e-05
    timeunit = 4.8377687e-17
    cond_unit = unitcharge * 1.0e10 / (bohr2ang * ryd2ev * timeunit)
    t_cond_unit = kelvin2ev * (unitcharge / timeunit) * 1.0e10 / bohr2ang

    # meanfp basic
    basic = meanfp_data.get("basic data", {})
    alat = float(basic["alat"])
    vol_bohr3 = float(basic["volume"])
    pref = float((spin_deg / vol_bohr3) * (alat**2))

    # Load tet
    with h5py_module.File(tet_path, "r") as f_tet:
        kpt2ir = np.asarray(f_tet["kpt2ir"][:], dtype=np.int64)
        kgrid_dim = np.asarray(f_tet["kgrid_dim"][:], dtype=np.int64)
        tetra = np.asarray(f_tet["tetra"][:], dtype=np.int64)
    Nk_full = int(np.prod(kgrid_dim))
    if Nk_full <= 0:
        raise SystemExit(f"Invalid kgrid_dim in tet.h5: {kgrid_dim.tolist()}")
    tweight = 1.0 / (6.0 * float(Nk_full))

    Nk_all = int(kpt2ir.shape[0])
    k_idx = tetra[:, :, 1] - 1
    if k_idx.min() < 0 or k_idx.max() >= Nk_all:
        raise SystemExit("tet.h5 tetra contains out-of-range k indices")

    meanfp_to_tet_irrpos = _meanfp_reorder_to_tet_irr(meanfp_data, tet_path, h5py_module)

    # Determine configs
    cfg_meanfp = set(_configs_in_meanfp(meanfp_data))
    cfg_tdf = set(_configs_in_tdf(tdf_path, h5py_module))
    cfgs = sorted(cfg_meanfp & cfg_tdf)
    if not cfgs:
        raise SystemExit("No overlapping configuration indices between meanfp.yml and tdf.h5")

    # Read common energy_grid and per-config (efermi,kBT) + mfd arrays
    cfg_meta: list[tuple[int, float, float]] = []
    mfd_conv_by_cfg: dict[int, np.ndarray] = {}
    mfd_rta_by_cfg: dict[int, np.ndarray] = {}
    with h5py_module.File(tdf_path, "r") as f_tdf:
        energy_grid = np.asarray(f_tdf["energy_grid"][:], dtype=float)
        for c in cfgs:
            gname = f"configuration_{c}"
            g = f_tdf[gname]
            efermi_ry = float(g["efermi"][()])
            kBT_ry = float(g["temperature"][()])
            mfd_conv_by_cfg[c] = np.asarray(g["mfd"][:], dtype=float)
            mfd_rta_by_cfg[c] = np.asarray(g["iterations"]["mfd_1"][:], dtype=float)
            cfg_meta.append((c, efermi_ry, kBT_ry))

    cfg_ids = [c for (c, _, _) in cfg_meta]
    efermi_arr = np.array([x for (_, x, _) in cfg_meta], dtype=float)
    kBT_arr = np.array([x for (_, _, x) in cfg_meta], dtype=float)

    # Temperature in K for matching trans_coef
    kB_eV_per_K = 8.617333262e-5
    RY_TO_EV = 13.605693122994
    T_K_arr = (kBT_arr * RY_TO_EV) / kB_eV_per_K

    dir = str(direction).lower()
    if dir == "avg":
        diag_axes = [0, 1, 2]
    elif dir == "xx":
        diag_axes = [0]
    elif dir == "yy":
        diag_axes = [1]
    elif dir == "zz":
        diag_axes = [2]
    else:
        raise SystemExit(f"Unsupported direction for check: {direction!r}")

    e_by_band = get_energy_by_band(meanfp_data)
    bands = sorted(e_by_band.keys())
    Nb = int(len(bands))

    # Sanity: mfd band count
    sample_cfg = cfg_ids[0]
    if mfd_conv_by_cfg[sample_cfg].shape[1] != Nb:
        raise SystemExit(
            f"Band count mismatch: meanfp has {Nb} bands, but mfd has {mfd_conv_by_cfg[sample_cfg].shape[1]}"
        )

    cond_tot = np.zeros((len(cfg_ids),), dtype=float)
    alpha_tot = np.zeros((len(cfg_ids),), dtype=float)
    cs_tot = np.zeros((len(cfg_ids),), dtype=float)
    kk_tot = np.zeros((len(cfg_ids),), dtype=float)

    # Precompute E_all_ry per band (independent of config)
    for b_idx, b in enumerate(bands):
        E_irr_eV = np.asarray(e_by_band[b], dtype=float)
        E_irr_ry = E_irr_eV / RY_TO_EV
        E_irr_tet = np.empty_like(E_irr_ry)
        E_irr_tet[meanfp_to_tet_irrpos] = E_irr_ry
        E_all_ry = _expand_irr_to_all(E_irr_tet, kpt2ir)
        e_corner = E_all_ry[k_idx]

        # Tetra weights for all configs for this band
        W0_all, W1_all = _tetra_weights_W0_W1_multi_cfg(
            e_corner=e_corner,
            energy_grid=energy_grid,
            efermi_ry=efermi_arr,
            kBT_ry=kBT_arr,
        )

        # Per-config contributions
        for icfg, cfg_id in enumerate(cfg_ids):
            mfd_conv = mfd_conv_by_cfg[cfg_id]
            mfd_rta = mfd_rta_by_cfg[cfg_id]

            # tau in t0 for this (cfg, band)
            tau_fs = np.asarray(get_config_band_series(meanfp_data, cfg_id, b, "relaxation time"), dtype=float)
            tau_irr_tet = np.empty_like(tau_fs)
            tau_irr_tet[meanfp_to_tet_irrpos] = tau_fs
            tau_all_t0 = _expand_irr_to_all((tau_irr_tet * 1e-15 / timeunit), kpt2ir)
            tau_all_t0 = np.where(tau_all_t0 > 0, tau_all_t0, np.nan)

            F_E = mfd_conv[:, b_idx, 0:3]
            F_T = mfd_conv[:, b_idx, 3:6]
            F_E_rta = mfd_rta[:, b_idx, 0:3]
            vnk = F_E_rta / tau_all_t0[:, None]

            # Average diagonal axes if requested
            nax = float(len(diag_axes))
            kBT = float(kBT_arr[icfg])
            for ax_i in diag_axes:
                gE = vnk[:, ax_i] * F_E[:, ax_i]
                gT = vnk[:, ax_i] * F_T[:, ax_i]
                gE_v = gE[k_idx]
                gT_v = gT[k_idx]

                W0 = W0_all[icfg]
                W1 = W1_all[icfg]
                cond_tot[icfg] += np.sum(W0 * gE_v) / nax
                alpha_tot[icfg] += np.sum((-W1) * gE_v) / nax
                cs_tot[icfg] += np.sum((-W0) * gT_v) / (kBT * nax)
                kk_tot[icfg] += np.sum(W1 * gT_v) / (kBT * nax)

    # Apply global prefactor tweight*(spin/vol)*alat^2
    scale = float(tweight * pref)
    cond_tot *= scale
    alpha_tot *= scale
    cs_tot *= scale
    kk_tot *= scale

    seebeck = cs_tot / np.where(np.abs(cond_tot) > 0, cond_tot, 1e-300)
    kappa_atomic = kk_tot - alpha_tot * seebeck
    sigma_SI = cond_tot * cond_unit
    kappa_SI = kappa_atomic * t_cond_unit

    # Print report
    print(f"\n[check] {Path(meanfp_path).name} vs {trans_path.name} (dir={dir})")
    print("cfg   T(K)   sigma_calc(S/m)  sigma_ref(S/m)  rel_err     kappa_calc(W/mK)  kappa_ref(W/mK)  rel_err")
    for icfg, cfg_id in enumerate(cfg_ids):
        T = float(T_K_arr[icfg])

        def pick_ref(tbl: dict[float, dict[str, float]]) -> tuple[float, float] | None:
            if not tbl:
                return None
            T_ref = min(tbl.keys(), key=lambda x: abs(float(x) - T))
            if abs(float(T_ref) - T) > 0.6:
                return None

            if dir == "avg":
                v = (float(tbl[T_ref]["xx"]) + float(tbl[T_ref]["yy"]) + float(tbl[T_ref]["zz"])) / 3.0
            else:
                v = float(tbl[T_ref][dir])
            return float(T_ref), float(v)

        s_ref = pick_ref(ref_sigma)
        k_ref = pick_ref(ref_kappa)

        s_calc = float(sigma_SI[icfg])
        k_calc = float(kappa_SI[icfg])
        if s_ref is not None:
            _, s_r = s_ref
            s_re = (s_calc - s_r) / s_r if s_r != 0 else np.nan
        else:
            s_r, s_re = np.nan, np.nan

        if k_ref is not None:
            _, k_r = k_ref
            k_re = (k_calc - k_r) / k_r if k_r != 0 else np.nan
        else:
            k_r, k_re = np.nan, np.nan

        print(
            f"{cfg_id:>3d}  {T:6.1f}  {s_calc:14.6e}  {s_r:14.6e}  {s_re:9.3e}   {k_calc:14.6f}  {k_r:14.6f}  {k_re:9.3e}"
        )


def _cumulative_from_mfd_tetra(
    *,
    meanfp_data: dict,
    tdf_h5_path: str,
    tet_path: Path,
    config: int,
    direction: str,
    spin_deg: float,
    h5py_module,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute cumulative sigma/kappa vs MFP using per-state mfd and tetra integration.

    This path does NOT use the energy-resolved tdf(E) dataset (which is not per-mode).
    Instead, it reproduces Perturbo's tetra weights (weight_dos/tetra_int) and applies
    them directly to per-state quantities built from mfd (E-field and T-field).

    Returns:
        mfp_sorted_nm, cum_sigma_SI, cum_kappa_SI, T_K
    """

    # Constants from Perturbo (pert_const.f90 / boltz_trans_output.f90)
    unitcharge = 1.60217733e-19
    bohr2ang = 0.52917721092
    ryd2ev = 13.605698066
    kelvin2ev = 8.6173427909e-05
    timeunit = 4.8377687e-17

    cond_unit = unitcharge * 1.0e10 / (bohr2ang * ryd2ev * timeunit)  # S/m per a.u.
    t_cond_unit = kelvin2ev * (unitcharge / timeunit) * 1.0e10 / bohr2ang  # W/m/K per a.u.

    # From meanfp.yml
    basic = meanfp_data.get("basic data", {})
    try:
        alat = float(basic["alat"])  # in bohr
        vol_bohr3 = float(basic["volume"])  # in bohr^3
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"meanfp.yml missing basic data (alat/volume): {exc}") from exc
    if not np.isfinite(alat) or alat <= 0:
        raise SystemExit(f"Invalid alat from meanfp.yml: {alat}")
    if not np.isfinite(vol_bohr3) or vol_bohr3 <= 0:
        raise SystemExit(f"Invalid volume from meanfp.yml: {vol_bohr3}")

    # Read tet mapping + tetra list
    with h5py_module.File(tet_path, "r") as f_tet:
        kpt2ir = np.asarray(f_tet["kpt2ir"][:], dtype=np.int64)  # 1..Nirr, length Nk_all
        kgrid_dim = np.asarray(f_tet["kgrid_dim"][:], dtype=np.int64)
        tetra = np.asarray(f_tet["tetra"][:], dtype=np.int64)

    Nk_full = int(np.prod(kgrid_dim))
    if Nk_full <= 0:
        raise SystemExit(f"Invalid kgrid_dim in tet.h5: {kgrid_dim.tolist()}")

    # In Perturbo: tweight = 1/tot_tet_fbz; for a uniform grid tot_tet_fbz = 6*Nk_full.
    tweight = 1.0 / (6.0 * float(Nk_full))

    Nk_all = int(kpt2ir.shape[0])
    if tetra.ndim != 3 or tetra.shape[1] != 4 or tetra.shape[2] != 2:
        raise SystemExit(f"Unexpected tet.h5 tetra shape: {tetra.shape}; expected (Nt,4,2)")
    k_idx = tetra[:, :, 1] - 1  # (Nt,4) 0-based k indices into the selected k list
    if k_idx.min() < 0 or k_idx.max() >= Nk_all:
        raise SystemExit("tet.h5 tetra contains out-of-range k indices")

    meanfp_to_tet_irrpos = _meanfp_reorder_to_tet_irr(meanfp_data, tet_path, h5py_module)

    # Read mfd (per-state) and energy grid (used only for tetra weights)
    with h5py_module.File(tdf_h5_path, "r") as f_tdf:
        energy_grid = np.asarray(f_tdf["energy_grid"][:], dtype=float)
        cfg_group = f"configuration_{config}"
        if cfg_group not in f_tdf:
            raise SystemExit(f"Missing group {cfg_group!r} in {tdf_h5_path}")
        g = f_tdf[cfg_group]
        efermi_ry = float(g["efermi"][()])
        kBT_ry = float(g["temperature"][()])
        mfd_conv = np.asarray(g["mfd"][:], dtype=float)
        if "iterations" not in g or "mfd_1" not in g["iterations"]:
            raise SystemExit(
                "tdf.h5 is missing configuration_X/iterations/mfd_1; "
                "cannot recover per-state velocity on the reducible grid without symmetry expansion."
            )
        mfd_rta = np.asarray(g["iterations"]["mfd_1"][:], dtype=float)

    if mfd_conv.shape != mfd_rta.shape:
        raise SystemExit(f"mfd shape mismatch: mfd={mfd_conv.shape}, mfd_1={mfd_rta.shape}")
    if mfd_conv.ndim != 3 or mfd_conv.shape[0] != Nk_all:
        raise SystemExit(f"Unexpected mfd shape: {mfd_conv.shape}; expected (Nk_all,Nb,Ncomp)")
    if mfd_conv.shape[2] < 6:
        raise SystemExit(
            f"mfd has only {mfd_conv.shape[2]} components; need >=6 (E-field 3 + T-field 3) to compute kappa consistently."
        )

    # Temperature in K
    kB_eV_per_K = 8.617333262e-5
    RY_TO_EV = 13.605693122994
    T_K = float((kBT_ry * RY_TO_EV) / kB_eV_per_K)

    # Prefactors: include spin degeneracy, volume, tweight, and the missing alat^2.
    pref = float(tweight * spin_deg / vol_bohr3 * (alat**2))

    # Direction handling (diagonal only)
    dir = str(direction).lower()
    if dir == "avg":
        diag_axes = [0, 1, 2]
    elif dir == "xx":
        diag_axes = [0]
    elif dir == "yy":
        diag_axes = [1]
    elif dir == "zz":
        diag_axes = [2]
    else:
        raise SystemExit(f"Unsupported direction for mfd tetra path: {direction!r}")

    e_by_band = get_energy_by_band(meanfp_data)
    bands = sorted(e_by_band.keys())
    if len(bands) != mfd_conv.shape[1]:
        raise SystemExit(
            f"Band count mismatch: meanfp has {len(bands)} bands, but mfd has {mfd_conv.shape[1]}"
        )

    # Collect per-mode (k,b) contributions
    mfp_flat: list[np.ndarray] = []
    cond_flat: list[np.ndarray] = []
    alpha_flat: list[np.ndarray] = []
    cs_flat: list[np.ndarray] = []
    kk_flat: list[np.ndarray] = []

    for b_idx, b in enumerate(bands):
        # Energies on reducible k list (Ry)
        E_irr_eV = np.asarray(e_by_band[b], dtype=float)
        E_irr_ry = E_irr_eV / RY_TO_EV
        if E_irr_ry.shape[0] != meanfp_to_tet_irrpos.shape[0]:
            raise SystemExit(
                f"meanfp k-point count mismatch: band {b} has {E_irr_ry.shape[0]} points, "
                f"but mapping has {meanfp_to_tet_irrpos.shape[0]}"
            )
        E_irr_tet = np.empty_like(E_irr_ry)
        E_irr_tet[meanfp_to_tet_irrpos] = E_irr_ry
        E_all_ry = _expand_irr_to_all(E_irr_tet, kpt2ir)

        # MFP (nm) and tau (fs) on reducible k list
        mfp_irr_nm = np.asarray(get_config_band_series(meanfp_data, config, b, "MFP"), dtype=float)
        tau_fs = np.asarray(get_config_band_series(meanfp_data, config, b, "relaxation time"), dtype=float)

        if mfp_irr_nm.shape[0] != E_irr_ry.shape[0] or tau_fs.shape[0] != E_irr_ry.shape[0]:
            raise SystemExit(
                f"meanfp length mismatch for band {b}: E={E_irr_ry.shape[0]} MFP={mfp_irr_nm.shape[0]} tau={tau_fs.shape[0]}"
            )

        mfp_irr_tet = np.empty_like(mfp_irr_nm)
        tau_irr_tet = np.empty_like(tau_fs)
        mfp_irr_tet[meanfp_to_tet_irrpos] = mfp_irr_nm
        tau_irr_tet[meanfp_to_tet_irrpos] = tau_fs
        mfp_all_nm = _expand_irr_to_all(mfp_irr_tet, kpt2ir)

        # tau in t0: tau_fs = tau_t0 * timeunit * 1e15  => tau_t0 = tau_fs / (timeunit*1e15)
        tau_all_t0 = _expand_irr_to_all((tau_irr_tet * 1e-15 / timeunit), kpt2ir)
        if np.any(tau_all_t0 <= 0):
            # Rare but can happen if some states are masked; keep them but avoid div-zero.
            tau_all_t0 = np.where(tau_all_t0 > 0, tau_all_t0, np.nan)

        # Per-state mfd vectors
        F_E = mfd_conv[:, b_idx, 0:3]
        F_T = mfd_conv[:, b_idx, 3:6]
        F_E_rta = mfd_rta[:, b_idx, 0:3]

        # Recover vnk on the reducible grid: mfd_1(E-field) = tau * vnk
        vnk = F_E_rta / tau_all_t0[:, None]

        # Precompute tetra-integrated weights for this band's energies
        e_corner = E_all_ry[k_idx]  # (Nt,4)
        W0, W1 = _tetra_weights_W0_W1(
            e_corner=e_corner,
            energy_grid=energy_grid,
            efermi_ry=efermi_ry,
            kBT_ry=kBT_ry,
        )

        # Build per-kpoint contributions for each diagonal component and average if requested
        # We only need diagonal components for this plotting tool.
        cond_axes = []
        alpha_axes = []
        cs_axes = []
        kk_axes = []

        for ax_i in diag_axes:
            gE = vnk[:, ax_i] * F_E[:, ax_i]
            gT = vnk[:, ax_i] * F_T[:, ax_i]

            gE_v = gE[k_idx]  # (Nt,4)
            gT_v = gT[k_idx]

            c_cond = np.zeros((Nk_all,), dtype=float)
            c_alpha = np.zeros((Nk_all,), dtype=float)
            c_cs = np.zeros((Nk_all,), dtype=float)
            c_kk = np.zeros((Nk_all,), dtype=float)

            for col in range(4):
                kk_i = k_idx[:, col]
                np.add.at(c_cond, kk_i, W0[:, col] * gE_v[:, col])
                np.add.at(c_alpha, kk_i, -W1[:, col] * gE_v[:, col])
                np.add.at(c_cs, kk_i, (-W0[:, col] * gT_v[:, col]) / kBT_ry)
                np.add.at(c_kk, kk_i, (W1[:, col] * gT_v[:, col]) / kBT_ry)

            cond_axes.append(c_cond)
            alpha_axes.append(c_alpha)
            cs_axes.append(c_cs)
            kk_axes.append(c_kk)

        cond_band = np.sum(cond_axes, axis=0) / float(len(diag_axes))
        alpha_band = np.sum(alpha_axes, axis=0) / float(len(diag_axes))
        cs_band = np.sum(cs_axes, axis=0) / float(len(diag_axes))
        kk_band = np.sum(kk_axes, axis=0) / float(len(diag_axes))

        # Apply global prefactor (tweight, spin, volume, alat^2)
        cond_band *= pref
        alpha_band *= pref
        cs_band *= pref
        kk_band *= pref

        mfp_flat.append(mfp_all_nm)
        cond_flat.append(cond_band)
        alpha_flat.append(alpha_band)
        cs_flat.append(cs_band)
        kk_flat.append(kk_band)

    mfp_v = np.concatenate(mfp_flat)
    cond_v = np.concatenate(cond_flat)
    alpha_v = np.concatenate(alpha_flat)
    cs_v = np.concatenate(cs_flat)
    kk_v = np.concatenate(kk_flat)

    mask = np.isfinite(mfp_v) & np.isfinite(cond_v) & np.isfinite(alpha_v) & np.isfinite(cs_v) & np.isfinite(kk_v) & (mfp_v > 0)
    mfp_v = mfp_v[mask]
    cond_v = cond_v[mask]
    alpha_v = alpha_v[mask]
    cs_v = cs_v[mask]
    kk_v = kk_v[mask]

    idx = np.argsort(mfp_v)
    mfp_v = mfp_v[idx]
    cond_v = cond_v[idx]
    alpha_v = alpha_v[idx]
    cs_v = cs_v[idx]
    kk_v = kk_v[idx]

    c_cond = np.cumsum(cond_v)
    c_alpha = np.cumsum(alpha_v)
    c_cs = np.cumsum(cs_v)
    c_kk = np.cumsum(kk_v)

    c_cond_safe = np.where(np.abs(c_cond) > 0, c_cond, 1e-300)
    seebeck_dimless = c_cs / c_cond_safe
    c_kappa_atomic = c_kk - c_alpha * seebeck_dimless

    cum_sigma_SI = c_cond * cond_unit
    cum_kappa_SI = c_kappa_atomic * t_cond_unit

    return mfp_v, cum_sigma_SI, cum_kappa_SI, T_K

def main() -> None:
    args = build_parser().parse_args()

    if args.style == "prb":
        apply_scienceplots_prb_style()
    else:
        apply_default_bold_rcparams()
    if args.fontsize is not None:
        apply_global_fontsize(float(args.fontsize))
    else:
        apply_global_fontsize(12)

    # Make mathtext follow the global bold-ish styling so subscripts like _{avg} also look bold.
    # (Mathtext does not always honor font.weight for LaTeX fragments.)
    plt.rcParams["mathtext.default"] = "bf"

    want_bold_fonts = bool(args.bold_fonts)
    if want_bold_fonts:
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"

    fig = None
    ax = None
    if not args.check_trans_coef:
        figsize = _parse_figsize(args.figsize)
        if figsize is None:
            fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=150)

    kB = 8.617333262e-5  # eV/K
    EV_TO_J = 1.602176634e-19
    E_CHARGE = 1.602176634e-19

    spin_deg = float(args.spin_deg)
    if not np.isfinite(spin_deg) or spin_deg <= 0:
        raise SystemExit("--spin-deg must be a finite positive number")

    # h5py is needed for the mfd+tetr path and for validation mode.
    try:
        import h5py as h5py  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Missing dependency h5py (needed for this script). Install with: pip install h5py "
            "(or run with your conda env that has h5py)."
        ) from exc

    if args.check_trans_coef and args.config != 0:
        raise SystemExit("--check-trans-coef requires --config 0 (iterate all temperature points).")

    def _normalize_list(val: list[str] | None) -> list[str]:
        if not val:
            return []
        return [str(x) for x in val]

    datasets: list[tuple[str, str, str]] = []

    # Deprecated input path
    if args.pair:
        for yaml_file, tdf_file, label in args.pair:
            datasets.append((str(yaml_file), str(tdf_file), str(label)))

    # New explicit input path
    meanfp_list = _normalize_list(args.meanfp)
    tdf_list = _normalize_list(args.tdf)
    legend_label_list = _normalize_list(args.legend)

    if meanfp_list or tdf_list or legend_label_list:
        if not meanfp_list or not tdf_list:
            raise SystemExit("Provide both --meanfp and --tdf (same count).")
        if len(meanfp_list) != len(tdf_list):
            raise SystemExit(f"Mismatched counts: --meanfp={len(meanfp_list)} vs --tdf={len(tdf_list)}")

        if legend_label_list:
            if len(legend_label_list) == 1 and len(meanfp_list) > 1:
                legend_label_list = legend_label_list * len(meanfp_list)
            if len(legend_label_list) != len(meanfp_list):
                raise SystemExit(
                    f"Mismatched counts: --legend={len(legend_label_list)} vs --meanfp={len(meanfp_list)} (or provide one label to reuse)"
                )
        else:
            if args.system and len(meanfp_list) == 1:
                legend_label_list = [str(args.system)]
            else:
                legend_label_list = [Path(p).stem.replace("_meanfp", "") for p in meanfp_list]

        for yaml_file, tdf_file, label in zip(meanfp_list, tdf_list, legend_label_list, strict=True):
            datasets.append((yaml_file, tdf_file, label))

    if not datasets:
        raise SystemExit("No inputs provided. Use --meanfp/--tdf[/--legend] (recommended) or deprecated --pair.")

    colors: list[str] = []
    if args.color:
        colors = [str(c) for c in args.color if str(c).strip()]
        if len(colors) == 1 and len(datasets) > 1:
            colors = colors * len(datasets)

    linestyles: list[str] = []
    if args.ls is not None:
        ls_in = [_normalize_linestyle_token(str(x)) for x in args.ls]
        linestyles = _broadcast_list(ls_in, len(datasets), "--ls")

    curve_mfp_list: list[np.ndarray] = []
    curve_y_list: list[np.ndarray] = []
    curve_color_list: list[str | None] = []

    for i_ds, (yaml_file, tdf_file, label) in enumerate(datasets):
        print(f"正在解析: {label}...")
        
        # 1. 从 YAML 提取能量和 MFP
        data = load_meanfp_yaml(yaml_file)
        e_by_band = get_energy_by_band(data)
        v_by_band = get_velocity_by_band(data)
        bands = sorted(e_by_band.keys())

        if args.check_trans_coef:
            tet_path = Path(args.tet) if args.tet else _infer_tet_path(tdf_file)
            _check_all_temperatures_against_trans_coef(
                meanfp_path=yaml_file,
                tdf_path=tdf_file,
                tet_path=tet_path,
                meanfp_data=data,
                direction=args.dir,
                spin_deg=spin_deg,
                h5py_module=h5py,
                trans_coef_path=(Path(args.trans_coef) if args.trans_coef else None),
            )
            continue

        if args.dir not in {"xx", "yy", "zz", "avg"}:
            raise SystemExit(f"Unsupported --dir={args.dir!r}")
        tet_path = Path(args.tet) if args.tet else _infer_tet_path(tdf_file)
        mfp_v, cum_sigma, cum_kappa, T_K = _cumulative_from_mfd_tetra(
            meanfp_data=data,
            tdf_h5_path=tdf_file,
            tet_path=tet_path,
            config=args.config,
            direction=args.dir,
            spin_deg=spin_deg,
            h5py_module=h5py,
        )

        y_plot = cum_kappa if args.qty == "kappa" else cum_sigma
        final_val = float(y_plot[-1])
        unit = "W/mK" if args.qty == "kappa" else "S/m"
        if args.qty == "sigma" and args.sigma_calib != "none":
            ref_val = _get_ref_total(
                method=args.sigma_calib,
                qty="sigma",
                direction=args.dir,
                T_K=T_K,
                tdf_path=tdf_file,
                trans_coef_path=(Path(args.trans_coef) if args.trans_coef else None),
                trans_ita_path=(Path(args.trans_ita) if args.trans_ita else None),
            )
            if ref_val is not None and np.isfinite(ref_val) and abs(final_val) > 0:
                scale = float(ref_val / final_val)
                y_plot = y_plot * scale
                final_val = float(y_plot[-1])
                print(f"   [calib] sigma <- {args.sigma_calib}: ref={float(ref_val):.6g} {unit}, scale={scale:.6f}")

        if args.qty == "kappa" and args.kappa_calib != "none":
            ref_val = _get_ref_total(
                method=args.kappa_calib,
                qty="kappa",
                direction=args.dir,
                T_K=T_K,
                tdf_path=tdf_file,
                trans_coef_path=(Path(args.trans_coef) if args.trans_coef else None),
                trans_ita_path=(Path(args.trans_ita) if args.trans_ita else None),
            )
            if ref_val is not None and np.isfinite(ref_val) and abs(final_val) > 0:
                scale = float(ref_val / final_val)
                y_plot = y_plot * scale
                final_val = float(y_plot[-1])
                print(f"   [calib] kappa <- {args.kappa_calib}: ref={float(ref_val):.6g} {unit}, scale={scale:.6f}")

        if args.qty == "kappa":
            print(f"   {args.dir} (mfd-tetra) kappa_el: {float(final_val):.6g} {unit}")
        else:
            print(f"   {args.dir} (mfd-tetra) sigma: {float(final_val):.6g} {unit}")

        assert ax is not None
        label_fmt = _format_chem_label(str(label), str(args.legend_format))
        line_label = f"{label_fmt}"
        plot_kwargs = {"lw": float(args.lw), "label": line_label}
        if colors and i_ds < len(colors):
            plot_kwargs["color"] = colors[i_ds]
        if linestyles:
            plot_kwargs["linestyle"] = linestyles[i_ds]

        # Extend the curve horizontally so it doesn't visually stop early.
        # Prefer extending to the user-provided x-limit upper bound.
        try:
            x_end = None
            if args.xlim is not None:
                x_end = float(args.xlim[1])
            else:
                x_end = float(mfp_v[-1]) * 1.2
            if np.isfinite(x_end) and x_end > float(mfp_v[-1]):
                mfp_v = np.concatenate([np.asarray(mfp_v, float), np.array([x_end], float)])
                y_plot = np.concatenate([np.asarray(y_plot, float), np.array([float(y_plot[-1])], float)])
        except Exception:
            pass

        line = ax.plot(mfp_v, y_plot, **plot_kwargs)[0]

        curve_mfp_list.append(np.asarray(mfp_v, float))
        curve_y_list.append(np.asarray(y_plot, float))
        try:
            curve_color_list.append(str(line.get_color()))
        except Exception:
            curve_color_list.append(None)

    if args.check_trans_coef:
        return

    assert ax is not None and fig is not None

    ax.set_xlabel(str(args.xlabel))

    if args.xlog:
        ax.set_xscale("log")
    if args.ylog:
        ax.set_yscale("log")
    
    if args.ylabel is not None:
        ax.set_ylabel(str(args.ylabel))
    else:
        unit_str = "W/mK" if args.qty == 'kappa' else "S/m"
        if args.qty == "kappa":
            if args.dir == "avg":
                ax.set_ylabel(f"Cumulative $\\boldsymbol{{\\kappa}}_{{\\boldsymbol{{avg}}}}$ ({unit_str})")
            else:
                ax.set_ylabel(f"Cumulative $\\boldsymbol{{\\kappa}}_{{\\boldsymbol{{el}}}}$ ({unit_str})")
        else:
            if args.dir == "avg":
                ax.set_ylabel(f"Cumulative $\\boldsymbol{{\\sigma}}_{{\\boldsymbol{{avg}}}}$ ({unit_str})")
            else:
                ax.set_ylabel(f"Cumulative $\\boldsymbol{{\\sigma}}$ ({unit_str})")
    
    if args.xlim is not None:
        ax.set_xlim(float(args.xlim[0]), float(args.xlim[1]))
    if args.ylim is not None:
        ax.set_ylim(float(args.ylim[0]), float(args.ylim[1]))

    if args.gb_mfp is not None and np.isfinite(float(args.gb_mfp)) and float(args.gb_mfp) > 0:
        gb_x = float(args.gb_mfp)
        ax.axvline(x=gb_x, color="grey", linestyle="--", linewidth=1.5, zorder=0)

        if args.gb_xlabel:
            try:
                txt = str(args.gb_xlabel_format).format(x=gb_x)
            except Exception:
                txt = f"{gb_x:g}"
            ax.annotate(
                txt,
                xy=(gb_x, 0.0),
                xycoords=("data", "axes fraction"),
                xytext=(0.0, float(args.gb_xlabel_dy)),
                textcoords="offset points",
                ha="center",
                va="top",
                color="grey",
            )

        # Annotate percentages for each dataset (one or many).
        if curve_mfp_list and curve_y_list:
            n_curves = len(curve_y_list)

            # Normalize multi-value args (broadcast to number of datasets).
            y_list = list(args.gb_text_y) if isinstance(args.gb_text_y, list) else [float(args.gb_text_y)]
            xpad_list = list(args.gb_text_xpad) if isinstance(args.gb_text_xpad, list) else [float(args.gb_text_xpad)]
            color_list = list(args.gb_text_color) if isinstance(args.gb_text_color, list) else [str(args.gb_text_color)]

            # If multiple datasets and only one y is provided, auto-stagger to avoid overlap.
            if n_curves > 1 and len(y_list) == 1:
                y0 = float(y_list[0])
                step = 0.06
                y_list = [max(0.05, min(0.95, y0 - i * step)) for i in range(n_curves)]

            y_list = _broadcast_list([float(v) for v in y_list], n_curves, "--gb-text-y")
            xpad_list = _broadcast_list([float(v) for v in xpad_list], n_curves, "--gb-text-xpad")
            color_list = _broadcast_list([str(v) for v in color_list], n_curves, "--gb-text-color")

            xmin, xmax = ax.get_xlim()

            text_common = {"transform": ax.get_xaxis_transform()}
            if args.legend_fontsize is not None:
                text_common["fontsize"] = float(args.legend_fontsize)

            outline_on = bool(args.gb_text_outline)
            outline_width = float(args.gb_text_outline_width)
            outline_color = str(args.gb_text_outline_color)
            if outline_width < 0:
                raise SystemExit("--gb-text-outline-width must be >= 0")

            for i in range(n_curves):
                mfp_i = curve_mfp_list[i]
                y_i = curve_y_list[i]
                if y_i.size < 2:
                    continue
                y_final = float(y_i[-1])
                if not (np.isfinite(y_final) and y_final != 0):
                    continue

                y_at = float(np.interp(gb_x, mfp_i, y_i, left=0.0, right=float(y_i[-1])))
                pct_left = max(0.0, min(100.0, 100.0 * y_at / y_final))
                pct_right = 100.0 - pct_left

                y_frac = float(y_list[i])
                if not (0.0 <= y_frac <= 1.0):
                    raise SystemExit("--gb-text-y must be within [0, 1]")

                xpad = float(xpad_list[i])
                if not np.isfinite(xpad) or xpad < 1.0:
                    raise SystemExit("--gb-text-xpad must be a finite number >= 1")

                # Avoid placing text exactly on the line when xpad==1.
                xpad_eff = xpad if xpad > 1.0 else 1.0001
                x_left = max(float(xmin), gb_x / xpad_eff)
                x_right = min(float(xmax), gb_x * xpad_eff)

                copt = str(color_list[i]).strip()
                if copt.lower() == "line":
                    c = curve_color_list[i]
                    text_color = c if c else None
                else:
                    text_color = copt

                kw = dict(text_common)
                if text_color is not None:
                    kw["color"] = text_color

                t_left = ax.text(x_left, y_frac, f"{pct_left:.1f}%", ha="right", va="top", **kw)
                t_right = ax.text(x_right, y_frac, f"{pct_right:.1f}%", ha="left", va="top", **kw)

                if outline_on and outline_width > 0:
                    pe = [
                        patheffects.Stroke(linewidth=outline_width, foreground=outline_color),
                        patheffects.Normal(),
                    ]
                    t_left.set_path_effects(pe)
                    t_right.set_path_effects(pe)

                if want_bold_fonts:
                    t_left.set_fontweight("bold")
                    t_right.set_fontweight("bold")

    if args.label_fontsize is not None:
        fs = float(args.label_fontsize)
        if fs <= 0:
            raise SystemExit("--label-fontsize must be > 0")
        ax.xaxis.label.set_size(fs)
        ax.yaxis.label.set_size(fs)
        ax.tick_params(axis="both", which="both", labelsize=fs)
        try:
            ax.xaxis.get_offset_text().set_size(fs)
        except Exception:
            pass
        try:
            ax.yaxis.get_offset_text().set_size(fs)
        except Exception:
            pass

    if args.grid:
        ax.grid(True, linestyle="--", alpha=0.3)

    leg_main = None
    if args.show_legend:
        legend_bbox = _parse_xy(args.legend_bbox)
        legend_loc = " ".join(args.legend_loc) if isinstance(args.legend_loc, list) else str(args.legend_loc)
        if (legend_bbox is not None) and (legend_loc.strip().lower() == "best"):
            legend_loc = "upper left"

        kwargs = dict(
            loc=legend_loc,
            frameon=bool(args.legend_alpha is not None),
            borderaxespad=(0.0 if legend_bbox is not None else 0.2),
            handlelength=1.5,
            handletextpad=0.4,
            labelspacing=0.35,
        )
        if args.legend_fontsize is not None:
            kwargs["fontsize"] = float(args.legend_fontsize)

        if legend_bbox is None:
            leg_main = ax.legend(**kwargs)
        else:
            leg_main = ax.legend(
                **kwargs,
                bbox_to_anchor=legend_bbox,
                bbox_transform=ax.transAxes,
            )

        if args.legend_alpha is not None and leg_main is not None:
            _apply_legend_frame_local(leg_main, alpha=float(args.legend_alpha))

        if want_bold_fonts and leg_main is not None:
            for t in leg_main.get_texts():
                t.set_fontweight("bold")

    # Optional system annotation as a separate legend entry.
    if args.system is not None and str(args.system).strip():
        sys_lab = _format_chem_label(str(args.system), str(args.system_format))
        h = Line2D([], [], color="none", label=sys_lab)

        sys_fs = args.system_fontsize
        if sys_fs is None:
            try:
                sys_fs = float(ax.yaxis.label.get_size()) * 1.15
            except Exception:
                sys_fs = None

        if leg_main is not None:
            ax.add_artist(leg_main)

        system_bbox = _parse_xy(args.system_bbox)
        sys_kwargs = {
            "frameon": bool(args.system_alpha is not None),
            "handlelength": 0,
            "handletextpad": 0.0,
            "borderaxespad": 0.2,
            "fontsize": sys_fs,
        }

        sys_loc = " ".join(args.system_loc) if isinstance(args.system_loc, list) else str(args.system_loc)

        if system_bbox is None:
            leg_sys = ax.legend(handles=[h], loc=sys_loc, **sys_kwargs)
        else:
            leg_sys = ax.legend(
                handles=[h],
                loc=sys_loc,
                bbox_to_anchor=system_bbox,
                bbox_transform=ax.transAxes,
                **sys_kwargs,
            )

        if args.system_alpha is not None and leg_sys is not None:
            _apply_legend_frame_local(leg_sys, alpha=float(args.system_alpha))

        if want_bold_fonts and leg_sys is not None:
            for t in leg_sys.get_texts():
                t.set_fontweight("bold")

    apply_plot_style(ax, legend=leg_main, bold=True)

    if want_bold_fonts:
        _set_figure_text_weight(fig, "bold")
    
    fig.tight_layout()
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out)
    if args.show or (not args.out):
        plt.show()

if __name__ == "__main__":
    main()