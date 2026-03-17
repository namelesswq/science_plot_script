#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from perturbo_meanfp_io import apply_plot_style


@dataclass(frozen=True)
class KappaSeries:
    temperatures: List[float]
    values: List[float]


def format_formula_subscripts(label: str) -> str:
    """Convert digits in a plain-text label into mathtext subscripts.

    Example: 'Zr2SC' -> 'Zr$_{2}$SC'
    If label already contains '$', it is returned unchanged.
    """
    if "$" in label:
        return label
    return re.sub(r"(\d+)", r"$_{\1}$", label)


def _to_float(tok: str) -> float:
    # Some Fortran outputs may use D exponent
    return float(tok.replace("D", "E").replace("d", "E"))


def read_shengbte_kappa_tensor_vs_t(path: str) -> Dict[str, KappaSeries]:
    """Read ShengBTE BTE.KappaTensorVsT_CONV.

    Supported formats (common):
    - T + 9 tensor components (xx xy xz yx yy yz zx zy zz) [+ optional last column like iteration count]
    - T + 6 components (xx yy zz xy xz yz)
    - T + 3 components (xx yy zz)

    Returns mapping component -> series.
    """

    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                _ = _to_float(parts[0])
            except Exception:
                continue

            vals: List[float] = []
            ok = True
            for p in parts:
                try:
                    vals.append(_to_float(p))
                except Exception:
                    ok = False
                    break
            if ok:
                rows.append(vals)

    if not rows:
        raise RuntimeError(f"No numeric data parsed from: {path}")

    ncols = max(len(r) for r in rows)

    # Normalize each row to same length by truncation
    # (some files may have an extra integer column at the end)
    def pick(row: Sequence[float], n: int) -> List[float]:
        return list(row[:n])

    out: Dict[str, Tuple[List[float], List[float]]] = {}

    if ncols >= 10:
        # Assume: T + 9 tensor entries
        order = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
        for comp_i, comp in enumerate(order):
            ts: List[float] = []
            ys: List[float] = []
            for r in rows:
                rr = pick(r, 10)
                if len(rr) < 10:
                    continue
                ts.append(float(rr[0]))
                ys.append(float(rr[1 + comp_i]))
            out[comp] = (ts, ys)

    elif ncols >= 7:
        # Assume: T + (xx yy zz xy xz yz)
        order6 = ["xx", "yy", "zz", "xy", "xz", "yz"]
        for comp_i, comp in enumerate(order6):
            ts: List[float] = []
            ys: List[float] = []
            for r in rows:
                rr = pick(r, 7)
                if len(rr) < 7:
                    continue
                ts.append(float(rr[0]))
                ys.append(float(rr[1 + comp_i]))
            out[comp] = (ts, ys)
        # mirror symmetric terms
        if "xy" in out:
            out["yx"] = out["xy"]
        if "xz" in out:
            out["zx"] = out["xz"]
        if "yz" in out:
            out["zy"] = out["yz"]

    elif ncols >= 4:
        # Assume: T + (xx yy zz)
        order3 = ["xx", "yy", "zz"]
        for comp_i, comp in enumerate(order3):
            ts: List[float] = []
            ys: List[float] = []
            for r in rows:
                rr = pick(r, 4)
                if len(rr) < 4:
                    continue
                ts.append(float(rr[0]))
                ys.append(float(rr[1 + comp_i]))
            out[comp] = (ts, ys)

    else:
        raise RuntimeError(
            f"Unsupported column count in {path}: {ncols}. Expected at least 4 columns."
        )

    return {k: KappaSeries(v[0], v[1]) for k, v in out.items()}


def read_perturbo_trans_ita_e_kappa(path: str) -> Dict[str, KappaSeries]:
    """Read electronic thermal conductivity from Perturbo trans-ita YAML."""

    try:
        import yaml  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Missing dependency PyYAML. Install with: pip install pyyaml") from exc

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "trans" not in data:
        raise RuntimeError(f"Not a Perturbo trans YAML: {path}")

    trans = data["trans"]
    cfgs = trans.get("configuration index")
    if not isinstance(cfgs, dict):
        raise RuntimeError(f"Missing trans.configuration index in: {path}")

    comp_map: Dict[str, List[Tuple[float, float]]] = {}

    for _, cfg in cfgs.items():
        if not isinstance(cfg, dict):
            continue
        t = float(cfg.get("temperature"))
        tc = cfg.get("thermal conductivity")
        if not isinstance(tc, dict):
            continue
        comps = tc.get("components")
        if not isinstance(comps, dict):
            continue
        for comp, val in comps.items():
            c = str(comp).lower()
            if c not in {"xx", "yy", "zz", "xy", "xz", "yz", "yx", "zx", "zy"}:
                continue
            comp_map.setdefault(c, []).append((t, float(val)))

    if not comp_map:
        raise RuntimeError(f"No thermal conductivity components found in: {path}")

    out: Dict[str, KappaSeries] = {}
    for c, pairs in comp_map.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        out[c] = KappaSeries([p[0] for p in pairs_sorted], [p[1] for p in pairs_sorted])

    return out


def _interp_on_grid(x_src: Sequence[float], y_src: Sequence[float], x_tgt: Sequence[float]) -> List[float]:
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Missing dependency numpy. Install with: pip install numpy") from exc

    xs = np.asarray(list(x_src), dtype=float)
    ys = np.asarray(list(y_src), dtype=float)
    xt = np.asarray(list(x_tgt), dtype=float)

    if xs.size == 0 or xt.size == 0:
        return [float("nan")] * len(x_tgt)

    # sort if necessary
    if xs.size >= 2 and not np.all(np.diff(xs) >= 0):
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]

    y = np.interp(xt, xs, ys, left=np.nan, right=np.nan)
    return [float(v) for v in y]


def _assert_same_temperature_grid(
    t_latt: Sequence[float],
    t_el: Sequence[float],
    *,
    tol: float = 1e-6,
    latt_label: str = "lattice",
    el_label: str = "electron",
) -> None:
    if len(t_latt) != len(t_el):
        raise SystemExit(
            f"Temperature grid mismatch: {latt_label} has {len(t_latt)} points, {el_label} has {len(t_el)} points."
        )

    mism: List[Tuple[int, float, float]] = []
    for i, (a, b) in enumerate(zip(t_latt, t_el)):
        if abs(float(a) - float(b)) > tol:
            mism.append((i, float(a), float(b)))

    if mism:
        preview = mism[:8]
        details = ", ".join([f"i={i}: {ta} vs {tb}" for i, ta, tb in preview])
        more = "" if len(mism) <= 8 else f" (and {len(mism) - 8} more)"
        raise SystemExit(
            "Temperature grid mismatch between lattice and electron files. "
            "This script does not interpolate; please rerun with matching temperature lists. "
            f"Mismatches: {details}{more}"
        )


def _get_component_series(
    series_by_comp: Dict[str, KappaSeries],
    component: str,
    *,
    series_label: str,
    tol: float = 1e-6,
) -> KappaSeries:
    comp = component.lower()
    if comp != "avg":
        if comp not in series_by_comp:
            raise SystemExit(
                f"Component {comp} not found in {series_label}. Available: {sorted(series_by_comp.keys())}"
            )
        return series_by_comp[comp]

    for c in ("xx", "yy", "zz"):
        if c not in series_by_comp:
            raise SystemExit(
                f"Component {c} required for avg in {series_label}, but not found. Available: {sorted(series_by_comp.keys())}"
            )

    sxx = series_by_comp["xx"]
    syy = series_by_comp["yy"]
    szz = series_by_comp["zz"]

    _assert_same_temperature_grid(sxx.temperatures, syy.temperatures, tol=tol, latt_label=f"{series_label} xx", el_label=f"{series_label} yy")
    _assert_same_temperature_grid(sxx.temperatures, szz.temperatures, tol=tol, latt_label=f"{series_label} xx", el_label=f"{series_label} zz")

    temps = list(sxx.temperatures)
    vals = [(a + b + c) / 3.0 for a, b, c in zip(sxx.values, syy.values, szz.values)]
    return KappaSeries(temps, vals)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot lattice/electronic/total thermal conductivity vs temperature by combining "
            "ShengBTE BTE.KappaTensorVsT_CONV and Perturbo trans-ita.yml."
        )
    )

    p.add_argument(
        "--set",
        action="append",
        nargs=3,
        metavar=("LABEL", "LATTICE_FILE", "ELECTRON_FILE"),
        help=(
            "Add one dataset: LABEL BTE.KappaTensorVsT_CONV zr2sc_trans-ita.yml. "
            "Repeat --set to compare multiple datasets."
        ),
        required=True,
    )

    p.add_argument(
        "--component",
        default="xx",
        choices=["xx", "yy", "zz", "xy", "xz", "yz", "yx", "zx", "zy", "avg"],
        help="Tensor component to plot [default: xx]. Use 'avg' for (xx+yy+zz)/3.",
    )

    p.add_argument(
        "--total-grid",
        choices=["lattice", "electron"],
        default="lattice",
        help="(Deprecated) Kept for compatibility. Temperatures must match; no interpolation is performed.",
    )

    p.add_argument("--ylog", action="store_true", help="Use log scale on y-axis")
    p.add_argument("--no-bold", action="store_true", help="Disable bold text in the figure")
    p.add_argument(
        "--sci-y",
        choices=["auto", "on", "off"],
        default="auto",
        help="Y-axis scientific notation factor (×10^n) [default: auto]",
    )

    p.add_argument("--xlim", default=None, help='x limits "xmin,xmax" in K')
    p.add_argument("--ylim", default=None, help='y limits "ymin,ymax" in W/m/K')
    p.add_argument("--title", default=None, help="Plot title")
    p.add_argument("--out", default=None, help="Output image path (png/pdf/svg). If omitted, show interactively.")
    return p


def _parse_lim(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = s.split(",", 1)
    return float(a), float(b)


def main() -> None:
    args = _build_parser().parse_args()

    xlim = _parse_lim(args.xlim)
    ylim = _parse_lim(args.ylim)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)

    # Style: solid lines only; use different colors to distinguish curves.
    for raw_label, latt_path, ele_path in args.set:
        label = format_formula_subscripts(raw_label)
        latt = read_shengbte_kappa_tensor_vs_t(latt_path)
        ele = read_perturbo_trans_ita_e_kappa(ele_path)

        comp = args.component.lower()
        latt_s = _get_component_series(latt, comp, series_label=f"{label} lattice")
        ele_s = _get_component_series(ele, comp, series_label=f"{label} electron")

        tL, kL = latt_s.temperatures, latt_s.values
        tE, kE = ele_s.temperatures, ele_s.values

        _assert_same_temperature_grid(
            tL,
            tE,
            tol=1e-6,
            latt_label=f"{label} κ_latt",
            el_label=f"{label} κ_el",
        )

        # plot lattice and electron
        ax.plot(tL, kL, lw=2.2, linestyle="-", label=f"{label} κ_latt")
        ax.plot(tE, kE, lw=2.2, linestyle="-", label=f"{label} κ_el")

        # total (no interpolation allowed)
        kT = [kl + ke for kl, ke in zip(kL, kE)]
        ax.plot(tL, kT, lw=2.4, linestyle="-", label=f"{label} κ_total")

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Thermal conductivity $\kappa$ (W/m/K)")

    if args.ylog:
        ax.set_yscale("log")

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title(f"Thermal conductivity vs T ({args.component})")

    ax.grid(True, alpha=0.25)
    leg = ax.legend(frameon=False, ncols=1)
    apply_plot_style(ax, legend=leg, bold=not args.no_bold, sci_y=args.sci_y, ylog=args.ylog)

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
