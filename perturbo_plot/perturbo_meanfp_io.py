#!/usr/bin/env python3
"""Utilities to read Perturbo meanfp YAML outputs.

Designed for YAML produced by `calc_mode: meanfp`.

Key structure observed in typical files:
- meanfp:
    energy:
      band index:
         1: [E(k1), E(k2), ...]
         2: [...]
    band velocity:
      band index:
         1: [|v|(k1), ...]
    configuration index:
      1:
        temperature: ...
        chemical potential: ...
        band index:
          1:
            MFP: [ ... ]
            relaxation time: [ ... ]
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def apply_scienceplots_prb_style() -> None:
    """Apply a PRB-like plotting style via SciencePlots.

    This is a best-effort helper. If SciencePlots is not installed, it falls back
    to matplotlib defaults.
    """

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except Exception:
        # Keep default matplotlib style.
        return


def apply_default_bold_rcparams() -> None:
    """Apply a QE-like 'default' bold look (non-SciencePlots).

    QE scripts historically used bold labels/ticks and thicker axes lines when not
    using the PRB SciencePlots preset.
    """

    try:
        import matplotlib.pyplot as plt

        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["axes.linewidth"] = 2
    except Exception:
        return


def format_label(label: str, mode: str) -> str:
    """Format a label as chemical formula (subscripts) or raw text.

    mode: 'chem' or 'raw'
    - raw: returns label unchanged
    - chem: renders digits appearing after letters or ')' as subscripts
    """

    if not label:
        return label
    mode = (mode or "raw").lower()
    if mode == "raw":
        return label
    if "$" in label:
        return label
    return re.sub(r"(?<=[A-Za-z\)])(\d+)", r"$_{\1}$", label)


def parse_xy(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = str(s).split(",", 1)
    return float(a), float(b)


def parse_figsize(s: Optional[str]) -> Optional[Tuple[float, float]]:
    """Parse figure size token 'w,h' (in inches)."""

    if not s:
        return None
    a, b = str(s).split(",", 1)
    return float(a), float(b)


def flatten_tokens(tokens: Optional[Sequence[str]]) -> List[str]:
    """Flatten argparse tokens supporting comma-separated inputs."""

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


def broadcast_list(xs: Sequence[str], n: int, name: str) -> List[str]:
    if n <= 0:
        return []
    if len(xs) == n:
        return list(xs)
    if len(xs) == 1:
        return [str(xs[0])] * n
    raise SystemExit(f"{name} expects 1 value (broadcast) or {n} values, got {len(xs)}")


def apply_plot_style(
    ax,
    *,
    legend=None,
    bold: bool = True,
    sci_y: str = "auto",
    ylog: bool = False,
) -> None:
    """Apply consistent plot styling.

    - bold: make labels/ticks/title/legend bold.
    - sci_y: 'auto'|'on'|'off' to show a ×10^n factor on y-axis.
      (Ignored when ylog=True.)
    """

    if bold:
        try:
            ax.title.set_fontweight("bold")
        except Exception:
            pass
        try:
            ax.xaxis.label.set_fontweight("bold")
            ax.yaxis.label.set_fontweight("bold")
        except Exception:
            pass

        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            try:
                t.set_fontweight("bold")
            except Exception:
                pass

        if legend is not None:
            for t in legend.get_texts():
                try:
                    t.set_fontweight("bold")
                except Exception:
                    pass

    sci_y = (sci_y or "auto").lower()
    if ylog:
        sci_y = "off"

    if sci_y not in {"auto", "on", "off"}:
        raise ValueError("sci_y must be one of: auto, on, off")

    def _should_use_sci() -> bool:
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

    if _should_use_sci():
        try:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            if bold:
                ax.yaxis.get_offset_text().set_fontweight("bold")
        except Exception:
            pass


class MeanFPFormatError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConfigInfo:
    index: int
    temperature_mev: float
    chemical_potential_ev: float


def _require_mapping(d, path: str):
    if not isinstance(d, dict):
        raise MeanFPFormatError(f"Expected mapping at {path}, got {type(d).__name__}")
    return d


def _as_float_list(x, path: str) -> List[float]:
    if not isinstance(x, list):
        raise MeanFPFormatError(f"Expected list at {path}, got {type(x).__name__}")
    out: List[float] = []
    for i, v in enumerate(x):
        try:
            out.append(float(v))
        except Exception as exc:  # noqa: BLE001
            raise MeanFPFormatError(f"Non-numeric at {path}[{i}]: {v!r}") from exc
    return out


def load_meanfp_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Missing dependency PyYAML. Install with: pip install pyyaml"
        ) from exc

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise MeanFPFormatError("Top-level YAML is not a mapping")
    if "meanfp" not in data:
        raise MeanFPFormatError("Missing top-level key 'meanfp'")
    return data


def get_energy_by_band(data: dict) -> Dict[int, List[float]]:
    meanfp = _require_mapping(data.get("meanfp"), "meanfp")
    energy = _require_mapping(meanfp.get("energy"), "meanfp.energy")
    band_index = _require_mapping(energy.get("band index"), "meanfp.energy.band index")

    out: Dict[int, List[float]] = {}
    for k, v in band_index.items():
        band = int(k)
        out[band] = _as_float_list(v, f"meanfp.energy.band index.{k}")
    return out


def get_velocity_by_band(data: dict) -> Dict[int, List[float]]:
    meanfp = _require_mapping(data.get("meanfp"), "meanfp")
    vel = _require_mapping(meanfp.get("band velocity"), "meanfp.band velocity")
    band_index = _require_mapping(vel.get("band index"), "meanfp.band velocity.band index")

    out: Dict[int, List[float]] = {}
    for k, v in band_index.items():
        band = int(k)
        out[band] = _as_float_list(v, f"meanfp.band velocity.band index.{k}")
    return out


def list_configurations(data: dict) -> List[ConfigInfo]:
    meanfp = _require_mapping(data.get("meanfp"), "meanfp")
    cfg_root = _require_mapping(meanfp.get("configuration index"), "meanfp.configuration index")

    out: List[ConfigInfo] = []
    for k, cfg in cfg_root.items():
        cfg_map = _require_mapping(cfg, f"meanfp.configuration index.{k}")
        idx = int(k)
        t_mev = float(cfg_map.get("temperature"))
        mu_ev = float(cfg_map.get("chemical potential"))
        out.append(ConfigInfo(index=idx, temperature_mev=t_mev, chemical_potential_ev=mu_ev))
    out.sort(key=lambda c: c.index)
    return out


def get_config_band_series(
    data: dict,
    config_index: int,
    band: int,
    key: str,
) -> List[float]:
    """Return per-(kpoint) series for a given config/band.

    key: "MFP" or "relaxation time".
    """
    meanfp = _require_mapping(data.get("meanfp"), "meanfp")
    cfg_root = _require_mapping(meanfp.get("configuration index"), "meanfp.configuration index")

    cfg = _require_mapping(cfg_root.get(config_index), f"meanfp.configuration index.{config_index}")
    bands = _require_mapping(cfg.get("band index"), f"meanfp.configuration index.{config_index}.band index")
    band_map = _require_mapping(bands.get(band), f"...band index.{band}")

    if key not in band_map:
        raise MeanFPFormatError(
            f"Missing key {key!r} for config {config_index}, band {band}"
        )
    return _as_float_list(band_map[key], f"...band index.{band}.{key}")


def get_mu_ev(data: dict, config_index: int) -> float:
    meanfp = _require_mapping(data.get("meanfp"), "meanfp")
    cfg_root = _require_mapping(meanfp.get("configuration index"), "meanfp.configuration index")
    cfg = _require_mapping(cfg_root.get(config_index), f"meanfp.configuration index.{config_index}")
    return float(cfg.get("chemical potential"))


def default_label(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def parse_band_selection(selection: Optional[str], available: Sequence[int]) -> List[int]:
    """Parse band selection like: "1", "1,3,5", "1-6".

    If selection is None: return all available.
    """
    av = sorted(set(int(x) for x in available))
    if not selection:
        return av

    chosen: List[int] = []
    parts = [p.strip() for p in selection.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            chosen.extend(list(range(min(lo, hi), max(lo, hi) + 1)))
        else:
            chosen.append(int(part))

    chosen = sorted(set(chosen))
    missing = [b for b in chosen if b not in av]
    if missing:
        raise ValueError(f"Bands not available in file: {missing}. Available: {av}")
    return chosen


def flatten_by_band(
    x_by_band: Dict[int, Sequence[float]],
    y_by_band: Dict[int, Sequence[float]],
    bands: Sequence[int],
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for b in bands:
        xb = list(x_by_band[b])
        yb = list(y_by_band[b])
        if len(xb) != len(yb):
            raise MeanFPFormatError(f"Length mismatch for band {b}: {len(xb)} vs {len(yb)}")
        xs.extend(xb)
        ys.extend(yb)
    return xs, ys


def bin_statistics(
    xs: Sequence[float],
    ys: Sequence[float],
    bin_width: float,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    reducer: str = "median",
) -> Tuple[List[float], List[float]]:
    """Bin (x,y) and compute a representative y per bin.

    reducer: 'median' or 'mean'
    Returns bin_centers, y_stat
    """
    import numpy as np

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if x.size == 0:
        return [], []

    lo = float(np.min(x) if x_min is None else x_min)
    hi = float(np.max(x) if x_max is None else x_max)
    if hi <= lo:
        return [], []

    nbins = int(np.ceil((hi - lo) / bin_width))
    edges = lo + bin_width * np.arange(nbins + 1)
    inds = np.digitize(x, edges) - 1

    centers: List[float] = []
    stats: List[float] = []

    for i in range(nbins):
        mask = inds == i
        if not np.any(mask):
            continue
        yc = y[mask]
        if reducer == "mean":
            val = float(np.mean(yc))
        else:
            val = float(np.median(yc))
        centers.append(float((edges[i] + edges[i + 1]) / 2.0))
        stats.append(val)

    return centers, stats
