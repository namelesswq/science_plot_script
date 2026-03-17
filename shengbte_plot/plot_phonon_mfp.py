#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot phonon mean free path (MFP) from ShengBTE outputs. "
            "Uses BTE.omega + BTE.v + one or more BTE.w_* files. "
            "Default units assume: omega [rad/ps], v [km/s], w [ps^-1] => MFP [nm] = |v|/w."
        )
    )

    p.add_argument("--omega", default="BTE.omega", help="Phonon frequency file [default: BTE.omega]")
    p.add_argument("--v", default="BTE.v", help="Phonon group velocity vector file [default: BTE.v]")
    p.add_argument("w_files", nargs="+", help="One or more scattering-rate files (e.g. BTE.w_final, BTE.w_isotopic)")
    p.add_argument("--labels", default=None, help="Comma-separated labels for each w file")

    p.add_argument(
        "--v-unit",
        choices=["km/s", "m/s"],
        default="km/s",
        help="Unit of |v| in BTE.v [default: km/s]",
    )

    p.add_argument("--x", choices=["omega", "wfile"], default="omega", help="x-axis source [default: omega]")
    p.add_argument("--ylog", action="store_true", help="Use log scale for y-axis")
    p.add_argument("--xlog", action="store_true", help="Use log scale for x-axis")

    p.add_argument("--alpha", type=float, default=0.35, help="Scatter alpha [default: 0.35]")
    p.add_argument("--s", type=float, default=8.0, help="Marker size [default: 8]")

    p.add_argument("--xlim", default=None, help='x limits "xmin,xmax" in THz')
    p.add_argument("--ylim", default=None, help='y limits "ymin,ymax" in nm')

    p.add_argument("--title", default="Phonon mean free path", help="Plot title")
    p.add_argument("--out", default="phonon_mfp_scatter.png", help="Output image [default: phonon_mfp_scatter.png]")
    p.add_argument("--show", action="store_true", help="Show interactively (may require display)")
    return p


def _parse_lim(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = s.split(",", 1)
    return float(a), float(b)


def _load_omega_thz(path: str) -> np.ndarray:
    # ShengBTE BTE.omega is typically angular frequency in rad/ps.
    # Convert: THz = (rad/ps) / (2*pi)
    omega_mat = np.loadtxt(path)
    omega_flat = np.asarray(omega_mat, dtype=float).flatten()
    return omega_flat / (2.0 * np.pi)


def _load_v_mag(path: str) -> np.ndarray:
    v_vec = np.loadtxt(path)
    v_vec = np.asarray(v_vec, dtype=float)
    if v_vec.ndim != 2 or v_vec.shape[1] < 3:
        raise RuntimeError(f"Unexpected BTE.v shape: {v_vec.shape} (expected N x 3)")
    return np.linalg.norm(v_vec[:, :3], axis=1)


def _load_w_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise RuntimeError(f"Unexpected w-file shape: {data.shape} (expected N x >=2)")
    omega = data[:, 0]
    w = data[:, 1]
    return omega, w


def _mfp_nm(vmag: np.ndarray, w: np.ndarray, v_unit: str) -> np.ndarray:
    # w in ps^-1, tau = 1/w (ps)
    # If v is km/s: 1 km/s = 1 nm/ps => MFP[nm] = v / w
    # If v is m/s:  1 m/s  = 1e-3 nm/ps => MFP[nm] = (v*1e-3) / w
    if v_unit == "km/s":
        factor = 1.0
    else:
        factor = 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        mfp = (vmag * factor) / w
    return mfp


def main() -> None:
    args = _build_parser().parse_args()

    if args.labels is None:
        labels: List[str] = [os.path.basename(f) for f in args.w_files]
    else:
        labels = [x.strip() for x in args.labels.split(",") if x.strip()]
        if len(labels) != len(args.w_files):
            raise SystemExit(f"--labels count {len(labels)} != w_files count {len(args.w_files)}")

    # bold style similar to your other scripts
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.linewidth"] = 2

    try:
        x_omega_thz = _load_omega_thz(args.omega)
    except Exception as e:
        raise SystemExit(f"[错误] 读取 {args.omega} 失败: {e}")

    try:
        vmag = _load_v_mag(args.v)
    except Exception as e:
        raise SystemExit(f"[错误] 读取 {args.v} 失败: {e}")

    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)

    for w_path, label in zip(args.w_files, labels):
        try:
            w_omega, w_rate = _load_w_file(w_path)
        except Exception as e:
            print(f"[警告] 读取 {w_path} 失败: {e} (跳过)")
            continue

        # Choose x-axis
        if args.x == "wfile":
            x_thz = np.asarray(w_omega, dtype=float) / (2.0 * np.pi)
        else:
            x_thz = x_omega_thz

        # Align lengths
        n = min(len(x_thz), len(vmag), len(w_rate))
        if n == 0:
            print(f"[警告] {w_path}: 没有可用数据 (跳过)")
            continue
        if len(x_thz) != n or len(vmag) != n or len(w_rate) != n:
            print(
                f"[警告] 长度不匹配，将截断到 {n}: x={len(x_thz)} v={len(vmag)} w={len(w_rate)} (file={w_path})"
            )

        x = np.asarray(x_thz[:n], dtype=float)
        v = np.asarray(vmag[:n], dtype=float)
        w = np.asarray(w_rate[:n], dtype=float)

        # Filter invalid / non-positive rates
        m = np.isfinite(x) & np.isfinite(v) & np.isfinite(w) & (w > 0.0)
        x = x[m]
        v = v[m]
        w = w[m]

        y = _mfp_nm(v, w, args.v_unit)
        m2 = np.isfinite(y) & (y > 0.0)
        x = x[m2]
        y = y[m2]

        ax.scatter(x, y, s=args.s, alpha=args.alpha, edgecolors="none", label=label)

    ax.set_title(args.title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Frequency (THz)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean free path (nm)", fontsize=14, fontweight="bold")

    if args.ylog:
        ax.set_yscale("log")
    if args.xlog:
        ax.set_xscale("log")

    xlim = _parse_lim(args.xlim)
    ylim = _parse_lim(args.ylim)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    ax.grid(True, linestyle="--", alpha=0.3)
    leg = ax.legend(loc="best", fontsize=12, frameon=True, edgecolor="black")
    for t in leg.get_texts():
        t.set_fontweight("bold")

    for t in (ax.get_xticklabels() + ax.get_yticklabels()):
        t.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"绘图完成！图片已保存为 {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
