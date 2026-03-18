# science_plot_script

这是一组用于科研绘图的 Python 脚本集合，覆盖：
- Quantum ESPRESSO：能带、DOS/PDOS（projwfc）以及“能带 + PDOS”合图
- Perturbo：meanfp（群速度、散射率、电子 MFP）与 κ(T)（结合 ShengBTE + Perturbo）
- ShengBTE：声子群速度散点、散射率散点、声子 MFP 散点

## 依赖安装

推荐使用 conda 环境（你自己的 python310 环境已验证可正常运行）。

- 最小依赖（绝大多数脚本都需要）：

```bash
pip install numpy matplotlib pyyaml
```

- QE 能带/PDOS 脚本默认 `--style prb` 会使用 SciencePlots（投稿风格）：

```bash
pip install SciencePlots
```

无显示器的服务器环境建议：

```bash
export MPLBACKEND=Agg
```

## 目录结构

- `qe_plot/`：Quantum ESPRESSO（能带、DOS/PDOS、合图）
- `perturbo_plot/`：Perturbo meanfp（|v|、1/τ、MFP）+ κ(T)
- `shengbte_plot/`：ShengBTE 声子相关散点图

---

# QE 绘图（qe_plot）

## 1) `qe_plot/plot_qe_dos_pdos_overlay.py`

用途：在一张图里叠加绘制总 DOS + 按元素/轨道（可选按 wfc# 区分）的 PDOS。

输入文件：
- 总 DOS：`<prefix>.pdos.pdos_tot`
- PDOS：`<prefix>.pdos.pdos_atm#N(El)_wfc#M(orb)`（projwfc 输出）

最常用示例（自动从 `--tot` 推断并 glob PDOS）：

```bash
python qe_plot/plot_qe_dos_pdos_overlay.py \
  --tot zr2sc.pdos.pdos_tot \
  --elements Zr,S,C \
  --orbitals s,p,d \
  --merge-wfc \
  --fermi 14.776 \
  --fermi-line \
  --system Zr2SC \
  --system-fontsize 12 \
  --system-bbox 0,0.75 \
  --xlim -5,5 \
  --out dos_pdos.png
```

多体系对比示例（多组总 DOS + PDOS 叠加；每组可有自己的 fermi 与 PDOS glob）：

```bash
python qe_plot/plot_qe_dos_pdos_overlay.py \
  --tot prim/zr2sc.pdos.pdos_tot sc222/zr2sc.pdos.pdos_tot \
  --pdos-glob 'prim/zr2sc.pdos.pdos_atm#*' 'sc222/zr2sc.pdos.pdos_atm#*' \
  --elements Zr,S,C \
  --orbitals s,p,d \
  --merge-wfc \
  --fermi 14.776,14.812 \
  --fermi-line \
  --system Zr2SC,2x2x2 \
  --system-fontsize 12 \
  --system-bbox 0,0.75 \
  --xlim -5,5 \
  --out dos_pdos_compare.png
```

常用参数：
- `--pdos`：显式给出 PDOS 文件列表（不常用，文件太多时可用）
- `--pdos-glob`：自定义 glob 模式（覆盖自动推断）
- `--merge-wfc`：将同一元素同一轨道的不同 `wfc#` 合并为一条曲线
- `--n0 'Zr=4,S=3,C=2'`：不合并 wfc 时，把 `wfc#` 重新标成“主量子数标签”（如 Zr-4d、Zr-5d…）
- `--tot-col` / `--pdos-col`：指定读哪一列（0-based；默认都取第 2 列即 `col=1`）
- `--style prb|default`、`--figsize 3.4,2.6`、`--ylog`、`--sci-y auto|on|off`
- `--system` / `--system-format chem|raw` / `--system-fontsize` / `--system-loc` / `--system-bbox x,y`：在图内标注体系名称（`--system-bbox` 用轴坐标 0~1 精确定位）

---

## 2) `qe_plot/plot_qe_bands.py`

用途：从 `bands.out.gnu` 画能带，并结合 `band.in` 与 `KPATH.in` 标注高对称点。

输入文件：
- `bands.out.gnu`：两列（k 距离、能量），空行分隔不同 band
- `band.in`：QE 输入，包含 `K_POINTS crystal_b`，每个 k 点行第 4 列为该段点数 `N`；其中 `N=1` 代表“跳跃/断点”
- `KPATH.in`（可选）：VASPKIT 格式的高对称点标签

示例：

```bash
python qe_plot/plot_qe_bands.py \
  --bands bands.out.gnu \
  --band-in band.in \
  --kpath KPATH.in \
  --fermi 14.776 \
  --fermi-line \
  --ylim -5,5 \
  --system Zr2SC \
  --system-fontsize 14 \
  --system-bbox 0,0.75 \
  --lw 0.6 \
  --out bands.png
```

多体系对比示例（两套能带叠加对比；脚本会按高对称点分段线性映射把 x 轴对齐到第一套数据）：

```bash
python qe_plot/plot_qe_bands.py \
  --bands prim/bands.out.gnu sc222/bands.out.gnu \
  --band-in prim/band.in sc222/band.in \
  --kpath KPATH.in \
  --fermi 14.776,14.812 \
  --fermi-line \
  --ylim -5,5 \
  --system Zr2SC,2x2x2 \
  --system-fontsize 14 \
  --system-bbox 0,0.75 \
  --lw 0.6 \
  --out bands_compare.png
```

说明：多体系叠加要求“高对称点标签序列一致”（例如都是 `Γ-X-M-Γ` 且断点位置一致）；否则脚本会报错提示。

说明：
- 当 `band.in` 中某行 `N=1` 时，该位置视作不连续点：刻度标签会被合并成 `A|L`（终点|起点），且能带不会跨越该点连线。
- x 轴只保留高对称点处的竖线（`axvline`），不再显示刻度短线；高对称点标签仍会显示。
- 可用 `--system` 在图内标注体系名称（默认会把化学式中的数字渲染为下标，如 `Zr2SC → Zr$_{2}$SC`）；若不想下标，用 `--system-format raw`。
- 体系标注位置：用 `--system-loc` 选大致方位；如需精确摆放（或放到图外），用 `--system-bbox x,y`（轴坐标系 0~1），例如 `--system-bbox 1.02,1.0`。

---

## 3) `qe_plot/plot_qe_bands_with_pdos.py`

用途：一张图同时画“能带 + PDOS”。右侧 PDOS 面板会旋转 90°（DOS 在 x，能量在共享 y）。

输入文件：
- 能带：`bands.out.gnu`、`band.in`、（可选）`KPATH.in`
- DOS/PDOS：同 `plot_qe_dos_pdos_overlay.py`（`--tot` + 自动 glob PDOS）

最常用示例（你此前常用的排版参数）：

```bash
python qe_plot/plot_qe_bands_with_pdos.py \
  --bands bands.out.gnu \
  --band-in band.in \
  --kpath KPATH.in \
  --tot zr2sc.pdos.pdos_tot \
  --elements Zr,S,C \
  --orbitals s,p,d \
  --merge-wfc \
  --fermi 14.776 \
  --fermi-line \
  --ylim -5,5 \
  --figsize-bands 7,3 \
  --figsize-dos 1.5,3 \
  --legend-fontsize 7 \
  --legend-loc best \
  --system Zr2SC \
  --system-fontsize 14 \
  --system-bbox 0,0.75 \
  --lw 0.6 \
  --out bands_pdos.png
```

多体系对比示例（两套“能带 + (P)DOS”合图叠加；bands/total DOS 用黑/红/蓝…区分体系，PDOS 用全局唯一色池）：

```bash
python qe_plot/plot_qe_bands_with_pdos.py \
  --bands prim/bands.out.gnu sc222/bands.out.gnu \
  --band-in prim/band.in sc222/band.in \
  --kpath KPATH.in \
  --tot prim/zr2sc.pdos.pdos_tot sc222/zr2sc.pdos.pdos_tot \
  --pdos-glob 'prim/zr2sc.pdos.pdos_atm#*' 'sc222/zr2sc.pdos.pdos_atm#*' \
  --elements Zr,S,C \
  --orbitals s,p,d \
  --merge-wfc \
  --fermi 14.776,14.812 \
  --fermi-line \
  --ylim -5,5 \
  --figsize-bands 7,3 \
  --figsize-dos 1.8,3 \
  --legend-fontsize 7 \
  --legend-loc best \
  --system Zr2SC,2x2x2 \
  --system-fontsize 14 \
  --system-bbox 0,0.75 \
  --lw 0.6 \
  --out bands_pdos_compare.png
```

说明：
- 多体系模式下不支持用 `--pdos` 直接给“所有 PDOS 文件列表”（无法自动分组）；请用每套一个 `--pdos-glob`，或依赖脚本对每个 `--tot` 的自动推断。

常用参数：
- `--dos-xlim xmin,xmax`：限制右侧 DOS 轴范围
- `--ratios 3,1`：面板宽度比（若没用 `--figsize-bands/--figsize-dos`，可用这个快速调比例）
- `--n0`、`--tot-col`、`--pdos-col`：同上
- `--system` / `--system-format chem|raw` / `--system-fontsize` / `--system-loc` / `--system-bbox x,y`：在左侧能带面板标注体系（`--system-bbox` 用轴坐标精确定位）

---

## 4) `qe_plot/plot_qe_phonon_bands.py`

用途：绘制 Quantum ESPRESSO `matdyn.x` 的声子谱（色散关系）。

输入文件：
- `*.freq.gp`：第一列是路径坐标 x（已累计的路径距离），后面每一列是一条声子带
- `matdyn.in`：包含 q 点路径（通常 `q_in_band_form = .true.`），每行第 4 列是该段点数 `N`；其中 `N=1` 表示跳跃/断点
- `KPATH.in`（可选）：VASPKIT 格式高对称点名称（坐标匹配，Γ 会自动归一化为 `Γ`）

示例（与你当前目录结构一致）：

```bash
python qe_plot/plot_qe_phonon_bands.py \
  --freq zr2sc.freq.gp \
  --matdyn-in matdyn.in \
  --kpath KPATH.in \
  --unit THz \
  --ylim 0,8 \
  --system Zr2SC \
  --system-fontsize 14 \
  --system-bbox 0,0.75 \
  --lw 0.6 \
  --figsize 7,3 \
  --out phonon_bands.png
```

说明：
- 默认会把 `*.freq.gp` 里的频率从 `cm^-1` 转为 `THz`（除以 `33.35641`）；如需保留 `cm^-1` 用 `--unit cm^-1`
- 脚本会按 `matdyn.in` 的 `N=1` 断点自动“断开连线”，并把该点刻度合并为 `A|L`
- 高对称点标签若过密，脚本会只对“发生遮挡的少数标签”做处理（缩字号/最多 45°/两行错位），不遮挡的标签保持水平
- x 轴只保留高对称点处的竖线（`axvline`），不再显示刻度短线；高对称点标签仍会显示。
- 可用 `--system` 标注体系名称（默认会把化学式数字渲染为下标；不想下标用 `--system-format raw`）。
- 体系标注位置：用 `--system-loc` 选方位；用 `--system-bbox x,y` 做精确定位（轴坐标 0~1）。

---

## 5) `qe_plot/plot_qe_phonon_dos_pdos.py`

用途：绘制声子总 DOS + PDOS（每原子一列），并可选择按元素求和。

输入文件：
- `*.dos`：第一列频率（cm^-1），第二列总 DOS，后面每一列是每个原子的 PDOS
- `scf.in`：用于解析 `ATOMIC_POSITIONS` 的原子顺序，从而给 PDOS 列贴标签/按元素求和

示例（按元素求和，单位 THz，y 轴单位为 `states/THz/unit cell`）：

```bash
python qe_plot/plot_qe_phonon_dos_pdos.py \
  --dos zr2sc.dos \
  --scf-in scf.in \
  --group element \
  --system Zr2SC \
  --system-fontsize 12 \
  --system-bbox 0,0.75 \
  --xlim 0,17.2 \
  --lw 0.6 \
  --figsize 6,3 \
  --out phonon_dos.png
```

多组数据对比（例如：原胞 vs 2×2×2 超胞）。多组模式下默认只画 total DOS（不画 PDOS，避免曲线/图例过多）：

```bash
python qe_plot/plot_qe_phonon_dos_pdos.py \
  --dos prim.dos sc222.dos \
  --scf-in prim_scf.in sc222_scf.in \
  --dos-norm 1,8 \
  --labels prim,2x2x2 \
  --xlim 0,17.2 \
  --lw 0.6 \
  --figsize 6,3 \
  --out phonon_dos_compare.png
```

说明：
- 默认 `--unit THz`：横轴 `cm^-1 → THz`（除以 `33.35641`）
- 为保持单位一致，默认会对 DOS/PDOS 做雅可比缩放：`g(THz) = g(cm^-1) * 33.35641`（可用 `--no-jacobian` 关闭）
- 可用 `--system` 在图内标注体系名称；位置用 `--system-loc` 或 `--system-bbox x,y` 精确控制（轴坐标 0~1）。
- 多组对比时，可用 `--dos-norm 1,8,...` 为每组指定一个整数归一化因子（读取后分别除以该因子），常用于将超胞 DOS 换算到“每原胞”可比的量。
- `--group atom` 时可用 `--atoms 1,2,5-8` 选原子；两种分组都可用 `--elements Zr,S` 过滤

---

## 6) `qe_plot/plot_qe_phonon_bands_with_dos.py`

用途：一张图同时画“声子谱 + 声子 DOS/PDOS”（布局与电子 `plot_qe_bands_with_pdos.py` 类似）。

输入文件：
- 声子谱：`*.freq.gp` + `matdyn.in` +（可选）`KPATH.in`
- DOS/PDOS：`*.dos` + `scf.in`

示例（左：声子谱；右：按元素求和的 PDOS，单位 THz）：

```bash
python qe_plot/plot_qe_phonon_bands_with_dos.py \
  --freq zr2sc.freq.gp \
  --matdyn-in matdyn.in \
  --kpath KPATH.in \
  --dos zr2sc.dos \
  --scf-in scf.in \
  --group element \
  --unit THz \
  --ylim 0,8 \
  --system Zr2SC \
  --system-fontsize 14 \
  --system-bbox 0,0.75 \
  --lw 0.6 \
  --figsize-bands 7,3 \
  --figsize-dos 2,3 \
  --out phonon_bands_dos.png
```

多组对比（bands + total DOS）：

```bash
python qe_plot/plot_qe_phonon_bands_with_dos.py \
  --freq prim.freq.gp sc222.freq.gp \
  --matdyn-in prim_matdyn.in sc222_matdyn.in \
  --kpath prim_KPATH.in sc222_KPATH.in \
  --dos prim.dos sc222.dos \
  --scf-in prim_scf.in sc222_scf.in \
  --dos-norm 1,8 \
  --system Zr2SC,Zr16S8C8 \
  --unit THz \
  --ylim 0,8 \
  --lw 0.6 \
  --figsize-bands 7,3 \
  --figsize-dos 2,3 \
  --out phonon_bands_dos_compare.png
```

说明：
- 默认会压缩断点处的 x 轴跳跃空白；如需保留跳跃，用 `--keep-jumps`
- 默认 `--unit THz`：频率 `cm^-1 → THz`（除以 `33.35641`）
- 为保持单位一致，默认会对 DOS/PDOS 做雅可比缩放：`g(THz) = g(cm^-1) * 33.35641`（可用 `--no-jacobian` 关闭）
- 多组对比时，可用 `--dos-norm 1,8,...` 为每组指定一个整数归一化因子（读取后分别除以该因子）；右侧面板默认只画 total DOS。
- 右侧 DOS 面板与左侧共享频率 y 轴（DOS 在 x，频率在 y）
- 左侧声子谱 x 轴只保留高对称点处的竖线，不再显示刻度短线；高对称点标签仍会显示。
- `--system` / `--system-format chem|raw` / `--system-fontsize` / `--system-loc` / `--system-bbox x,y`：在左侧面板标注体系（`--system-bbox` 用轴坐标精确定位）。

# Perturbo 绘图（perturbo_plot）

说明：meanfp 系列脚本读取 Perturbo 的 `*_meanfp.yml`。

通用参数要点：
- `--x energy|e_minus_mu`：横轴用 $E$ 或 $E-\mu$（默认 `e_minus_mu`）
- `--mode scatter|binned`：散点或分箱曲线
- `--bands "1-6"` / `"1,3,5"`：选带（YAML 内的 band index）
- `--xlim a,b`、`--ylim a,b`
- `--sci-y auto|on|off`：纵轴科学计数法“×10^n”

## 1) `perturbo_plot/plot_group_velocity.py`

用途：画电子群速度 $|v|$。

示例（多文件对比 + 散点）：

```bash
python perturbo_plot/plot_group_velocity.py \
  zr2sc_meanfp.yml zr2sc_defect_meanfp.yml \
  --labels pristine,defect \
  --bands 1-6 \
  --x e_minus_mu \
  --mode scatter \
  --alpha 0.15 --s 4 \
  --xlim -1,1 \
  --out v_scatter.png
```

示例（分箱曲线）：

```bash
python perturbo_plot/plot_group_velocity.py \
  zr2sc_meanfp.yml \
  --mode binned \
  --bin-width 0.02 \
  --reducer median \
  --out v_binned.png
```

---

## 2) `perturbo_plot/plot_scattering_rate.py`

用途：画电子散射率 $1/\tau$（由 `relaxation time` 取倒数得到）。

示例：

```bash
python perturbo_plot/plot_scattering_rate.py \
  zr2sc_meanfp.yml \
  --config 1 \
  --unit ps^-1 \
  --mode scatter \
  --ylog \
  --xlim -1,1 \
  --out rate.png
```

---

## 3) `perturbo_plot/plot_mean_free_path.py`

用途：画电子平均自由程（MFP，nm）。

示例：

```bash
python perturbo_plot/plot_mean_free_path.py \
  zr2sc_meanfp.yml \
  --config 1 \
  --mode scatter \
  --ylog \
  --xlim -1,1 \
  --out mfp.png
```

---

## 4) `perturbo_plot/plot_kappa_vs_temperature.py`

用途：在一张图里同时画：
- `κ_latt(T)`：来自 ShengBTE `BTE.KappaTensorVsT_CONV`
- `κ_el(T)`：来自 Perturbo `*_trans-ita.yml`
- `κ_total(T)=κ_latt+κ_el`

重要约束：温度网格必须完全一致；脚本不会插值，不一致会直接报错。

单组数据示例：

```bash
python perturbo_plot/plot_kappa_vs_temperature.py \
  --set Zr2SC BTE.KappaTensorVsT_CONV zr2sc_trans-ita.yml \
  --component avg \
  --out kappa_vs_T.png
```

多组对比示例（重复 `--set`）：

```bash
python perturbo_plot/plot_kappa_vs_temperature.py \
  --set pristine BTE.KappaTensorVsT_CONV zr2sc_trans-ita.yml \
  --set defect   defect/BTE.KappaTensorVsT_CONV defect/zr2sc_trans-ita.yml \
  --component xx \
  --xlim 200,800 \
  --out kappa_compare.png
```

---

## 5) `perturbo_plot/perturbo_meanfp_io.py`

这是 meanfp 系列脚本共用的 I/O 与绘图工具模块（读取 YAML、分箱统计、统一粗体/科学计数法样式等），一般不需要直接运行。

---

# ShengBTE 绘图（shengbte_plot）

## 1) `shengbte_plot/plot_phonon_mfp.py`

用途：从 ShengBTE 输出画声子 MFP 散点图。

默认物理单位假设：
- `BTE.omega`：`rad/ps`
- `BTE.v`：`km/s`（可用 `--v-unit m/s`）
- `BTE.w_*`：`ps^-1`

则 $\mathrm{MFP} (\mathrm{nm}) = |v|/w$。

示例（同一套 `BTE.omega` + `BTE.v`，对比多种散射率文件）：

```bash
python shengbte_plot/plot_phonon_mfp.py \
  --omega BTE.omega \
  --v BTE.v \
  BTE.w_isotopic T300K/BTE.w_final \
  --labels isotopic,3ph \
  --ylog \
  --xlim 0,25 \
  --out phonon_mfp.png
```

常用参数：
- `--x omega|wfile`：x 轴频率取 `BTE.omega` 或取每个 w 文件的第一列
- `--xlog/--ylog`、`--xlim/--ylim`、`--alpha/--s`

---

## 2) `shengbte_plot/plot_phonon_v.py`

用途：画声子群速度 $|v|$ vs 频率散点图。

这是一个“配置区写死参数”的脚本（没有命令行参数）：
- 默认读 `BTE.omega` 与 `BTE.v`
- 输出 `phonon_velocity_scatter.png`
- 脚本顶部的“配置区域”可改：输入文件名、点大小、透明度、颜色等

运行：

```bash
python shengbte_plot/plot_phonon_v.py
```

服务器上不想弹窗：把脚本末尾的 `plt.show()` 保持注释状态（当前脚本默认就是注释）。

---

## 3) `shengbte_plot/plot_lattice_scatter.py`

用途：画晶格散射率散点图（支持把多个 `BTE.w_*` 文件画在同一张图里，并给不同 legend）。

同样是“配置区写死参数”的脚本（没有命令行参数）：
- 在脚本顶部 `files_and_labels = {...}` 里添加/修改文件路径与图例名
- 可以在顶部设置：`use_log_scale_x/y`、`ylim`、点大小与透明度等

运行：

```bash
python shengbte_plot/plot_lattice_scatter.py
```

注意：该脚本默认会 `plt.show()`，若在无显示环境运行：
- 设 `export MPLBACKEND=Agg` 或
- 注释掉 `plt.show()`
