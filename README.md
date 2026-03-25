# science_plot_script

这是一组用于科研绘图的 Python 脚本集合，覆盖：
- Quantum ESPRESSO：能带、DOS/PDOS（projwfc）以及“能带 + PDOS”合图
- Perturbo：meanfp（群速度、散射率、电子 MFP）与电子 κ(T)
- ShengBTE：声子群速度/散射率/MFP 散点、\$\\kappa(T)\$ 张量、累计 \$\\kappa(\\omega)\$ 张量

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
  --legend Pristine \
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

多体系对比示例（多组总 DOS + PDOS 叠加；每组可有自己的 `--fermi`/`--norm`/PDOS glob；曲线图例会自动变成 `legend:Total` 与 `legend:El-orb`）：

```bash
python qe_plot/plot_qe_dos_pdos_overlay.py \
  --tot prim/zr2sc.pdos.pdos_tot sc222/zr2sc.pdos.pdos_tot \
  --pdos-glob 'prim/zr2sc.pdos.pdos_atm#*' 'sc222/zr2sc.pdos.pdos_atm#*' \
  --legend Pristine 2x2x2 \
  --elements Zr,S,C \
  --orbitals s,p,d \
  --merge-wfc \
  --fermi 14.776,14.812 \
  --norm 1,4 \
  --fermi-line \
  --system Zr2SC \
  --system-fontsize 12 \
  --system-bbox 0,0.75 \
  --xlim -5,5 \
  --out dos_pdos_compare.png
```

只画总 DOS（tot-only）示例：当显式传了 `--elements` 但内容为空白时，会自动关闭 PDOS，只绘制总 DOS：

```bash
python qe_plot/plot_qe_dos_pdos_overlay.py \
  --tot zr2sc.pdos.pdos_tot \
  --legend Pristine \
  --elements ' ' \
  --fermi 14.776 \
  --fermi-line \
  --system Zr2SC \
  --out dos_only.png
```

只画“按元素求和的 PDOS”（不分轨道）示例：当显式传了 `--orbitals` 但内容为空白时，会进入“element-sum mode”，即：
- 仍然会读取 PDOS 文件
- 但只按元素把所有轨道/投影求和后画一条曲线（每个元素 1 条）

```bash
python qe_plot/plot_qe_dos_pdos_overlay.py \
  --tot zr2sc.pdos.pdos_tot \
  --legend Pristine \
  --elements Zr,S,C \
  --orbitals ' ' \
  --merge-wfc \
  --fermi 14.776 \
  --fermi-line \
  --system Zr2SC \
  --out dos_pdos_element_sum.png
```

只画 d 的各个分量（不画 total DOS，也不画 d 的总和曲线）示例：
- `--orbitals d,no-tot`：只保留 d 轨道，并关闭 total DOS
- `--pdos-components d=all`：把 d 的 5 个分量都画出来
- `--pdos-components-only`：只画分量，不画 d 的总和（ldos）

```bash
python qe_plot/plot_qe_dos_pdos_overlay.py \
  --tot zr2sc.pdos.pdos_tot \
  --legend Pristine \
  --elements Zr,S,C \
  --orbitals d,no-tot \
  --pdos-components 'd=all' \
  --pdos-components-only \
  --merge-wfc \
  --fermi 14.776 \
  --fermi-line \
  --system Zr2SC \
  --out dos_pdos_d_components_only.png
```

常用参数：
- `--pdos`：显式给出 PDOS 文件列表（不常用，文件太多时可用）
- `--pdos-glob`：自定义 glob 模式（覆盖自动推断）
- `--merge-wfc`：将同一元素同一轨道的不同 `wfc#` 合并为一条曲线
- `--n0 'Zr=4,S=3,C=2'`：不合并 wfc 时，把 `wfc#` 重新标成“主量子数标签”（如 Zr-4d、Zr-5d…）
- `--tot-col` / `--pdos-col`：指定读哪一列（0-based；默认都取第 2 列即 `col=1`）
- `--pdos-components`：选择并绘制指定轨道的分量（m-components），如 `p=all;d=1,3,5`
- `--pdos-components-only`：配合 `--pdos-components` 使用，只画分量，不画该轨道的总和（ldos）
- `--legend`：每组数据的前缀标签（支持 1 个值广播到所有体系，或给 N 个值对应 N 组）
- `--norm`：对每组 DOS/PDOS 的 y 值做归一化（除以 `norm`；同样支持广播/逐组）
- `--style prb|default`、`--figsize 3.4,2.6`、`--ylog`、`--sci-y auto|on|off`
- `--label-fontsize`：x/y 轴标签字号
- `--linestyle` / `--linewidth`：按规则为不同曲线指定线型/线宽（支持 `tot`、`pdos`、`<orb>`、`<el>-<orb>`、以及 `1:`/`2:` 这类“第 N 组数据覆盖规则”；当 `--orbitals` 为空白时也可用 `zr=--` 这种“按元素”规则）
- `--system` / `--system-format chem|raw` / `--system-fontsize` / `--system-loc` / `--system-bbox x,y`：在图内标注一个全局体系名称（`--system-bbox` 用轴坐标 0~1 精确定位）

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
  --legend Pristine \
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
  --legend Pristine 2x2x2 \
  --fermi 14.776,14.812 \
  --norm 1,1 \
  --fermi-line \
  --ylim -5,5 \
  --system Zr2SC \
  --system-fontsize 14 \
  --system-bbox 0,0.75 \
  --lw 0.6 \
  --out bands_compare.png
```

说明：多体系叠加要求“高对称点标签序列一致”（例如都是 `Γ-X-M-Γ` 且断点位置一致）；否则脚本会报错提示。

说明：
- 当 `band.in` 中某行 `N=1` 时，该位置视作不连续点：刻度标签会被合并成 `A|L`（终点|起点），且能带不会跨越该点连线。
- x 轴只保留高对称点处的竖线（`axvline`），不再显示刻度短线；高对称点标签仍会显示。
- 多体系叠加时用 `--legend` 区分每组数据；`--system` 只用于图内的一个全局标注。
- 可用 `--system` 在图内标注体系名称（默认会把化学式中的数字渲染为下标，如 `Zr2SC → Zr$_{2}$SC`）；若不想下标，用 `--system-format raw`。
- 体系标注位置：用 `--system-loc` 选大致方位；如需精确摆放（或放到图外），用 `--system-bbox x,y`（轴坐标系 0~1），例如 `--system-bbox 1.02,1.0`。
- `--fermi`/`--norm` 支持逐组设置：能量轴会先做费米能级平移 $E\rightarrow E-E_f$，再做归一化 $E\rightarrow (E-E_f)/\text{norm}$（未提供 `--norm` 时等价于 1）。

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
  --legend Pristine \
  --elements Zr,S,C \
  --orbitals s,p,d \
  --merge-wfc \
  --fermi 14.776 \
  --fermi-line \
  --ylim -5,5 \
  --figsize-bands 7,3 \
  --figsize-dos 1.5,3 \
  --pdos-legend-fontsize 7 \
  --pdos-legend-loc best \
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
  --legend Pristine 2x2x2 \
  --elements Zr,S,C \
  --orbitals s,p,d \
  --merge-wfc \
  --fermi 14.776,14.812 \
  --norm 1,4 \
  --fermi-line \
  --ylim -5,5 \
  --figsize-bands 7,3 \
  --figsize-dos 1.8,3 \
  --pdos-legend-fontsize 7 \
  --pdos-legend-loc best \
  --system Zr2SC \
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
- `--pdos-components`：选择并绘制指定轨道的分量（m-components），如 `p=all;d=1,3,5`
- `--legend`：每组数据的前缀标签（多体系时用它来区分；支持广播/逐组）
- `--pdos-legend-loc` / `--pdos-legend-fontsize`：右侧 (P)DOS 面板的曲线图例位置/字号
- `--norm`：对每组 DOS/PDOS 的 x 值做归一化（除以 `norm`；同样支持广播/逐组）
- `--label-fontsize`：坐标轴标签字号（作用于能带与 DOS 面板）
- `--system` / `--system-format chem|raw` / `--system-fontsize` / `--system-loc` / `--system-bbox x,y`：在左侧能带面板标注一个全局体系名称（`--system-bbox` 用轴坐标精确定位）

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
- `--figsize w,h`：图尺寸（英寸），如 `--figsize 7.2,4.6`
- `--sci-y auto|on|off`：纵轴科学计数法“×10^n”
- `--style prb|default`：默认 `prb`；若安装了 SciencePlots 会启用 `science,no-latex` 风格
- `--legend ...`：多组数据对比时的数据集图例（支持 1 个值广播或逐组）
- `--system ...`：全局体系标注（独立于数据集图例，纯文字、粗体）

## 1) `perturbo_plot/plot_group_velocity.py`

用途：画电子群速度 $|v|$。

示例（多文件对比 + 散点）：

```bash
python perturbo_plot/plot_group_velocity.py \
  zr2sc_meanfp.yml zr2sc_defect_meanfp.yml \
  --legend pristine defect \
  --system Zr2SC \
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
  --legend pristine \
  --system Zr2SC \
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
  --legend pristine \
  --system Zr2SC \
  --config 1 \
  --mode scatter \
  --ylog \
  --xlim -1,1 \
  --out mfp.png
```

---

## 4) `perturbo_plot/plot_kappa_vs_temperature.py`

用途：绘制 Perturbo 电子热导 `κ_el(T)`（从 `*_trans-ita.yml` 读取），支持多文件对比；`--component` 可一次给多个分量并画在同一张图上。

补充：多文件对比时也支持按文件分别设置 `--color` / `--marker` / `--ms`（支持 1 个值广播或逐文件指定）；线宽用 `--lw` 全局控制。

单组数据示例：

```bash
python perturbo_plot/plot_kappa_vs_temperature.py \
  zr2sc_trans-ita.yml \
  --legend pristine \
  --out kappa_el_vs_T.png
```

多组对比示例（多个文件 + `--legend`）：

```bash
python perturbo_plot/plot_kappa_vs_temperature.py \
  zr2sc_trans-ita.yml defect/zr2sc_trans-ita.yml \
  --legend pristine defect \
  --component xx \
  --xlim 200,800 \
  --out kappa_el_compare.png
```

---

## 5) `perturbo_plot/plot_sigma_kappa_vs_temperature.py`

用途：从 `*_trans-ita.yml` 同时绘制电导率 $\sigma(T)$ 与电子热导率 $\kappa(T)$（双 y 轴：左 $\sigma$，右 $\kappa$）。

补充：该脚本对样式参数采用“每个文件两条线”的输入方式：
- `--ls/--color/--marker/--ms/--lw` 需要给 `文件数×2`（或给 2 个值广播为“σ/κ”，或给 1 个值广播到所有线）
- 顺序固定为：`σ(file1), κ(file1), σ(file2), κ(file2), ...`

也可分别用 `--ylim-sigma` / `--ylim-kappa` 设置左右 y 轴范围。

示例（单文件，默认 `--component avg`）：

```bash
python perturbo_plot/plot_sigma_kappa_vs_temperature.py \
  zr2sc_trans-ita.yml \
  --legend pristine \
  --system Zr2SC \
  --out sigma_kappa_vs_T.png
```

示例（多文件对比 + 选择分量 + 为 σ/κ 分别指定样式）：

```bash
python perturbo_plot/plot_sigma_kappa_vs_temperature.py \
  zr2sc_trans-ita.yml defect/zr2sc_trans-ita.yml \
  --legend pristine defect \
  --component avg \
  --ls - -- -- - -- \
  --color tab:red tab:red tab:blue tab:blue \
  --marker o none s none \
  --lw 2 2 2 2 \
  --xlim 200,800 \
  --out sigma_kappa_compare.png
```

---

## 6) `perturbo_plot/plot_lorenz_vs_temperature.py`

用途：从 `*_trans-ita.yml` 计算并绘制洛伦兹数

$$
L(T)=\frac{\kappa(T)}{\sigma(T)\,T}
$$

并用一条横向虚线标出理论值 $L_0=2.44\times10^{-8}\;\mathrm{W\Omega/K^2}$。

示例：

```bash
python perturbo_plot/plot_lorenz_vs_temperature.py \
  zr2sc_trans-ita.yml \
  --legend pristine \
  --component avg \
  --out lorenz_vs_T.png
```

---

## 7) `perturbo_plot/plot_sigma_kappa_lorenz_vs_temperature.py`

用途：从 `*_trans-ita.yml` 在一张图里同时绘制：
- 电导率 $\sigma(T)$（左 y 轴）
- 电子热导率 $\kappa_{el}(T)$（右 y 轴）
- 洛伦兹数 $L(T)=\kappa/(\sigma T)$（第二个右 y 轴，会向右偏移一段距离）

补充：
- 该脚本使用 `--color-sigma/--color-kappa/--color-lorenz` 分别控制三类曲线颜色；`--color`（逐文件）已废弃，会直接报错提示。
- `--ls-kappa` 默认是 `dashed`（因为命令行里裸的 `--` 会被 argparse 当作“选项结束”）。

示例（单文件，默认 `--component avg`）：

```bash
python perturbo_plot/plot_sigma_kappa_lorenz_vs_temperature.py \
  zr2sc_trans-ita.yml \
  --legend pristine \
  --system Zr2SC \
  --label-fontsize 11 \
  --out sigma_kappa_lorenz_vs_T.png
```

示例（多文件对比 + 调整右侧两根 y 轴间距）：

```bash
python perturbo_plot/plot_sigma_kappa_lorenz_vs_temperature.py \
  pristine/zr2sc_trans-ita.yml defect/zr2sc_trans-ita.yml \
  --legend pristine defect \
  --component xx \
  --right-axis-spacing 0.08 \
  --xlim 200,800 \
  --out sigma_kappa_lorenz_compare.png
```

---

## 8) `perturbo_plot/perturbo_meanfp_io.py`

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

用途：绘制 ShengBTE 散射率（或任意两列数据）的散点分布，并支持多文件叠加对比。

默认物理单位假设：
- 输入文件第 1 列是频率 $\omega$（`rad/ps`），脚本会自动换算为 `THz`：$\omega_{\mathrm{THz}}=\omega/(2\pi)$
- 输入文件第 2 列是散射率 $w$（`ps^-1`）

单文件示例：

```bash
python shengbte_plot/plot_lattice_scatter.py \
  --file T300K/BTE.w_final \
  --legend 3ph \
  --system Zr2SC \
  --ylog \
  --ylim 1e-3,2e2 \
  --figsize 7,5 \
  --out scattering.png
```

多文件对比示例（原胞 vs 缺陷）：

```bash
python shengbte_plot/plot_lattice_scatter.py \
  --file \
    T300K/BTE.w_final \
    ../../../../Zr2SC_defect_Zr/shengbte/Zr_3.0/20-20/T300K/BTE.w_final \
    ../../../../Zr2SC_defect_S/shengbte/S_3.0/20-20/T300K/BTE.w_final \
  --legend Pristine 'V[Zr]=3.0%' 'V[S]=3.0%' \
  --legend-fontsize 10 \
  --legend-bbox 0.18,0.95 \
  --system Zr2SC \
  --ms 10 --alpha 0.4 \
  --ylog \
  --ylim 3e-3,3e4 \
  --figsize 7,5 \
  --out scattering_compare.png
```

图例/标注参数约定（与 QE 脚本统一风格）：
- 数据集图例（带彩色点标识）：`--legend/--legend-format/--legend-fontsize/--legend-loc/--legend-bbox`
- 整体体系标注（纯文字，粗体）：`--system/--system-format/--system-fontsize/--system-loc/--system-bbox`

说明：
- `--xlim`/`--xlabel` 的单位是 `THz`（因为脚本会把输入 x 自动换算到 THz）
- 对数坐标下会自动丢弃非正值点，并给出 warning

---

## 4) `shengbte_plot/plot_kappa_tensor_vs_temperature.py`

用途：绘制 ShengBTE 的晶格热导率张量随温度变化曲线（来自 `BTE.KappaTensorVsT_CONV`），并支持多组数据叠加对比。

数据结构约定：
- 第 1 列：温度 $T$（K）
- 后 9 列：$\kappa$ 张量分量，顺序为 `xx,xy,xz,yx,yy,yz,zx,zy,zz`
- 最后一列：不使用（脚本会忽略）

单文件示例（默认 `--component avg`，即 $(\kappa_{xx}+\kappa_{yy}+\kappa_{zz})/3$）：

```bash
python shengbte_plot/plot_kappa_tensor_vs_temperature.py \
  --file BTE.KappaTensorVsT_CONV \
  --legend pristine \
  --system Zr2SC \
  --component avg \
  --xlim 200,800 \
  --out kappa_vs_T.png
```

多组对比示例：

```bash
python shengbte_plot/plot_kappa_tensor_vs_temperature.py \
  --file \
    prim/BTE.KappaTensorVsT_CONV \
    zr_vac/BTE.KappaTensorVsT_CONV \
    s_vac/BTE.KappaTensorVsT_CONV \
  --legend Pristine 'V[Zr]' 'V[S]' \
  --legend-bbox 0.18,0.95 \
  --legend-fontsize 10 \
  --system Zr2SC \
  --system-bbox 0,0.75 \
  --component xx \
  --xlim 200,800 \
  --out kappa_compare.png
```

常用参数：
- `--component xx|yy|zz|xy|...|avg|trace`：选择绘制哪个分量（`avg` 为对角平均）
- `--color ...` / `--marker ...` / `--ms ...`：为每组数据指定颜色、点样式（marker）与点大小（marker size；`--ms` 支持逐组给值或给 1 个值广播），并在 legend 中同步显示
- 数据集图例（彩色线段）：`--legend/--legend-format/--legend-fontsize/--legend-loc/--legend-bbox`
- 整体体系标注（纯文字，粗体）：`--system/--system-format/--system-fontsize/--system-loc/--system-bbox`

示例（显式指定每组颜色与 marker）：

```bash
python shengbte_plot/plot_kappa_tensor_vs_temperature.py \
  --file prim/BTE.KappaTensorVsT_CONV zr_vac/BTE.KappaTensorVsT_CONV s_vac/BTE.KappaTensorVsT_CONV \
  --legend Pristine 'V[Zr]' 'V[S]' \
  --color black tab:red tab:blue \
  --marker o s ^ \
  --ms 4.5 6 6 \
  --component avg \
  --xlim 200,700 \
  --out kappa_compare_marker.png
```

---

## 5) `shengbte_plot/plot_cumulative_kappa_vs_omega_tensor.py`

用途：绘制 ShengBTE 的“随频率累计的晶格热导率张量”曲线（来自 `BTE.cumulative_kappaVsOmega_tensor`），并支持多组数据叠加对比。

数据结构约定：
- 第 1 列：声子角频率 \$\\omega\$（`rad/ps`），脚本会自动换算为 `THz`：\$f_{\\mathrm{THz}}=\\omega/(2\\pi)\$
- 后 9 列：累计热导率张量分量，顺序为 `xx,xy,xz,yx,yy,yz,zx,zy,zz`

单文件示例（默认 `--component avg`，即对角平均）：

```bash
python shengbte_plot/plot_cumulative_kappa_vs_omega_tensor.py \
  --file T300K/BTE.cumulative_kappaVsOmega_tensor \
  --legend T300K \
  --system Zr2SC \
  --component avg \
  --xlim 0,25 \
  --out cumulative_kappa_vs_freq.png
```

多组对比示例（原胞 vs 缺陷；每组可指定颜色）：

```bash
python shengbte_plot/plot_cumulative_kappa_vs_omega_tensor.py \
  --file \
    prim/T300K/BTE.cumulative_kappaVsOmega_tensor \
    zr_vac/T300K/BTE.cumulative_kappaVsOmega_tensor \
    s_vac/T300K/BTE.cumulative_kappaVsOmega_tensor \
  --legend Pristine 'V[Zr]' 'V[S]' \
  --color black tab:red tab:blue \
  --system Zr2SC \
  --system-bbox 0,0.75 \
  --component xx \
  --xlim 0,25 \
  --out cumulative_kappa_compare.png
```

常用参数：
- `--component xx|yy|zz|xy|...|avg|trace`：选择绘制哪个分量（`avg` 为对角平均）
- `--color ...`：逐组指定线条颜色（支持逐组给值或给 1 个值广播）
- 数据集图例（彩色线段）：`--legend/--legend-format/--legend-fontsize/--legend-loc/--legend-bbox`
- 整体体系标注（纯文字，粗体）：`--system/--system-format/--system-fontsize/--system-loc/--system-bbox`
