import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================

# 1. 在这里配置每一个文件对应的图例名称
#    格式: "你的文件名.txt": "你想显示的图例名"
files_and_labels = {
    "BTE.w_isotopic": "isotopic",
    "T300K/BTE.w_final": "3ph",
    # "/data/home/wangqian/workplace/Zr2AC/Zr2SC_defect_Zr/shengbte/Zr_0.5/20-20/BTE.w_isotopic": "isotopic(Zr_0.5)",
    # "/data/home/wangqian/workplace/Zr2AC/Zr2SC_defect_Zr/shengbte/Zr_0.5/20-20/T300K/BTE.w_final": "3ph(Zr_0.5)",
    "/data/home/wangqian/workplace/Zr2AC/Zr2SC_defect_S/shengbte/S_3.0/20-20/BTE.w_isotopic": "isotopic(S_3.0)",
    "/data/home/wangqian/workplace/Zr2AC/Zr2SC_defect_S/shengbte/S_3.0/20-20/T300K/BTE.w_final": "3ph(S_3.0)"
    # 你可以在这里继续添加更多的文件...
}

# 2. 图片保存名称
output_image_name = "scattering.png"

# 3. 坐标轴设置
use_log_scale_x = False  # X轴 (通常是频率/能量)
use_log_scale_y = True   # Y轴 (散射率通常跨度大，建议开启)

# 4. 散点设置
point_size = 10          # 点的大小
point_alpha = 0.4        # 点的透明度 (0-1)，重叠时可以看出密度

# 5. 坐标范围
ylim = [1e-5, 1e3]

# 全局粗体设置
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 2  # 把坐标轴线也加粗一点，更有质感

# ===========================================

def plot_scatter_with_custom_legends():
    # 设置画图大小和清晰度
    plt.figure(figsize=(10, 7), dpi=120)
    
    # 获取所有要画的文件列表
    target_files = list(files_and_labels.keys())
    
    # 生成颜色映射，确保不同文件颜色不同
    if len(target_files) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, len(target_files)))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(target_files)))

    files_plotted = 0

    print(f"准备读取 {len(target_files)} 个指定文件...")

    # 遍历字典，同时获取 文件名(filename) 和 图例名(legend_label)
    for i, (filename, legend_label) in enumerate(files_and_labels.items()):
        
        # 检查文件是否存在
        if not os.path.exists(filename):
            print(f"[警告] 找不到文件: {filename} (跳过)")
            continue
            
        try:
            # 读取数据
            data = np.loadtxt(filename)
            
            # 提取 X (第一列) 和 Y (第二列)
            x = data[:, 0]
            y = data[:, 1]
            
            # --- 绘制散点图 ---
            plt.scatter(x, y, 
                        s=point_size,       # 点的大小
                        alpha=point_alpha,  # 透明度
                        label=legend_label, # <--- 关键：这里使用了你配置的图例名
                        color=colors[i],    # 颜色
                        edgecolors='none')  # 去掉点的边框，保持圆点清晰
            
            print(f"成功绘制: {filename} -> 图例: {legend_label}")
            files_plotted += 1
            
        except Exception as e:
            print(f"[错误] 读取文件 {filename} 失败: {e}")

    if files_plotted == 0:
        print("没有绘制任何文件，请检查文件名列表是否正确。")
        return

    # --- 图表美化 ---
    # 使用 fontweight='bold' 确保所有文字加粗
    plt.title("Scattering Rate Distribution", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Frequency (rad/ps)", fontsize=14, fontweight='bold') 
    plt.ylabel("Scattering Rate (ps^-1)", fontsize=14, fontweight='bold')
    
    # 刻度设置：labelsize 控制大小，并通过 list comprehension 手动设置粗体（针对某些 backend）
    plt.tick_params(axis='both', which='major', labelsize=12, width=2)
    
    # 坐标轴对数设置
    if use_log_scale_y:
        plt.yscale('log')
    if use_log_scale_x:
        plt.xscale('log')

    # --- 修改图例：移入图中 ---
    # loc='upper right' 将图例放在右上角，你也可以改为 'upper left' 或 'best'
    # frameon=True 加上边框有助于在散点密集的图中看清图例
    plt.legend(loc='upper right', 
               fontsize=12, 
               markerscale=1.5, 
               frameon=True, 
               edgecolor='black') 
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.ylim(ylim)

    # 强制刷新刻度字体为粗体
    ax = plt.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')

    # 保存并显示
    plt.tight_layout()
    plt.savefig(output_image_name, dpi=300)
    print(f"绘图完成！图片已保存为 {output_image_name}")
    plt.show()

if __name__ == "__main__":
    plot_scatter_with_custom_legends()