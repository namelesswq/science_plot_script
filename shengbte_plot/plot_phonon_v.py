import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================
frequency_file = "BTE.omega"  # 频率文件
velocity_file = "BTE.v"       # 速度文件
output_image_name = "phonon_velocity_scatter.png"

# 单位转换: rad/ps -> THz
# 1 THz = 2*pi rad/ps
to_THz = 1.0 / (2 * np.pi)

# 绘图样式
point_size = 8          # 点的大小
point_alpha = 0.4       # 透明度 (0-1)，设低一点可以看到点的密度
point_color = 'blue'    # 颜色
# ===========================================

def plot_group_velocity():
    print("正在处理数据...")
    
    # 1. 读取频率 (Matrix: n_qpoints x n_bands)
    try:
        omega_matrix = np.loadtxt(frequency_file)
        # 【关键修正】: 将矩阵展平为一维数组，使其变成 [mode1, mode2, mode3, ...]
        omega_flat = omega_matrix.flatten() 
        
        # 转换为 THz
        x_data = omega_flat * to_THz
        print(f"频率数据读取成功: 原始形状 {omega_matrix.shape} -> 展平后 {x_data.shape}")
        
    except Exception as e:
        print(f"[错误] 读取 {frequency_file} 失败: {e}")
        return

    # 2. 读取速度 (List: n_modes x 3)
    try:
        v_vectors = np.loadtxt(velocity_file)
        # 计算速度大小 |v|
        y_data = np.linalg.norm(v_vectors, axis=1)
        print(f"速度数据读取成功: 形状 {y_data.shape}")
        
    except Exception as e:
        print(f"[错误] 读取 {velocity_file} 失败: {e}")
        return

    # 3. 二次检查数据长度
    if len(x_data) != len(y_data):
        print(f"[严重错误] 数据长度依然不匹配！")
        print(f"频率点数: {len(x_data)}")
        print(f"速度点数: {len(y_data)}")
        # 尝试通过截断来强制画图（防止脚本崩溃，但通常不建议）
        min_len = min(len(x_data), len(y_data))
        x_data = x_data[:min_len]
        y_data = y_data[:min_len]
        print(f"已自动截断数据至 {min_len} 个点以进行绘图。")

    # 4. 绘图
    plt.figure(figsize=(10, 7), dpi=120)
    
    plt.scatter(x_data, y_data, 
                s=point_size, 
                alpha=point_alpha, 
                color=point_color, 
                edgecolors='none')

    # 美化
    plt.title("Phonon Group Velocity vs. Frequency", fontsize=16, fontweight='bold')
    plt.xlabel("Frequency (THz)", fontsize=14)
    plt.ylabel("Group Velocity (km/s)", fontsize=14) # 注意确认你的BTE.v单位，通常是km/s
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 这里的 ylim 可以把负值去掉（有些虚频可能会导致负数或者计算误差）
    plt.ylim(bottom=0)
    plt.xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_image_name, dpi=300)
    print(f"绘图完成！已保存为 {output_image_name}")
    # plt.show() # 如果在服务器上运行，请注释掉这一行

if __name__ == "__main__":
    plot_group_velocity()