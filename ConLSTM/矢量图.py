import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 全局样式设置（学术论文规范+清新字体）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.facecolor'] = 'white'

fig = plt.figure(figsize=(18, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()  # 隐藏坐标轴

# -------------------------- 国风青色调色板（清新活泼） --------------------------
color_input = '#F4FDFF'       # 天缥（Input，最浅）
color_conv1 = '#C4E7E9'       # 影青（ConvLSTM1）
color_conv2 = '#86C0CA'       # 西子（ConvLSTM2）
color_conv3 = '#A0D6B4'       # 沧浪（ConvLSTM3）
color_conv4 = '#75C2C3'       # 松石（ConvLSTM4）
color_pred = '#00A9BF'        # 碧青（Prediction，醒目）
line_color = '#2B6968'        # 空青（线条/箭头统一色）

# -------------------------- 长方体绘制函数（修复ndim报错） --------------------------
def draw_box(x0, y0, z0, dx, dy, dz, color, alpha=0.8, edgecolor=line_color, linewidth=1.5):
    # 1. 顶点全部转为numpy数组，彻底解决ndim报错
    x = np.array([x0, x0+dx, x0+dx, x0, x0, x0+dx, x0+dx, x0])
    y = np.array([y0, y0, y0+dy, y0+dy, y0, y0, y0+dy, y0+dy])
    z = np.array([z0, z0, z0, z0, z0+dz, z0+dz, z0+dz, z0+dz])
    
    # 2. 绘制12条棱
    edges = np.array([[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]])
    for edge in edges:
        ax.plot3D(x[edge], y[edge], z[edge], color=edgecolor, linewidth=linewidth)
    
    # 3. 绘制顶面：全部转为numpy数组，修复list无ndim的报错
    xx = np.array([[x0, x0+dx], [x0, x0+dx]])
    yy = np.array([[y0, y0], [y0+dy, y0+dy]])
    zz = np.array([[z0+dz, z0+dz], [z0+dz, z0+dz]])
    ax.plot_surface(xx, yy, zz, color=color, alpha=alpha, linewidth=0, antialiased=True)

    # 4. 内部特征小矩形
    cx0, cy0 = x0 + dx*0.2, y0 + dy*0.2
    cdx, cdy = dx*0.6, dy*0.6
    ax.plot3D(
        np.array([cx0, cx0+cdx, cx0+cdx, cx0, cx0]),
        np.array([cy0, cy0, cy0+cdy, cy0+cdy, cy0]),
        np.array([z0+dz+0.01]*5), 
        color=edgecolor, linewidth=1.2
    )

# -------------------------- 通用箭头绘制函数（无重复参数） --------------------------
def draw_arrow(x_start, y_start, z_start, x_end, y_end, z_end, color=line_color, linewidth=2, alpha=0.8):
    # 绘制箭头线
    ax.plot3D([x_start, x_end], [y_start, y_end], [z_start, z_end], 
              color=color, linewidth=linewidth, alpha=alpha)
    # 绘制箭头头部
    ax.quiver(x_end, y_end, z_end, 
              x_end - x_start, y_end - y_start, z_end - z_start,
              color=color, linewidth=linewidth, length=0.15, 
              arrow_length_ratio=0.3, normalize=True)

# -------------------------- 层内时间循环箭头 --------------------------
def draw_loop(x, y, z, radius=0.3, color=line_color, linewidth=1.5):
    theta = np.linspace(0, 2*np.pi, 50)
    ax.plot3D(x + radius*np.cos(theta), y + radius*np.sin(theta), [z]*50, 
              color=color, linewidth=linewidth)
    ax.quiver(x + radius*np.cos(np.pi/4), y + radius*np.sin(np.pi/4), z,
              -np.sin(np.pi/4), np.cos(np.pi/4), 0, color=color, linewidth=linewidth, 
              length=0.12, arrow_length_ratio=0.3)

# -------------------------- 时间步点线 --------------------------
def draw_dots(x0, y0, z0, dx, dy, num=12):
    x = np.linspace(x0, x0+dx, num)
    y = np.linspace(y0, y0+dy, num)
    ax.scatter(x, y, [z0]*num, color=line_color, s=3, alpha=0.6, edgecolors='none')

# -------------------------- 1. 绘制编码网络 --------------------------
# Input
draw_box(x0=0, y0=0, z0=0, dx=3, dy=2, dz=0.1, color=color_input)
ax.text(1.5, -1.0, 0.05, 'Input', fontsize=16, ha='center', weight='medium')

# ConvLSTM1
draw_box(x0=0.2, y0=0.2, z0=0.4, dx=2.6, dy=1.6, dz=0.1, color=color_conv1)
ax.text(1.5, -1.0, 0.45, r'$ConvLSTM_1$', fontsize=16, ha='center', weight='medium')

# ConvLSTM2
draw_box(x0=0.4, y0=0.4, z0=0.8, dx=2.2, dy=1.2, dz=0.1, color=color_conv2)
ax.text(1.5, -1.0, 0.85, r'$ConvLSTM_2$', fontsize=16, ha='center', weight='medium')

# 编码网络标题
ax.text(1.5, 1.2, 0.95, 'Encoding Network', fontsize=18, ha='center', weight='bold', color=line_color)

# -------------------------- 2. 编码网络内部箭头（Input→ConvLSTM1→ConvLSTM2） --------------------------
# 修复重复参数问题，每个参数只写一次
draw_arrow(x_start=1.5, y_start=0, z_start=0.1, x_end=1.5, y_end=0.2, z_end=0.4, color=line_color)
draw_arrow(x_start=1.5, y_start=0.2, z_start=0.5, x_end=1.5, y_end=0.4, z_end=0.8, color=line_color)

# -------------------------- 3. 绘制解码网络 --------------------------
# ConvLSTM3
draw_box(x0=6.0, y0=0.2, z0=0.4, dx=2.6, dy=1.6, dz=0.1, color=color_conv3)
ax.text(7.3, -1.0, 0.45, r'$ConvLSTM_3$', fontsize=16, ha='center', weight='medium')

# ConvLSTM4
draw_box(x0=6.2, y0=0.4, z0=0.8, dx=2.2, dy=1.2, dz=0.1, color=color_conv4)
ax.text(7.3, -1.0, 0.85, r'$ConvLSTM_4$', fontsize=16, ha='center', weight='medium')

# Prediction
draw_box(x0=6.4, y0=0.6, z0=1.2, dx=2.0, dy=1.0, dz=0.1, color=color_pred, edgecolor='#006D8F', linewidth=2)
ax.text(7.4, 1.8, 1.25, 'Prediction', fontsize=16, ha='center', color='#006D8F', weight='bold')

# 解码网络标题
ax.text(7.3, 1.2, 0.95, 'Forecasting Network', fontsize=18, ha='center', weight='bold', color=line_color)

# -------------------------- 4. 解码网络内部箭头（ConvLSTM3→ConvLSTM4→Prediction） --------------------------
draw_arrow(x_start=7.3, y_start=0.2, z_start=0.5, x_end=7.3, y_end=0.4, z_end=0.8, color=line_color)
draw_arrow(x_start=7.3, y_start=0.4, z_start=0.9, x_end=7.4, y_end=0.6, z_end=1.2, color='#00A9BF', linewidth=2.5)

# -------------------------- 5. 编码→解码的Copy箭头 --------------------------
# ConvLSTM1 → ConvLSTM3
draw_arrow(x_start=2.8, y_start=1, z_start=0.45, x_end=6.0, y_end=1, z_end=0.45, color=line_color, linewidth=2)
ax.text(4.4, 1.1, 0.45, 'Copy', fontsize=14, ha='center', weight='medium', color=line_color)

# ConvLSTM2 → ConvLSTM4
draw_arrow(x_start=2.6, y_start=1, z_start=0.85, x_end=6.2, y_end=1, z_end=0.85, color=line_color, linewidth=2)
ax.text(4.4, 1.1, 0.85, 'Copy', fontsize=14, ha='center', weight='medium', color=line_color)

# -------------------------- 6. 层内时间循环箭头 --------------------------
# 编码网络循环
draw_loop(x=2.8, y=1, z=0.45)
draw_loop(x=2.6, y=1, z=0.85)

# 解码网络循环
draw_loop(x=8.6, y=1, z=0.45)
draw_loop(x=8.4, y=1, z=0.85)

# -------------------------- 7. 时间步点线 --------------------------
# 编码网络点线
draw_dots(0, 0, 0.05, 3, 2)
draw_dots(0.2, 0.2, 0.45, 2.6, 1.6)
draw_dots(0.4, 0.4, 0.85, 2.2, 1.2)

# 解码网络点线
draw_dots(6.0, 0.2, 0.45, 2.6, 1.6)
draw_dots(6.2, 0.4, 0.85, 2.2, 1.2)
draw_dots(6.4, 0.6, 1.25, 2.0, 1.0)

# -------------------------- 视角调整 --------------------------
ax.view_init(elev=22, azim=-65)
ax.set_box_aspect([2.5, 1, 0.5])
plt.tight_layout()
plt.show()