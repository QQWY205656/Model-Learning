import matplotlib.pyplot as plt
import numpy as np

# ===================== 全局样式设置（论文级规范）=====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'none'

# 创建画布
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(-1, 15)
ax.set_ylim(-1, 10)
ax.axis('off')

# ===================== 清新青色调色板（和原图一致）=====================
colors = {
    "input": "#F4FDFF",       # Input（最浅）
    "conv1": "#C4E7E9",       # ConvLSTM1（浅青）
    "conv2": "#86C0CA",       # ConvLSTM2（深青）
    "conv3": "#A0D6B4",       # ConvLSTM3（暖青）
    "conv4": "#75C2C3",       # ConvLSTM4（深青）
    "pred": "#00A9BF",        # Prediction（醒目碧青）
    "line": "#2B6968",        # 线条/箭头统一色
    "edge": "#2B6968"         # 边框色
}

# ===================== 核心函数：高真实感立体长方体 =====================
def draw_3d_box(x, y, w, h, d, color, label, label_y_offset=-0.8):
    """
    绘制更真实的立体长方体
    x,y: 长方体正面左下角坐标
    w: 正面宽度, h: 正面高度, d: 深度（立体效果）
    """
    # 1. 定义长方体8个顶点（更自然的透视）
    skew = d * 0.6  # 透视倾斜度
    vertices = np.array([
        # 正面（z=0）
        [x, y], [x+w, y], [x+w, y+h], [x, y+h],
        # 背面（z=d）
        [x+skew, y+skew], [x+w+skew, y+skew], 
        [x+w+skew, y+h+skew], [x+skew, y+h+skew]
    ])
    
    # 2. 绘制3个可见面（正面、右侧面、顶面），增强立体感
    # 正面
    ax.fill(vertices[[0,1,2,3],0], vertices[[0,1,2,3],1], 
            color=color, alpha=0.9, edgecolor=colors["edge"], linewidth=1.5)
    # 右侧面
    ax.fill(vertices[[1,5,6,2],0], vertices[[1,5,6,2],1], 
            color=color, alpha=0.7, edgecolor=colors["edge"], linewidth=1.5)
    # 顶面
    ax.fill(vertices[[3,2,6,7],0], vertices[[3,2,6,7],1], 
            color=color, alpha=0.8, edgecolor=colors["edge"], linewidth=1.5)
    
    # 3. 绘制内部特征小矩形（还原原图细节）
    cx, cy = x + w*0.2, y + h*0.3
    cw, ch = w*0.6, h*0.4
    ax.plot([cx, cx+cw, cx+cw, cx, cx],
            [cy, cy, cy+ch, cy+ch, cy],
            color=colors["edge"], linewidth=1.2)
    # 小矩形的立体透视
    ax.plot([cx+skew, cx+cw+skew, cx+cw+skew, cx+skew, cx+skew],
            [cy+skew, cy+skew, cy+ch+skew, cy+ch+skew, cy+skew],
            color=colors["edge"], linewidth=1.2, alpha=0.7)
    
    # 4. 文字标注（远离图形，无遮挡）
    ax.text(x + w/2, y + label_y_offset, label, 
            fontsize=14, ha='center', va='top', 
            weight='medium', color=colors["line"])

# ===================== 箭头绘制函数（干净实线，无圆圈）=====================
def draw_arrow(x1, y1, x2, y2, color=colors["line"], lw=2, arrowstyle='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=arrowstyle, color=color, lw=lw))

# ===================== 层内时间循环箭头 =====================
def draw_loop(x, y, color=colors["line"]):
    # 绘制环形箭头，更精致
    theta = np.linspace(0, 2*np.pi, 50)
    r = 0.3
    ax.plot(x + r*np.cos(theta), y + r*np.sin(theta), 
            color=color, linewidth=1.5)
    # 箭头头部
    ax.arrow(x + r*np.cos(np.pi/4), y + r*np.sin(np.pi/4),
             -r*np.sin(np.pi/4)*0.3, r*np.cos(np.pi/4)*0.3,
             head_width=0.1, color=color)

# ===================== 1. 绘制编码网络（左侧）=====================
ax.text(2, 9.2, "Encoding Network", 
        fontsize=16, ha='center', weight='bold', color=colors["line"])

# Input（最上层）
draw_3d_box(x=0.5, y=7, w=3, h=1.5, d=0.5, 
            color=colors["input"], label="Input", label_y_offset=-0.8)

# ConvLSTM1（中间层）
draw_3d_box(x=0.5, y=4.8, w=3, h=1.5, d=0.5, 
            color=colors["conv1"], label="$ConvLSTM_1$", label_y_offset=-0.8)
draw_loop(x=3.5, y=5.55)  # 层内循环箭头

# ConvLSTM2（最下层）
draw_3d_box(x=0.5, y=2.6, w=3, h=1.5, d=0.5, 
            color=colors["conv2"], label="$ConvLSTM_2$", label_y_offset=-0.8)
draw_loop(x=3.5, y=3.35)  # 层内循环箭头

# ===================== 2. 绘制解码网络（右侧）=====================
ax.text(10, 9.2, "Forecasting Network", 
        fontsize=16, ha='center', weight='bold', color=colors["line"])

# ConvLSTM3（上层）
draw_3d_box(x=7, y=4.8, w=3, h=1.5, d=0.5, 
            color=colors["conv3"], label="$ConvLSTM_3$", label_y_offset=-0.8)
draw_loop(x=10.5, y=5.55)  # 层内循环箭头

# ConvLSTM4（中层）
draw_3d_box(x=7, y=2.6, w=3, h=1.5, d=0.5, 
            color=colors["conv4"], label="$ConvLSTM_4$", label_y_offset=-0.8)
draw_loop(x=10.5, y=3.35)  # 层内循环箭头

# Prediction（最下层，醒目）
draw_3d_box(x=7.5, y=0.4, w=2, h=1.2, d=0.4, 
            color=colors["pred"], label="Prediction", label_y_offset=-0.8)

# ===================== 3. 全链路箭头（删除圆圈，干净实线）=====================
# 编码网络内部箭头（Input→Conv1→Conv2）
draw_arrow(x1=2, y1=7, x2=2, y2=5.5)    # Input → ConvLSTM1
draw_arrow(x1=2, y1=4.8, x2=2, y2=3.3)  # ConvLSTM1 → ConvLSTM2

# Copy箭头（编码→解码，删除圆圈，纯实线）
draw_arrow(x1=3.5, y1=5.55, x2=7, y2=5.55, lw=2)  # Conv1 → Conv3
ax.text(5.2, 5.7, "Copy", fontsize=12, ha='center', weight='bold', color=colors["line"])

draw_arrow(x1=3.5, y1=3.35, x2=7, y2=3.35, lw=2)  # Conv2 → Conv4
ax.text(5.2, 3.5, "Copy", fontsize=12, ha='center', weight='bold', color=colors["line"])

# 解码网络内部箭头（Conv3→Conv4→Prediction）
draw_arrow(x1=8.5, y1=4.8, x2=8.5, y2=3.3)  # ConvLSTM3 → ConvLSTM4
draw_arrow(x1=8.5, y1=2.6, x2=8.5, y2=1.6)  # ConvLSTM4 → Prediction

# ===================== 最终布局优化 =====================
plt.tight_layout()
plt.show()