# =============================================================================
# 第一部分：导入必要的库
# =============================================================================
import numpy as np                              # 数值计算库，用于数据处理
import torch                                    # PyTorch深度学习框架
import torch.nn as nn                           # 神经网络模块，包含LSTM等层
import matplotlib.pyplot as plt                 # 绘图库，用于结果可视化
from sklearn.preprocessing import MinMaxScaler  # 数据归一化工具
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # 评估指标

# -----------------------------------------------------------------------------
# 设置matplotlib中文显示
# -----------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# =============================================================================
# 第二部分：数据准备 —— 自主生成模拟房价时间序列数据
# =============================================================================

def generate_house_price_data(n_samples=1000):
    """
    生成模拟房价数据（仿照股票生成逻辑，符合真实房价趋势）
    
    房价通常包含：
    1. 长期上涨趋势（城市化、通胀）
    2. 周期性波动（房地产周期）
    3. 随机波动（市场情绪）
    
    参数：
        n_samples: 时间步数量
    
    返回：
        price: 房价序列
    """
    # 时间轴
    t = np.linspace(0, 20, n_samples)
    
    # 房价生成公式（完全仿照股票格式）
    # 基础价 + 逐年上涨 + 周期波动 + 随机噪声
    price = 200 + 8*t + 6*np.sin(2 * np.pi * t / 2.0) + 3*np.random.randn(n_samples)
    
    # 确保房价不为负
    price = np.maximum(price, 100)
    
    return price

# 生成1000个时间步的模拟房价数据
prices = generate_house_price_data(1000)
print(f"生成房价数据形状: {prices.shape}")

# -----------------------------------------------------------------------------
# 数据归一化
# -----------------------------------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
print(f"归一化后数据范围: [{prices_scaled.min():.4f}, {prices_scaled.max():.4f}]")

# -----------------------------------------------------------------------------
# 滑动窗口构建监督学习数据集
# -----------------------------------------------------------------------------
def create_dataset(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# 用过去25个时间步的房价，预测下一个时间步的房价
SEQ_LENGTH = 25
X, y = create_dataset(prices_scaled, SEQ_LENGTH)

print(f"总样本数: {len(X)}")
print(f"输入X形状: {X.shape}")
print(f"输出y形状: {y.shape}")

# -----------------------------------------------------------------------------
# 划分训练集 / 测试集（时间序列不能打乱）
# -----------------------------------------------------------------------------
train_size = int(len(X) * 0.8)

X_train = torch.FloatTensor(X[:train_size])
y_train = torch.FloatTensor(y[:train_size])
X_test = torch.FloatTensor(X[train_size:])
y_test = torch.FloatTensor(y[train_size:])

print(f"\n数据集划分:")
print(f"  训练集: {X_train.shape}")
print(f"  测试集: {X_test.shape}")

# =============================================================================
# 第三部分：LSTM模型定义
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output

# 实例化模型
model = LSTMModel(
    input_size=1,
    hidden_size=64,
    num_layers=2,
    output_size=1
)

print(f"\n模型结构:")
print(model)
print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")

# =============================================================================
# 第四部分：模型训练
# =============================================================================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 32
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

num_epochs = 100
train_losses = []

print(f"\n开始训练房价预测模型.")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

print("训练完成。")

# =============================================================================
# 第五部分：模型预测
# =============================================================================
model.eval()
with torch.no_grad():
    train_pred = model(X_train)
    test_pred = model(X_test)

# 反归一化
train_pred = scaler.inverse_transform(train_pred.numpy())
y_train_actual = scaler.inverse_transform(y_train.numpy())
test_pred = scaler.inverse_transform(test_pred.numpy())
y_test_actual = scaler.inverse_transform(y_test.numpy())

print(f"\n房价预测完成。")

# =============================================================================
# 第六部分：模型评估与可视化（6张图，完全仿照股票版）
# =============================================================================
mse = mean_squared_error(y_test_actual, test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, test_pred)
r2 = r2_score(y_test_actual, test_pred)

# =============================================================================
# 第七部分：可视化 —— 核心配色修改部分（哥伦比亚比娅配色）
# 配色方案参考：
# 深紫: #77035D (R:119, G:41, B:93)
# 玫红: #C3489A (R:195, G:79, B:162)
# 纯白: #FFFFFF (R:237, G:241, B:253)
# 天蓝: #71C4E0 (R:113, G:196, B:224)
# 深蓝: #538BB8 (R:83, G:139, B:184)
# 对比色: #F5B841 (用于平均误差线)
# =============================================================================

# 定义配色常量
COLOR_DEEP_PURPLE = '#77035D'      # 深紫（标题/边框）
COLOR_PINK = '#C3489A'           # 玫红（主要数据）
COLOR_WHITE = '#FFFFFF'         # 纯白（背景）
COLOR_SKY_BLUE = '#71C4E0'      # 天蓝（次要数据）
COLOR_NAVY = '#538BB8'         # 深蓝（辅助数据）
COLOR_ACCENT = '#F5B841'      # 亮黄（平均误差线）

# 创建画布
fig = plt.figure(figsize=(20, 12))

# 子图1：原始房价数据
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(prices, color=COLOR_DEEP_PURPLE, linewidth=1.5, alpha=0.8, label='原始房价')
ax1.set_title('原始房价数据', fontsize=12, fontweight='bold', color=COLOR_DEEP_PURPLE)
ax1.set_xlabel('时间步', color=COLOR_DEEP_PURPLE)
ax1.set_ylabel('房价（万元）', color=COLOR_DEEP_PURPLE)
ax1.grid(True, alpha=0.3)
# 分割线
ax1.axvline(x=len(prices)*0.8, color=COLOR_PINK, linestyle='--', linewidth=2, label='训练/测试分割点')
ax1.legend(labelcolor=COLOR_DEEP_PURPLE)

# 子图2：训练损失曲线
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(train_losses, color=COLOR_NAVY, linewidth=2, label='训练损失')
ax2.fill_between(range(len(train_losses)), train_losses, alpha=0.3, color=COLOR_SKY_BLUE)
ax2.set_title('训练损失曲线', fontsize=12, fontweight='bold', color=COLOR_DEEP_PURPLE)
ax2.set_xlabel('Epoch', color=COLOR_DEEP_PURPLE)
ax2.set_ylabel('MSE Loss', color=COLOR_DEEP_PURPLE)
ax2.grid(True, alpha=0.3)

# 子图3：训练集房价预测对比
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(y_train_actual, color=COLOR_DEEP_PURPLE, linewidth=1.5, alpha=0.7, label='真实房价')
ax3.plot(train_pred, color=COLOR_PINK, linestyle='--', linewidth=1.5, alpha=0.7, label='预测房价')
ax3.set_title('训练集房价预测对比', fontsize=12, fontweight='bold', color=COLOR_DEEP_PURPLE)
ax3.set_xlabel('样本', color=COLOR_DEEP_PURPLE)
ax3.set_ylabel('房价', color=COLOR_DEEP_PURPLE)
ax3.legend(labelcolor=COLOR_DEEP_PURPLE)
ax3.grid(True, alpha=0.3)

# 子图4：测试集房价预测对比
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(y_test_actual, color=COLOR_DEEP_PURPLE, linewidth=1.5, label='真实房价')
ax4.plot(test_pred, color=COLOR_PINK, linestyle='--', linewidth=1.5, label='预测房价')
ax4.set_title('测试集房价预测对比', fontsize=12, fontweight='bold', color=COLOR_DEEP_PURPLE)
ax4.set_xlabel('样本', color=COLOR_DEEP_PURPLE)
ax4.set_ylabel('房价', color=COLOR_DEEP_PURPLE)
ax4.legend(labelcolor=COLOR_DEEP_PURPLE)
ax4.grid(True, alpha=0.3)

# 子图5：预测误差分布
ax5 = fig.add_subplot(2, 3, 5)
test_errors = y_test_actual.flatten() - test_pred.flatten()
ax5.hist(test_errors, bins=30, color=COLOR_SKY_BLUE, edgecolor=COLOR_DEEP_PURPLE, alpha=0.7)
ax5.axvline(0, color=COLOR_PINK, linestyle='--', linewidth=2, label="零误差")
mean_error = np.mean(test_errors)
ax5.axvline(mean_error, color=COLOR_ACCENT, linewidth=2, label=f'平均误差: {mean_error:.2f}')
ax5.set_title('预测误差分布', fontsize=12, fontweight='bold', color=COLOR_DEEP_PURPLE)
ax5.set_xlabel('误差', color=COLOR_DEEP_PURPLE)
ax5.set_ylabel('频次', color=COLOR_DEEP_PURPLE)
ax5.legend(labelcolor=COLOR_DEEP_PURPLE)
ax5.grid(True, alpha=0.3)

# 子图6：真实值 vs 预测值散点图
ax6 = fig.add_subplot(2, 3, 6)
ax6.scatter(y_test_actual.flatten(), test_pred.flatten(), alpha=0.6, c=COLOR_NAVY, s=15)
# 理想预测线
min_val = min(y_test_actual.min(), test_pred.min())
max_val = max(y_test_actual.max(), test_pred.max())
ax6.plot([min_val, max_val], [min_val, max_val], color=COLOR_PINK, linestyle='--', linewidth=2, label='理想预测线')
ax6.set_title('真实房价 vs 预测房价', fontsize=12, fontweight='bold', color=COLOR_DEEP_PURPLE)
ax6.set_xlabel('真实房价', color=COLOR_DEEP_PURPLE)
ax6.set_ylabel('预测房价', color=COLOR_DEEP_PURPLE)
ax6.legend(labelcolor=COLOR_DEEP_PURPLE)
ax6.grid(True, alpha=0.3)

# 总标题（包含评估指标）
fig.suptitle(
    f'LSTM房价预测结果 | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}',
    fontsize=14, fontweight='bold', color=COLOR_DEEP_PURPLE, y=1.02
)

plt.tight_layout()
plt.savefig('房价预测结果_哥伦比亚比娅版.png', dpi=150, bbox_inches='tight', facecolor=COLOR_WHITE)
plt.show()

# -----------------------------------------------------------------------------
# 输出评估报告
# -----------------------------------------------------------------------------
print(f"\n{'='*60}")
print("                    房价预测模型评估报告")
print(f"{'='*60}")
print(f"【误差指标】")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f} 万元")
print(f"  MAE:  {mae:.4f} 万元")
print(f"\n【拟合优度】")
print(f"  R²:    {r2:.4f}")
print(f"{'='*60}")