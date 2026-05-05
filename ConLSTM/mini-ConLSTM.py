import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """
    ConvLSTM 的核心细胞单元 (Cell)。
    将传统 LSTM 中的全连接层(Linear)替换为卷积层(Conv2d)，
    从而在提取时序特征的同时，保留空间拓扑结构。
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 确保 kernel_size 是元组
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        # 自动计算 padding 以保证输入输出的空间维度一致 (保持 H 和 W 不变)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        # 核心：将输入 x_t 和前一时刻的隐藏状态 h_{t-1} 在通道维度上拼接后，只做一次卷积操作
        # 输出通道数为 4 * hidden_dim，分别对应 i(输入门), f(遗忘门), o(输出门), g(细胞状态候选)
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # 在通道维度 (dim=1) 将输入和前一状态进行拼接
        # input_tensor 形状: [batch_size, input_dim, height, width]
        # h_cur 形状: [batch_size, hidden_dim, height, width]
        combined = torch.cat([input_tensor, h_cur], dim=1)  
        
        # 通过卷积操作计算所有的门控信号
        combined_conv = self.conv(combined)
        
        # 将结果在通道维度上切分为 4 份
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # 门控激活函数
        i = torch.sigmoid(cc_i) # 输入门 (Input gate)
        f = torch.sigmoid(cc_f) # 遗忘门 (Forget gate)
        o = torch.sigmoid(cc_o) # 输出门 (Output gate)
        g = torch.tanh(cc_g)    # 细胞状态候选 (Cell candidate)

        # 细胞状态与隐藏状态更新 (哈达玛乘积)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """初始化隐藏状态和细胞状态为全零张量"""
        height, width = image_size
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class MiniConvLSTM(nn.Module):
    """
    基于 ConvLSTMCell 的编码-预测 (Encoding-Forecasting) 时空序列模型。
    用于模拟论文中预测未来雷达回波图的过程。
    """
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(MiniConvLSTM, self).__init__()
        # 实例化核心细胞
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        
        # 使用 1x1 卷积将高维的隐藏状态映射回原始图像的通道数 (例如：从 16 通道变回单通道灰度图)
        self.out_conv = nn.Conv2d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=1)

    def forward(self, x, future_steps=0):
        """
        x: 历史图像序列，形状为 [batch_size, seq_len, channels, height, width]
        future_steps: 需要往未来预测多少帧
        """
        batch_size, seq_len, _, h, w = x.size()
        
        # 初始化状态
        hidden_state = self.cell.init_hidden(batch_size, (h, w))
        
        # 1. 编码阶段 (Encoder): 逐步读取历史序列，累积时空特征
        for t in range(seq_len):
            input_t = x[:, t, :, :, :]
            h_next, c_next = self.cell(input_t, hidden_state)
            hidden_state = (h_next, c_next)

        outputs = []
        # 2. 预测阶段 (Forecaster): 典型的自回归推演 (Autoregressive)
        # 用最后一帧作为初始输入，结合编码器最后的状态推演未来
        current_input = x[:, -1, :, :, :] 
        
        for _ in range(future_steps):
            h_next, c_next = self.cell(current_input, hidden_state)
            hidden_state = (h_next, c_next)
            
            # 将隐藏状态映射为预测的雷达图/图像帧
            pred_frame = self.out_conv(h_next)
            outputs.append(pred_frame.unsqueeze(1)) # 增加时间维度
            
            # 将当前预测出的帧作为下一时刻的输入
            current_input = pred_frame 

        if future_steps > 0:
            # 沿着序列长度(seq_len)维度拼接预测帧
            return torch.cat(outputs, dim=1) 
        else:
            return h_next


# ==========================================
# 测试与运行用例 (Mock Test)
# ==========================================
if __name__ == "__main__":
    print("开始初始化 Mini ConvLSTM 模型...")
    
    # 假设场景：输入的是单通道灰度雷达图 (channels=1)
    # 隐藏层特征维度设为 16，卷积核大小为 3x3
    model = MiniConvLSTM(input_dim=1, hidden_dim=16, kernel_size=3)
    
    # 创建模拟的输入数据 (模拟过去 5 个时间步的雷达回波图)
    # 维度: [Batch大小, 序列长度(时间步), 通道数, 高度, 宽度]
    batch_size = 2
    past_frames = 5
    channels = 1
    height, width = 64, 64
    
    x = torch.randn(batch_size, past_frames, channels, height, width)
    print(f"输入张量形状 (历史数据): {x.shape}  -> [Batch={batch_size}, Seq={past_frames}, C={channels}, H={height}, W={width}]")
    
    # 我们希望模型根据过去 5 帧，预测未来 3 帧
    future_frames = 3
    predictions = model(x, future_steps=future_frames)
    
    print(f"预测张量形状 (未来数据): {predictions.shape}  -> [Batch={batch_size}, Seq={future_frames}, C={channels}, H={height}, W={width}]")
    print("模型前向传播运行成功！")