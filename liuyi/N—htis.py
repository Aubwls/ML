# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： 03-main.py
# 编码实现 Coding by： Hongjie Liu , Suiwen Zhang 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest  University of Nationality，or Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师
# coding=utf-8
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

start_time = time.time()

# 加载数据集
print("加载数据集...")
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')

columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 准备数据
X_train = train_dataSet[noise_columns].values
y_train = train_dataSet[columns].values
X_test = test_dataSet[noise_columns].values
y_test = test_dataSet[columns].values

print(f"训练集形状: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"测试集形状: X_test: {X_test.shape}, y_test: {y_test.shape}")

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)  # 添加这行，使用transform而不是fit_transform


# 创建序列数据
def create_sequences(X, y, sequence_length=24):
    """创建时间序列数据"""
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:(i + sequence_length)])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)


# 序列参数
SEQUENCE_LENGTH = 24  # 使用24个时间步预测下一个时间步
BATCH_SIZE = 64

# 创建序列数据集
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH)  # 现在y_test_scaled已定义

print(f"序列数据形状 - 训练集: {X_train_seq.shape}, 测试集: {X_test_seq.shape}")

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
y_test_tensor = torch.FloatTensor(y_test_seq).to(device)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 定义简化的N-HiTS模型
class SimplifiedNHITS(nn.Module):
    """简化版N-HiTS模型"""

    def __init__(self, input_size, output_size, sequence_length, hidden_size=256, num_blocks=4):
        super(SimplifiedNHITS, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks

        # 输入投影层
        self.input_projection = nn.Linear(sequence_length * input_size, hidden_size)

        # 多个预测块
        self.blocks = nn.ModuleList([
            self._create_block(hidden_size) for _ in range(num_blocks)
        ])

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * num_blocks, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )

    def _create_block(self, hidden_dim):
        """创建单个预测块"""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)

        # 展平序列
        x_flat = x.reshape(batch_size, -1)

        # 输入投影
        x_proj = self.input_projection(x_flat)

        # 通过各个块
        block_outputs = []
        current_input = x_proj

        for block in self.blocks:
            block_out = block(current_input)
            block_outputs.append(block_out)
            # 残差连接
            current_input = current_input + block_out

        # 合并所有块的输出
        combined = torch.cat(block_outputs, dim=1)

        # 最终输出
        output = self.output_layer(combined)
        return output


# 初始化模型
input_size = X_train_seq.shape[2]  # 特征数量
output_size = y_train_seq.shape[1]  # 目标数量

model = SimplifiedNHITS(
    input_size=input_size,
    output_size=output_size,
    sequence_length=SEQUENCE_LENGTH,
    hidden_size=256,
    num_blocks=4
).to(device)

print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# 训练模型
print("开始训练N-HiTS模型...")
num_epochs = 50
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

    val_loss /= len(test_loader)
    val_losses.append(val_loss)

    # 学习率调度
    scheduler.step(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

# 最终预测
print("进行最终预测...")
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

# 合并预测结果
y_pred_scaled = np.vstack(all_predictions)
y_true_scaled = np.vstack(all_targets)

# 反标准化
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_true_scaled)

# 由于序列数据长度变化，我们需要对齐原始测试集
# 取最后一部分与预测结果对齐
start_idx = SEQUENCE_LENGTH
end_idx = start_idx + len(y_pred)
y_test_aligned = y_test[start_idx:end_idx]

# 保存预测结果
results = []
for True_Value, Predicted_Value in zip(y_test_aligned, y_pred):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_NHITS.csv", index=False)
print("\nN-HiTS结果已保存到: result_NHITS.csv")

# 计算平均误差
data = pd.read_csv("result_NHITS.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("\nN-HiTS模型6个数据的平均值为：")
print(means)
print(f"总平均误差: {means.mean():.6f}")