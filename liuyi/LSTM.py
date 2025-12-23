import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 设置随机种子以保证结果可复现
torch.manual_seed(217)
np.random.seed(217)

start_time = time.time()

# 设备配置（使用GPU如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 加载数据集 (请根据您的实际文件路径修改)
# 假设您的数据格式与之前一致
train_dataSet = pd.read_csv(
    r'C:\Users\63189\Desktop\ML期末代码比赛\foundation\数据集（含真实值）\modified_数据集Time_Series661.dat')
test_dataSet = pd.read_csv(
    r'C:\Users\63189\Desktop\ML期末代码比赛\foundation\数据集（含真实值）\modified_数据集Time_Series662.dat')

# 定义特征列
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']

# 准备数据
X_train = train_dataSet[noise_columns].values
y_train = train_dataSet[columns].values
X_test = test_dataSet[noise_columns].values
y_test = test_dataSet[columns].values

# 2. 数据标准化 (非常重要!)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)  # 注意：对y_test使用transform而非fit_transform


# 3. 构建时序数据样本 (关键步骤!)
# LSTM需要将输入数据构造成(samples, timesteps, features)的形式。
# 这里我们假设每个样本是一个独立的“时间步”，创建一个简单的序列。
# 如果您的数据本身有时序关系，需要按真实顺序构造序列。
def create_sequences(data, targets, sequence_length=1):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        xs.append(data[i:(i + sequence_length)])
        ys.append(targets[i + sequence_length])  # 预测下一个时间步的目标
    return np.array(xs), np.array(ys)


sequence_length = 3  # 可以调整，表示用过去3个时间点预测下一个
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
y_test_tensor = torch.FloatTensor(y_test_seq).to(device)

# 创建DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 4. 定义LSTM模型
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))

        # 只取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


# 模型参数
input_size = X_train_seq.shape[2]  # 特征数量
hidden_size = 64
num_layers = 2
output_size = y_train_seq.shape[1]  # 目标数量
learning_rate = 0.001
num_epochs = 100  # 可以调整

model = LSTMRegressor(input_size, hidden_size, num_layers, output_size).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 5. 训练模型
model.train()
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 6. 模型预测
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).cpu().numpy()

# 将预测值反标准化回原始范围
y_pred = scaler_y.inverse_transform(y_pred_scaled)
# 对应的真实值也需要是未标准化的，注意y_test_seq是对应X_test_seq的，需要对齐
y_actual = scaler_y.inverse_transform(y_test_seq)

# 7. 评估模型
mse = mean_squared_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)  # 当输出为多变量时，请留意scikit-learn中r2_score的multioutput参数

print('*' * 50)
print(f"LSTM 模型评估:")
print(f'均方误差 (MSE): {mse:.4f}')
print(f'决定系数 (R²): {r2:.4f}')

# 计算每个目标变量的平均绝对误差
mae_per_column = np.mean(np.abs(y_actual - y_pred), axis=0)
print("\n各目标变量的平均绝对误差 (MAE):")
for col, mae_val in zip(columns, mae_per_column):
    print(f"  {col}: {mae_val:.4f}")
print(f"整体平均MAE: {np.mean(mae_per_column):.4f}")

end_time = time.time()
print(f"\n总耗时：{end_time - start_time:.3f} 秒")