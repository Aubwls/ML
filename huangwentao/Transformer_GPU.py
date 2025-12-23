# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： 03-main.py
# 编码实现 Coding by： Hongjie Liu , Suiwen Zhang 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest University of Nationality，or Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师
# coding=utf-8
import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

start_time = time.time()

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 加载数据集时指定数据类型，加速读取
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')

# columns表示原始列，noise_columns表示添加噪声的列
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 划分训练集和测试集
X_train = train_dataSet[noise_columns].values
y_train = train_dataSet[columns].values
X_test = test_dataSet[noise_columns].values
y_test = test_dataSet[columns].values

# 数据标准化
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.fit_transform(y_train)

# 转换为PyTorch张量并移动到设备，使用float32减少内存占用
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)

# 创建数据加载器，增大batch_size加速训练
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # 增大batch_size


# 简化Transformer模型 - 减小模型复杂度
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=32, nhead=2, num_layers=1):  # 减小模型参数
        super(SimpleTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
            dropout=0.1  # 添加dropout防止过拟合
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加序列维度
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # 取最后一个时间步
        return self.output_layer(x)


# 初始化模型并移动到设备
model = SimpleTransformer(
    input_dim=X_train.shape[1],
    output_dim=y_train.shape[1]
).to(device)

# 训练配置
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 减少训练轮数
num_epochs = 20  # 从50减少到20
print("开始训练Transformer模型...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:  # 每5轮输出一次
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}')

print("训练完成！")

# 预测
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).cpu().numpy()

# 反标准化
y_predict = y_scaler.inverse_transform(y_pred_scaled)

# 计算并保存结果 - 优化保存过程
results = []
for i in range(len(y_test)):
    true_val = y_test[i]
    pred_val = y_predict[i]
    error = np.abs(true_val - pred_val)
    results.append([
        ' '.join(f"{x:.6f}" for x in true_val),
        ' '.join(f"{x:.6f}" for x in pred_val),
        ' '.join(f"{x:.6f}" for x in error)
    ])

# 保存结果
result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_Transformer_GPU.csv", index=False)

# 计算平均误差
error_data = pd.read_csv("result_Transformer_GPU.csv")
column3 = error_data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("6个数据的平均值为：\n", means)
print(f"总体平均误差: {means.mean():.6f}")

end_time = time.time()
total_time = end_time - start_time
print(f"总耗时：{total_time:.3f}秒")
