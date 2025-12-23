# coding=utf-8
import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim

start_time = time.time()

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载数据集
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')

columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

print("数据加载完成...")

# 数据预处理
scaler = StandardScaler()
X_train = scaler.fit_transform(train_dataSet[noise_columns])
y_train = train_dataSet[columns].values
X_test = scaler.transform(test_dataSet[noise_columns])
y_test = test_dataSet[columns].values

# 转换为PyTorch张量并移动到GPU
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)

print("开始训练简化但有效的MLP和随机森林融合模型...")


# 简化的PyTorch MLP模型 - 专注于有效性
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)


# 初始化MLP模型
input_size = X_train.shape[1]
output_size = y_train.shape[1]
mlp_model = MLP(input_size, output_size).to(device)

# 简化的训练配置
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# 训练MLP模型
print("训练PyTorch MLP...")
mlp_model.train()
epochs = 150
batch_size = 2048

for epoch in range(epochs):
    epoch_loss = 0
    batch_count = 0

    # 手动批处理
    for i in range(0, len(X_train_tensor), batch_size):
        end_idx = min(i + batch_size, len(X_train_tensor))
        batch_X = X_train_tensor[i:end_idx]
        batch_y = y_train_tensor[i:end_idx]

        optimizer.zero_grad()
        outputs = mlp_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    if (epoch + 1) % 30 == 0:
        avg_loss = epoch_loss / batch_count
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

# 预测
mlp_model.eval()
with torch.no_grad():
    mlp_pred_list = []
    for i in range(0, len(X_test_tensor), batch_size):
        end_idx = min(i + batch_size, len(X_test_tensor))
        batch = X_test_tensor[i:end_idx]
        mlp_pred_list.append(mlp_model(batch).cpu().numpy())

    mlp_pred = np.vstack(mlp_pred_list)

# 优化的随机森林
print("训练随机森林...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=14,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# 基于每个目标变量的性能调整权重
with torch.no_grad():
    mlp_train_pred = mlp_model(X_train_tensor).cpu().numpy()

# 计算每个模型在每个目标变量上的MSE
rf_mse = []
mlp_mse = []
for i in range(len(columns)):
    rf_mse.append(mean_squared_error(y_train[:, i], rf_model.predict(X_train)[:, i]))
    mlp_mse.append(mean_squared_error(y_train[:, i], mlp_train_pred[:, i]))

# 基于MSE计算权重（MSE越小权重越大）
rf_weights = 1 / np.array(rf_mse)
mlp_weights = 1 / np.array(mlp_mse)

# 归一化权重
total_weights = rf_weights + mlp_weights
rf_weights /= total_weights
mlp_weights /= total_weights

print("模型权重分配:")
for i, col in enumerate(columns):
    print(f"  {col}: RF={rf_weights[i]:.3f}, MLP={mlp_weights[i]:.3f}")

# 应用加权融合
y_predict = np.zeros_like(rf_pred)
for i in range(len(columns)):
    y_predict[:, i] = rf_weights[i] * rf_pred[:, i] + mlp_weights[i] * mlp_pred[:, i]

# 保存结果
results = []
for True_Value, Predicted_Value in zip(y_test, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_MLP_RF_Simple_Effective.csv", index=False)

print("<*>" * 50)

# 评估结果
data = pd.read_csv("result_MLP_RF_Simple_Effective.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("6个目标变量的平均误差为：")
for i, col in enumerate(columns):
    print(f"  {col}: {means[i]:.6f}")
print(f"总体平均误差: {means.mean():.6f}")

r2 = r2_score(y_test, y_predict)
print(f"R² Score: {r2:.6f}")

mse = mean_squared_error(y_test, y_predict)
print(f"MSE: {mse:.6f}")

end_time = time.time()
print(f"总耗时：{end_time - start_time : .3f}秒")