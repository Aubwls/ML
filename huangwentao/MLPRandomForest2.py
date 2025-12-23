# coding=utf-8
import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
print(f"训练数据形状: {train_dataSet.shape}, 测试数据形状: {test_dataSet.shape}")

# 保存原始训练目标值
y_train_original = train_dataSet[columns].values
y_test_original = test_dataSet[columns].values

# 数据预处理 - 尝试不同的标准化策略
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(train_dataSet[noise_columns])
X_test = scaler_X.transform(test_dataSet[noise_columns])

# 不对y进行标准化，直接使用原始值
y_train = y_train_original
y_test = y_test_original

print("开始训练优化的MLP和随机森林融合模型...")

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)


# 优化的MLP模型 - 稍微增加复杂度但保持稳健
class OptimizedMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(OptimizedMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.network(x)


# 初始化MLP模型
input_size = X_train.shape[1]
output_size = y_train.shape[1]
mlp_model = OptimizedMLP(input_size, output_size).to(device)

# 训练配置 - 使用更精细的参数
criterion = nn.SmoothL1Loss()  # 对异常值更稳健
optimizer = optim.AdamW(mlp_model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# 训练MLP模型
print("训练优化的PyTorch MLP...")
mlp_model.train()
epochs = 120
batch_size = 1024  # 增加批处理大小

best_mlp_loss = float('inf')
patience = 15
patience_counter = 0

for epoch in range(epochs):
    # 随机批处理
    indices = torch.randperm(len(X_train_tensor))
    epoch_loss = 0
    batch_count = 0

    for i in range(0, len(X_train_tensor), batch_size):
        end_idx = min(i + batch_size, len(X_train_tensor))
        batch_indices = indices[i:end_idx]

        batch_X = X_train_tensor[batch_indices]
        batch_y = y_train_tensor[batch_indices]

        optimizer.zero_grad()
        outputs = mlp_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    avg_loss = epoch_loss / batch_count
    scheduler.step()

    # 早停机制
    if avg_loss < best_mlp_loss:
        best_mlp_loss = avg_loss
        patience_counter = 0
        torch.save(mlp_model.state_dict(), 'best_mlp_model.pth')
    else:
        patience_counter += 1

    if (epoch + 1) % 20 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")

    if patience_counter >= patience:
        print(f"早停于第 {epoch + 1} 轮")
        break

# 加载最佳MLP模型
mlp_model.load_state_dict(torch.load('best_mlp_model.pth'))

# MLP预测
mlp_model.eval()
with torch.no_grad():
    mlp_pred_list = []
    inference_batch_size = 8192
    for i in range(0, len(X_test_tensor), inference_batch_size):
        end_idx = min(i + inference_batch_size, len(X_test_tensor))
        batch = X_test_tensor[i:end_idx]
        mlp_pred_list.append(mlp_model(batch).cpu().numpy())

    mlp_pred = np.vstack(mlp_pred_list)

# 优化的随机森林 - 增加复杂度但控制过拟合
print("训练优化的随机森林...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features=0.8,  # 限制特征数量防止过拟合
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)
print(f"随机森林OOB Score: {rf_model.oob_score_:.6f}")

rf_pred = rf_model.predict(X_test)

print("计算精细化的融合权重...")

# 创建验证集
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# 重新训练MLP在训练子集上以获得验证集预测
mlp_val_model = OptimizedMLP(input_size, output_size).to(device)
mlp_val_optimizer = optim.Adam(mlp_val_model.parameters(), lr=0.001)

# 快速训练MLP用于验证
X_train_split_tensor = torch.FloatTensor(X_train_split).to(device)
y_train_split_tensor = torch.FloatTensor(y_train_split).to(device)
X_val_split_tensor = torch.FloatTensor(X_val_split).to(device)

mlp_val_model.train()
for epoch in range(50):
    mlp_val_optimizer.zero_grad()
    outputs = mlp_val_model(X_train_split_tensor)
    loss = criterion(outputs, y_train_split_tensor)
    loss.backward()
    mlp_val_optimizer.step()

# 获取验证集预测
mlp_val_model.eval()
with torch.no_grad():
    mlp_val_pred = mlp_val_model(X_val_split_tensor).cpu().numpy()

# 重新训练RF在训练子集上
rf_val_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_val_model.fit(X_train_split, y_train_split)
rf_val_pred = rf_val_model.predict(X_val_split)

# 计算每个目标变量的最佳权重
rf_weights = []
mlp_weights = []

for i in range(len(columns)):
    rf_mse = mean_squared_error(y_val_split[:, i], rf_val_pred[:, i])
    mlp_mse = mean_squared_error(y_val_split[:, i], mlp_val_pred[:, i])

    # 基于MSE比例分配权重
    total_mse = rf_mse + mlp_mse
    rf_weight = mlp_mse / total_mse  # RF的MSE越小，其权重越大
    mlp_weight = rf_mse / total_mse  # MLP的MSE越小，其权重越大

    rf_weights.append(rf_weight)
    mlp_weights.append(mlp_weight)

print("各目标变量的优化权重分配:")
for i, col in enumerate(columns):
    print(f"  {col}: RF={rf_weights[i]:.3f}, MLP={mlp_weights[i]:.3f}")

# 应用加权融合
y_predict = np.zeros_like(rf_pred)
for i in range(len(columns)):
    y_predict[:, i] = rf_weights[i] * rf_pred[:, i] + mlp_weights[i] * mlp_pred[:, i]


# 后处理：使用移动平均平滑预测结果
def smooth_predictions(predictions, window_size=5):
    """对预测结果进行平滑处理"""
    smoothed = np.zeros_like(predictions)
    for i in range(predictions.shape[1]):
        # 使用移动平均
        df = pd.DataFrame(predictions[:, i])
        smoothed[:, i] = df.rolling(window=window_size, center=True, min_periods=1).mean().values.flatten()
    return smoothed


y_predict_smoothed = smooth_predictions(y_predict)

# 保存结果
results = []
for True_Value, Predicted_Value in zip(y_test_original, y_predict_smoothed):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_MLP_RF_Final.csv", index=False)

print("<*>" * 50)

# 评估结果
data = pd.read_csv("result_MLP_RF_Final.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("6个目标变量的平均误差为：")
for i, col in enumerate(columns):
    print(f"  {col}: {means[i]:.6f}")
print(f"总体平均误差: {means.mean():.6f}")

r2 = r2_score(y_test_original, y_predict_smoothed)
print(f"R² Score: {r2:.6f}")

mse = mean_squared_error(y_test_original, y_predict_smoothed)
print(f"MSE: {mse:.6f}")

# 与原始结果比较
original_error = 0.579386
error_change = (means.mean() - original_error) / original_error * 100
print(f"相对于原始结果的误差变化: {error_change:+.2f}%")

end_time = time.time()
print(f"总耗时：{end_time - start_time : .3f}秒")

# 内存清理
torch.cuda.empty_cache()
