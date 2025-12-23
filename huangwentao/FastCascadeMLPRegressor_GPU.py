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

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 针对单卡优化CUDA配置
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # 设置GPU内存使用限制
    torch.cuda.set_per_process_memory_fraction(0.9)  # 使用90%的GPU内存
start_time = time.time()

# 加载数据集
print("加载数据...")
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')

columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 划分训练集和测试集
X_train = train_dataSet[noise_columns]
y_train = train_dataSet[columns]
X_test = test_dataSet[noise_columns]
y_test = test_dataSet[columns]

# 数据标准化
print("数据标准化...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)

# 转换为PyTorch张量并预加载到GPU
print("数据加载到GPU...")
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device, non_blocking=True)
y_train_tensor = torch.FloatTensor(y_train_scaled).to(device, non_blocking=True)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device, non_blocking=True)

# 使用最大可能的批处理大小 - 864,000行数据，12GB GPU内存
batch_size = 131072  # 大幅增加批处理大小
print(f"使用批处理大小: {batch_size}")

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          pin_memory=False, num_workers=0)


# 优化的MLP模型 - 充分利用GPU内存
class HighCapacityMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(HighCapacityMLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Linear(64, output_size)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


# 高效级联MLP类 - 平衡速度与性能
class OptimizedCascadeMLP:
    def __init__(self, input_size, output_size=1, max_layers=2,
                 learning_rate=0.001, max_epochs=40):
        self.input_size = input_size
        self.output_size = output_size
        self.max_layers = max_layers
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.layers = []
        self.criterion = nn.MSELoss()

    def fit(self, train_loader, X, y, target_idx=0):
        current_features = X
        input_dim = self.input_size

        for layer in range(self.max_layers):
            print(f"训练第 {layer + 1} 层...")

            # 创建模型
            model = HighCapacityMLP(
                input_size=input_dim,
                output_size=self.output_size
            ).to(device)

            # 优化器 - 使用AdamW并提高学习率
            optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

            # 使用余弦退火学习率调度
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)

            # 训练模型
            model.train()
            for epoch in range(self.max_epochs):
                epoch_loss = 0.0
                batch_count = 0

                for batch_X, batch_y in train_loader:
                    # 为当前层准备输入特征
                    if layer > 0:
                        with torch.no_grad():
                            features = batch_X
                            for prev_model in self.layers:
                                pred = prev_model(features)
                                features = torch.cat([batch_X, pred], dim=1)
                            batch_X_current = features
                    else:
                        batch_X_current = batch_X

                    optimizer.zero_grad(set_to_none=True)

                    outputs = model(batch_X_current)
                    # 选择对应的输出维度
                    batch_y_single = batch_y[:, target_idx].unsqueeze(1) if self.output_size == 1 else batch_y
                    loss = self.criterion(outputs, batch_y_single)
                    loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1

                # 更新学习率
                scheduler.step()

                # 每10个epoch打印一次
                if (epoch + 1) % 10 == 0:
                    avg_loss = epoch_loss / batch_count
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"层 {layer + 1}, 轮次 {epoch + 1}, 平均损失: {avg_loss:.6f}, 学习率: {current_lr:.6f}")

            # 保存当前层
            self.layers.append(model)

            # 更新下一层的输入维度
            input_dim = self.input_size + self.output_size * (layer + 1)

        return self

    def predict(self, X):
        current_features = X
        y_pred = None

        with torch.no_grad():
            for i, model in enumerate(self.layers):
                y_pred = model(current_features)

                # 如果是最后一层，返回结果
                if i == len(self.layers) - 1:
                    return y_pred

                # 为下一层准备特征
                current_features = torch.cat([X, y_pred], dim=1)

        return y_pred if y_pred is not None else torch.zeros((X.shape[0], self.output_size)).to(device)


# 快速训练函数
def train_fast_sequential():
    print("开始快速顺序训练...")
    y_predict_scaled = np.zeros_like(y_test.values)

    for i, column in enumerate(columns):
        print(f"\n训练第 {i + 1}/{len(columns)} 个输出变量: {column}")
        print("=" * 50)

        # 创建并训练级联MLP模型
        cascade_model = OptimizedCascadeMLP(
            input_size=X_train_tensor.shape[1],
            output_size=1,
            max_layers=2,
            learning_rate=0.001,
            max_epochs=40  # 适当增加轮数但使用更好的优化策略
        )

        # 训练模型
        cascade_model.fit(train_loader, X_train_tensor, y_train_tensor, target_idx=i)

        # 预测
        print(f"进行预测...")
        y_pred_single = cascade_model.predict(X_test_tensor)
        y_predict_scaled[:, i] = y_pred_single.cpu().numpy().flatten()

        # 计算当前变量的R²分数
        y_test_single = y_test[column].values
        y_pred_single_original = scaler_y.inverse_transform(
            np.column_stack([y_predict_scaled[:, j] if j == i else np.zeros_like(y_predict_scaled[:, i])
                             for j in range(len(columns))])
        )[:, i]

        r2 = r2_score(y_test_single, y_pred_single_original)
        print(f"{column} 的 R² 分数: {r2:.4f}")

        # 清理模型释放GPU内存
        del cascade_model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # 打印进度和内存使用情况
        elapsed_time = time.time() - start_time
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            print(f"进度: {i + 1}/{len(columns)}，耗时: {elapsed_time / 60:.2f} 分钟, GPU内存: {allocated:.2f}GB")
        else:
            print(f"进度: {i + 1}/{len(columns)}，耗时: {elapsed_time / 60:.2f} 分钟")

    return y_predict_scaled


# 主训练流程
print(
    f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}GB" if device.type == 'cuda' else "使用CPU")

# 使用快速顺序训练
y_predict_scaled = train_fast_sequential()

# 反标准化预测结果
y_predict = scaler_y.inverse_transform(y_predict_scaled)

# 保存预测结果
print("\n保存结果...")
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_Optimized_CascadeMLP.csv", index=False)
print("结果已保存到: result_Optimized_CascadeMLP.csv")

# 计算平均误差
data = pd.read_csv("result_Optimized_CascadeMLP.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("\n各变量的平均误差：")
for i, column in enumerate(columns):
    print(f"{column}: {means[i]:.6f}")
print(f"总平均误差: {means.mean():.6f}")

# 计算总体R²分数
overall_r2 = r2_score(y_test, y_predict)
print(f"总体 R² 分数: {overall_r2:.4f}")

end_time = time.time()
total_time = end_time - start_time
print(f"\n总耗时：{total_time / 60:.2f} 分钟")

# 最终内存清理
if device.type == 'cuda':
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    print(f"训练完成后GPU内存使用: {allocated:.2f}GB")