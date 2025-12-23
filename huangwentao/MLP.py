# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： 03-main.py
# 编码实现 Coding by： Hongjie Liu , Suiwen Zhang 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest  University of Nationality，or Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师
# coding=utf-8
import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import warnings

warnings.filterwarnings('ignore')

start_time = time.time()

# 加载数据集
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')

columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 划分训练集和测试集
X_train_raw = train_dataSet[noise_columns]
y_train = train_dataSet[columns]
X_test_raw = test_dataSet[noise_columns]
y_test = test_dataSet[columns]

# ==================== 简化但有效的特征工程 ====================
print("开始特征工程...")

# 1. 标准化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)


# 2. 添加关键特征（基于之前分析的重要性）
def add_key_features(X):
    X_df = pd.DataFrame(X, columns=noise_columns)

    # 关键交互特征
    X_df['T_CO2_ratio'] = X_df['Error_T_SONIC'] / (X_df['Error_CO2_density'] + 1e-8)
    X_df['H2O_CO2_signal_ratio'] = X_df['Error_H2O_sig_strgth'] / (X_df['Error_CO2_sig_strgth'] + 1e-8)
    X_df['signal_strength_sum'] = X_df['Error_H2O_sig_strgth'] + X_df['Error_CO2_sig_strgth']
    X_df['signal_strength_diff'] = X_df['Error_H2O_sig_strgth'] - X_df['Error_CO2_sig_strgth']

    # 基本统计特征
    X_df['error_mean'] = X_df.mean(axis=1)
    X_df['error_std'] = X_df.std(axis=1)

    return X_df


X_train = add_key_features(X_train_scaled)
X_test = add_key_features(X_test_scaled)

print(f"特征工程后维度：训练集 {X_train.shape}, 测试集 {X_test.shape}")

# ==================== 数据标准化（输出变量也需要） ====================
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# ==================== MLP模型 ====================
print("开始训练MLP模型...")

# 创建MLP模型 - 针对你的数据特点调整
mlp_model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),  # 三层隐藏层
    activation='relu',  # 激活函数
    solver='adam',  # 优化器
    alpha=0.0001,  # L2正则化
    batch_size=256,  # 批大小
    learning_rate='adaptive',  # 自适应学习率
    learning_rate_init=0.001,
    max_iter=500,  # 最大迭代次数
    early_stopping=True,  # 早停
    validation_fraction=0.1,  # 验证集比例
    n_iter_no_change=20,  # 早停耐心值
    random_state=217,
    verbose=True
)

# 训练模型
mlp_model.fit(X_train, y_train_scaled)

# 预测
y_predict_scaled = mlp_model.predict(X_test)

# 反标准化
y_predict = scaler_y.inverse_transform(y_predict_scaled)

# ==================== 结果保存与评估 ====================
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_MLP.csv", index=False)
print("\nMLP结果已保存到: result_MLP.csv")

# 计算平均误差
data = pd.read_csv("result_MLP.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("\nMLP 6个数据的平均误差为：")
for col, mean in zip(columns, means):
    print(f"{col}: {mean:.6f}")
print(f"总平均误差: {means.mean():.6f}")

# 评估指标
print("\nMLP模型评估指标:")
total_r2 = 0
for i, col in enumerate(columns):
    r2 = r2_score(y_test.iloc[:, i], y_predict[:, i])
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_predict[:, i]))
    mae = np.mean(np.abs(y_test.iloc[:, i] - y_predict[:, i]))
    total_r2 += r2
    print(f"{col}: R² = {r2:.6f}, RMSE = {rmse:.6f}, MAE = {mae:.6f}")

print(f"\n平均R²: {total_r2 / 6:.6f}")

# 训练损失曲线
if hasattr(mlp_model, 'loss_curve_'):
    print(f"\n最终训练损失: {mlp_model.loss_curve_[-1]:.6f}")

end_time = time.time()
print(f"\n总耗时：{end_time - start_time:.3f}秒")

# ==================== 可选的MLP参数优化 ====================
print("\n" + "=" * 50)
print("MLP参数优化建议（如果需要进一步提升）:")
print("1. 调整隐藏层结构: hidden_layer_sizes=(256, 128, 64, 32)")
print("2. 增加批大小: batch_size=512 或 1024")
print("3. 调整学习率: learning_rate_init=0.0005")
print("4. 增加最大迭代次数: max_iter=1000")
print("5. 尝试不同的激活函数: activation='tanh'")
print("6. 调整正则化强度: alpha=0.00001 或 0.001")