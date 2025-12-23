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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats

start_time = time.time()

# 加载数据集
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')

columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 划分训练集和测试集
X_train = train_dataSet[noise_columns]
y_train = train_dataSet[columns]
X_test = test_dataSet[noise_columns]
y_test = test_dataSet[columns]

# === 简单优化1: 数据标准化 ===
print("进行数据标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 简单优化2: 优化随机森林参数 ===
rf_params = {
    'n_estimators': 300,           # 增加树的数量
    'max_depth': 20,               # 适度增加深度
    'min_samples_split': 3,        # 更细的分裂条件
    'min_samples_leaf': 1,         # 更小的叶子节点
    'max_features': 0.6,           # 使用60%的特征（比sqrt更多）
    'bootstrap': True,
    'random_state': 217,
    'n_jobs': -1,
    'verbose': 1                   # 显示训练进度
}

# 创建并训练随机森林模型
print("开始训练优化后的随机森林模型...")
rf_model = RandomForestRegressor(**rf_params)
rf_model.fit(X_train_scaled, y_train)  # 使用标准化后的数据

# 模型预测
y_predict = rf_model.predict(X_test_scaled)

# === 简单优化3: 计算更多评估指标 ===
# R²分数
r2_scores = r2_score(y_test, y_predict, multioutput='raw_values')
print("\n各目标变量的R²分数:")
for i, col in enumerate(columns):
    print(f"{col}: {r2_scores[i]:.6f}")
print(f"平均R²分数: {np.mean(r2_scores):.6f}")

# 均方根误差
rmse_scores = np.sqrt(mean_squared_error(y_test, y_predict, multioutput='raw_values'))
print("\n各目标变量的RMSE:")
for i, col in enumerate(columns):
    print(f"{col}: {rmse_scores[i]:.6f}")
print(f"平均RMSE: {np.mean(rmse_scores):.6f}")

# 保存预测结果
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_RF_optimized.csv", index=False)
print("\n优化后的随机森林结果已保存到: result_RF_optimized.csv")

# 计算平均误差
data = pd.read_csv("result_RF_optimized.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("\n优化后随机森林6个数据的平均绝对误差为：")
for i, col in enumerate(columns):
    print(f"{col}: {means[i]:.6f}")
print(f"总平均误差: {means.mean():.6f}")

# === 简单优化4: 特征重要性分析 ===
print("\n特征重要性分析:")
feature_importance = rf_model.feature_importances_
for i, col in enumerate(noise_columns):
    print(f"{col}: {feature_importance[i]:.6f}")

end_time = time.time()
print(f"\n总耗时：{end_time - start_time:.3f}秒")