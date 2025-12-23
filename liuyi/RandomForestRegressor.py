# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： 03-main.py
# 编码实现 Coding by： Hongjie Liu , Suiwen Zhang 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest  University of Nationality，or Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师
# coding=utf-8
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

start_time = time.time()

# 加载数据集
print("加载数据集...")
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

print(f"训练集形状: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"测试集形状: X_test: {X_test.shape}, y_test: {y_test.shape}")

# 基于原始成功参数进行优化
# 原始参数: n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2
# 误差: 0.503526, 时间: 114.360秒

# 优化策略: 保持相似的模型复杂度但优化训练速度
rf_params = {
    'n_estimators': 350,           # 略微减少树的数量
    'max_depth': 15,               # 略微增加深度以提高精度
    'min_samples_split': 3,        # 减少分割样本数，增加模型灵活性
    'min_samples_leaf': 1,         # 减少叶节点样本数
    'max_features': 0.6,        # 使用sqrt特征数，原始成功设置
    'bootstrap': True,
    'random_state': 217,
    'n_jobs': -1,                  # 使用所有CPU核心
    'verbose': 1,
}

print("开始训练随机森林模型...")
rf_model = RandomForestRegressor(**rf_params)
rf_model.fit(X_train, y_train)

# 模型预测
print("进行预测...")
y_predict = rf_model.predict(X_test)

# 保存预测结果 - 使用原始代码的保存方式
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_RF.csv", index=False)
print("\n随机森林结果已保存到: result_RF.csv")

# 计算平均误差 - 使用原始代码的计算方式
data = pd.read_csv("result_RF.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("\n随机森林6个数据的平均值为：")
print(means)
print(f"总平均误差: {means.mean():.6f}")


end_time = time.time()
print(f"\n总耗时：{end_time - start_time:.3f}秒")

# 特征重要性
feature_importance = rf_model.feature_importances_
print("\n特征重要性:")
for i, col in enumerate(noise_columns):
    print(f"{col}: {feature_importance[i]:.4f}")