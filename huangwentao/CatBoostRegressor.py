import time
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

start_time = time.time()

# 加载数据集
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')

# 定义列名
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 划分训练集和测试集
X_train = train_dataSet[noise_columns]
y_train = train_dataSet[columns]
X_test = test_dataSet[noise_columns]
y_test = test_dataSet[columns]

# CatBoost模型参数
model_params = {
    'random_seed': 217,
    'iterations': 200,
    'learning_rate': 0.1,
    'depth': 6,
    'l2_leaf_reg': 3,
    'loss_function': 'MultiRMSE',
    'verbose': False,
    'task_type': 'CPU',
    'devices': '0:0'  # 使用第一个GPU设备
}

# 创建并训练模型
model = CatBoostRegressor(**model_params)
model.fit(X_train, y_train)

# 预测
y_predict = model.predict(X_test)

# 计算误差并保存结果
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

# 保存结果到CSV
result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_CatBoost.csv", index=False)

# 计算平均误差
data = pd.read_csv("result_CatBoost.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("6个数据的平均值为：")
print(means)
print(f"总体平均值: {means.mean()}")

end_time = time.time()
print(f"总耗时：{end_time - start_time:.3f}秒")