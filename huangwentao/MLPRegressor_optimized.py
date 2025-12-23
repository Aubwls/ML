# coding=utf-8
import time
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

start_time = time.time()

# 加载数据
print("正在加载数据...")
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')

# 定义特征列和目标列
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

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

# 优化MLP模型 - 减少复杂度以加速训练
print("训练MLP模型...")
model_mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),  # 减少网络深度和宽度
    activation='relu',
    solver='adam',
    alpha=0.001,  # 适当增加正则化防止过拟合
    learning_rate='constant',  # 使用固定学习率
    learning_rate_init=0.01,  # 提高学习率加速收敛
    max_iter=300,  # 减少迭代次数
    batch_size=256,  # 增加批量大小
    early_stopping=False,  # 关闭早停以加速
    random_state=217,
    verbose=True
)

# 训练模型
model_mlp.fit(X_train_scaled, y_train_scaled)

# 预测
print("进行预测...")
y_predict_scaled = model_mlp.predict(X_test_scaled)
y_predict = scaler_y.inverse_transform(y_predict_scaled)

# 保存预测结果到CSV文件
print("保存预测结果...")
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_MLP_fast.csv", index=False)

# 计算并显示误差
data = pd.read_csv("result_MLP_fast.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("\n" + "="*50)
print("6个数据的平均误差为：\n", means)
print(f"总体平均误差：{means.mean():.4f}")
print("="*50)

end_time = time.time()
total_time = end_time - start_time
print(f"总耗时：{total_time:.2f}秒")