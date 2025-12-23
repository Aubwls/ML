import pandas as pd
import numpy as np
import time
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

start_time = time.time()

# 加载数据集
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')

# 定义特征和目标列
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr',
                 'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density',
           'H2O_sig_strgth', 'CO2_sig_strgth']

X_train = train_dataSet[noise_columns]
y_train = train_dataSet[columns]
X_test = test_dataSet[noise_columns]
y_test = test_dataSet[columns]

# 优化后的LightGBM参数
lgbm_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 127,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_child_samples': 20,
    'random_state': 217,
    'n_jobs': -1,
    'verbose': -1
}

# 训练模型并预测
models = {}
predictions = np.zeros_like(y_test.values)

print("开始训练模型...")
for i, col in enumerate(columns):
    print(f"训练目标变量 {i + 1}/6: {col}")

    model = LGBMRegressor(**lgbm_params)
    model.fit(X_train, y_train[col])

    pred = model.predict(X_test)
    predictions[:, i] = pred
    models[col] = model

# 计算误差
errors = np.abs(y_test.values - predictions)

# 保存结果
results = []
for i in range(len(y_test)):
    true_values = ' '.join(map(str, y_test.values[i]))
    pred_values = ' '.join(map(str, predictions[i]))
    error_values = ' '.join(map(str, errors[i]))
    results.append([true_values, pred_values, error_values])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("optimized_LGBM_results.csv", index=False)

# 计算并显示平均误差
column_errors = errors.mean(axis=0)
print("\n各目标变量的平均绝对误差:")
for col, error in zip(columns, column_errors):
    print(f"{col}: {error:.6f}")

print(f"\n总体平均绝对误差: {errors.mean():.6f}")

# 模型评估
print("\n模型性能评估:")
for i, col in enumerate(columns):
    r2 = r2_score(y_test[col], predictions[:, i])
    mse = mean_squared_error(y_test[col], predictions[:, i])
    print(f"{col}: R²={r2:.4f}, MSE={mse:.6f}")

end_time = time.time()
print(f"\n总耗时: {end_time - start_time:.2f} 秒")