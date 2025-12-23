# coding=utf-8
import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import xgboost as xgb

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


class FastCascadeForest:
    def __init__(self, n_estimators=15, max_layers=3, early_stopping_rounds=1):
        self.n_estimators = n_estimators
        self.max_layers = max_layers
        self.early_stopping_rounds = early_stopping_rounds
        self.layers = []

    def fit(self, X, y):
        current_features = X
        best_score = -np.inf
        no_improvement = 0

        for layer in range(self.max_layers):
            # 使用XGBoost GPU版本
            rf = xgb.XGBRegressor(
                n_estimators=100,
                random_state=42 + layer,
                tree_method='hist',
                device='cuda',
                learning_rate=0.1,
                max_depth=15
            )

            rf.fit(current_features, y)
            y_pred = rf.predict(current_features)
            score = r2_score(y, y_pred)

            # 检查是否提前停止
            if score > best_score:
                best_score = score
                no_improvement = 0
            else:
                no_improvement += 1

            self.layers.append(rf)

            # 提前停止
            if no_improvement >= self.early_stopping_rounds:
                break

            # 为下一层准备特征
            if layer < self.max_layers - 1:
                y_pred_expanded = y_pred.reshape(-1, 1) if len(y_pred.shape) == 1 else y_pred
                current_features = np.column_stack([X, y_pred_expanded])

        return self

    def predict(self, X):
        current_features = X

        for i, rf in enumerate(self.layers):
            y_pred = rf.predict(current_features)

            if i == len(self.layers) - 1:
                return y_pred

            # 为下一层准备特征
            y_pred_expanded = y_pred.reshape(-1, 1) if len(y_pred.shape) == 1 else y_pred
            current_features = np.column_stack([X, y_pred_expanded])

        return y_pred


# 为每个输出变量训练模型
print("开始训练级联森林模型（GPU加速）...")
y_predict = np.zeros_like(y_test.values)

for i, column in enumerate(columns):
    print(f"训练第 {i + 1}/{len(columns)} 个输出变量: {column}")

    cascade_model = FastCascadeForest()
    cascade_model.fit(X_train.values, y_train[column].values)
    y_predict[:, i] = cascade_model.predict(X_test.values)

# 保存预测结果
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_FastCascade_GPU.csv", index=False)
print("结果已保存到: result_FastCascade_GPU.csv")

# 计算平均误差
data = pd.read_csv("result_FastCascade_GPU.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("\n6个数据的平均值为：")
print(means)
print(f"总平均误差: {means.mean():.6f}")

# 计算总体R²分数
overall_r2 = r2_score(y_test, y_predict)
print(f"总体 R² 分数: {overall_r2:.4f}")

end_time = time.time()
print(f"\n总耗时：{end_time - start_time:.3f}秒")