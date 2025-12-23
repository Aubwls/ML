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


# 优化的级联森林实现
class FastCascadeForest:
    def __init__(self, n_estimators=100, max_layers=3, early_stopping_rounds=2):
        self.n_estimators = n_estimators
        self.max_layers = max_layers
        self.early_stopping_rounds = early_stopping_rounds
        self.layers = []

    def fit(self, X, y):
        current_features = X
        best_score = -np.inf
        no_improvement = 0

        for layer in range(self.max_layers):
            # 使用优化的随机森林参数
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42 + layer,
                n_jobs=-1
            )
            # rf = RandomForestRegressor(
            #     n_estimators=self.n_estimators,
            #     max_depth=14,
            #     min_samples_split=3,
            #     min_samples_leaf=1,
            #     max_features=0.7,
            #     bootstrap=True,
            #     random_state=42 + layer,
            #     n_jobs=-1
            # )
            rf.fit(current_features, y)

            # 预测
            y_pred = rf.predict(current_features)
            score = r2_score(y, y_pred)

            # 检查是否提前停止
            if score > best_score:
                best_score = score
                no_improvement = 0
            else:
                no_improvement += 1

            # 保存当前层
            self.layers.append(rf)

            # 提前停止
            if no_improvement >= self.early_stopping_rounds:
                break

            # 为下一层准备特征
            if layer < self.max_layers - 1:
                current_features = np.column_stack([X, y_pred])

        return self

    def predict(self, X):
        current_features = X

        for i, rf in enumerate(self.layers):
            y_pred = rf.predict(current_features)

            # 如果是最后一层，返回结果
            if i == len(self.layers) - 1:
                return y_pred

            # 为下一层准备特征
            current_features = np.column_stack([X, y_pred])

        return y_pred


# 为每个输出变量训练一个简化的级联森林模型
print("开始训练优化级联森林模型...")
y_predict = np.zeros_like(y_test.values)

for i, column in enumerate(columns):
    print(f"训练第 {i + 1}/{len(columns)} 个输出变量: {column}")

    # 创建并训练简化级联森林模型
    cascade_model = FastCascadeForest(
        n_estimators=100,  # 增加树的数量
        max_layers=3,
        early_stopping_rounds=1
    )

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
result_df.to_csv("result_FastCascade.csv", index=False)
print("结果已保存到: result_FastCascade.csv")

# 计算平均误差
data = pd.read_csv("result_FastCascade.csv")
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