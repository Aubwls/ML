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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)

# 简化的级联森林实现（使用MLP）
class FastCascadeMLP:
    def __init__(self, hidden_layer_sizes=(100,), max_layers=3, early_stopping_rounds=2,
                 learning_rate_init=0.001, max_iter=500, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_layers = max_layers
        self.early_stopping_rounds = early_stopping_rounds
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.layers = []

    def fit(self, X, y):
        current_features = X
        best_score = -np.inf
        no_improvement = 0

        for layer in range(self.max_layers):
            # 训练MLP
            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                random_state=self.random_state + layer,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20
            )
            mlp.fit(current_features, y)

            # 预测
            y_pred = mlp.predict(current_features)
            score = r2_score(y, y_pred)

            # 检查是否提前停止
            if score > best_score:
                best_score = score
                no_improvement = 0
            else:
                no_improvement += 1

            # 保存当前层
            self.layers.append(mlp)

            # 提前停止
            if no_improvement >= self.early_stopping_rounds:
                print(f"提前停止在第 {layer + 1} 层")
                break

            # 为下一层准备特征
            if layer < self.max_layers - 1:
                current_features = np.column_stack([X, y_pred])

        return self

    def predict(self, X):
        current_features = X

        for i, mlp in enumerate(self.layers):
            y_pred = mlp.predict(current_features)

            # 如果是最后一层，返回结果
            if i == len(self.layers) - 1:
                return y_pred

            # 为下一层准备特征
            current_features = np.column_stack([X, y_pred])

        return y_pred


# 为每个输出变量训练一个简化的级联MLP模型
print("开始训练简化级联MLP模型...")
y_predict_scaled = np.zeros_like(y_test.values)

for i, column in enumerate(columns):
    print(f"训练第 {i + 1}/{len(columns)} 个输出变量: {column}")

    # 创建并训练简化级联MLP模型
    cascade_model = FastCascadeMLP(
        hidden_layer_sizes=(100, 50),  # 两层隐藏层
        max_layers=3,  # 减少层数
        early_stopping_rounds=1,
        learning_rate_init=0.001,
        max_iter=500
    )

    cascade_model.fit(X_train_scaled, y_train_scaled[:, i])
    y_predict_scaled[:, i] = cascade_model.predict(X_test_scaled)

# 反标准化预测结果
y_predict = scaler_y.inverse_transform(y_predict_scaled)

# 保存预测结果
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_FastCascadeMLP.csv", index=False)
print("结果已保存到: result_FastCascadeMLP.csv")

# 计算平均误差
data = pd.read_csv("result_FastCascadeMLP.csv")
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