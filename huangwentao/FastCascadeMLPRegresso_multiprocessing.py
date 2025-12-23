# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： 03-main_optimized_simple.py
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
from joblib import Parallel, delayed

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


# 优化的级联MLP实现
class OptimizedCascadeMLP:
    def __init__(self, hidden_layer_sizes=(200, 150, 100), max_layers=4,
                 learning_rate_init=0.001, max_iter=1000, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_layers = max_layers
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.layers = []

    def fit(self, X, y):
        current_features = X
        best_score = -np.inf

        for layer in range(self.max_layers):
            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                random_state=self.random_state + layer,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                alpha=0.01,  # 正则化
                solver='adam',
                batch_size=256
            )

            mlp.fit(current_features, y)
            y_pred = mlp.predict(current_features)
            score = r2_score(y, y_pred)

            print(f"层 {layer + 1} R² 分数: {score:.4f}")

            # 保存当前层
            self.layers.append(mlp)

            # 为下一层准备特征
            if layer < self.max_layers - 1:
                current_features = np.column_stack([X, y_pred])

        return self

    def predict(self, X):
        current_features = X

        for i, mlp in enumerate(self.layers):
            y_pred = mlp.predict(current_features)

            if i == len(self.layers) - 1:
                return y_pred

            current_features = np.column_stack([X, y_pred])

        return y_pred


# 定义训练单个模型的函数
def train_single_model(i, X_train_scaled, y_train_scaled, X_test_scaled, column):
    print(f"开始训练第 {i + 1}/{len(columns)} 个输出变量: {column}")

    cascade_model = OptimizedCascadeMLP(
        hidden_layer_sizes=(200, 150, 100),
        max_layers=4,
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42 + i
    )

    cascade_model.fit(X_train_scaled, y_train_scaled[:, i])
    y_pred_single = cascade_model.predict(X_test_scaled)

    print(f"完成训练第 {i + 1}/{len(columns)} 个输出变量: {column}")
    return i, y_pred_single


# 并行训练
print("开始并行训练优化级联MLP模型...")

results_parallel = Parallel(n_jobs=-1, verbose=10)(
    delayed(train_single_model)(i, X_train_scaled, y_train_scaled, X_test_scaled, column)
    for i, column in enumerate(columns)
)

# 整理结果
y_predict_scaled = np.zeros_like(y_test.values)
for i, y_pred_single in results_parallel:
    y_predict_scaled[:, i] = y_pred_single

# 反标准化
y_predict = scaler_y.inverse_transform(y_predict_scaled)

# 保存结果
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_OptimizedCascadeMLP.csv", index=False)
print("结果已保存到: result_OptimizedCascadeMLP.csv")

# 计算平均误差
data = pd.read_csv("result_OptimizedCascadeMLP.csv")
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