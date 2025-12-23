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
from sklearn.tree import DecisionTreeRegressor


class CascadeGradientBoosting:
    def __init__(self, n_cascades=3, learning_rate=0.1, max_depth=3, random_state=217):
        self.n_cascades = n_cascades
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.cascades = []

    def fit(self, X, y):
        self.cascades = []
        n_samples, n_features = X.shape
        n_targets = y.shape[1]

        # 初始化预测值
        current_predictions = np.zeros_like(y)

        for cascade in range(self.n_cascades):
            print(f"训练第 {cascade + 1} 层级联...")
            cascade_models = []

            for target_idx in range(n_targets):
                # 计算当前残差
                if cascade == 0:
                    residual = y.iloc[:, target_idx] if hasattr(y, 'iloc') else y[:, target_idx]
                else:
                    residual = (y.iloc[:, target_idx] if hasattr(y, 'iloc') else y[:,
                                                                                 target_idx]) - current_predictions[:,
                                                                                                target_idx]

                # 创建并训练基学习器
                model = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    random_state=self.random_state
                )
                model.fit(X, residual)

                # 更新预测值
                pred = model.predict(X)
                current_predictions[:, target_idx] += self.learning_rate * pred

                cascade_models.append(model)

            self.cascades.append(cascade_models)

            # 计算当前层的训练误差
            mse = mean_squared_error(y, current_predictions)
            r2 = r2_score(y, current_predictions)
            print(f"第 {cascade + 1} 层训练完成 - MSE: {mse:.4f}, R2: {r2:.4f}")

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        n_targets = len(self.cascades[0])

        predictions = np.zeros((n_samples, n_targets))

        for cascade_models in self.cascades:
            for target_idx, model in enumerate(cascade_models):
                pred = model.predict(X)
                predictions[:, target_idx] += self.learning_rate * pred

        return predictions


start_time = time.time()

# 加载数据集
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')

# 列定义
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 划分训练集和测试集
X_train = train_dataSet[noise_columns]
y_train = train_dataSet[columns]
X_test = test_dataSet[noise_columns]
y_test = test_dataSet[columns]

# 创建级联梯度下降模型
cascade_model = CascadeGradientBoosting(
    n_cascades=50,  # 5层级联
    learning_rate=0.01,  # 学习率
    max_depth=10,  # 树的最大深度
    random_state=217
)

# 训练模型
print("开始训练级联梯度下降模型...")
cascade_model.fit(X_train, y_train)

# 预测
print("开始预测...")
y_predict = cascade_model.predict(X_test)

# 评估模型性能
test_mse = mean_squared_error(y_test, y_predict)
test_r2 = r2_score(y_test, y_predict)
print(f"测试集性能 - MSE: {test_mse:.4f}, R2: {test_r2:.4f}")

# 保存结果
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_Cascade_GD.csv", index=False)

print("<*>" * 50)

# 计算平均误差
data = pd.read_csv("result_Cascade_GD.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("6个数据的平均值为：\n", means)
print(f"总体平均误差: {means.mean():.6f}")

end_time = time.time()
print(f"总耗时：{end_time - start_time:.3f}秒")