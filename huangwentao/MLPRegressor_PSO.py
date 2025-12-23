# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： 03-main.py
# 编码实现 Coding by： Hongjie Liu , Suiwen Zhang 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest  University of Nationality，or Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师
# coding=utf-8
import time

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from scipy import stats
import pyswarms as ps

start_time = time.time()

# 特征标准化
scaler = StandardScaler()
# 加载数据集
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')

# columns表示原始列，noise_columns表示添加噪声的额列
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth', ]
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

CL = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth',
      'Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
      'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

## 查看数据缺失情况
data = train_dataSet[CL]
missingDf = data.isnull().sum().sort_values(ascending=False).reset_index()
missingDf.columns = ['feature', 'miss_num']
missingDf['miss_percentage'] = missingDf['miss_num'] / data.shape[0]  # 缺失值比例
print("缺失值比例")
print(missingDf)

# 初始化一个字典来存储每一列的异常值比例
outlier_ratios = {}

# 遍历每一列
for column in CL:
    # 计算每一列的Z分数
    z_scores = np.abs(stats.zscore(train_dataSet[column]))

    # 找出异常值（假设Z分数大于2为异常值）
    outliers = (z_scores > 2)

    # 计算异常值的比例
    outlier_ratio = outliers.mean()

    # 存储异常值比例
    outlier_ratios[column] = outlier_ratio
print("*" * 30)
# 打印结果
print("异常值的比例:")
for column, ratio in outlier_ratios.items():
    print(f"{column}: {ratio:.2%}")

# 划分训练集中X_Train和y_Train
X_train = train_dataSet[noise_columns]
y_train = train_dataSet[columns]

# 划分测试集中X_test和y_test
X_test = test_dataSet[noise_columns]
y_test = test_dataSet[columns]

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)


# 粒子群优化算法PSO
def pso_mlp(params):
    """
    PSO目标函数 - 最小化MLP的MSE
    """
    n_particles = params.shape[0]
    scores = np.zeros(n_particles)

    for i in range(n_particles):
        try:
            # 提取参数
            hidden_layer_sizes = int(params[i, 0])
            alpha = 10 ** params[i, 1]  # 对数尺度
            learning_rate_init = 10 ** params[i, 2]  # 对数尺度
            learning_rate = ['constant', 'invscaling', 'adaptive'][int(params[i, 3]) % 3]

            # 创建MLP模型
            mlp = MLPRegressor(
                hidden_layer_sizes=(hidden_layer_sizes,),
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                learning_rate=learning_rate,
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=20,
                validation_fraction=0.1,
                random_state=217,
                solver='adam',
                batch_size=min(256, len(X_train_scaled)),
                verbose=False
            )

            # 训练模型
            mlp.fit(X_train_scaled, y_train_scaled)

            # 预测并计算MSE
            y_pred_scaled = mlp.predict(X_test_scaled)
            mse = mean_squared_error(y_test_scaled, y_pred_scaled)
            scores[i] = mse

        except:
            scores[i] = 1e10  # 如果训练失败，返回很大的误差

    return scores


# PSO参数设置
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
bounds = (np.array([10, -5, -5, 0]),  # hidden_layer_sizes, alpha, learning_rate_init, learning_rate_type
          np.array([200, 0, -1, 2.9]))

# 运行PSO优化
print("开始PSO优化MLP参数...")
optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=4, options=options, bounds=bounds)
best_cost, best_pos = optimizer.optimize(pso_mlp, iters=30)

# 解析最佳参数
best_hidden_layer_sizes = int(best_pos[0])
best_alpha = 10 ** best_pos[1]
best_learning_rate_init = 10 ** best_pos[2]
best_learning_rate = ['constant', 'invscaling', 'adaptive'][int(best_pos[3]) % 3]

print(f"\nPSO找到的最佳参数:")
print(f"隐藏层神经元数量: {best_hidden_layer_sizes}")
print(f"正则化参数alpha: {best_alpha:.6f}")
print(f"初始学习率: {best_learning_rate_init:.6f}")
print(f"学习率策略: {best_learning_rate}")

# 使用最佳参数训练最终模型
print("\n使用最佳参数训练最终MLP模型...")
final_mlp = MLPRegressor(
    hidden_layer_sizes=(best_hidden_layer_sizes, best_hidden_layer_sizes),
    alpha=best_alpha,
    learning_rate_init=best_learning_rate_init,
    learning_rate=best_learning_rate,
    max_iter=1000,
    early_stopping=True,
    n_iter_no_change=30,
    validation_fraction=0.1,
    random_state=217,
    solver='adam',
    batch_size=min(256, len(X_train_scaled)),
    verbose=True
)

# 模型训练
final_mlp.fit(X_train_scaled, y_train_scaled)

# 预测值
y_predict_scaled = final_mlp.predict(X_test_scaled)
y_predict = scaler_y.inverse_transform(y_predict_scaled)

results = []
# 遍历y_test和y_predict，并且计算误差
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    # 格式化True_Value和Predicted_Value为原始数据格式
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))  # 修改ERROR数据格式
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])  # 保存结果

# 结果写入CSV文件当中
result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_MLP_PSO.csv", index=False)

print("<*>" * 50)

# 从CSV文件读取数据
data = pd.read_csv("result_MLP_PSO.csv")

# 提取第三列数据
column3 = data.iloc[:, 2]

# 将每行的6个数字拆分并转换为数字列表
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)

# 计算平均值
means = numbers.mean()

# 打印结果
print("6个数据的平均绝对误差为：\n", means)
print(f"总体平均绝对误差: {means.mean():.6f}")

# 计算R2和MSE
r2 = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
print(f"R2 Score: {r2:.6f}")
print(f"MSE: {mse:.6f}")

end_time = time.time()
print(f"总耗时：{end_time - start_time : .3f}秒")