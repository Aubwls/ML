# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： catboost_model.py
# 编码实现 Coding by： Hongjie Liu , Suiwen Zhang 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest  University of Nationality，or Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师
# coding=utf-8
import time
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

start_time = time.time()

# 加载数据集
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series662.dat')

columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 划分训练集和测试集
X_train = train_dataSet[noise_columns]
y_train = train_dataSet[columns]
X_test = test_dataSet[noise_columns]
y_test = test_dataSet[columns]

# 针对大数据集的快速CatBoost参数设置
catboost_params = {
    'iterations': 500,           # 迭代次数
    'depth': 6,                  # 树深度
    'learning_rate': 0.1,        # 学习率
    'l2_leaf_reg': 5,            # L2正则化
    'border_count': 64,          # 分箱数，减少以加速
    'loss_function': 'MultiRMSE', # 多输出RMSE
    'verbose': 100,              # 每100轮输出一次
    'random_seed': 217,
    'task_type': 'CPU',
    'thread_count': -1,          # 使用所有CPU核心
    'early_stopping_rounds': 20, # 早停轮数
    'use_best_model': True,      # 使用早停最佳模型
}

print("开始训练CatBoost模型...")
cb_model = CatBoostRegressor(**catboost_params)

# 使用测试集作为验证集进行训练
cb_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    plot=False,   # 不显示训练图
    silent=False  # 显示训练日志
)

# 模型预测
print("进行预测...")
y_predict = cb_model.predict(X_test)

# 保存预测结果
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_CatBoost.csv", index=False)
print("CatBoost结果已保存到: result_CatBoost.csv")

# 计算平均误差
data = pd.read_csv("result_CatBoost.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("\nCatBoost 6个数据的平均值为：")
print(means)
print(f"总平均误差: {means.mean():.6f}")

end_time = time.time()
print(f"\n总耗时：{end_time - start_time:.3f}秒")