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
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy import stats

start_time = time.time()

# 特征标准化
scaler = StandardScaler()
# 加载数据集
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')

test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')


# columns表示原始列，noise_columns表示添加噪声的额列
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth',]
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

CL = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth','Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

## 查看数据缺失情况
data = train_dataSet[CL]
missingDf=data.isnull().sum().sort_values(ascending=False).reset_index()
missingDf.columns=['feature','miss_num']
missingDf['miss_percentage']=missingDf['miss_num']/data.shape[0]  #缺失值比例
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
print("*"*30)
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

"""模型调参"""
params = {
    # 'n_estimators': [120, 150, 180, 200],
    # 'learning_rate': [0.1, 0.15, 0.2],
    # 'max_depth': [2, 3, 4, 5],
    # "reg_alpha": [8, 10, 20, 30],
    # "reg_lambda": [6, 12],
    # "min_child_weight": [2, 3, 4, 5, 6],
    # 'subsample': [i / 100.0 for i in range(60, 80, 5)],
    # 'colsample_bytree': [i / 100.0 for i in range(80, 100, 5)]
}
other_params = {
    'seed': 217,
    'booster': 'gbtree',
    'max_depth': 5,
    'n_estimators': 200,
    'learning_rate': 0.1,
    'gamma': 0,
    'reg_alpha': 10,
    'reg_lambda': 6,
    'min_child_weight': 5,
    'colsample_bytree': 0.85,
    'subsample': 0.6,
}

model_adj = XGBRegressor(**other_params)

# # sklearn提供的调参工具，训练集k折交叉验证(消除数据切分产生数据分布不均匀的影响)
# optimized_param = GridSearchCV(estimator=model_adj, param_grid=params, scoring='r2', cv=5, verbose=1)
# # 模型训练
# optimized_param.fit(X_train, y_train)
#
# # 对应参数的k折交叉验证平均得分
# means = optimized_param.cv_results_['mean_test_score']
# params = optimized_param.cv_results_['params']
# for mean, param in zip(means, params):
#     print("mean_score: %f,  params: %r" % (mean, param))
# # 最佳模型参数
# print('参数的最佳取值：{0}'.format(optimized_param.best_params_))
# # 最佳参数模型得分
# print('最佳模型得分:{0}'.format(optimized_param.best_score_))


# 模型训练
model_adj.fit(X_train, y_train)

# # 模型保存
# model_adj.save_model('xgb_regressor.json')
#
# # 模型加载
# model_adj = XGBRegressor()
# model_adj.load_model('xgb_regressor.json')

# 预测值
y_predict = model_adj.predict(X_test)


# def metrics_sklearn(y_valid, y_pred_):
#     """模型效果评估"""
#     r2 = r2_score(y_valid, y_pred_)
#     print('r2_score:{0}'.format(r2))
#
#     mse = mean_squared_error(y_valid, y_pred_)
#     print('mse:{0}'.format(mse))
#
#
#
# """模型效果评估"""
# metrics_sklearn(y_test, y_predict)

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
result_df.to_csv("result_XGB.csv", index=False)

print("<*>"*50)

# 从CSV文件读取数据
data = pd.read_csv("result_XGB.csv")

# 提取第三列数据
column3 = data.iloc[:, 2]

# 将每行的7个数字拆分并转换为数字列表
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)

# 计算平均值
means = numbers.mean()

# 打印结果
print("6个数据的平均值为：\n", means)
print(means.mean())

end_time = time.time()
print(f"总耗时：{end_time - start_time : .3f}秒")