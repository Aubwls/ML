import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats

start_time = time.time()

# 特征标准化
scaler = StandardScaler()
# 加载数据集
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')

# 定义特征列和目标列
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

CL = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth','Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 检查数据缺失情况
data = train_dataSet[CL]
missingDf=data.isnull().sum().sort_values(ascending=False).reset_index()
missingDf.columns=['feature','miss_num']
missingDf['miss_percentage']=missingDf['miss_num']/data.shape[0]
print("缺失值比例")
print(missingDf)

# 检查异常值比例
outlier_ratios = {}
for column in CL:
    z_scores = np.abs(stats.zscore(train_dataSet[column]))
    outliers = (z_scores > 2)
    outlier_ratio = outliers.mean()
    outlier_ratios[column] = outlier_ratio
print("*"*30)
print("异常值的比例:")
for column, ratio in outlier_ratios.items():
    print(f"{column}: {ratio:.2%}")

# 划分训练集和测试集
X_train = train_dataSet[noise_columns]
y_train = train_dataSet[columns]
X_test = test_dataSet[noise_columns]
y_test = test_dataSet[columns]

# 使用SVR模型
model_adj = MultiOutputRegressor(SVR())

# 模型训练
model_adj.fit(X_train, y_train)

# 预测
y_predict = model_adj.predict(X_test)

# 计算误差并保存结果
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

# 保存结果到CSV文件
result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_SVR.csv", index=False)

print("<*>"*50)

# 读取结果并计算平均误差
data = pd.read_csv("result_SVR.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("6个数据的平均值为：\n", means)
print(means.mean())

end_time = time.time()
print(f"总耗时：{end_time - start_time : .3f}秒")