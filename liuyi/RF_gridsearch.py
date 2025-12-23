# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： 03-main.py
# 编码实现 Coding by： Hongjie Liu , Suiwen Zhang 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest  University of Nationality，or Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师
# coding=utf-8
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import joblib

start_time = time.time()

# 加载数据集
print("加载数据集...")
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

print(f"训练集形状: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"测试集形状: X_test: {X_test.shape}, y_test: {y_test.shape}")

# 网格搜索参数设置
print("开始网格搜索寻找最优参数...")
grid_search_start = time.time()

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.6, 0.8]
}

# 创建基础模型
rf_base = RandomForestRegressor(
    bootstrap=True,
    random_state=217,
    n_jobs=-1,
    verbose=0
)

# 创建网格搜索对象
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    scoring='r2',  # 使用R²作为评分标准
    cv=3,          # 3折交叉验证
    verbose=2,     # 显示详细进度
    n_jobs=-1      # 使用所有可用的CPU核心
)

# 执行网格搜索
print("正在进行网格搜索，这可能需要一些时间...")
grid_search.fit(X_train, y_train)

grid_search_end = time.time()
print(f"网格搜索完成，耗时: {grid_search_end - grid_search_start:.2f}秒")

# 输出最优参数
print("\n最优参数找到:")
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数 (R²): {grid_search.best_score_:.6f}")

# 使用最优参数训练最终模型
print("\n使用最优参数训练最终模型...")
best_params = grid_search.best_params_

# 确保所有参数都传递给模型
rf_final_params = {
    'n_estimators': best_params['n_estimators'],
    'max_depth': best_params['max_depth'],
    'min_samples_split': best_params['min_samples_split'],
    'min_samples_leaf': best_params['min_samples_leaf'],
    'max_features': best_params['max_features'],
    'bootstrap': True,
    'random_state': 217,
    'n_jobs': -1,
    'verbose': 1
}

print(f"最终模型参数: {rf_final_params}")

rf_model = RandomForestRegressor(**rf_final_params)
rf_model.fit(X_train, y_train)

# 模型预测
print("进行预测...")
y_predict = rf_model.predict(X_test)

# 保存预测结果
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_RF_gridsearch.csv", index=False)
print("\n随机森林结果已保存到: result_RF_gridsearch.csv")