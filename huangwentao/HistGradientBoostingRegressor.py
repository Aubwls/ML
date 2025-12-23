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
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import warnings

warnings.filterwarnings('ignore')

start_time = time.time()

# 加载数据集
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')

columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

CL = columns + noise_columns


# 高效数据预处理
def preprocess_data(data):
    data_processed = data[CL].copy()
    # 使用中位数填充缺失值
    for col in CL:
        if data_processed[col].isnull().sum() > 0:
            data_processed[col].fillna(data_processed[col].median(), inplace=True)
    return data_processed


print("数据预处理中...")
train_data_processed = preprocess_data(train_dataSet)
test_data_processed = preprocess_data(test_dataSet)

# 使用RobustScaler处理异常值
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(train_data_processed[noise_columns])
X_test_scaled = scaler.transform(test_data_processed[noise_columns])

y_train = train_data_processed[columns]
y_test = test_data_processed[columns]


# HistGradientBoostingRegressor参数优化
def optimize_hist_gbm(X_train, y_train, X_test, y_test):
    """优化HistGradientBoostingRegressor模型"""

    # 定义参数网格
    param_grid = {
        'estimator__max_iter': [100, 150],
        'estimator__learning_rate': [0.05, 0.1],
        'estimator__max_depth': [8, 12],
        'estimator__min_samples_leaf': [15, 20],
        'estimator__l2_regularization': [0, 0.1]
    }

    # 使用采样数据进行快速参数搜索
    sample_size = min(50000, len(X_train))
    if sample_size < len(X_train):
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train[indices]
        y_sample = y_train.iloc[indices]
    else:
        X_sample = X_train
        y_sample = y_train

    print(f"使用 {sample_size} 个样本进行参数优化...")

    # 创建基础模型并使用MultiOutputRegressor包装
    base_model = HistGradientBoostingRegressor(
        random_state=42,
        early_stopping=True,
        scoring='loss',
        n_iter_no_change=10,
        verbose=0
    )

    multi_output_model = MultiOutputRegressor(base_model)

    # 网格搜索
    print("开始HistGradientBoosting参数优化...")
    grid_search = GridSearchCV(
        multi_output_model,
        param_grid,
        cv=2,  # 减少交叉验证折数以加快速度
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_sample, y_sample)

    # 用完整数据训练最佳参数模型
    print("使用最佳参数训练完整模型...")
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # 评估模型
    y_pred = best_model.predict(X_test)
    score = r2_score(y_test, y_pred, multioutput='variance_weighted')

    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳 R² 分数: {score:.4f}")

    return best_model, score


# 执行模型优化
best_model, best_score = optimize_hist_gbm(X_train_scaled, y_train, X_test_scaled, y_test)

print(f"\n最佳 R² 分数: {best_score:.4f}")

# 使用最佳模型进行预测
y_predict = best_model.predict(X_test_scaled)

# 计算并保存结果
results = []
for True_Value, Predicted_Value in zip(y_test.values, y_predict):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

# 结果写入CSV文件
result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_HistGBM_optimized.csv", index=False)

# 计算平均误差和评估指标
print("\n" + "<*>" * 50)
data = pd.read_csv("result_HistGBM_optimized.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("6个数据的平均绝对误差为：")
for i, col in enumerate(columns):
    print(f"{col}: {means[i]:.6f}")

print(f"总体平均误差: {means.mean():.6f}")

# 计算其他评估指标
mse = mean_squared_error(y_test, y_predict, multioutput='raw_values')
r2 = r2_score(y_test, y_predict, multioutput='raw_values')

print("\n各目标变量的评估指标:")
for i, col in enumerate(columns):
    print(f"{col}: MSE={mse[i]:.6f}, R²={r2[i]:.4f}")

end_time = time.time()
print(f"\n总耗时：{end_time - start_time:.3f}秒")
print("使用的模型: MultiOutputRegressor(HistGradientBoostingRegressor)")