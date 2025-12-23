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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings

warnings.filterwarnings('ignore')

start_time = time.time()

# 加载数据集
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')

columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 划分训练集和测试集
X_train_raw = train_dataSet[noise_columns]
y_train = train_dataSet[columns]
X_test_raw = test_dataSet[noise_columns]
y_test = test_dataSet[columns]

# ==================== 增强的特征工程 ====================
print("开始特征工程...")


def create_advanced_features(X):
    """创建更丰富的特征集合"""
    X_enhanced = X.copy()

    # 1. 对数变换（处理偏态分布）
    for col in X.columns:
        if 'Error' in col:
            X_enhanced[f'log_{col}'] = np.log1p(np.abs(X[col])) * np.sign(X[col])

    # 2. 交互特征（基于特征重要性分析）
    # T_SONIC 相关
    X_enhanced['T_SONIC_CO2_density_interact'] = X['Error_T_SONIC'] * X['Error_CO2_density']
    X_enhanced['T_SONIC_PCA_interact'] = X['Error_T_SONIC'] * X['Error_CO2_density_fast_tmpr']

    # 信号强度相关（针对R²低的变量）
    X_enhanced['H2O_CO2_signal_ratio'] = X['Error_H2O_sig_strgth'] / (X['Error_CO2_sig_strgth'] + 1e-8)
    X_enhanced['signal_strength_product'] = X['Error_H2O_sig_strgth'] * X['Error_CO2_sig_strgth']

    # 3. 多项式特征（二阶）
    for col1 in ['Error_T_SONIC', 'Error_CO2_density']:
        for col2 in ['Error_CO2_density_fast_tmpr', 'Error_H2O_density']:
            X_enhanced[f'{col1}_{col2}_interact'] = X[col1] * X[col2]

    # 4. 统计特征增强
    X_enhanced['error_mean'] = X.mean(axis=1)
    X_enhanced['error_std'] = X.std(axis=1)
    X_enhanced['error_skew'] = X.apply(lambda row: row.skew(), axis=1)
    X_enhanced['error_kurtosis'] = X.apply(lambda row: row.kurtosis(), axis=1)

    # 5. 分箱特征
    for col in X.columns:
        X_enhanced[f'{col}_bin'] = pd.qcut(X[col], q=5, labels=False, duplicates='drop')

    return X_enhanced


# 应用特征工程
X_train = create_advanced_features(X_train_raw)
X_test = create_advanced_features(X_test_raw)

# 标准化
scaler = RobustScaler()  # 使用RobustScaler处理异常值
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(f"特征工程后维度：训练集 {X_train.shape}, 测试集 {X_test.shape}")

# ==================== 针对每个变量的优化模型 ====================
rf_params = {
    'n_estimators': 100,
    'max_depth': 13,
    'min_samples_split': 3,
    'min_samples_leaf': 1,
    'max_features': 0.7,
    'bootstrap': True,
    'random_state': 217,
    'n_jobs': -1
}

# 针对每个变量的特征选择
selected_features = {}
for i, col in enumerate(columns):
    # 使用互信息选择特征
    selector = SelectKBest(score_func=mutual_info_regression, k=min(20, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train[col])
    X_test_selected = selector.transform(X_test)

    # 保存选择的特征
    selected_features[col] = {
        'selector': selector,
        'features': X_train.columns[selector.get_support()]
    }

    # 更新数据
    if i == 0:
        X_train_all_selected = X_train_selected
        X_test_all_selected = X_test_selected
    else:
        # 对于每个变量，我们可能会使用不同的特征子集
        pass

print("开始训练优化模型...")

# 为每个目标变量创建专门的模型
models = {}
predictions = {}

for i, col in enumerate(columns):
    print(f"训练 {col} 的优化模型...")

    # 根据R²值选择不同的模型策略
    if col in ['H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']:
        # R²较低的变量使用更复杂的模型
        # 使用特征选择后的数据
        selector = SelectKBest(score_func=mutual_info_regression, k=min(25, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train[col])
        X_test_selected = selector.transform(X_test)

        # 使用梯度提升树（更适合非线性关系）
        gb_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=217,
            subsample=0.8
        )
        gb_model.fit(X_train_selected, y_train[col])
        models[col] = gb_model
        predictions[col] = gb_model.predict(X_test_selected)

    elif col == 'T_SONIC':
        # T_SONIC使用更深的树
        rf_deep = RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=0.6,
            random_state=217,
            n_jobs=-1
        )
        rf_deep.fit(X_train, y_train[col])
        models[col] = rf_deep
        predictions[col] = rf_deep.predict(X_test)

    else:
        # 其他变量使用标准随机森林
        rf_model = RandomForestRegressor(**rf_params)
        rf_model.fit(X_train, y_train[col])
        models[col] = rf_model
        predictions[col] = rf_model.predict(X_test)

# 组合所有预测结果
y_predict = np.column_stack([predictions[col] for col in columns])

# ==================== 使用堆叠集成进一步提升 ====================
print("训练堆叠集成模型...")

# 创建基学习器的预测作为元特征
meta_features_train = []
meta_features_test = []

for col in columns:
    model = models[col]
    # 使用交叉验证风格生成训练集的预测
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=217)
    train_pred = np.zeros_like(y_train[col])

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold = y_train[col].iloc[train_idx]

        # 重新训练模型
        if col in ['H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']:
            temp_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=217
            )
        else:
            temp_model = RandomForestRegressor(**rf_params)

        temp_model.fit(X_train_fold, y_train_fold)
        train_pred[val_idx] = temp_model.predict(X_val_fold)

    meta_features_train.append(train_pred)
    meta_features_test.append(predictions[col])

# 转换为DataFrame
meta_train_df = pd.DataFrame(np.column_stack(meta_features_train),
                             columns=[f'meta_{col}' for col in columns])
meta_test_df = pd.DataFrame(np.column_stack(meta_features_test),
                            columns=[f'meta_{col}' for col in columns])

# 添加原始特征
meta_train_df = pd.concat([pd.DataFrame(X_train_scaled), meta_train_df], axis=1)
meta_test_df = pd.concat([pd.DataFrame(X_test_scaled), meta_test_df], axis=1)

# 使用线性模型进行堆叠
from sklearn.linear_model import Ridge

final_predictions = np.zeros_like(y_predict)
for i, col in enumerate(columns):
    ridge = Ridge(alpha=1.0, random_state=217)
    ridge.fit(meta_train_df, y_train[col])
    final_predictions[:, i] = ridge.predict(meta_test_df)

# ==================== 后处理和校准 ====================
# 1. 计算并应用偏差校正
train_predictions = np.column_stack([models[col].predict(X_train) for col in columns])
bias = y_train.values - train_predictions
bias_correction = np.mean(bias, axis=0)

final_predictions_corrected = final_predictions + bias_correction

# 2. 应用边界约束（基于物理意义）
# 假设信号强度应该非负
for i, col in enumerate(columns):
    if 'sig_strgth' in col:
        final_predictions_corrected[:, i] = np.maximum(final_predictions_corrected[:, i], 0)

# ==================== 结果保存与评估 ====================
results = []
for True_Value, Predicted_Value in zip(y_test.values, final_predictions_corrected):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_Final_RF.csv", index=False)
print("\n最终优化结果已保存到: result_Final_RF.csv")

# 计算平均误差
data = pd.read_csv("result_Final_RF.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("\n最终优化6个数据的平均误差为：")
for col, mean in zip(columns, means):
    print(f"{col}: {mean:.6f}")
print(f"总平均误差: {means.mean():.6f}")

# 评估指标
print("\n最终模型评估指标:")
total_r2 = 0
for i, col in enumerate(columns):
    r2 = r2_score(y_test.iloc[:, i], final_predictions_corrected[:, i])
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], final_predictions_corrected[:, i]))
    mae = np.mean(np.abs(y_test.iloc[:, i] - final_predictions_corrected[:, i]))
    total_r2 += r2
    print(f"{col}: R² = {r2:.6f}, RMSE = {rmse:.6f}, MAE = {mae:.6f}")

print(f"\n平均R²: {total_r2 / 6:.6f}")

# 误差降低分析
print("\n误差降低分析:")
prev_means = [2.048608, 9.644039, 9.271078, 0.774052, 0.072309, 0.076907]
for i, (prev, curr) in enumerate(zip(prev_means, means)):
    reduction = (prev - curr) / prev * 100
    print(f"{columns[i]}: {prev:.6f} → {curr:.6f}, 降低 {reduction:.2f}%")

end_time = time.time()
print(f"\n总耗时：{end_time - start_time:.3f}秒")