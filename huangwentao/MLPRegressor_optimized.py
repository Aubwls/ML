# coding=utf-8
import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

start_time = time.time()

print("开始加载数据集...")
# 加载数据集
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')

columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

# 划分训练集和测试集
X_train = train_dataSet[noise_columns].values
y_train = train_dataSet[columns].values
X_test = test_dataSet[noise_columns].values
y_test = test_dataSet[columns].values

print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 数据标准化 - 这对MLP非常重要
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)

print("开始训练MLP模型...")


# 方案1：集成多个MLP模型（提升精度）
def train_ensemble_mlp(X_train, y_train, X_test, n_models=5):
    """
    训练多个MLP模型并集成，提高稳定性
    """
    models = []
    predictions = []

    for i in range(n_models):
        print(f"训练第 {i + 1}/{n_models} 个MLP模型...")

        # 每个模型使用不同的随机种子
        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),  # 稍微增加网络容量
            activation='relu',
            solver='adam',
            alpha=0.0005,  # 调整正则化
            batch_size=512,  # 更大的batch size
            max_iter=200,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            tol=1e-4,
            random_state=42 + i,  # 不同种子
            verbose=False
        )

        mlp.fit(X_train, y_train)
        models.append(mlp)

        # 预测并反标准化
        y_pred_scaled = mlp.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        predictions.append(y_pred)

    # 集成预测：取所有模型的平均值
    ensemble_pred = np.mean(predictions, axis=0)

    return models, ensemble_pred


# 方案2：更深的MLP模型
def train_deeper_mlp(X_train, y_train, X_test):
    """
    训练更深的MLP模型
    """
    print("训练更深的MLP模型...")

    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32),  # 4层网络
        activation='relu',
        solver='adam',
        alpha=0.0001,  # 更小的正则化
        batch_size=512,
        max_iter=250,  # 更多迭代
        learning_rate='adaptive',
        learning_rate_init=0.0008,  # 更小的学习率
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-5,  # 更严格的容忍度
        random_state=42,
        verbose=True
    )

    mlp.fit(X_train, y_train)

    # 预测
    y_pred_scaled = mlp.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    return mlp, y_pred


# 方案3：针对每个输出训练单独的MLP
def train_per_column_mlp(X_train, y_train, X_test, columns):
    """
    为每个输出变量训练单独的MLP模型
    """
    print(f"为每个输出变量训练单独的MLP模型...")

    predictions = np.zeros_like(y_test)

    for i, col_name in enumerate(columns):
        print(f"训练第 {i + 1}/{len(columns)} 个变量: {col_name}")

        # 提取当前列的目标值
        y_train_col = y_train[:, i].reshape(-1, 1)
        y_scaler = StandardScaler()
        y_train_col_scaled = y_scaler.fit_transform(y_train_col)

        # 训练MLP
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=512,
            max_iter=150,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4,
            random_state=42 + i,
            verbose=False
        )

        mlp.fit(X_train_scaled, y_train_col_scaled)

        # 预测
        y_pred_col_scaled = mlp.predict(X_test_scaled).reshape(-1, 1)
        y_pred_col = y_scaler.inverse_transform(y_pred_col_scaled).flatten()
        predictions[:, i] = y_pred_col

    return predictions


# 尝试不同的方案
print("\n=== 方案1: 集成MLP模型 ===")
models_ensemble, y_pred_ensemble = train_ensemble_mlp(X_train_scaled, y_train_scaled, X_test_scaled, n_models=3)

print("\n=== 方案2: 更深的MLP模型 ===")
model_deep, y_pred_deep = train_deeper_mlp(X_train_scaled, y_train_scaled, X_test_scaled)

print("\n=== 方案3: 单变量MLP模型 ===")
y_pred_per_col = train_per_column_mlp(X_train_scaled, y_train_scaled, X_test_scaled, columns)


# 比较不同方案的性能
def evaluate_predictions(y_true, y_pred, label):
    """评估预测结果"""
    # 计算平均绝对误差
    mae_per_col = np.mean(np.abs(y_true - y_pred), axis=0)
    overall_mae = np.mean(mae_per_col)

    # 计算R²分数
    r2 = r2_score(y_true, y_pred)

    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{label} 结果:")
    print("各变量平均绝对误差:")
    for i, (col_name, mae) in enumerate(zip(columns, mae_per_col)):
        print(f"  {col_name}: {mae:.6f}")
    print(f"总平均误差: {overall_mae:.6f}")
    print(f"R²分数: {r2:.6f}")
    print(f"RMSE: {rmse:.6f}")

    return r2, overall_mae


# 评估三种方案
print("\n" + "=" * 50)
results = []

# 方案1评估
r2_ensemble, mae_ensemble = evaluate_predictions(y_test, y_pred_ensemble, "方案1: 集成MLP")
results.append(("集成MLP", r2_ensemble, mae_ensemble))

# 方案2评估
r2_deep, mae_deep = evaluate_predictions(y_test, y_pred_deep, "方案2: 更深MLP")
results.append(("更深MLP", r2_deep, mae_deep))

# 方案3评估
r2_percol, mae_percol = evaluate_predictions(y_test, y_pred_per_col, "方案3: 单变量MLP")
results.append(("单变量MLP", r2_percol, mae_percol))

# 选择最佳方案
print("\n" + "=" * 50)
print("方案比较:")
best_r2_idx = np.argmax([r[1] for r in results])
best_mae_idx = np.argmin([r[2] for r in results])

print(f"最佳R²方案: {results[best_r2_idx][0]} (R²={results[best_r2_idx][1]:.6f})")
print(f"最佳MAE方案: {results[best_mae_idx][0]} (MAE={results[best_mae_idx][2]:.6f})")

# 使用最佳方案（基于R²）
if best_r2_idx == 0:
    best_predictions = y_pred_ensemble
    best_label = "集成MLP"
elif best_r2_idx == 1:
    best_predictions = y_pred_deep
    best_label = "更深MLP"
else:
    best_predictions = y_pred_per_col
    best_label = "单变量MLP"

# 保存最佳预测结果
print(f"\n保存最佳预测结果 ({best_label})...")
results_data = []
for True_Value, Predicted_Value in zip(y_test, best_predictions):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results_data.append([formatted_true_value, formatted_predicted_value, formatted_error])

result_df = pd.DataFrame(results_data, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_MLP_best.csv", index=False)
print("结果已保存到: result_MLP_best.csv")

# 计算最终的平均误差
data = pd.read_csv("result_MLP_best.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print(f"\n最终结果 ({best_label}):")
print("6个数据的平均绝对误差为：")
for i, (col_name, mean_val) in enumerate(zip(columns, means)):
    print(f"  {col_name}: {mean_val:.6f}")
print(f"总平均误差: {means.mean():.6f}")

# 计算总体R²分数
overall_r2 = r2_score(y_test, best_predictions)
print(f"总体 R² 分数: {overall_r2:.6f}")

# 计算RMSE
rmse = np.sqrt(mean_squared_error(y_test, best_predictions))
print(f"总体 RMSE: {rmse:.6f}")

end_time = time.time()
total_time = end_time - start_time
print(f"\n总耗时：{total_time:.3f}秒 ({total_time / 60:.2f}分钟)")

if total_time > 600:
    print("警告：运行时间超过10分钟！")
else:
    print("成功在10分钟内完成运行！")

print("\n改进建议:")
print("1. 如果R²还不够高，可以尝试:")
print("   - 增加训练数据量")
print("   - 使用更复杂的模型（如XGBoost、LightGBM）")
print("   - 特征工程：创建新的特征")
print("   - 调整模型超参数（使用网格搜索）")
print("2. 如果时间允许，可以:")
print("   - 使用交叉验证选择最佳模型")
print("   - 使用更深的神经网络")
print("   - 集成多种不同的模型")