import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import time
from gcForest.lib.gcforest.gcforest import GCForest


def main():
    start_time = time.time()

    # 加载数据
    try:
        train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
        test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')
        print("数据加载成功")
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        return

    # 定义特征和目标
    noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr',
                     'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
    columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density',
               'H2O_sig_strgth', 'CO2_sig_strgth']

    X_train = train_dataSet[noise_columns].values
    y_train = train_dataSet[columns].values
    X_test = test_dataSet[noise_columns].values
    y_test = test_dataSet[columns].values

    print("数据加载完成")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # 配置gcForest参数
    config = {
        "cascade": {
            "random_state": 0,
            "max_layers": 3,  # 最大层数
            "early_stopping_rounds": 2,  # 添加这个参数
            "n_trees": 50,  # 每层中的树的数量
            "n_folds": 5,  # K折交叉验证
            "estimators": [
                {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 50, "max_depth": None, "n_jobs": -1},
                {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 50, "max_depth": None, "n_jobs": -1},
            ],
            "tol": 0.0,  # 容差
            "use_predict_proba": True,  # 使用概率预测
            "seed": 0,  # 随机种子
        },
        "data_cache": {
            "dir": "cache",  # 缓存目录
            "name": "temp",  # 缓存名称
            "ignore": False,  # 是否忽略缓存
            "delete": False,  # 是否删除缓存
        }
    }

    print("\n初始化gcForest模型...")

    # 为每个目标变量创建一个gcForest模型
    all_predictions = []
    model_details = {}

    for target_idx, target_name in enumerate(columns):
        print(f"\n训练目标变量 {target_idx + 1}/{len(columns)}: {target_name}")

        # 创建并训练gcForest
        gcf = GCForest(config)

        # 训练当前目标变量的模型
        gcf.fit_transform(X_train_scaled, y_train_scaled[:, target_idx].reshape(-1, 1),
                          X_test_scaled, y_test[:, target_idx].reshape(-1, 1))

        # 预测
        y_pred_scaled = gcf.predict(X_test_scaled).reshape(-1, 1)
        all_predictions.append(y_pred_scaled.flatten())

        # 反标准化
        y_pred_temp = scaler_y.inverse_transform(
            np.column_stack([y_pred_scaled] * len(columns))
        )[:, target_idx]

        # 计算指标
        mse = mean_squared_error(y_test[:, target_idx], y_pred_temp)
        r2 = r2_score(y_test[:, target_idx], y_pred_temp)
        mae = np.mean(np.abs(y_test[:, target_idx] - y_pred_temp))

        model_details[target_name] = {
            'model': gcf,
            'mse': mse,
            'r2': r2,
            'mae': mae
        }

        print(f"{target_name} - MSE: {mse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")

    # 合并所有预测结果
    y_pred_scaled_combined = np.column_stack(all_predictions)
    y_pred = scaler_y.inverse_transform(y_pred_scaled_combined)

    # 整体评估
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('\n' + '=' * 60)
    print("gcForest模型评估结果:")
    print('=' * 60)
    print(f"整体均方误差 (MSE): {mse:.6f}")
    print(f"整体决定系数 (R²): {r2:.6f}")

    # 各目标变量的性能
    print("\n各目标变量详细评估:")
    print("-" * 50)
    mae_per_column = np.mean(np.abs(y_test - y_pred), axis=0)
    mse_per_column = np.mean((y_test - y_pred) ** 2, axis=0)
    r2_per_column = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(len(columns))]

    for i, col in enumerate(columns):
        print(f"{col:>30}: MAE = {mae_per_column[i]:.4f}, MSE = {mse_per_column[i]:.4f}, R² = {r2_per_column[i]:.4f}")

    print(f"\n平均MAE: {np.mean(mae_per_column):.4f}")
    print(f"平均MSE: {np.mean(mse_per_column):.4f}")
    print(f"平均R²: {np.mean(r2_per_column):.4f}")

    # 保存预测结果
    results = []
    for True_Value, Predicted_Value in zip(y_test, y_pred):
        error = np.abs(True_Value - Predicted_Value)
        formatted_true_value = ' '.join(map(str, True_Value))
        formatted_predicted_value = ' '.join(map(str, Predicted_Value))
        formatted_error = ' '.join(map(str, error))
        results.append([formatted_true_value, formatted_predicted_value, formatted_error])

    result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
    result_df.to_csv("result_gcForest.csv", index=False)
    print("\ngcForest结果已保存到: result_gcForest.csv")

    # 计算并显示平均误差
    print("\n各目标变量的平均绝对误差:")
    error_data = []
    for i, col in enumerate(columns):
        avg_error = np.mean(np.abs(y_test[:, i] - y_pred[:, i]))
        error_data.append(avg_error)
        print(f"{col}: {avg_error:.4f}")

    print(f"总体平均绝对误差: {np.mean(error_data):.4f}")

    end_time = time.time()
    print(f"\n总耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    main()