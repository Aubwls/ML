import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import time
import warnings
import gc

warnings.filterwarnings('ignore')


class OptimizedCascadeForest:
    """
    优化参数设置的级联森林
    增加内存管理和性能优化
    """

    def __init__(self, n_layers=2, n_estimators=80, random_state=217,
                 use_early_stopping=True, target_specific_params=None,
                 use_subsample=False, subsample_ratio=0.1):
        self.n_layers = n_layers
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.use_early_stopping = use_early_stopping
        self.target_specific_params = target_specific_params or {}
        self.use_subsample = use_subsample
        self.subsample_ratio = subsample_ratio
        self.layers = []
        self.best_layer = 0

    def _create_optimized_forests(self, X, y, layer_idx, target_name):
        """创建优化参数设置的森林"""
        forests = []

        # 获取目标特定参数
        target_params = self.target_specific_params.get(target_name, {})

        # 减少树的数量和深度以节省内存
        n_est_for_layer = max(10, self.n_estimators // ((layer_idx + 1) * 2))
        max_depth_for_layer = min(15, target_params.get('max_depth', 15))

        # 基础参数配置 - 减少内存使用
        base_rf_params = {
            'n_estimators': n_est_for_layer,
            'max_depth': max_depth_for_layer,
            'min_samples_split': target_params.get('min_samples_split', 10),
            'min_samples_leaf': target_params.get('min_samples_leaf', 4),
            'max_features': target_params.get('max_features', 0.6),
            'bootstrap': True,
            'random_state': self.random_state + layer_idx * 100,
            'n_jobs': 1,
            'verbose': 0
        }

        base_et_params = {
            'n_estimators': n_est_for_layer,
            'max_depth': max_depth_for_layer,
            'min_samples_split': target_params.get('min_samples_split', 10),
            'min_samples_leaf': target_params.get('min_samples_leaf', 4),
            'max_features': target_params.get('max_features', 0.6),
            'bootstrap': False,
            'random_state': self.random_state + layer_idx * 100 + 50,
            'n_jobs': 1,
            'verbose': 0
        }

        # 森林配置 - 减少森林数量以节省内存
        forest_configs = [
            # 配置1: 标准随机森林
            {
                'model': RandomForestRegressor,
                'params': base_rf_params
            },
            # 配置2: 极端随机树
            {
                'model': ExtraTreesRegressor,
                'params': base_et_params
            }
        ]

        # 训练所有森林
        for config in forest_configs:
            try:
                model = config['model'](**config['params'])
                model.fit(X, y)
                forests.append(model)
                # 清理内存
                gc.collect()
            except Exception as e:
                print(f"  森林训练失败: {e}")
                continue

        return forests

    def fit(self, X, y, target_name):
        """训练优化级联森林"""
        print(f"开始训练 {target_name} 的优化级联森林...")
        start_fit = time.time()

        self.layers = []
        X_current = X.copy()
        best_score = -np.inf
        self.best_layer = 0

        # 修改：移除层数限制，使用self.n_layers
        for layer in range(self.n_layers):
            print(f"  训练第 {layer + 1}/{self.n_layers} 层级联...")

            # 创建优化森林
            forests = self._create_optimized_forests(X_current, y, layer, target_name)

            if len(forests) == 0:
                print(f"  第 {layer + 1} 层无法创建森林，跳过...")
                continue

            self.layers.append(forests)

            # 评估当前层性能
            try:
                layer_predictions = []
                for forest in forests:
                    pred = forest.predict(X_current)
                    layer_predictions.append(pred.reshape(-1, 1))

                # 计算当前层的平均预测
                current_pred = np.mean(layer_predictions, axis=0).flatten()
                current_r2 = r2_score(y, current_pred)

                print(f"  第 {layer + 1} 层 R²: {current_r2:.6f}")

                # 早停检查
                if self.use_early_stopping and current_r2 > best_score:
                    best_score = current_r2
                    self.best_layer = layer
                elif self.use_early_stopping and layer > 0:
                    improvement = current_r2 - best_score
                    if improvement < 0.0005:  # 早停阈值
                        print(f"  早停触发在第 {layer + 1} 层，改进仅为 {improvement:.6f}")
                        break
            except Exception as e:
                print(f"  第 {layer + 1} 层评估失败: {e}")
                continue

            # 如果不是最后一层，生成增强特征
            # 修改：使用self.n_layers - 1代替硬编码的2
            if layer < self.n_layers - 1 and len(layer_predictions) > 0:
                enhanced_features = np.hstack(layer_predictions)
                X_current = np.hstack([X, enhanced_features])
                print(f"  级联层 {layer + 1} 完成，特征维度: {X_current.shape[1]}")

            # 清理内存
            del layer_predictions
            gc.collect()

        # 最终预测器 - 使用更简单的模型
        if len(self.layers) == 0:
            print("  使用简单随机森林作为后备")
            self.fallback_model = RandomForestRegressor(
                n_estimators=30,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=1,
                verbose=0
            )
            self.fallback_model.fit(X, y)
            self.use_fallback = True
        else:
            print("  训练最终集成预测器...")
            # 获取目标特定参数用于最终预测器
            target_params = self.target_specific_params.get(target_name, {})
            self.final_estimator = RandomForestRegressor(
                n_estimators=max(20, self.n_estimators // 2),
                max_depth=min(15, target_params.get('max_depth', 15)),
                min_samples_split=target_params.get('min_samples_split', 10),
                min_samples_leaf=target_params.get('min_samples_leaf', 4),
                max_features=target_params.get('max_features', 0.6),
                random_state=self.random_state,
                n_jobs=1,
                verbose=0
            )
            self.final_estimator.fit(X_current, y)
            self.use_fallback = False

            # 清理内存
            del X_current
            gc.collect()

        end_fit = time.time()
        print(f"  {target_name} 级联森林训练完成，耗时: {end_fit - start_fit:.2f}秒")
        if not self.use_fallback:
            print(f"  最佳层: {self.best_layer + 1}")
            print(f"  总层数: {len(self.layers)}")

        return self

    def predict(self, X):
        """使用优化级联森林进行预测"""
        if hasattr(self, 'use_fallback') and self.use_fallback:
            return self.fallback_model.predict(X)

        X_current = X.copy()

        # 只使用最佳层之前的层进行特征增强
        for layer_idx, forests in enumerate(self.layers):
            if layer_idx > self.best_layer:
                break

            predictions = []
            for forest in forests:
                pred = forest.predict(X_current)
                predictions.append(pred.reshape(-1, 1))

            if layer_idx < len(self.layers) - 1 and len(predictions) > 0:
                enhanced_features = np.hstack(predictions)
                X_current = np.hstack([X, enhanced_features])

        result = self.final_estimator.predict(X_current)

        # 清理内存
        del X_current
        gc.collect()

        return result


def cascade_forest_predict():
    """级联森林预测方法 - 优化内存使用"""
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

    # 使用部分数据以减少内存使用
    print("注意: 使用部分数据进行训练以节省内存...")
    sample_fraction = 0.2  # 使用20%的数据

    # 采样数据
    train_indices = np.random.choice(len(train_dataSet),
                                     size=int(len(train_dataSet) * sample_fraction),
                                     replace=False)
    test_indices = np.random.choice(len(test_dataSet),
                                    size=int(len(test_dataSet) * sample_fraction),
                                    replace=False)

    X_train = train_dataSet.loc[train_indices, noise_columns].values
    y_train = train_dataSet.loc[train_indices, columns].values
    X_test = test_dataSet.loc[test_indices, noise_columns].values
    y_test = test_dataSet.loc[test_indices, columns].values

    print(f"使用 {sample_fraction * 100:.0f}% 数据")
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)

    print("\n开始优化级联森林训练...")

    # 目标特定的参数配置 - 取消层数限制
    target_specific_params = {
        'T_SONIC': {
            'n_layers': 10,  # 设置6层
            'n_estimators': 50,  # 减少树的数量
            'max_depth': 20,  # 减少深度
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 0.6
        },
        'CO2_density': {
            'n_layers': 8,
            'n_estimators': 50,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 0.6
        },
        'CO2_density_fast_tmpr': {
            'n_layers': 8,
            'n_estimators': 50,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 0.6
        },
        'H2O_density': {
            'n_layers': 8,
            'n_estimators': 50,
            'max_depth': 15,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 0.6
        },
        'H2O_sig_strgth': {
            'n_layers': 8,
            'n_estimators': 50,
            'max_depth': 15,
            'min_samples_split': 15,
            'min_samples_leaf': 6,
            'max_features': 0.5
        },
        'CO2_sig_strgth': {
            'n_layers': 8,
            'n_estimators': 50,
            'max_depth': 15,
            'min_samples_split': 15,
            'min_samples_leaf': 6,
            'max_features': 0.5
        }
    }

    # 为每个目标训练优化级联森林
    cascade_predictions = []
    cascade_models = {}

    print("\n" + "=" * 60)
    print("优化级联森林训练 (支持多层架构)")
    print("=" * 60)

    for target_idx, target_name in enumerate(columns):
        print(f"\n训练目标变量 {target_idx + 1}/{len(columns)}: {target_name}")

        # 获取目标特定配置
        target_config = target_specific_params[target_name]

        # 创建优化级联森林
        cascade_model = OptimizedCascadeForest(
            n_layers=target_config['n_layers'],
            n_estimators=target_config['n_estimators'],
            random_state=217 + target_idx,
            use_early_stopping=True,
            target_specific_params=target_specific_params
        )

        # 训练模型
        cascade_model.fit(X_train_scaled, y_train_scaled[:, target_idx], target_name)

        # 预测
        pred_scaled = cascade_model.predict(X_test_scaled)
        cascade_predictions.append(pred_scaled)
        cascade_models[target_name] = cascade_model

        # 立即评估
        pred_temp = scaler_y.inverse_transform(
            np.column_stack([pred_scaled] * len(columns))
        )[:, target_idx]
        mae = np.mean(np.abs(y_test[:, target_idx] - pred_temp))
        print(f"  {target_name} 测试MAE: {mae:.4f}")

        # 清理内存
        gc.collect()

    # 合并预测结果
    y_cascade_scaled = np.column_stack(cascade_predictions)
    y_cascade = scaler_y.inverse_transform(y_cascade_scaled)

    # 性能评估
    print("\n" + "=" * 60)
    print("级联森林性能评估")
    print("=" * 60)

    mse = mean_squared_error(y_test, y_cascade)
    r2 = r2_score(y_test, y_cascade)
    mae = np.mean(np.abs(y_test - y_cascade))

    print(f"MSE: {mse:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"MAE: {mae:.6f}")

    # 详细分析
    print("\n" + "=" * 60)
    print("级联森林详细分析")
    print("=" * 60)

    mae_per_column = np.mean(np.abs(y_test - y_cascade), axis=0)
    mse_per_column = np.mean((y_test - y_cascade) ** 2, axis=0)
    r2_per_column = [r2_score(y_test[:, i], y_cascade[:, i]) for i in range(len(columns))]

    for i, col in enumerate(columns):
        print(f"{col:>25}: MAE = {mae_per_column[i]:.4f}, MSE = {mse_per_column[i]:.4f}, R² = {r2_per_column[i]:.4f}")

    avg_mae = np.mean(mae_per_column)
    avg_mse = np.mean(mse_per_column)
    avg_r2 = np.mean(r2_per_column)

    print(f"\n平均MAE: {avg_mae:.4f}")
    print(f"平均MSE: {avg_mse:.4f}")
    print(f"平均R²: {avg_r2:.4f}")

    # 保存预测结果
    results_final = []
    for True_Value, Predicted_Value in zip(y_test, y_cascade):
        error = np.abs(True_Value - Predicted_Value)
        formatted_true_value = ' '.join(map(str, True_Value))
        formatted_predicted_value = ' '.join(map(str, Predicted_Value))
        formatted_error = ' '.join(map(str, error))
        results_final.append([formatted_true_value, formatted_predicted_value, formatted_error])

    result_df = pd.DataFrame(results_final, columns=['True_Value', 'Predicted_Value', 'Error'])
    result_df.to_csv("cascade_forest_predictions_memory_optimized.csv", index=False)
    print(f"\n预测结果已保存到: cascade_forest_predictions_memory_optimized.csv")

    # 级联森林分析
    print("\n" + "=" * 60)
    print("级联森林架构分析")
    print("=" * 60)

    for target_name in columns:
        model = cascade_models[target_name]
        if hasattr(model, 'best_layer'):
            print(f"{target_name}: 使用 {model.best_layer + 1} 层级联 (总共训练了 {len(model.layers)} 层)")
        else:
            print(f"{target_name}: 使用后备模型")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n总耗时: {total_time:.2f} 秒 ({total_time / 60:.2f} 分钟)")

    return avg_mae, avg_r2


if __name__ == "__main__":
    print("开始级联森林训练与预测 (支持多层架构)...")
    avg_mae, avg_r2 = cascade_forest_predict()
    print(f"\n🎉 级联森林预测完成!")
    print(f"平均MAE: {avg_mae:.4f}")
    print(f"平均R²: {avg_r2:.4f}")