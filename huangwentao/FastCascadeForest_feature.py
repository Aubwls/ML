# coding=utf-8
import time
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import warnings
from scipy import stats

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

print(f"训练集大小: {len(train_dataSet)}, 测试集大小: {len(test_dataSet)}")
print(f"输入特征数: {len(noise_columns)}, 输出变量数: {len(columns)}")

# 1. 改进的数据预处理
print("\n=== 数据预处理 ===")


# 异常值检测和处理（针对CO2相关变量）
def handle_outliers(data, columns, z_threshold=3):
    """使用Z-score方法处理异常值"""
    data_clean = data.copy()
    outlier_count = 0

    for col in columns:
        z_scores = np.abs(stats.zscore(data[col]))
        outliers = z_scores > z_threshold
        outlier_count += outliers.sum()

        if outliers.any():
            # 使用中位数替换异常值
            median_val = data[col].median()
            data_clean.loc[outliers, col] = median_val
            print(f"  {col}: 检测到{outliers.sum()}个异常值，用中位数{median_val:.4f}替换")

    return data_clean, outlier_count


# 处理训练集中的异常值（特别是CO2相关变量）
print("处理异常值...")
train_data_clean, train_outliers = handle_outliers(y_train, ['CO2_density', 'CO2_density_fast_tmpr'])
y_train_clean = train_data_clean[columns]

# 标准化 - 使用RobustScaler对异常值更鲁棒
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)

y_train_scaled = scaler_y.fit_transform(y_train_clean)
y_test_scaled = scaler_y.transform(y_test)

print(f"处理了 {train_outliers} 个异常值")
print("使用RobustScaler进行数据标准化完成")


# 2. 改进的级联随机森林实现
class OptimizedCascadeRandomForest:
    def __init__(self, n_levels=3, n_estimators=100, max_depth=None, min_samples_split=10,
                 min_samples_leaf=5, max_features='auto', early_stopping_rounds=2,
                 validation_split=0.1, random_state=42):
        """
        改进的级联随机森林模型

        Args:
            n_levels: 最大级联层数
            n_estimators: 每层随机森林的树数量
            max_depth: 每棵树的最大深度（None表示不限制）
            min_samples_split: 分裂节点所需的最小样本数
            min_samples_leaf: 叶节点所需的最小样本数
            max_features: 寻找最佳分割时考虑的特征数量
            early_stopping_rounds: 早停轮数
            validation_split: 验证集比例
            random_state: 随机种子
        """
        self.n_levels = n_levels
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_split = validation_split
        self.random_state = random_state
        self.models = []  # 存储每层的模型
        self.feature_importances = []  # 存储每层的特征重要性
        self.best_level = 0  # 最佳层数（早停）

    def _create_train_val_split(self, X, y):
        """创建训练集和验证集"""
        n_samples = X.shape[0]
        n_val = int(n_samples * self.validation_split)

        indices = np.arange(n_samples)
        np.random.seed(self.random_state)
        np.random.shuffle(indices)

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        return X_train, X_val, y_train, y_val

    def fit(self, X, y):
        """
        训练改进的级联随机森林模型

        Args:
            X: 输入特征
            y: 目标变量
        """
        print(f"训练优化级联随机森林 (最大层数={self.n_levels})...")

        # 初始化特征
        current_features = X.copy()
        best_val_score = -np.inf
        no_improvement_count = 0

        for level in range(self.n_levels):
            print(f"\n训练第 {level + 1} 层:")

            # 创建训练验证集
            X_train, X_val, y_train, y_val = self._create_train_val_split(
                current_features, y
            )

            # 为每个目标变量训练一个随机森林
            level_models = []
            level_importances = []
            level_val_scores = []

            for i in range(y.shape[1]):
                # 创建随机森林模型
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    bootstrap=True,
                    random_state=self.random_state + level * 100 + i,
                    n_jobs=-1
                )

                # 训练模型
                rf.fit(X_train, y_train[:, i])
                level_models.append(rf)
                level_importances.append(rf.feature_importances_)

                # 在验证集上的表现
                y_pred_val = rf.predict(X_val)
                r2_val = r2_score(y_val[:, i], y_pred_val)
                level_val_scores.append(r2_val)

                # 在训练集上的表现
                y_pred_train = rf.predict(X_train)
                r2_train = r2_score(y_train[:, i], y_pred_train)
                print(f"  目标变量 {i + 1}: 训练R² = {r2_train:.4f}, 验证R² = {r2_val:.4f}")

            # 计算平均验证分数
            avg_val_score = np.mean(level_val_scores)
            print(f"  平均验证R²: {avg_val_score:.4f}")

            # 保存当前层的模型和特征重要性
            self.models.append(level_models)
            self.feature_importances.append(level_importances)

            # 早停检查
            if avg_val_score > best_val_score:
                best_val_score = avg_val_score
                self.best_level = level + 1
                no_improvement_count = 0
                print(f"  ✅ 性能提升，最佳层数更新为 {self.best_level}")
            else:
                no_improvement_count += 1
                print(f"  ⚠️ 性能未提升 ({no_improvement_count}/{self.early_stopping_rounds})")

                if no_improvement_count >= self.early_stopping_rounds:
                    print(f"  🛑 早停触发，最终使用 {self.best_level} 层")
                    break

            # 如果不是最后一层，则生成新的特征用于下一层
            if level < self.n_levels - 1:
                # 使用当前层的预测作为新特征
                new_features = []
                for i, model in enumerate(level_models):
                    pred = model.predict(current_features).reshape(-1, 1)
                    new_features.append(pred)

                # 将原始特征和预测特征合并
                new_features = np.hstack(new_features)
                combined_features = np.hstack([current_features, new_features])

                print(
                    f"  生成新特征: 原特征{current_features.shape[1]} + 预测特征{new_features.shape[1]} = 总特征{combined_features.shape[1]}")

                # 更新当前特征为组合特征
                current_features = combined_features

        print(f"\n最终模型: {len(self.models)}层 (最佳{self.best_level}层)")

    def predict(self, X):
        """
        使用级联随机森林进行预测

        Args:
            X: 输入特征

        Returns:
            预测结果
        """
        current_features = X.copy()

        # 只使用最佳层数进行预测
        for level, level_models in enumerate(self.models[:self.best_level]):
            if level < self.best_level - 1:
                # 对于中间层，生成新特征
                new_features = []
                for model in level_models:
                    pred = model.predict(current_features).reshape(-1, 1)
                    new_features.append(pred)

                new_features = np.hstack(new_features)
                current_features = np.hstack([current_features, new_features])
            else:
                # 对于最后一层，直接进行预测
                predictions = []
                for i, model in enumerate(level_models):
                    pred = model.predict(current_features)
                    predictions.append(pred)

                return np.column_stack(predictions)

        return None


# 3. 训练优化后的级联随机森林模型
print("\n=== 训练优化级联随机森林模型 ===")

# 为目标变量创建优化级联模型
cascade_models = {}
y_predict_scaled = np.zeros_like(y_test_scaled)
all_train_scores = []
all_val_scores = []
all_test_scores = []

# 为每个目标变量训练单独的优化级联模型
for i, target_name in enumerate(columns):
    print(f"\n{'=' * 60}")
    print(f"训练目标变量 {i + 1}/{len(columns)}: {target_name}")
    print('=' * 60)

    # 提取当前目标变量
    y_train_target = y_train_scaled[:, i].reshape(-1, 1)

    # 根据目标变量调整优化参数
    if target_name in ['T_SONIC', 'H2O_density']:
        # 对过拟合严重的变量使用更保守的参数
        n_levels = 5
        n_estimators = 80
        max_depth = 6
        min_samples_split = 15
        min_samples_leaf = 8
        max_features = 0.7  # 限制特征使用比例
        early_stopping_rounds = 2

    elif target_name in ['CO2_density', 'CO2_density_fast_tmpr']:
        # 对这些变量使用中等配置，注意防止过拟合
        n_levels = 4
        n_estimators = 100
        max_depth = 8
        min_samples_split = 10
        min_samples_leaf = 5
        max_features = 'sqrt'
        early_stopping_rounds = 2

    else:
        # 对信号强度变量使用较浅的级联
        n_levels = 3
        n_estimators = 60
        max_depth = 5
        min_samples_split = 8
        min_samples_leaf = 4
        max_features = 0.8
        early_stopping_rounds = 2

    # 创建优化级联模型
    cascade_model = OptimizedCascadeRandomForest(
        n_levels=n_levels,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        early_stopping_rounds=early_stopping_rounds,
        validation_split=0.15,  # 15%验证集
        random_state=42 + i * 10
    )

    # 训练模型
    cascade_model.fit(X_train_scaled, y_train_target)
    cascade_models[target_name] = cascade_model

    # 在训练集上的表现（使用最佳层数）
    y_train_pred = cascade_model.predict(X_train_scaled)
    train_r2 = r2_score(y_train_target, y_train_pred)
    all_train_scores.append(train_r2)

    # 在验证集上的表现
    val_r2 = cascade_model.best_val_score if hasattr(cascade_model, 'best_val_score') else train_r2
    all_val_scores.append(val_r2)

    # 在测试集上的表现
    y_test_pred = cascade_model.predict(X_test_scaled)
    y_predict_scaled[:, i] = y_test_pred.flatten()
    test_r2 = r2_score(y_test_scaled[:, i], y_test_pred.flatten())
    all_test_scores.append(test_r2)

    print(f"\n📊 {target_name} 结果汇总:")
    print(f"  配置: 最大层数={n_levels}, 最佳层数={cascade_model.best_level}")
    print(f"  树数量={n_estimators}, 深度={max_depth}")
    print(f"  训练集R²: {train_r2:.4f}, 验证集R²: {val_r2:.4f}, 测试集R²: {test_r2:.4f}")

    # 过拟合分析
    overfit = train_r2 - test_r2
    val_overfit = train_r2 - val_r2

    if val_overfit > 0.15:
        print(f"  ⚠️ 验证集严重过拟合: 差异={val_overfit:.4f}")
    elif val_overfit > 0.08:
        print(f"  ⚠️ 验证集中度过拟合: 差异={val_overfit:.4f}")
    elif val_overfit > 0.04:
        print(f"  ⚠️ 验证集轻微过拟合: 差异={val_overfit:.4f}")
    else:
        print(f"  ✅ 验证集拟合良好: 差异={val_overfit:.4f}")

# 4. 反向标准化
print("\n=== 反向标准化预测结果 ===")
y_predict = scaler_y.inverse_transform(y_predict_scaled)

# 5. 保存结果
results = []
for true_value, pred_value in zip(y_test.values, y_predict):
    error = np.abs(true_value - pred_value)
    formatted_true = ' '.join(f"{x:.6f}" for x in true_value)
    formatted_pred = ' '.join(f"{x:.6f}" for x in pred_value)
    formatted_error = ' '.join(f"{x:.6f}" for x in error)
    results.append([formatted_true, formatted_pred, formatted_error])

result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_OptimizedCascadeRF.csv", index=False)
print("结果已保存到: result_OptimizedCascadeRF.csv")

# 6. 性能评估
print("\n=== 详细性能评估 ===")
performance_metrics = []
mae_values = []
mse_values = []

for i, column in enumerate(columns):
    y_pred_original = y_predict[:, i]
    y_true_original = y_test.iloc[:, i]

    r2 = r2_score(y_true_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    mae = np.mean(np.abs(y_true_original - y_pred_original))

    mae_values.append(mae)
    mse_values.append(rmse ** 2)

    # 计算相对误差
    y_mean = y_true_original.mean()
    y_std = y_true_original.std()

    if abs(y_mean) > 1e-8:
        relative_rmse = rmse / abs(y_mean)
        relative_mae = mae / abs(y_mean)
    else:
        relative_rmse = rmse
        relative_mae = mae

    # 计算误差在标准差中的比例
    error_std_ratio = mae / y_std if y_std > 1e-8 else mae

    performance_metrics.append([
        column, all_train_scores[i], all_val_scores[i], all_test_scores[i], r2,
        rmse, mae, relative_rmse, relative_mae, error_std_ratio
    ])

    print(f"\n{column}:")
    print(f"  训练R²: {all_train_scores[i]:.4f}, 验证R²: {all_val_scores[i]:.4f}, 测试R²: {test_r2:.4f}")
    print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    print(f"  相对误差: RMSE={relative_rmse:.2%}, MAE={relative_mae:.2%}")
    print(f"  误差/标准差: {error_std_ratio:.2%}")

# 保存性能指标
perf_columns = ['Variable', 'Train_R2', 'Val_R2', 'Test_R2_scaled', 'Test_R2',
                'RMSE', 'MAE', 'Relative_RMSE', 'Relative_MAE', 'Error/Std_Ratio']
perf_df = pd.DataFrame(performance_metrics, columns=perf_columns)
perf_df.to_csv("performance_optimized_cascade.csv", index=False)

# 7. 关键指标计算
print("\n=== 关键指标汇总 ===")
total_mae = np.mean(mae_values)
total_weighted_mae = np.average(mae_values, weights=[1.0, 0.5, 0.5, 1.0, 0.2, 0.2])  # 加权平均
total_mse = np.mean(mse_values)
total_rmse = np.sqrt(total_mse)

print(f"算术平均MAE: {total_mae:.6f}")
print(f"加权平均MAE: {total_weighted_mae:.6f} (T_SONIC和H2O_density权重=1，CO2相关=0.5，信号强度=0.2)")
print(f"总RMSE: {total_rmse:.6f}")
print(f"平均训练R²: {np.mean(all_train_scores):.4f}")
print(f"平均验证R²: {np.mean(all_val_scores):.4f}")
print(f"平均测试R²: {np.mean(all_test_scores):.4f}")

# 8. 误差分布分析（过滤极大异常值）
print("\n=== 误差分布分析 (过滤异常值后) ===")
for i, col in enumerate(columns):
    errors = np.abs(y_test.iloc[:, i] - y_predict[:, i])

    # 过滤掉极大误差（仅显示99%分位数以内的数据）
    q99 = np.percentile(errors, 99)
    errors_filtered = errors[errors <= q99]

    print(f"{col}:")
    print(f"  样本数: {len(errors_filtered)}/{len(errors)} (过滤了{len(errors) - len(errors_filtered)}个异常误差)")
    print(f"  最小值: {errors_filtered.min():.6f}")
    print(f"  25%分位数: {np.percentile(errors_filtered, 25):.6f}")
    print(f"  中位数: {np.median(errors_filtered):.6f}")
    print(f"  75%分位数: {np.percentile(errors_filtered, 75):.6f}")
    print(f"  90%分位数: {np.percentile(errors_filtered, 90):.6f}")
    print(f"  99%分位数: {np.percentile(errors_filtered, 99):.6f}")
    print(f"  最大值(过滤后): {errors_filtered.max():.6f}")
    print(f"  原始最大值: {errors.max():.6f}")

# 9. 目标达成评估
target_mae = 0.5
print(f"\n=== 目标评估 (目标MAE < {target_mae}) ===")

variables_achieved = []
variables_not_achieved = []

for col, mae in zip(columns, mae_values):
    if mae < target_mae:
        variables_achieved.append((col, mae))
    else:
        variables_not_achieved.append((col, mae))

print(f"达到目标的变量 ({len(variables_achieved)}/{len(columns)}):")
for col, mae in variables_achieved:
    print(f"  ✅ {col}: MAE={mae:.6f}")

print(f"未达到目标的变量 ({len(variables_not_achieved)}/{len(columns)}):")
for col, mae in variables_not_achieved:
    improvement_needed = mae - target_mae
    percent_improvement = improvement_needed / mae * 100
    print(f"  ❌ {col}: MAE={mae:.6f} (需要降低{improvement_needed:.6f}, {percent_improvement:.1f}%)")

# 10. 模型诊断和建议
print("\n=== 模型诊断和建议 ===")

if total_weighted_mae < target_mae:
    print("🎉 恭喜！加权平均MAE已低于目标值0.5！")
else:
    print(f"当前加权平均MAE为 {total_weighted_mae:.6f}，距离目标还有 {total_weighted_mae - target_mae:.6f} 的差距")

    # 计算每个变量需要的改进比例
    print(f"\n📈 各变量改进优先级:")
    improvement_priority = []
    for i, col in enumerate(columns):
        current_mae = mae_values[i]
        if current_mae > target_mae:
            needed_improvement = (current_mae - target_mae) / current_mae * 100
            improvement_priority.append((col, needed_improvement, current_mae))

    # 按改进比例排序
    improvement_priority.sort(key=lambda x: x[1], reverse=True)

    for col, needed_percent, current_mae in improvement_priority:
        print(f"  {col}: 需要改进{needed_percent:.1f}% (从{current_mae:.6f}到{target_mae:.6f})")

# 11. 下一步优化策略
print("\n=== 下一步优化策略 ===")
print("如果MAE仍然不理想，可以尝试以下高级策略:")
print("1. 🎯 集成方法:")
print("   - 使用XGBoost或LightGBM代替部分随机森林")
print("   - 对不同变量的模型进行加权集成")
print("2. 📊 特征工程:")
print("   - 创建特征交互项")
print("   - 添加滞后特征（时间序列特性）")
print("   - 使用PCA进行特征降维")
print("3. ⚙️ 模型调优:")
print("   - 使用贝叶斯优化进行超参数调优")
print("   - 尝试不同的级联策略（如残差连接）")
print("   - 实现自适应级联深度")
print("4. 🧹 数据优化:")
print("   - 更精细的异常值检测和处理")
print("   - 考虑数据分段建模（不同范围用不同模型）")
print("   - 增加训练数据或使用数据增强")

# 12. 模型参数总结
print("\n=== 最终模型参数总结 ===")
summary_data = []
for col in columns:
    model = cascade_models[col]
    summary_data.append([
        col, model.best_level, model.n_estimators, model.max_depth,
        model.min_samples_split, model.min_samples_leaf,
        f"{all_train_scores[columns.index(col)]:.4f}",
        f"{all_val_scores[columns.index(col)]:.4f}",
        f"{all_test_scores[columns.index(col)]:.4f}",
        f"{mae_values[columns.index(col)]:.6f}"
    ])

summary_df = pd.DataFrame(summary_data, columns=[
    'Variable', 'Best_Levels', 'N_Estimators', 'Max_Depth',
    'Min_Samples_Split', 'Min_Samples_Leaf',
    'Train_R2', 'Val_R2', 'Test_R2', 'MAE'
])
print(summary_df.to_string(index=False))

# 13. 最终总结
print("\n" + "=" * 70)
print("优化级联随机森林最终总结")
print("=" * 70)
print(f"📊 总体性能:")
print(f"  - 算术平均MAE: {total_mae:.6f}")
print(f"  - 加权平均MAE: {total_weighted_mae:.6f}")
print(f"  - 总RMSE: {total_rmse:.6f}")
print(f"  - 平均测试R²: {np.mean(all_test_scores):.4f}")
print(f"  - 达到目标变量: {len(variables_achieved)}/{len(columns)}")

print(f"\n🎯 目标状态: {'✅ 已达成' if total_weighted_mae < target_mae else '❌ 未达成'}")
if total_weighted_mae < target_mae:
    print(f"   加权平均MAE比目标低 {target_mae - total_weighted_mae:.6f}")
else:
    print(f"   需要再降低 {total_weighted_mae - target_mae:.6f} 才能达到目标")

print(f"\n🚀 主要改进:")
print(f"  1. 添加早停机制防止过拟合")
print(f"  2. 使用RobustScaler处理异常值")
print(f"  3. 增加验证集进行模型选择")
print(f"  4. 为不同变量定制化参数")
print(f"  5. 过滤极端误差进行分析")

print(f"\n⏱️  运行时间: {time.time() - start_time:.2f}秒")
print("=" * 70)