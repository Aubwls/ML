# 时间：2024年6月8号  Date： June 16, 2024
# 文件名称 Filename： RF_with_denoising.py
# 编码实现 Coding by： Hongjie Liu , Suiwen Zhang 邮箱 Mailbox：redsocks1043@163.com
# 所属单位：中国 成都，西南民族大学（Southwest  University of Nationality，or Southwest Minzu University）, 计算机科学与工程学院.
# 指导老师：周伟老师
# coding=utf-8
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


# 去噪方法实现
class DenoisingMethods:
    """多种时间序列去噪方法"""

    @staticmethod
    def moving_average(signal, window_size=5):
        """移动平均去噪"""
        return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

    @staticmethod
    def exponential_smoothing(signal, alpha=0.3):
        """指数平滑去噪"""
        smoothed = np.zeros_like(signal)
        smoothed[0] = signal[0]
        for i in range(1, len(signal)):
            smoothed[i] = alpha * signal[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    @staticmethod
    def median_filter(signal, window_size=5):
        """中值滤波去噪"""
        padded = np.pad(signal, (window_size // 2, window_size // 2), mode='edge')
        result = np.zeros_like(signal)
        for i in range(len(signal)):
            result[i] = np.median(padded[i:i + window_size])
        return result

    @staticmethod
    def savgol_filter(signal, window_length=11, polyorder=2):
        """Savitzky-Golay滤波器"""
        try:
            from scipy.signal import savgol_filter
            return savgol_filter(signal, window_length, polyorder)
        except ImportError:
            print("scipy未安装，使用移动平均替代")
            return DenoisingMethods.moving_average(signal, window_length)

    @staticmethod
    def wavelet_denoise(signal, wavelet='db4', level=1):
        """小波去噪"""
        try:
            import pywt
            # 小波分解
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            # 计算阈值
            sigma = np.median(np.abs(coeffs[-level])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(signal)))
            # 应用软阈值
            coeffs_thresh = []
            for i, coeff in enumerate(coeffs):
                if i == 0:  # 近似系数保留
                    coeffs_thresh.append(coeff)
                else:  # 细节系数阈值处理
                    coeffs_thresh.append(pywt.threshold(coeff, threshold, mode='soft'))
            # 小波重构
            denoised_signal = pywt.waverec(coeffs_thresh, wavelet)
            # 确保长度一致
            if len(denoised_signal) > len(signal):
                denoised_signal = denoised_signal[:len(signal)]
            elif len(denoised_signal) < len(signal):
                denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)))
            return denoised_signal
        except ImportError:
            print("PyWavelets未安装，使用移动平均替代")
            return DenoisingMethods.moving_average(signal)


class DenoisingRandomForest:
    """带去噪处理的随机森林回归器"""

    def __init__(self, denoising_method='moving_average', denoising_params=None, rf_params=None):
        self.denoising_method = denoising_method
        self.denoising_params = denoising_params or {}
        self.rf_params = rf_params or {}
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.models = {}

    def apply_denoising(self, data):
        """应用去噪处理"""
        denoised_data = np.zeros_like(data)

        for i in range(data.shape[1]):
            signal = data[:, i]

            if self.denoising_method == 'moving_average':
                window_size = self.denoising_params.get('window_size', 5)
                denoised_data[:, i] = DenoisingMethods.moving_average(signal, window_size)

            elif self.denoising_method == 'exponential_smoothing':
                alpha = self.denoising_params.get('alpha', 0.3)
                denoised_data[:, i] = DenoisingMethods.exponential_smoothing(signal, alpha)

            elif self.denoising_method == 'median_filter':
                window_size = self.denoising_params.get('window_size', 5)
                denoised_data[:, i] = DenoisingMethods.median_filter(signal, window_size)

            elif self.denoising_method == 'savgol_filter':
                window_length = self.denoising_params.get('window_length', 11)
                polyorder = self.denoising_params.get('polyorder', 2)
                denoised_data[:, i] = DenoisingMethods.savgol_filter(signal, window_length, polyorder)

            elif self.denoising_method == 'wavelet':
                wavelet = self.denoising_params.get('wavelet', 'db4')
                level = self.denoising_params.get('level', 1)
                denoised_data[:, i] = DenoisingMethods.wavelet_denoise(signal, wavelet, level)

            else:
                # 默认不使用去噪
                denoised_data[:, i] = signal

        return denoised_data

    def fit(self, X, y):
        """训练模型"""
        print(f"应用 {self.denoising_method} 去噪处理...")

        # 对特征进行去噪
        X_denoised = self.apply_denoising(X)

        # 数据标准化
        X_scaled = self.scaler_X.fit_transform(X_denoised)
        y_scaled = self.scaler_y.fit_transform(y)

        # 为每个目标变量训练一个随机森林模型
        self.models = {}
        n_outputs = y.shape[1]

        # 默认随机森林参数
        default_rf_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 217,
            'n_jobs': -1,
            'verbose': 0
        }

        # 更新参数
        rf_params = {**default_rf_params, **self.rf_params}

        for i in range(n_outputs):
            print(f"训练目标变量 {i + 1}/{n_outputs}")

            rf = RandomForestRegressor(**rf_params)
            rf.fit(X_scaled, y_scaled[:, i])
            self.models[i] = rf

        return self

    def predict(self, X):
        """预测"""
        # 对测试集特征进行去噪
        X_denoised = self.apply_denoising(X)
        X_scaled = self.scaler_X.transform(X_denoised)

        predictions = []
        for i in range(len(self.models)):
            pred_scaled = self.models[i].predict(X_scaled)
            predictions.append(pred_scaled)

        # 合并预测结果并反标准化
        predictions_scaled = np.column_stack(predictions)
        predictions_original = self.scaler_y.inverse_transform(predictions_scaled)

        return predictions_original


def compare_denoising_methods(X_train, y_train, X_test, y_test, columns):
    """比较不同去噪方法的效果"""

    # 定义不同的去噪方法和参数
    denoising_configs = [
        {
            'name': '无去噪',
            'method': 'none',
            'params': {}
        },
        {
            'name': '移动平均',
            'method': 'moving_average',
            'params': {'window_size': 5}
        },
        {
            'name': '指数平滑',
            'method': 'exponential_smoothing',
            'params': {'alpha': 0.3}
        },
        {
            'name': '中值滤波',
            'method': 'median_filter',
            'params': {'window_size': 5}
        },
        {
            'name': 'Savitzky-Golay',
            'method': 'savgol_filter',
            'params': {'window_length': 11, 'polyorder': 2}
        },
        {
            'name': '小波去噪',
            'method': 'wavelet',
            'params': {'wavelet': 'db4', 'level': 1}
        }
    ]

    # 随机森林参数
    rf_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
        'random_state': 217,
        'n_jobs': -1,
        'verbose': 0
    }

    results = []
    best_r2 = -np.inf
    best_method = None
    best_predictions = None

    for config in denoising_configs:
        print(f"\n{'=' * 60}")
        print(f"测试方法: {config['name']}")
        print('=' * 60)

        start_time = time.time()

        # 创建带去噪的随机森林模型
        model = DenoisingRandomForest(
            denoising_method=config['method'],
            denoising_params=config['params'],
            rf_params=rf_params
        )

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)

        # 评估
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))

        end_time = time.time()
        training_time = end_time - start_time

        print(f"{config['name']} - R²: {r2:.6f}, MSE: {mse:.6f}, MAE: {mae:.6f}, 时间: {training_time:.2f}s")

        # 保存结果
        results.append({
            'method': config['name'],
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'time': training_time
        })

        # 更新最佳方法
        if r2 > best_r2:
            best_r2 = r2
            best_method = config['name']
            best_predictions = y_pred

    # 显示比较结果
    print(f"\n{'=' * 80}")
    print("去噪方法比较结果")
    print('=' * 80)
    print(f"{'方法':<20} {'R²':<10} {'MSE':<12} {'MAE':<12} {'时间(s)':<10}")
    print('-' * 80)

    for result in results:
        print(f"{result['method']:<20} {result['r2']:<10.6f} {result['mse']:<12.6f} "
              f"{result['mae']:<12.6f} {result['time']:<10.2f}")

    print(f"\n最佳方法: {best_method}, R²: {best_r2:.6f}")

    return best_method, best_predictions, results


def main():
    start_time = time.time()

    # 加载数据集
    print("加载数据集...")
    train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
    test_dataSet = pd.read_csv(r'modified_数据集Time_Series662_detail.dat')

    columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
    noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                     'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']

    # 准备数据
    X_train = train_dataSet[noise_columns].values
    y_train = train_dataSet[columns].values
    X_test = test_dataSet[noise_columns].values
    y_test = test_dataSet[columns].values

    print(f"训练集形状: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"测试集形状: X_test: {X_test.shape}, y_test: {y_test.shape}")

    # 比较不同去噪方法
    best_method, best_predictions, comparison_results = compare_denoising_methods(
        X_train, y_train, X_test, y_test, columns
    )

    # 保存最佳预测结果
    results = []
    for True_Value, Predicted_Value in zip(y_test, best_predictions):
        error = np.abs(True_Value - Predicted_Value)
        formatted_true_value = ' '.join(map(str, True_Value))
        formatted_predicted_value = ' '.join(map(str, Predicted_Value))
        formatted_error = ' '.join(map(str, error))
        results.append([formatted_true_value, formatted_predicted_value, formatted_error])

    result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
    result_df.to_csv(f"result_RF_denoised_{best_method}.csv", index=False)
    print(f"\n去噪结果已保存到: result_RF_denoised_{best_method}.csv")

    # 计算平均误差
    data = pd.read_csv(f"result_RF_denoised_{best_method}.csv")
    column3 = data.iloc[:, 2]
    numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
    means = numbers.mean()

    print(f"\n{best_method} 6个数据的平均值为：")
    print(means)
    print(f"总平均误差: {means.mean():.6f}")

    # 各目标变量的R²分数
    print("\n各目标变量R²分数:")
    for i, col in enumerate(columns):
        r2_col = r2_score(y_test[:, i], best_predictions[:, i])
        print(f"{col}: {r2_col:.4f}")

    # 保存比较结果
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv("denoising_comparison_results.csv", index=False)
    print("\n去噪方法比较结果已保存到: denoising_comparison_results.csv")

    end_time = time.time()
    print(f"\n总耗时：{end_time - start_time:.3f}秒")


if __name__ == "__main__":
    main()