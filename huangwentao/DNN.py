# coding=utf-8
import os
import time
import pandas as pd
import numpy as np

# 设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

start_time = time.time()

# 加载数据集
train_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')
test_dataSet = pd.read_csv(r'modified_数据集Time_Series661_detail.dat')

# 特征列
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr',
                 'Error_H2O_density', 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density',
           'H2O_sig_strgth', 'CO2_sig_strgth']

# 准备数据
X_train = train_dataSet[noise_columns].values
y_train = train_dataSet[columns].values
X_test = test_dataSet[noise_columns].values
y_test = test_dataSet[columns].values

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)

# 使用深度神经网络而不是LSTM
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(y_train_scaled.shape[1])
])

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)

# 添加早停回调
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=10,
    restore_best_weights=True
)

# 训练模型
history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping]
)

# 预测
y_pred_scaled = model.predict(X_test_scaled, batch_size=32)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 计算并保存结果
results = []
for True_Value, Predicted_Value in zip(y_test, y_pred):
    error = np.abs(True_Value - Predicted_Value)
    formatted_true_value = ' '.join(map(str, True_Value))
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    formatted_error = ' '.join(map(str, error))
    results.append([formatted_true_value, formatted_predicted_value, formatted_error])

# 保存结果
result_df = pd.DataFrame(results, columns=['True_Value', 'Predicted_Value', 'Error'])
result_df.to_csv("result_DNN.csv", index=False)

# 计算平均误差
data = pd.read_csv("result_DNN.csv")
column3 = data.iloc[:, 2]
numbers = column3.str.split(' ', expand=True).apply(pd.to_numeric)
means = numbers.mean()

print("6个数据的平均值为：\n", means)
print(f"总平均误差: {means.mean()}")

# 计算R²分数
r2 = r2_score(y_test, y_pred)
print(f"R²分数: {r2:.4f}")

end_time = time.time()
print(f"总耗时：{end_time - start_time:.3f}秒")