import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from filterpy.kalman import KalmanFilter
from utils.load_data import read_file_data

def kalmanf_single(signal, noise_R=200):  # 单个传感器signal[2333,1]
    signal=signal.reshape([-1,1])
    filtered_sig = []
    dim_z = 1
    dim_x = dim_z * 3  # 位置 速度 加速度
    f = KalmanFilter(dim_x=dim_x, dim_z=dim_z)  # 加一个速度（传感器特征？）

    #     print(signal[0,:],np.zeros(dim_z))
    f.x = np.array([signal[0,0], 0, 0])  # 初始值,速度,加速度设为0
    # print(f.x)

    f.F = np.array([[1, 1, 0.5],  # 状态转移
           [0, 1, 1],
           [0, 0, 1]])
    f.H = np.array([[1, 0, 0]])  # 测量矩阵
    f.P = np.eye(dim_x) * 1  # 协方差
    f.R = np.eye(dim_z) * noise_R  # 测量误差
    for i in signal[:,0]:
        f.update(i)
        f.predict()
        filtered_sig.append(f.x)
    # print(len(filtered_sig))
    return filtered_sig

def kalmanf(signals, noise_R=200):
    filtered_sigs = []
    for signal_index in range(signals.shape[1]):
        filtered_sigs.append(kalmanf_single(signals[:,signal_index],noise_R=noise_R))
    return np.array(filtered_sigs)

file_path = "data/类别2/0.1.txt"
test_data = read_file_data(file_path)
# test_data_filterd_all =
test_data_filterd_all = np.einsum("ijk->jik",kalmanf(test_data, noise_R=100))
a = test_data_filterd_all[:,:,-1]
start_index = np.linalg.norm(a, axis=1)
print(666)
