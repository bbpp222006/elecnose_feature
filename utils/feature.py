# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from filterpy.kalman import KalmanFilter


# def kalmanf(signal, noise_R=66):
#     filtered_sig = []
#     dim_z = dim_x = signal.shape[1]
#     f = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
#
#     f.x = np.array(signal[0, :])  # 初始值
#     f.F = np.eye(dim_x)  # 状态转移
#     f.H = np.eye(dim_x) * 1  # 测量矩阵
#     f.P = np.eye(dim_x) * 1  # 协方差
#     f.R = np.eye(dim_x) * noise_R  # 测量误差
#     for i in signal:
#         f.update(i)
#         f.predict()
#         filtered_sig.append(f.x)
#     print(len(filtered_sig))
#     return np.array(filtered_sig)

def transpose_sig(signal):
    trans_sig = []
    for single_signal in signal:
        mean_sig = np.mean(single_signal)
        big_num = single_signal[single_signal.shape[0]//2]
        if big_num<mean_sig:
            single_signal = 2 * np.mean(single_signal) - single_signal
        trans_sig.append(single_signal)
    return np.array(trans_sig)


def get_base(signal):
    base = np.mean(signal[0:10, :], axis=0)
    return base


def partial_fraction(signal):
    sig_base = get_base(signal)
    cleaned_sig = difference(signal) / sig_base
    return cleaned_sig


def difference(signal):
    sig_base = get_base(signal)
    cleaned_sig = np.abs(signal - sig_base)
    return cleaned_sig


def log_sig(signal):
    sig_base = get_base(signal)
    cleaned_sig = np.log(np.abs(signal - sig_base))
    return cleaned_sig


def get_max_index(signal, top_k):
    signal_range = np.linalg.norm(signal, axis=1)
    top_k_idx = signal_range.argsort()[::-1][0:top_k]
    return top_k_idx


def get_max(signal, top_k=10):
    max_index = get_max_index(signal, top_k=top_k)
    max_sig_list = signal[max_index]
    max_feature = np.mean(max_sig_list, axis=0)
    return np.array(max_feature)


def get_feature(signal, num_sig):
    max_feature = get_max(signal[:, :num_sig])
    vis_feature = signal[0, num_sig:]
    feature = np.concatenate([max_feature, vis_feature])
    return feature


def get_start_end_time(test_data):
    b, a = signal.butter(8, 0.02, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    temp_sig = []
    for i in range(test_data.shape[1]):  # 滤波
        temp_sig.append(signal.filtfilt(b, a, test_data[:, i]))
    filterd_data = np.einsum('ij->ji', np.array(temp_sig))

    scaler = StandardScaler()
    test_data = scaler.fit_transform(test_data)
    filterd_data = scaler.fit_transform(filterd_data)  # 归一化，方便绘图

    diff = filterd_data[:-1, :] - filterd_data[1:, :]  # 一阶差分
    start_index = np.linalg.norm(diff[diff.shape[0] // 5:diff.shape[0] // 2, :], axis=1).argsort()[::-1][0]+diff.shape[0] // 5
    end_index = diff.shape[0] // 2 + np.linalg.norm(diff[diff.shape[0] // 2:diff.shape[0]*4 // 5, :], axis=1).argsort()[::-1][0]
    return start_index, end_index


def get_base_index(test_data_raw, top_k=20):
    start_index, end_index = get_start_end_time(test_data_raw)
    scaler = MinMaxScaler()
    test_data = scaler.fit_transform(test_data_raw)
    signal_range = np.linalg.norm(test_data[start_index - 50:start_index + 50, :], axis=1)
    base_index = signal_range.argsort()[::1][0:top_k] + start_index - 50
    #     sig_base=np.mean(test_data_raw[base_index], axis=0)
    return base_index


def get_high_index(test_data_raw, top_k=20):
    start_index, end_index = get_start_end_time(test_data_raw)
    scaler = MinMaxScaler()
    test_data = scaler.fit_transform(test_data_raw)
    signal_range = np.linalg.norm(test_data[start_index:end_index, :], axis=1)
    high_idx = signal_range.argsort()[::-1][0:top_k] + start_index
    #     sig_high=np.mean(test_data_raw[high_idx], axis=0)
    return high_idx


def get_sensitive(test_data_raw):
    base_index = get_base_index(test_data_raw)  # 基线
    sig_base = np.mean(test_data_raw[base_index], axis=0)

    high_idx = get_high_index(test_data_raw)  # 响应
    sig_high = np.mean(test_data_raw[high_idx], axis=0)

    return sig_high / sig_base


def get_sig_high_90_idx(test_data):
    start_index, end_index = get_start_end_time(test_data)
    high_idxs = get_high_index(test_data)
    high_idx = int(np.mean(high_idxs))
    sig_high = np.mean(test_data[high_idxs], axis=0)

    signal_range = np.linalg.norm(test_data[start_index:high_idx, :] - 0.9 * sig_high, axis=1)
    sig_high_90_idx = signal_range.argsort()[::1][0:1] + start_index
    return sig_high_90_idx[0]


def get_sig_base_90_idx(test_data):
    start_index, end_index = get_start_end_time(test_data)

    base_idxs = get_base_index(test_data)  # 找基线
    sig_base = np.mean(test_data[base_idxs], axis=0)

    high_idxs = get_high_index(test_data)  # 找响应
    sig_high = np.mean(test_data[high_idxs], axis=0)

    target_sig = sig_base + 0.1 * (sig_high - sig_base)
    signal_range = np.linalg.norm(test_data[end_index:, :] - target_sig, axis=1)
    sig_base_90_idx = signal_range.argsort()[::1][0:1] + end_index
    return sig_base_90_idx[0]


def Feature_sum(test_data):
    start_index, end_index = get_start_end_time(test_data)

    high_90_idx = get_sig_high_90_idx(test_data)
    base_90_idx = get_sig_base_90_idx(test_data)

    up_sum = np.sum(test_data[start_index:high_90_idx, :], axis=0)
    down_sum = np.sum(test_data[end_index:base_90_idx, :], axis=0)
    return np.concatenate([up_sum, down_sum])


def Feature_sensitive(signal):
    return np.array(get_sensitive(signal))


def Feature_time(signal):
    start_index, end_index = get_start_end_time(signal)
    high_90_idx = get_sig_high_90_idx(signal)
    base_90_idx = get_sig_base_90_idx(signal)
    return np.array([high_90_idx - start_index, base_90_idx - end_index])

def get_all_feature(test_data):
    feature_sum = Feature_sum(test_data)
    feature_sensitive = Feature_sensitive(test_data)
    feature_time = Feature_time(test_data)
    feature_all = np.concatenate([feature_sum,feature_sensitive,feature_time])  # feature_sum,feature_sensitive,feature_time

    return feature_all