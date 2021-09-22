# -*- coding: utf-8 -*-
import os
import numpy as np
import re


def transpose_sig(signal):
    trans_sig = []
    for single_signal in signal:
        mean_sig = np.mean(single_signal)
        big_num = single_signal[single_signal.shape[0] // 2]
        if big_num < mean_sig:
            single_signal = 2 * np.mean(single_signal) - single_signal
        trans_sig.append(single_signal)
    return np.array(trans_sig)


def read_file_data(file_path):
    data = np.genfromtxt(file_path, delimiter="\t")
    # 删除nan序列
    data = np.delete(data, np.argwhere(np.isnan(data[1, :])), axis=1)

    # 删除时间序列
    data = np.concatenate((np.arange(data.shape[0]).reshape([-1, 1]), data), axis=1)
    temp_a = np.corrcoef(data.transpose())
    timearray_index = np.argwhere(temp_a[0] >= 0.9999)
    data = np.delete(data, timearray_index, axis=1)
    print("自动检测出时间序列，已删除", timearray_index, end=",")
    nan_flag = data.sum()
    assert np.isnan(nan_flag) == False
    # data = transpose_sig(data)
    return data


def read_data(data_path):
    return_data = {}
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.split(".")[-1] != "txt":
                continue
            file_path = root + "/" + file
            signal = read_file_data(file_path, num_sig=5)
            signal_class = root.split("/")[1]
            if signal_class not in return_data:
                return_data[signal_class] = []
            return_data[signal_class].append(signal)
            print(file_path, "---->", root, signal.shape)
    print("end")
    return return_data


# file_path = "data/coffee/8.31下午（1280）/2.肯亚.txt"
# # raw_data = np.genfromtxt(file_path, delimiter='\t')[:,:-1]
# test_data=read_file_data(file_path)
