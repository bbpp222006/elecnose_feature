import os
import numpy as np
from feature import *
import re
import pandas as pd

def read_file_data(file_path, num_sig=8):
    data = np.genfromtxt(file_path, delimiter='\t')
    nan_flag = data.sum()
    if np.isnan(nan_flag):
        print('已删除nan', end=",")
        data = data[:, :-1]
    if data.shape[1] > num_sig:
        print('已删除时间序列', end=",")
        data = data[:, -num_sig:]
    nan_flag = data.sum()
    assert np.isnan(nan_flag) == False
    assert data.shape[1] == num_sig
    return data

def read_data(data_path):
    return_data={}
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.split(".")[-1]!="txt":
                continue
            file_path=root+"/"+file
            signal=read_file_data(file_path,num_sig=5)
            signal_class = root.split("\\")[1]
            if signal_class not in return_data:
                return_data[signal_class]=[]
            return_data[signal_class].append(signal)
            print(file_path,"---->", root,signal.shape)
    print("end")
    return return_data

# a = read_data("data")
# for key, data in a.items():
#     print(key, len(data))