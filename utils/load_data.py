import os
import numpy as np
from .feature import *
import re
import pandas as pd



# 这里只做简单的数据预处理，若需求变更复杂，需另外单独进行处理后再保存
def read_file_data(file_path): #至少20个以上的采样点！
    data = np.genfromtxt(file_path, delimiter='\t')
    # 删除nan序列
    data = np.delete(data,np.argwhere(np.isnan(data[1,:])),axis=1)
    data = np.concatenate((data,np.arange(data.shape[0]).reshape([-1,1])),axis=1)
    # 删除时间序列
    # if num_sig and data.shape[1] > num_sig:
    #     print('已删除时间序列', end=",")
    #     data = data[:, -num_sig:]
    # else:
    count_list=np.zeros(data.shape[1])
    sig_len = data.shape[0]
    for i in range(10): #取10次，找出时间序列
        rand_indexs =np.sort( np.random.randint(0,sig_len,3))
        temp_a=np.abs((rand_indexs[1]-rand_indexs[2])*data[rand_indexs[0],:]-(rand_indexs[0]-rand_indexs[2])*data[rand_indexs[1],:]+(rand_indexs[0]-rand_indexs[1])*data[rand_indexs[2],:])
        zero_index = np.argwhere(temp_a<1e-5)
        count_list[zero_index] += 1
    timearray_index = np.argwhere(count_list>=9)
    data = np.delete(data, timearray_index, axis=1)
    print('自动检测出时间序列，已处理', end=",")
    nan_flag = data.sum()
    assert np.isnan(nan_flag) == False
    # data = transpose_sig(data)
    # assert data.shape[1] == num_sig
    return data

def read_data(data_path): #,num_sig=None
    return_data={}
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.split(".")[-1]!="txt":
                continue
            file_path=root+"/"+file
            signal=read_file_data(file_path)#,num_sig=num_sig
            signal_class = root.split("\\")[1]
            if signal_class not in return_data:
                return_data[signal_class]=[]
            return_data[signal_class].append(signal)
            print(file_path,"---->", root,signal.shape)
    print("end")
    return return_data

# a = read_file_data(r"G:\pycharm_pro\elecnose_feature\data\类别1\1.txt")
# print(213)
# for key, data in a.items():
#     print(key, len(data))