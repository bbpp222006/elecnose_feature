from functools import reduce
import re
import numpy as np
import os
import utils.load_data 
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import utils.feature 
from sklearn.metrics import log_loss,classification_report
from tqdm import tqdm

def change2feature(data_dict,sig_index=None):
    """
    data_dict:正常的数据
    sig_index:要删除的信号列
    """
    if sig_index!= None:
        assert sig_index<list(data_dict.values())[0][0].shape[1]
    data_feature_dict = {}
    for i in tqdm(data_dict.items()):
        featured_data_list = []
        for j in i[1]:
            if sig_index!= None:
                j = np.delete(j,sig_index,axis=1)
            feature_data = utils.feature.get_all_feature(j)
            featured_data_list.append(feature_data)
        data_feature_dict[i[0]]=featured_data_list
    return data_feature_dict

def get_score(all_data,all_labels,test_data):
    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data)
    test_data = scaler.transform([test_data])
    clf = LinearDiscriminantAnalysis()
    clf.fit(all_data, all_labels)
    predict = clf.predict(test_data)
    scores = clf.predict_proba(test_data)
    return scores,predict

def get_all_score(feature_dict):
    data_score_dict = {}
    predict_score = [0,0]
    all_feature = np.concatenate(list(feature_dict.values()),axis=0)
    all_labels = reduce(lambda x, y: x+y, [[key]*len(value) for key,value in feature_dict.items()])
    for data_index,label in enumerate(all_labels):
        test_data = all_feature[data_index,:]
        test_feature  = np.delete(all_feature,data_index,axis=0)
        test_labels  = np.delete(all_labels,data_index,axis=0)
        scores,predict = get_score(test_feature,test_labels,test_data)
        if predict == label:
            predict_score[1]+=1
        else:
            predict_score[0]+=1
        if label not in data_score_dict:
            data_score_dict[label]=scores[0]
        else:
            data_score_dict[label]+=scores[0]
        # print(data_index,label,predict,scores[0])
    for key,values in feature_dict.items():
        data_score_dict[key] = data_score_dict[key]/len(values)
    y_score = np.array(list(data_score_dict.values()))
    y_test = np.eye(class_num)
    a = log_loss(y_true=y_test,y_pred=y_score)
    predict_score = predict_score[1]/(predict_score[0]+predict_score[1])*100
    return a,predict_score

path = "data"

data_dict = utils.load_data.read_data(path)

class_num = len(list(data_dict.keys()))
sig_num = list(data_dict.values())[0][0].shape[1]

feature_dict = change2feature(data_dict)
a,predict_score = get_all_score(feature_dict)
print("完整传感器阵列的交叉熵：",a,"准确率：",predict_score,"%")
for sig2test in range(sig_num):
    feature_dict = change2feature(data_dict,sig2test)
    a,predict_score = get_all_score(feature_dict)
    print("删除第",sig2test,"个传感器的交叉熵：",a,"准确率：",predict_score,"%")


scaler = MinMaxScaler(feature_range=())


input()