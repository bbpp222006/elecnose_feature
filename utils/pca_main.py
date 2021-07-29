import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from feature import *
import load_data

data_dict = load_data.read_data("data")


def get_all_feature(test_data):
    feature_sum = Feature_sum(test_data)
    feature_sensitive = Feature_sensitive(test_data)
    feature_time = Feature_time(test_data)
    feature_all = np.concatenate([feature_sum,feature_sensitive,feature_time])  # feature_sum,feature_sensitive,feature_time

    return feature_all


scalar_data = {}
for name, data_list in data_dict.items():
    print(name, len(data_list))
    scalar_data[name] = [get_all_feature(single_data) for single_data in data_list]

all_data_temp = np.array(list(scalar_data.values()))
all_data_temp = all_data_temp.reshape([-1, all_data_temp.shape[-1]])
scaler = StandardScaler()
all_data_temp = scaler.fit_transform(all_data_temp)

for name, data_list in scalar_data.items():
    scalar_data[name] = scaler.transform(data_list)


# fname中选择一个你本机查询出来的字体 若没有中文字体则需要你本人手动安装
fig = plt.figure()
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

pca = PCA(n_components=2)
pca.fit(all_data_temp[:,:10])
ax = plt.subplot2grid((3,3),(0,0))
for i,(name, data_list) in enumerate(scalar_data.items()):
    new_sig = pca.transform(data_list[:,:10])
    ax.scatter(new_sig[:, 0], new_sig[:, 1], color=colors[i], alpha=0.5, label=name)
ax.legend()
ax.set_title('Response')

pca = PCA(n_components=2)
pca.fit(all_data_temp[:,10:15])
ax = plt.subplot2grid((3,3),(0,1))
for i,(name, data_list) in enumerate(scalar_data.items()):
    new_sig = pca.transform(data_list[:,10:15])
    ax.scatter(new_sig[:, 0], new_sig[:, 1], color=colors[i], alpha=0.5, label=name)
ax.legend()
ax.set_title('Sum')

pca = PCA(n_components=2)
pca.fit(all_data_temp[:,15:])
ax = plt.subplot2grid((3,3),(0,2))
for i,(name, data_list) in enumerate(scalar_data.items()):
    new_sig = pca.transform(data_list[:,15:])
    ax.scatter(new_sig[:, 0], new_sig[:, 1], color=colors[i], alpha=0.5, label=name)
ax.legend()
ax.set_title('Time')

pca = PCA(n_components=2)
pca.fit(all_data_temp[:,:15])
ax = plt.subplot2grid((3,3),(1,0))
for i,(name, data_list) in enumerate(scalar_data.items()):
    new_sig = pca.transform(data_list[:,:15])
    ax.scatter(new_sig[:, 0], new_sig[:, 1], color=colors[i], alpha=0.5, label=name)
ax.legend()
ax.set_title('Response,Sum')


pca = PCA(n_components=2)
pca.fit(all_data_temp[:,10:])
ax = plt.subplot2grid((3,3),(1,1))
for i,(name, data_list) in enumerate(scalar_data.items()):
    new_sig = pca.transform(data_list[:,10:])
    ax.scatter(new_sig[:, 0], new_sig[:, 1], color=colors[i], alpha=0.5, label=name)
ax.legend()
ax.set_title('Sum,Time')


pca = PCA(n_components=2)
test_np = np.concatenate((all_data_temp[:,-2:], all_data_temp[:,:10]),axis=1)
pca.fit(test_np)
ax = plt.subplot2grid((3,3),(1,2))
for i,(name, data_list) in enumerate(scalar_data.items()):
    test_data_list = np.concatenate((data_list[:, -2:], data_list[:, :10]), axis=1)
    new_sig = pca.transform(test_data_list)
    ax.scatter(new_sig[:, 0], new_sig[:, 1], color=colors[i], alpha=0.5, label=name)
ax.legend()
ax.set_title('Time,Response')


pca = PCA(n_components=2)
pca.fit(all_data_temp)
ax = plt.subplot2grid((3,3),(2,0),colspan=2)
for i,(name, data_list) in enumerate(scalar_data.items()):
    new_sig = pca.transform(data_list)
    ax.scatter(new_sig[:, 0], new_sig[:, 1], color=colors[i], alpha=0.5, label=name)
ax.legend()
ax.set_title('All')
plt.show()
