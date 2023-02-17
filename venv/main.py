import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
from itertools import islice
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
import sys
import os
colors = ['red','green','blue','purple']
def window(seq, ws=2):
    it = iter(seq)
    result = tuple(islice(it, ws))
    if len(result) == ws:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
def cvt_gen2ary(inp):
    return np.array(list(inp))

def plot_raw(data):
    plt.scatter(data[:, 0], y=data[:, 1])
    plt.show()

def preprocess(data, fixed_t):
    del_idx = []
    for i in range (data.shape[0]):
        if abs(data[i, 1]) > fixed_t:
            del_idx.append(i)
    return np.delete(data, del_idx, axis=0)

def plot_cluster(data, labels):
    plt.scatter(data[:, 0], y=data[:, 1])
    plt.show()

folder = './data/*.npz'
WINDOW = 10

for i in glob.glob(folder):
    data = np.load(i)
    train_ts, train_dl, test_ts_1gal, test_dl_1gal = data['train_ts'], data['train_dl'], data['test_ts_1gal'], data['test_dl_1gal']
    train_tsw = window(train_ts[:, 1], WINDOW)
    train_dlw = window(train_dl[:, 1], WINDOW)
    test_tsw_1gal = window(test_ts_1gal[:, 1], WINDOW)
    test_dlw_1gal = window(test_dl_1gal[:, 1], WINDOW)
    X_ts, X_dl, X_ts_1gal, X_dl_1gal = cvt_gen2ary(train_tsw), cvt_gen2ary(train_dlw), cvt_gen2ary(
        test_tsw_1gal), cvt_gen2ary(test_dlw_1gal)
    break
train_dl = preprocess(train_dl, 4)
test_dl_1gal = preprocess(test_dl_1gal, 4)
# plot_raw(train_dl)
# plot_raw(test_dl_1gal)
arr = np.concatenate((train_dl, test_dl_1gal), axis=0)
arr_var = window(arr[:, 1], WINDOW)
arr_var = cvt_gen2ary(arr_var)

# db_ts = DBSCAN(eps=0.2, min_samples=10).fit(arr)
# labels = db_ts.labels_
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print("Estimated number of clusters: %d" % n_clusters_)
# db_ts.fit(X_ts_1gal)
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)
# print("Estimated number of clusters: %d" % n_clusters_)


kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(arr_var)
labels = kmeans.labels_
fig = plt.figure(figsize=(8, 3))
plt.scatter(x=arr[5:-4, 0], y=arr[5:-4, 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()