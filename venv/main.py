import numpy as np
import matplotlib.pyplot as plt
import glob
from itertools import islice
from sklearn.cluster import DBSCAN
from sklearn import metrics
import sys
import os

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

folder = './data/*.npz'
WINDOW = 20

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
plot_raw(train_dl)
plot_raw(test_dl_1gal)


# db_ts = DBSCAN(eps=0.3, min_samples=10).fit(X_ts)
# labels = db_ts.labels_
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)
# print("Estimated number of clusters: %d" % n_clusters_)
# db_ts.fit(X_ts_1gal)
# labels = db_ts.labels_
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)
# print("Estimated number of clusters: %d" % n_clusters_)