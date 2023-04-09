import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
from itertools import islice
# from sklearn.cluster import DBSCAN, KMeans, MeanShift
from sklearn.metrics.pairwise import euclidean_distances
from bayesian_changepoint_detection.bayesian_models import online_changepoint_detection
import bayesian_changepoint_detection.online_likelihoods as online_ll
from bayesian_changepoint_detection.hazard_functions import constant_hazard
from functools import partial
import matplotlib.cm as cm
from time import time
# from bayesian_changepoint_detection.bocd import BayesianOnlineChangePointDetection
# from bayesian_changepoint_detection.distribution import MultivariateT
from bayesian_changepoint_detection.bocpd import MultivariateT, BOCPD, BOCPD_UNI
# from bayesian_changepoint_detection.bayesian_online import ConstantHazard, Detector,

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

def dbscan(arr_var, i, arr):
    db = DBSCAN(eps=0.8, min_samples=20).fit(arr_var)
    labels = db.labels_
    fig = plt.figure(figsize=(20, 10))
    plt.scatter(x=arr[5:-4, 0], y=arr[5:-4, 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    # plt.show()
    plt.savefig(i + '.png')

def plot_cluster(data, labels):
    plt.scatter(data[:, 0], y=data[:, 1])
    plt.show()

def feature_extraction(arr):
    mean = np.mean(arr, axis=1)
    std = np.std(arr, axis=1)
    res = np.stack((mean, std), axis=1)
    return res

def split_buffer(arr, buffer_size):
    l = len(arr)
    for ndx in range(0, l, buffer_size):
        yield arr[ndx:min(ndx + buffer_size, l)]

def update_memory():
    pass

def normalisation(array, test):
    gmin_dl = np.min(array)
    gmax_dl = np.max(array)
    array_norm = (array - gmin_dl) / (gmax_dl - gmin_dl)
    test_norm = (test - gmin_dl) / (gmax_dl - gmin_dl)
    return array_norm, test_norm

if __name__ == '__main__':
    folder = './data3/*.npz'
    WINDOW = 10
    mean_lr, std_lr, scale_lr = 0.1, 0.00001, 1
    batch_size = 32
    train_epochs = 2
    buffer_size = 24
    minimum_size = 20
    maximum_storage_time = 3
    esp = 0.1
    hazard_function = partial(constant_hazard, 1440)
    lambda_ = 1440
    delay = 10
    fixed_threshold = 1.5

    for i in glob.glob(folder):
        data = np.load(i, allow_pickle=True)
        name = i[-19:-12]  # i[-11:-4]
        train_ts, train_dl, test_ts_1gal, test_dl_1gal, label = data['train_ts'], data['train_dl'], data['test_ts_2gal'], data['test_dl_2gal'], data['label'].item()
        dl = np.concatenate((train_dl, test_dl_1gal))
        test_dl_1gal = test_dl_1gal[~np.isnan(test_dl_1gal).any(axis=1)]
        test_ts_1gal = test_ts_1gal[~np.isnan(test_ts_1gal).any(axis=1)]

        test_dl_1gal = preprocess(test_dl_1gal, fixed_threshold)
        test_ts_1gal = preprocess(test_ts_1gal, fixed_threshold)
        train_dl = preprocess(train_dl, fixed_threshold)
        train_ts = preprocess(train_ts, fixed_threshold)

        ts = test_dl_1gal[:, 0]
        cps = label['test_2gal']

        train_var_dl = train_dl[:, 1]
        train_ht_dl = train_dl[:, 2]
        test_var_dl = test_dl_1gal[:, 1]
        test_ht_dl = test_dl_1gal[:, 2]

        # train_var_dl_norm, test_var_dl_norm = normalisation(train_var_dl, test_var_dl)
        # train_ht_dl_norm, test_ht_dl_norm = normalisation(train_ht_dl, test_ht_dl)
        # multi_test = np.stack((test_var_dl, test_ht_dl), axis=1)
        # fig = plt.figure(figsize=[25, 16])
        # plt.plot(ts, test_var_dl, marker="o")
        # plt.show()

        # Feature extraction on windows
        # statistical
        # train_dl = feature_extraction(train_var_dl)
        # test_dl = feature_extraction(test_var_dl)

        R, maxes = online_changepoint_detection(
            test_var_dl, hazard_function, online_ll.StudentT(kappa=10)
        )

        fig = plt.figure()
        fig, ax = plt.subplots(2, figsize=[18, 16], sharex=True)
        ax[0].plot(ts, test_var_dl)
        for cp in cps:
            ax[0].axvline(x=cp, color='g', alpha=0.6)

        ax[1].plot(ts[delay-1:-1], R[delay,delay:-1])
        # for cp in candcps:
        #     ax[1].axvline(x=ts[cp], color='g', alpha=0.6)

        plt.savefig(name + '.png')





