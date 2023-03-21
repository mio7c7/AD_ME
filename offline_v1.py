import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
from itertools import islice
from sklearn.cluster import DBSCAN, KMeans, MeanShift
from sklearn import metrics
import sys
import os
from rpca.omwrpca import omwrpca
from rpca.omwrpca_cp import omwrpca_cp
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import euclidean_distances

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

if __name__ == '__main__':
    folder = './data/*.npz'
    WINDOW = 10
    mean_lr, std_lr, scale_lr = 0.1, 0.00001, 1
    batch_size = 32
    train_epochs = 2
    buffer_size = 24
    minimum_size = 20
    maximum_storage_time = 3
    esp = 0.1


    for i in glob.glob(folder):
        data = np.load(i)
        train_ts, train_dl, test_ts_1gal, test_dl_1gal = data['train_ts'], data['train_dl'], data['test_ts_1gal'], data['test_dl_1gal']
        dl = np.concatenate((train_dl, test_dl_1gal))
        test_dl_1gal = test_dl_1gal[~np.isnan(test_dl_1gal).any(axis=1)]
        test_ts_1gal = test_ts_1gal[~np.isnan(test_ts_1gal).any(axis=1)]

        train_var_dl = train_dl[:, 1]
        gmin_dl = np.min(train_var_dl)
        gmax_dl = np.max(train_var_dl)
        train_var_dl_norm = (train_var_dl - gmin_dl) / (gmax_dl - gmin_dl)

        test_var_dl = test_dl_1gal[:, 1]
        test_var_dl_norm = (test_var_dl - gmin_dl) / (gmax_dl - gmin_dl)
        # test_var_dl_norm = test_var_dl_norm.reshape((1, test_var_dl_norm.shape[0]))

        # image generator for normalised vs org
        # fig = plt.figure(figsize=(20, 10))
        # fig, ax = plt.subplots()
        # ax.plot(test_dl_1gal[:, 0], test_var_dl, color="blue", marker="o")
        # ax.set_xlabel("time")
        # ax.set_ylabel("var_tc")
        # ax.plot(test_dl_1gal[:, 0], test_var_dl_norm, color="red", marker="o")
        # plt.show()

        # sliding window, np.array
        train_var_dl = window(train_var_dl, WINDOW)
        train_var_dl = cvt_gen2ary(train_var_dl)
        test_var_dl = window(test_var_dl, WINDOW)
        test_var_dl = cvt_gen2ary(test_var_dl)

        # Feature extraction on windows
        # statistical
        train_dl = feature_extraction(train_var_dl)
        test_dl = feature_extraction(test_var_dl)
        test_split = split_buffer(test_dl, buffer_size=buffer_size)

        # initialise omwrpca
        # Lhat, Shat, rank = omwrpca(train_var_dl.transpose(1, 0), burnin=20, win_size=20, lambda1=1.0 / np.sqrt(200), lambda2=1.0 / np.sqrt(200) * (10))
        # pass
        # for l1 in [0.0001, 0.01, 0.1]:
        #     for l2 in [0.00001, 0.0001, 0.01, 0.1]:
        #         Lhat, Shat, rank, cp, num_sparses = omwrpca_cp(test_var_dl_norm, burnin=50, win_size=25,
        #                                                        track_cp_burnin=50, n_check_cp=24, alpha=0.05, proportion=0.05, n_positive=3, min_test_size=100,
        #                                                        tolerance_num=3, lambda1=l1, lambda2=l2, factor=1)
        #         print(l1, l2, cp)
        # pass

        # Clustering
        model = KMeans(n_clusters=1, random_state=0)
        model.fit(train_dl)
        # compute the baseline distance
        k = model.cluster_centers_
        distances = euclidean_distances(train_dl, model.cluster_centers_)
        N = [train_dl.shape[0]] #store the no. of samples in each cluster
        baseline_statistics = [np.percentile(distances, 95)] # distance threshold
        cluster_centres = model.cluster_centers_
        candidates = [] #where the points which do not belong to any cluster are stored, may either be anomaly or new state data
        counter = 0
        cps = []

        for buffer in test_split:
            try:
                buffer_dists = euclidean_distances(buffer, cluster_centres) #compute the distances to existing clusters
            except ValueError:
                print(buffer)
                print(cluster_centres)
            js = np.argmin(buffer_dists, axis=1) #closest cluster j

            # remove candidates that has expired
            candidates = [(cpos, cctr) for (cpos, cctr) in candidates if counter - cctr > maximum_storage_time]

            for b in range(buffer.shape[0]):
                if buffer_dists[b, js[b]] < baseline_statistics[js[b]]: #nominal data
                    N[js[b]] += 1
                    cluster_centres[js[b]] += (1/N[js[b]])*(buffer[b]-cluster_centres[js[b]]) #update center

                else:
                    candidates.append((buffer[b], counter))

                    if len(candidates) >= 24:
                        #determine if a new cluster may be formed
                        temp = np.array([])
                        temp = [np.concatenate([temp, pos]) for (pos, ctr) in candidates]
                        temp = np.array(temp)
                        clusters = DBSCAN(eps=0.5, min_samples=20).fit(temp)
                        labels = clusters.labels_
                        if len(set(labels)) == 1:
                            if labels[0] == -1: #all regarded as anomaly
                                pass
                            else: #a new state started
                                new_ctr = np.mean(temp, axis=0)
                                cluster_centres = np.concatenate((cluster_centres, np.expand_dims(new_ctr, axis=0)))
                                N.append(temp.shape[0])
                                distances = euclidean_distances(temp, cluster_centres)
                                baseline_statistics.append(np.percentile(distances, 95))
                                cps.append(counter*buffer_size)
                        elif len(set(labels)) == 2: #a new state started 0 and -1
                            itemindex = np.where(labels == 0)[0]
                            new_j = temp[itemindex]
                            new_ctr = np.mean(new_j, axis=0)
                            cluster_centres = np.concatenate((cluster_centres, np.expand_dims(new_ctr, axis=0)))
                            N.append(new_j.shape[0])
                            distances = euclidean_distances(new_j, cluster_centres)
                            baseline_statistics.append(np.percentile(distances, 95))
                            cps.append(counter * buffer_size)
            counter += 1

        # image generator for normalised vs org
        fig = plt.figure(figsize=(20, 10))
        fig, ax = plt.subplots()
        ax.plot(test_dl_1gal[:, 0], test_dl_1gal[:, 1], color="blue", marker="o")
        ax.set_xlabel("time")
        ax.set_ylabel("var_tc")
        # ax.plot(test_dl_1gal[:, 0], test_var_dl_norm, color="red", marker="o")
        for pred in cps:
            plt.axvline(x=test_dl_1gal[pred, 0], color='g', alpha=1)
        # plt.show()
        plt.savefig(i + '.png')

        #test
        # test_tsw_1gal = window(test_ts_1gal[:, 1], WINDOW)
        # test_dlw_1gal = window(test_dl_1gal[:, 1], WINDOW)
        # X_ts_1gal, X_dl_1gal = cvt_gen2ary( test_tsw_1gal), cvt_gen2ary(test_dlw_1gal)

        # train_dl = preprocess(train_dl, 6)
        # test_dl_1gal = preprocess(test_dl_1gal, 4)
        # plot_raw(train_dl)
        # plot_raw(test_dl_1gal)
        # arr = np.concatenate((train_dl, test_dl_1gal), axis=0)
        # arr = np.where(np.isnan(arr), 0, arr)
        # arr_var = window(arr[:, 1], WINDOW)
        # arr_var = cvt_gen2ary(arr_var)
        # dbscan(arr_var, i, arr)


    # db_ts = DBSCAN(eps=0.2, min_samples=10).fit(arr)
    # labels = db_ts.labels_
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print("Estimated number of clusters: %d" % n_clusters_)
    # db_ts.fit(X_ts_1gal)
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)
    # print("Estimated number of clusters: %d" % n_clusters_)

    # kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(arr_var)
    # labels = kmeans.labels_
    # fig = plt.figure(figsize=(20, 10))
    # plt.scatter(x=arr[5:-4, 0], y=arr[5:-4, 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    # plt.show()

    # db = DBSCAN(eps=2, min_samples=10).fit(arr_var)
    # labels = db.labels_
    # fig = plt.figure(figsize=(20, 10))
    # plt.scatter(x=arr[5:-4, 0], y=arr[5:-4, 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    # plt.show()