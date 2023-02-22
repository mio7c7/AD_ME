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
import torch
from torch.utils.data import Dataset, DataLoader
from dain_utils import TempDataset, MLP, lob_epoch_trainer, train_evaluate_anchored

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


if __name__ == '__main__':
    folder = './data/*.npz'
    WINDOW = 10
    mean_lr, std_lr, scale_lr = 0.1, 0.00001, 1
    batch_size = 32
    train_epochs = 2

    for i in glob.glob(folder):
        data = np.load(i)
        train_ts, train_dl, test_ts_1gal, test_dl_1gal = data['train_ts'], data['train_dl'], data['test_ts_1gal'], data['test_dl_1gal']
        dl = np.concatenate((train_dl, test_dl_1gal))

        train_dl_norm = window(train_dl[:, 1], WINDOW)
        train_dl_norm = cvt_gen2ary(train_dl_norm)
        train_dl_norm = torch.from_numpy(train_dl_norm).float()
        train_dl_norm = train_dl_norm.reshape((train_dl_norm.shape[0], 1, train_dl_norm.shape[1]))[:-1, :, :]
        train_dl_target = torch.from_numpy(train_dl[10:, 1]).float()
        train_dataset = TempDataset(train_dl_norm, train_dl_target)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)

        dean = MLP(mode='adaptive_avg', mean_lr=mean_lr, gate_lr=scale_lr, scale_lr=std_lr)
        for epoch in range(train_epochs):
            loss = lob_epoch_trainer(model=dean, loader=train_loader)
            print("Epoch ", epoch, "loss: ", loss)

        #
        test_dl_norm = window(test_dl_1gal[:, 1], WINDOW)
        test_dl_norm = cvt_gen2ary(test_dl_norm)
        test_dl_norm = torch.from_numpy(test_dl_norm).float()
        test_dl_norm = test_dl_norm.reshape((test_dl_norm.shape[0], 1, test_dl_norm.shape[1]))[:-1, :, :]
        test_dl_target = torch.from_numpy(test_dl_1gal[10:, 1]).float()
        test_dataset = TempDataset(test_dl_norm, test_dl_target)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=True)

        # uncomment if want to see the comparison of normalised data
        norm = train_evaluate_anchored(dean, test_loader, train_epochs=5)

        fig = plt.figure(figsize=(20, 10))
        fig, ax = plt.subplots()
        ax.plot(test_dl_1gal[10:, 0], test_dl_1gal[10:, 1], color="blue", marker="o")
        ax.set_xlabel("time")
        ax.set_ylabel("var_tc")
        ax.plot(test_dl_1gal[10:, 0], norm.detach().numpy(), color="red", marker="o")
        plt.show()

        # initialise omwrpca
        # X_dl = norm.reshape((dl_norm.shape[0], dl_norm.shape[2])).cpu().detach().numpy()
        # Lhat, Shat, rank = omwrpca(X_dl.transpose(1, 0), burnin=20, win_size=20, lambda1=1.0 / np.sqrt(200), lambda2=1.0 / np.sqrt(200) * (10))


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