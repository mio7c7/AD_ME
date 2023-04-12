import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from btgym_ssa import SSA
from scipy import stats
import math

max_length = 100
window = 5
folder = '../data3/*.npz'
for i in glob.glob(folder):
    data = np.load(i, allow_pickle=True)
    name = i[-19:-12]
    train_ts, train_dl, test_ts_1gal, test_dl_1gal, label = data['train_ts'], data['train_dl'], data['test_ts_2gal'], \
                                                            data['test_dl_2gal'], data['label'].item()
    dl = np.concatenate((train_dl, test_dl_1gal))
    test_dl_1gal = test_dl_1gal[~np.isnan(test_dl_1gal).any(axis=1)]
    test_ts_1gal = test_ts_1gal[~np.isnan(test_ts_1gal).any(axis=1)]

    ts = test_dl_1gal[:, 0]
    cps = label['test_2gal']

    train_var_dl = train_dl[:, 1]
    train_ht_dl = train_dl[:, 2]
    test_var_dl = test_dl_1gal[:, 1]
    test_ht_dl = test_dl_1gal[:, 2]
    # multi_test = np.stack((test_var_dl, test_ht_dl), axis=1)
    # test_var_dl = np.reshape(test_var_dl, (test_var_dl.shape[0],1))
    # initialisation
    ssa = SSA(window=window, max_length=max_length)
    X_new = ssa.reset(test_var_dl[:max_length])
    state = ssa.get_state()
    X_new = ssa.transform(X_new, state=state)
    reconstructeds = X_new.sum(axis=0)
    residuals = test_var_dl[:max_length] - reconstructeds
    resmean = residuals.mean()
    M2 = ((residuals - resmean) ** 2).sum()*(len(residuals)-1)*residuals.var()

    j = max_length
    outliers = []
    while j < test_var_dl.shape[0]:
        new = np.array([test_var_dl[j]])
        updates = ssa.update(new)
        state = ssa.get_state()
        updates = ssa.transform(updates, state=state)[:, -1]

        reconstructed = updates.sum(axis=0)
        residual = new - reconstructed
        residuals = np.concatenate([residuals, residual])

        delta = residual - resmean
        resmean += delta/j
        M2 += delta * (residual - resmean)

        stdev = math.sqrt(M2/(j-1))
        threshold_upper = resmean + 2 * stdev
        threshold_lower = resmean - 2 * stdev

        if residual > threshold_upper or residual < threshold_lower:
            outliers.append(j)

        X_new = np.concatenate((X_new, updates[:,None]), axis=1)
        reconstructeds = np.concatenate((reconstructeds, reconstructed[None]))
        j += 1

    X_new = np.delete(X_new, outliers, axis=1)
    reconstructeds = np.delete(reconstructeds, outliers)
    e = X_new[1, :].reshape(-1,1)
    e1 = X_new[2, :].reshape(-1, 1)
    e2 = X_new[3, :].reshape(-1, 1)
    e3 = X_new[4, :].reshape(-1, 1)

    fig = plt.figure()
    fig, ax = plt.subplots(2, figsize=[18, 16], sharex=True)
    ax[0].plot(test_var_dl,'-gD', markevery=outliers)
    ax[1].plot(reconstructeds)
    # ax[2].plot(e1)
    # ax[3].plot(e2)
    # ax[4].plot(e3)
    # plt.show()
    plt.savefig(name + 'ssa.png')