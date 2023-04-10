import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from btgym_ssa import SSA

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

    j = max_length
    while j < test_var_dl.shape[0]:
        new = test_var_dl[j:j+10]
        updates = ssa.update(new)
        state = ssa.get_state()
        updates = ssa.transform(updates, state=state)[:, -10:]
        X_new = np.concatenate((X_new, updates), axis=1)
        j += 10

    e = X_new[1, :].reshape(-1,1)
    e1 = X_new[2, :].reshape(-1, 1)
    e2 = X_new[3, :].reshape(-1, 1)
    e3 = X_new[4, :].reshape(-1, 1)

    fig = plt.figure()
    fig, ax = plt.subplots(5, figsize=[18, 16], sharex=True)
    ax[0].plot(test_var_dl)
    ax[1].plot(e)
    ax[2].plot(e1)
    ax[3].plot(e2)
    ax[4].plot(e3)
    # plt.show()
    plt.savefig(name + 'ssa.png')