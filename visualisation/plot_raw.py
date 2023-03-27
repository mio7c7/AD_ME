import numpy as np
import matplotlib.pyplot as plt
import glob

if __name__ == '__main__':
    folder = '../data2/*.npz'

    for i in glob.glob(folder):
        data = np.load(i, allow_pickle=True)
        name = i
        train_ts, train_dl, test_ts_1gal, test_dl_1gal, label = data['train_ts'], data['train_dl'], data['test_ts_1gal'], data['test_dl_1gal'], data['label'].item()
        dl = np.concatenate((train_dl, test_dl_1gal))
        test_dl_1gal = test_dl_1gal[~np.isnan(test_dl_1gal).any(axis=1)]
        test_ts_1gal = test_ts_1gal[~np.isnan(test_ts_1gal).any(axis=1)]

        ts = test_dl_1gal[:, 0]
        cps = label['test_1gal']
        test_var_dl = test_dl_1gal[:, 1]
        test_ht_dl = test_dl_1gal[:, 2]

        fig = plt.figure(figsize=[25, 16])
        plt.plot(ts, test_var_dl)
        for cp in cps:
            plt.axvline(x=cp, color='g', alpha=0.6)
        plt.savefig(i + '01.png')