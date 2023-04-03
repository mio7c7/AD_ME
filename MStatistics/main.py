from MS import Kernel
import numpy as np
import glob
import matplotlib.pyplot as plt


def preprocess(data, fixed_t):
    del_idx = []
    for i in range(data.shape[0]):
        if abs(data[i, 1]) > fixed_t:
            del_idx.append(i)
    return np.delete(data, del_idx, axis=0)


if __name__ == '__main__':
    folder = '../data1/*.npz'
    fixed_threshold = 1.5
    bo = 30
    N = 10
    threshold = 5

    model = Kernel(b=threshold, bo=bo, N=N)

    for i in glob.glob(folder):
        data = np.load(i, allow_pickle=True)
        name = i[-11:-4]
        train_ts, train_dl, test_ts_1gal, test_dl_1gal, label = data['train_ts'], data['train_dl'], data[
            'test_ts_1gal'], data['test_dl_1gal'], data['label'].item()
        dl = np.concatenate((train_dl, test_dl_1gal))
        test_dl_1gal = test_dl_1gal[~np.isnan(test_dl_1gal).any(axis=1)]
        test_ts_1gal = test_ts_1gal[~np.isnan(test_ts_1gal).any(axis=1)]

        test_dl_1gal = preprocess(test_dl_1gal, fixed_threshold)
        test_ts_1gal = preprocess(test_ts_1gal, fixed_threshold)

        ts = test_dl_1gal[:, 0]
        cps = label['test_1gal']

        train_var_dl = train_dl[:, 1]
        train_ht_dl = train_dl[:, 2]
        test_var_dl = test_dl_1gal[:, 1]
        test_ht_dl = test_dl_1gal[:, 2]
        multi_test = np.stack((test_var_dl, test_ht_dl), axis=1)

        preds = []
        input = multi_test.copy()
        scores = []
        while True:
            pred, ss = model.change_detection(input)

            if pred != -1:
                if len(preds) == 0:
                    preds.append(pred)
                    scores = ss
                else:
                    preds.append(pred + preds[-1])
                    scores = scores + ss
                model = Kernel(b=threshold, bo=bo, N=N)
                if preds[-1] + bo * (N + 1) > multi_test.shape[0]:
                    print('Finish insufficient')
                    scores = scores + [0] * (multi_test.shape[0] - len(scores))
                    break
                input = input[pred + 1:]
            else:
                print('Finish the end')
                scores = scores + ss
                break

        fig = plt.figure()
        fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
        ax[0].plot(ts, multi_test[:, 0])
        for cp in cps:
            ax[0].axvline(x=cp, color='g', alpha=0.6)

        ax[1].plot(ts, scores)
        for cp in preds:
            ax[1].axvline(x=ts[cp], color='g', alpha=0.6)

        ax[2].plot(ts, multi_test[:, 1])
        plt.savefig(name + '.png')
