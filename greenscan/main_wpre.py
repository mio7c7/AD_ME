import numpy as np
import glob
import sys
import argparse
import matplotlib.pyplot as plt
import math
sys.path.append('./')
from evaluation import Evaluation_metrics
from ssa.btgym_ssa import SSA
from sklearn.linear_model import SGDRegressor

parser = argparse.ArgumentParser(description='Mstatistics evaluation on bottom 0.2 data')
parser.add_argument('--data', type=str, default='../data3/*.npz', help='directory of data')
parser.add_argument('--ssa_window', type=int, default=5, help='n_components for ssa preprocessing')
parser.add_argument('--bs', type=int, default=24, help='buffer size for ssa')
parser.add_argument('--threshold', type=float, default=5, help='threshold')
parser.add_argument('--fixed_outlier', type=float, default=1, help='preprocess outlier filter')
parser.add_argument('--outfile', type=str, default='15IQRMED11WND100', help='name of file to save results')
args = parser.parse_args()

def ssa_update(new, residuals, resmean, M2, j):
    updates = ssa.update(new)
    updates = ssa.transform(updates, state=ssa.get_state())[:, -1]
    reconstructed = updates.sum(axis=0)
    residual = new - reconstructed
    residuals = np.concatenate([residuals, residual])
    delta = residual - resmean
    resmean += delta / j
    M2 += delta * (residual - resmean)
    return residuals, resmean, M2

def preprocess(data, fixed_t):
    del_idx = []
    for i in range (data.shape[0]):
        if abs(data[i, 1]) > fixed_t:
            del_idx.append(i)
    return np.delete(data, del_idx, axis=0)

if __name__ == '__main__':
    folder = args.data
    fixed_threshold = args.fixed_outlier
    threshold = args.threshold

    error_margin = 604800  # 7 days
    no_CPs = 0
    no_preds = 0
    no_TPS = 0
    delays = []
    step = args.bs

    for i in glob.glob(folder):
        data = np.load(i, allow_pickle=True)
        name = i[-19:-12]
        train_ts, train_dl, test_ts_1gal, test_dl_1gal, label = data['train_ts'], data['train_dl'], data['test_ts_2gal'], data['test_dl_2gal'], data['label'].item()
        dl = np.concatenate((train_dl, test_dl_1gal))
        test_dl_1gal = test_dl_1gal[~np.isnan(test_dl_1gal).any(axis=1)]
        test_ts_1gal = test_ts_1gal[~np.isnan(test_ts_1gal).any(axis=1)]

        test_dl_1gal = preprocess(test_dl_1gal, fixed_threshold)
        test_ts_1gal = preprocess(test_ts_1gal, fixed_threshold)

        ts = test_dl_1gal[:, 0]
        cps = label['test_2gal']

        train_var_dl = train_dl[:, 1]
        train_ht_dl = train_dl[:, 2]
        test_var_dl = test_dl_1gal[:, 1]
        test_ht_dl = test_dl_1gal[:, 2]
        multi_test = np.stack((test_var_dl, test_ht_dl), axis=1)
        # test_var_dl = np.reshape(test_var_dl, (test_var_dl.shape[0], 1))

        # initialisation
        preds = []
        outliers = []
        input = test_var_dl.copy()
        coeffdiff = np.zeros(len(test_var_dl))
        csdiff = np.zeros(len(test_var_dl))

        ssa = SSA(window=args.ssa_window, max_length=100)
        ctr = 100
        X = test_var_dl[:ctr]
        X_pred = ssa.reset(X)
        X_pred = ssa.transform(X_pred, state=ssa.get_state())
        reconstructeds = X_pred.sum(axis=0)
        residuals = X - reconstructeds
        resmean = residuals.mean()
        M2 = ((residuals - resmean) ** 2).sum() * (len(residuals) - 1) * residuals.var()

        model = SGDRegressor(loss='huber', epsilon=1.35, learning_rate='constant', eta0=0.01, alpha=0.1)
        train_x = np.arange(1, ctr+1).reshape(-1, 1)
        train_y = np.cumsum(reconstructeds)
        model.fit(train_x, train_y)
        cs = train_y[-1]
        new = model

        while ctr <= len(test_var_dl):
            data = test_var_dl[ctr:ctr + step]
            updates = ssa.update(data)
            updates = ssa.transform(updates, state=ssa.get_state())[:, -step:]
            # reconstructed = updates.sum(axis=0)
            reconstructed = updates[1, :]
            residual = data - reconstructed
            residuals = np.concatenate([residuals, residual])
            reconstructeds = np.concatenate((reconstructeds, reconstructed))
            old = new
            new = SGDRegressor(loss='huber', epsilon=1.35, learning_rate='constant', eta0=0.01, alpha=0.1)

            xs = []
            ys = []
            for k in range(len(data)):
                delta = residual[k] - resmean
                resmean += delta / (ctr + k)
                M2 += delta * (residual[k] - resmean)
                stdev = math.sqrt(M2 / (ctr + k - 1))
                threshold_upper = resmean + 2 * stdev
                threshold_lower = resmean - 2 * stdev
                if residual[k] > threshold_upper or residual[k] < threshold_lower:
                    outliers.append(ctr + k)
                    continue
                xs.append(ctr+k+1-len(outliers))
                cs += reconstructed[k]
                ys.append(cs)
            batch_x = np.array(xs).reshape(-1, 1)
            batch_y = np.array(ys)
            train_x = np.concatenate((train_x, batch_x))
            train_y = np.concatenate((train_y, batch_y))

            pred = old.predict(np.array([ctr+k+1 - len(outliers)]).reshape(1, -1))
            cs += reconstructed[k]
            n = cs / pred
            csdiff[ctr+k] = n
            old_coeff = (old.coef_).copy()
            new.fit(train_x, train_y)
            m = new.coef_ / old_coeff
            coeffdiff[ctr+k] = m

                # if score >= threshold:
                #     preds.append(ctr + k)
                #     ctr += (args.N + 1) * args.bo - 1 + k
                #     warm_up = True
                #     score = 0
                #     break

            if len(test_var_dl) - ctr <= args.bs:
                break
            elif len(test_var_dl) - ctr <= 2 * args.bs:
                ctr += args.bs
                step = len(test_var_dl) - ctr
                print(step)
            else:
                ctr += args.bs

        # print(preds)
        fig = plt.figure()
        fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
        ax[0].plot(ts, multi_test[:, 0])
        for cp in cps:
            ax[0].axvline(x=cp, color='g', alpha=0.6)

        ax[1].plot(ts, coeffdiff)
        # for cp in preds:
        #     ax[1].axvline(x=ts[cp], color='g', alpha=0.6)

        ax[2].plot(ts, csdiff)
        # plt.show()
        plt.savefig(name + '.png')
    #     no_CPs += len(cps)
    #     no_preds += len(preds)
    #     for j in preds:
    #         timestamp = ts[j]
    #         for l in cps:
    #             if timestamp >= l and timestamp <= l + error_margin:
    #                 no_TPS += 1
    #                 delays.append(timestamp - l)
    # #
    # rec = Evaluation_metrics.recall(no_TPS, no_CPs)
    # FAR = Evaluation_metrics.False_Alarm_Rate(no_preds, no_TPS)
    # prec = Evaluation_metrics.precision(no_TPS, no_preds)
    # f1score = Evaluation_metrics.F1_score(rec, prec)
    # dd = Evaluation_metrics.detection_delay(delays)
    # print('recall: ', rec)
    # print('false alarm rate: ', FAR)
    # print('precision: ', prec)
    # print('F1 Score: ', f1score)
    # print('detection delay: ', dd)
    #
    # npz_filename = args.outfile
    # np.savez(npz_filename,
    #          rec=rec, FAR=FAR, prec=prec, f1score=f1score, dd=dd)