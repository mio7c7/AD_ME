from utils.Detector import Detector
import numpy as np
import glob
import sys
import os
import argparse
import matplotlib.pyplot as plt
import math
sys.path.append('../')
from evaluation import Evaluation_metrics
from ssa.btgym_ssa import SSA

parser = argparse.ArgumentParser(description='Mstatistics evaluation on bottom 0.2 data')
parser.add_argument('--data', type=str, default='../data3/*.npz', help='directory of data')
parser.add_argument('--ssa_window', type=int, default=5, help='n_components for ssa preprocessing')
parser.add_argument('--bs', type=int, default=50, help='buffer size for ssa')
parser.add_argument('--forgetting_factor', type=float, default=0.9, help='between 0.9 and 1')
parser.add_argument('--stabilisation_period', type=int, default=50, help='number of reference blocks')
parser.add_argument('--out_threshold', type=float, default=2, help='threshold for outlier filtering')
parser.add_argument('--normal_boundary', type=float, default=0.9, help='threshold for outlier filtering')
parser.add_argument('--guard_zone', type=float, default=0.95, help='threshold for outlier filtering')
parser.add_argument('--p', type=float, default=20, help='threshold')
parser.add_argument('--cs', type=float, default=1.5, help='c-separation')
parser.add_argument('--fixed_outlier', type=float, default=1, help='preprocess outlier filter')
parser.add_argument('--outfile', type=str, default='9_95_50', help='name of file to save results')
args = parser.parse_args()

def preprocess(data, fixed_t):
    del_idx = []
    for i in range(data.shape[0]):
        if abs(data[i, 1]) > fixed_t:
            del_idx.append(i)
    return np.delete(data, del_idx, axis=0)

if __name__ == '__main__':
    folder = args.data
    fixed_threshold = 1.5
    forgetting_factor = args.forgetting_factor
    stabilisation_period = args.stabilisation_period
    p = args.p
    c = args.cs

    error_margin = 864000  # 7 days
    no_CPs = 0
    no_preds = 0
    no_TPS = 0
    delays = []
    ignored = ['../data3\\A043_T2bottom02.npz', '../data3\\A441_T2bottom02.npz',
               '../data3\\B402_T3bottom02.npz', '../data3\\B402_T4bottom02.npz',
               '../data3\\B402_T4bottom06.npz', '../data3\\F257_T2bottom02.npz',
               '../data3\\F257_T2bottom05.npz', '../data3\\F289_T4bottom02.npz', ]

    if not os.path.exists(args.outfile):
        os.makedirs(args.outfile)

    for i in glob.glob(folder):
        if i in ignored:
            continue
        data = np.load(i, allow_pickle=True)
        name = i[-19:-12]
        train_ts, train_dl, test_ts_1gal, test_dl_1gal, cps = data['train_ts'], data['train_dl'], data['test_ts'], data['test_dl'], data['label'].item()
        dl = np.concatenate((train_dl, test_dl_1gal))
        test_dl_1gal = test_dl_1gal[~np.isnan(test_dl_1gal).any(axis=1)]
        test_ts_1gal = test_ts_1gal[~np.isnan(test_ts_1gal).any(axis=1)]
        test_dl_1gal = preprocess(test_dl_1gal, fixed_threshold)
        test_ts_1gal = preprocess(test_ts_1gal, fixed_threshold)
        ts = test_dl_1gal[:, 0]
        test_var_dl = test_dl_1gal[:, 1]
        # multi_test = np.stack((test_var_dl, test_ht_dl), axis=1)
        # test_var_dl = np.reshape(test_var_dl, (test_var_dl.shape[0], 1))

        train_dl_2gal = train_dl[~np.isnan(train_dl).any(axis=1)]
        train_dl_2gal = preprocess(train_dl_2gal, fixed_threshold)

        # initialisation
        # initialsets = multi_test[:stabilisation_period]
        X = train_dl_2gal[:, 1]
        ssa = SSA(window=args.ssa_window, max_length=len(X))
        X_pred = ssa.reset(X)
        X_pred = ssa.transform(X_pred, state=ssa.get_state())
        reconstructeds = X_pred.sum(axis=0)
        # reconstructeds = X_pred[1,:]
        residuals = X - reconstructeds
        resmean = residuals.mean()
        M2 = ((residuals - resmean) ** 2).sum()

        detector = Detector(forgetting_factor=forgetting_factor, stabilisation_period=stabilisation_period,
                            p=p, c=c, normal_boundary=args.normal_boundary, guard_zone=args.guard_zone)
        X = X.reshape(-1, 1)

        detector.initialisation(X)
        preds = []
        outliers = []
        ctr = 0
        step = args.bs
        filtered = []
        gt_margin = []
        for tt in cps:
            closest_element = ts[ts < tt].max()
            idx = np.where(ts == closest_element)[0][0]
            gt_margin.append((ts[idx - 10], tt + error_margin, tt))

        while ctr < len(test_var_dl):
            new = test_var_dl[ctr:ctr + step]
            updates = ssa.update(new)
            updates = ssa.transform(updates, state=ssa.get_state())[:, -step:]
            reconstructed = updates.sum(axis=0)
            # reconstructed = updates[1,:]
            residual = new - reconstructed
            residuals = np.concatenate([residuals, residual])
            # reconstructeds = np.concatenate((reconstructeds, reconstructed))

            for k in range(len(new)):
                delta = residual[k] - resmean
                resmean += delta / (ctr + k + len(train_dl_2gal))
                M2 += delta * (residual[k] - resmean)

                stdev = math.sqrt(M2 / (ctr + k + len(train_dl_2gal) - 1))
                threshold_upper = resmean + args.out_threshold * stdev
                threshold_lower = resmean - args.out_threshold * stdev

                if residual[k] > threshold_upper or residual[k] < threshold_lower:
                    outliers.append(ctr + k)
                    filtered.append(np.mean(filtered[-10:] if len(filtered) > 10 else 0))
                    # continue
                else:
                    filtered.append(new[k])
                try:
                    if detector.predict(np.array([filtered[-1]]), ctr+k):
                        preds.append(ctr + k)
                except:
                    print()

            if len(test_var_dl) - ctr <= args.bs:
                break
            elif len(test_var_dl) - ctr <= 2*args.bs:
                ctr += args.bs
                step = len(test_var_dl) - ctr
                print(step)
            else:
                ctr += args.bs

        fig = plt.figure()
        fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
        ax[0].plot(ts, test_var_dl)
        for cp in gt_margin:
            ax[0].axvline(x=cp[0], color='green', linestyle='--')
            ax[0].axvline(x=cp[1], color='green', linestyle='--')

        ax[1].plot(ts, detector.score)
        for cp in preds:
            ax[1].axvline(x=ts[cp], color='g', alpha=0.6)

        ax[2].plot(ts, filtered)
        # for cp in preds:
        #     ax[1].axvline(x=ts[cp], color='g', alpha=0.6)
        #
        # ax[2].plot(ts, multi_test[:, 1])
        # plt.show()
        plt.savefig(args.outfile + '/' + name + '.png')
        no_CPs += len(cps)
        no_preds += len(preds)
        mark = []
        for j in preds:
            timestamp = ts[j]
            for l in gt_margin:
                if timestamp >= l[0] and timestamp <= l[1]:
                    if l not in mark:
                        mark.append(l)
                    else:
                        no_preds -= 1
                        continue
                    no_TPS += 1
                    delays.append(timestamp - l[2])

    rec = Evaluation_metrics.recall(no_TPS, no_CPs)
    FAR = Evaluation_metrics.False_Alarm_Rate(no_preds, no_TPS)
    prec = Evaluation_metrics.precision(no_TPS, no_preds)
    f1score = Evaluation_metrics.F1_score(rec, prec)
    f2score = Evaluation_metrics.F2_score(rec, prec)
    dd = Evaluation_metrics.detection_delay(delays)
    print('recall: ', rec)
    print('false alarm rate: ', FAR)
    print('precision: ', prec)
    print('F1 Score: ', f1score)
    print('F2 Score: ', f2score)
    print('detection delay: ', dd)

    npz_filename = args.outfile
    np.savez(npz_filename,
             rec=rec, FAR=FAR, prec=prec, f1score=f1score, dd=dd)