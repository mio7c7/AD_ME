from Detector import Detector
import os
import numpy as np
import glob
import sys
import argparse
import matplotlib.pyplot as plt
import math
from scipy.stats import wasserstein_distance
sys.path.append('../')
from evaluation import Evaluation_metrics
from ssa.btgym_ssa import SSA

parser = argparse.ArgumentParser(description='LIFEWATCH')
parser.add_argument('--data', type=str, default='../005lr/*.npz', help='directory of data')
parser.add_argument('--ssa_window', type=int, default=5, help='n_components for ssa preprocessing')
parser.add_argument('--window_size', type=int, default=75, help='window_size')
parser.add_argument('--max_points', type=int, default=133, help='min blocks required in a distrib. before starting detection')
parser.add_argument('--min_batch_size', type=int, default=6, help='mini_batch_size')
parser.add_argument('--fixed_outlier', type=float, default=1, help='preprocess outlier filter')
parser.add_argument('--out_threshold', type=float, default=2, help='threshold for outlier filtering')
parser.add_argument('--epsilon', type=float, default=1.5, help='epsilon')
parser.add_argument('--outfile', type=str, default='25_400_20_15', help='name of file to save results')
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

def sliding_window(elements, window_size):
    if len(elements) <= window_size:
        return elements
    new = np.empty((0, window_size))
    for i in range(len(elements) - window_size + 1):
        new = np.vstack((new, elements[i:i+window_size]))
    return new

def preprocess(data, fixed_t):
    del_idx = []
    for i in range (data.shape[0]):
        if abs(data[i, 1]) > fixed_t:
            del_idx.append(i)
    return np.delete(data, del_idx, axis=0)

if __name__ == '__main__':
    folder = args.data
    fixed_threshold = args.fixed_outlier
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
        train_var_dl = train_dl[:, 1]
        test_var_dl = test_dl_1gal[:, 1]

        train_dl_2gal = train_dl[~np.isnan(train_dl).any(axis=1)]
        train_dl_2gal = preprocess(train_dl_2gal, fixed_threshold)

        # initialisation for preprocessing module
        X = train_dl_2gal[:, 1]
        ssa = SSA(window=args.ssa_window, max_length=len(X))
        X_pred = ssa.reset(X)
        X_pred = ssa.transform(X_pred, state=ssa.get_state())
        reconstructeds = X_pred.sum(axis=0)
        # reconstructeds = X_pred[1,:]
        residuals = X - reconstructeds
        resmean = residuals.mean()
        M2 = ((residuals - resmean) ** 2).sum()

        # initialisation
        ws = args.window_size
        preds = []
        outliers = []
        input = test_var_dl.copy()
        detector = Detector(ws, args.epsilon)
        ctr = 0
        filtered = []
        scores = []
        thresholds = []
        ct = 0
        gt_margin = []
        cp_ctr = []
        for tt in cps:
            closest_element = ts[ts < tt].max()
            idx = np.where(ts == closest_element)[0][0]
            gt_margin.append((ts[idx - 10], tt + error_margin, tt))

        mm = 0
        while mm < len(reconstructeds)-args.window_size:
            detector.current_distribution = np.append(detector.current_distribution, X[mm:mm+args.window_size].reshape((1, -1)), axis=0)
            mm += args.window_size
            if len(detector.current_distribution) == args.max_points:
                break

        if len(detector.current_distribution) >= args.min_batch_size:  # 9
            dis_threshold = detector.compute_threshold()  # 10
            detector.distribution_threshold.append(dis_threshold)  # 10
            ct = dis_threshold
            detector.distribution_pool.append(detector.current_distribution)  # 11
            if detector.current_distribution_index is None:
                detector.current_distribution_index = -1

        while ctr <= len(input):
            data = test_var_dl[ctr:ctr + ws]
            updates = ssa.update(data)
            updates = ssa.transform(updates, state=ssa.get_state())[:, -ws:]
            reconstructed = updates.sum(axis=0)
            # reconstructed = updates[1, :]
            residual = data - reconstructed
            residuals = np.concatenate([residuals, residual])
            # reconstructeds = np.concatenate((reconstructeds, reconstructed))

            ys = []
            for k in range(ws):
                delta = residual[k] - resmean
                resmean += delta / (ctr + k + len(train_dl_2gal))
                M2 += delta * (residual[k] - resmean)
                stdev = math.sqrt(M2 / (ctr + k + len(train_dl_2gal) - 1))
                threshold_upper = resmean + args.out_threshold * stdev
                threshold_lower = resmean - args.out_threshold * stdev
                if residual[k] > threshold_upper or residual[k] < threshold_lower:
                    outliers.append(ctr + k)
                    ys.append(np.mean(filtered[-10:] if len(filtered)>10 else 0))
                    continue
                ys.append(data[k])
            filtered = filtered + ys
            scores = scores + [0]*ws
            if detector.current_distribution_index is None:
                thresholds = thresholds + [0] * ws
            else:
                thresholds = thresholds + [detector.distribution_threshold[detector.current_distribution_index]] * ws
            # current Bi
            Bi = np.array(ys)
            if len(detector.current_distribution) < args.min_batch_size:
                detector.current_distribution = np.append(detector.current_distribution, [Bi], axis=0) #8
                if len(detector.current_distribution) >= args.min_batch_size: #9
                    dis_threshold = detector.compute_threshold() #10
                    detector.distribution_threshold.append(dis_threshold) #10
                    detector.distribution_pool.append(detector.current_distribution) #11
                    if detector.current_distribution_index is None:
                        detector.current_distribution_index = -1
            else:
                x = Bi
                y = detector.current_distribution
                distance = wasserstein_distance(x, y.reshape(-1))
                scores[-1] = distance
                if distance > detector.distribution_threshold[detector.current_distribution_index]:
                    min_dist = 100000
                    m = 0
                    dk = None
                    while m < len(detector.distribution_pool):
                        distribution = detector.distribution_pool[m]
                        cur_threshold = detector.distribution_threshold[m]
                        distance = wasserstein_distance(x, distribution.reshape(-1))
                        if distance < cur_threshold:
                            if distance < min_dist:
                                min_dist = distance
                                dk = m
                        m += 1
                    if dk is None:
                        detector.N.append(ctr+ws)
                        detector.current_distribution_index = -1
                        detector.current_distribution = np.empty((0, args.window_size))
                    else:
                        detector.R.append(ctr+ws)
                        detector.current_distribution_index = m - 1
                        detector.current_distribution = detector.distribution_pool[m - 1]

                if len(detector.current_distribution) < args.max_points:
                    detector.current_distribution = np.append(detector.current_distribution, [Bi], axis=0)
                    dis_threshold = detector.compute_threshold()
                    detector.distribution_threshold[detector.current_distribution_index] = dis_threshold

            if len(test_var_dl) - ctr <= ws:
                break
            elif len(test_var_dl) - ctr <= 2*ws:
                ctr += ws
                # ws = len(test_var_dl) - ctr
                filtered = filtered + [0]*(len(test_var_dl) - ctr)
                break
            else:
                ctr += ws

        preds = detector.N + detector.R
        scores = scores + [0] * (len(ts) - len(scores))
        thresholds = thresholds + [0] * (len(ts) - len(thresholds))
        fig = plt.figure()
        fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
        ax[0].plot(ts, test_var_dl)
        for cp in cps:
            ax[0].axvline(x=cp, color='g', alpha=0.6)

        ax[1].plot(ts, filtered)
        for cp in detector.N:
            ax[2].axvline(x=ts[cp], color='purple', alpha=0.6)
        for cp in detector.R:
            ax[2].axvline(x=ts[cp], color='r', alpha=0.6)
        ax[2].plot(ts, scores)
        ax[2].plot(ts, thresholds)
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
    #
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
             rec=rec, FAR=FAR, prec=prec, f1score=f1score, f2score=f2score, dd=dd)