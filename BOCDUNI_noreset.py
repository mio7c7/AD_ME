import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
from itertools import islice
import os
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
# from bayesian_changepoint_detection.bocpd import MultivariateT, BOCPD, BOCPD_UNI
# from bayesian_changepoint_detection.bayesian_online import ConstantHazard, Detector,
import argparse
import math
import sys
sys.path.append('./')
from evaluation import Evaluation_metrics
from ssa.btgym_ssa import SSA

parser = argparse.ArgumentParser(description='Mstatistics evaluation on bottom 0.2 data')
parser.add_argument('--data', type=str, default='./data3/*.npz', help='directory of data')
parser.add_argument('--ssa_window', type=int, default=5, help='n_components for ssa preprocessing')
parser.add_argument('--bs', type=int, default=24, help='buffer size for ssa')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold')
parser.add_argument('--delay', type=int, default=50, help='Threshold')
parser.add_argument('--initial_period', type=int, default=50, help='initialisation period')
parser.add_argument('--fixed_outlier', type=float, default=1, help='preprocess outlier filter')
parser.add_argument('--outfile', type=str, default='bocd', help='name of file to save results')
args = parser.parse_args()

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
    folder = args.data
    buffer_size = 24
    hazard_function = partial(constant_hazard, 1440)
    lambda_ = 1440
    delay = args.delay
    fixed_threshold = 1.5

    error_margin = 864000  # 7 days
    no_CPs = 0
    no_preds = 0
    no_TPS = 0
    delays = []
    initial_period = args.initial_period

    if not os.path.exists(args.outfile):
        os.makedirs(args.outfile)

    ignored = ['data3\\A043_T2bottom02.npz', 'data3\\A441_T2bottom02.npz',
               'data3\\B402_T3bottom02.npz', 'data3\\B402_T4bottom02.npz',
               'data3\\B402_T4bottom06.npz', 'data3\\F257_T2bottom02.npz',
               'data3\\F257_T2bottom05.npz', 'data3\\F289_T4bottom02.npz',]

    for i in glob.glob(folder):
        if i in ignored:
            continue
        # if i !='../data3\\J496_T5bottom02.npz' and i !='../data3\\J802_T1bottom02.npz' and i !='../data3\\J023_T1bottom02.npz' and i !='../data3\\Q152_T2bottom02.npz':
        # if i !='./data3\\H915_T2bottom02.npz':
        #     continue
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

        gt_margin = []
        for tt in cps:
            closest_element = ts[ts < tt].max()
            idx = np.where(ts == closest_element)[0][0]
            gt_margin.append((ts[idx - 10], tt + error_margin, tt))

        # initialisation for preprocessing module
        X = train_dl_2gal[:, 1]
        ssa = SSA(window=args.ssa_window, max_length=len(X))
        X_pred = ssa.reset(X)
        X_pred = ssa.transform(X_pred, state=ssa.get_state())
        reconstructeds = X_pred.sum(axis=0)
        residuals = X - reconstructeds
        resmean = residuals.mean()
        M2 = ((residuals - resmean) ** 2).sum()

        ctr = 0
        maxes = np.zeros(len(test_var_dl))
        R = np.zeros((len(test_var_dl)+1, len(test_var_dl)+1))
        R[0, 0] = 1
        log_likelihood_class = online_ll.StudentT(kappa=10)
        outliers = []
        preds = []
        scores = []
        step = args.bs
        filtered = []
        gt_margin = []
        cp_ctr = []
        for tt in cps:
            closest_element = ts[ts < tt].max()
            idx = np.where(ts == closest_element)[0][0]
            gt_margin.append((ts[idx - 10], tt + error_margin, tt))

        while ctr < len(test_var_dl):
            new = test_var_dl[ctr:ctr + step]
            updates = ssa.update(new)
            updates = ssa.transform(updates, state=ssa.get_state())[:, -step:]
            reconstructed = updates.sum(axis=0)
            residual = new - reconstructed
            residuals = np.concatenate([residuals, residual])

            for k in range(len(new)):
                delta = residual[k] - resmean
                resmean += delta / (ctr + k + len(train_dl_2gal))
                M2 += delta * (residual[k] - resmean)
                stdev = math.sqrt(M2 / (ctr + k + + len(train_dl_2gal) - 1))
                threshold_upper = resmean + 2 * stdev
                threshold_lower = resmean - 2 * stdev
                if residual[k] > threshold_upper or residual[k] < threshold_lower:
                    outliers.append(ctr + k)
                    filtered.append(np.mean(filtered[-10:] if len(filtered)>10 else 0))
                else:
                    filtered.append(new[k])

                t = ctr + k
                predprobs = log_likelihood_class.pdf(filtered[-1])
                H = hazard_function(np.array(range(t + 1)))
                R[1: t + 2, t + 1] = R[0: t + 1, t] * predprobs * (1 - H)
                R[0, t + 1] = np.sum(R[0: t + 1, t] * predprobs * H)
                R[:, t + 1] = R[:, t + 1] / np.sum(R[:, t + 1])
                log_likelihood_class.update_theta(filtered[-1], t=t)
                maxes[t] = R[:, t].argmax()
                scores.append(R[delay, t])

                if t > delay and R[delay, t] > args.threshold:
                    preds.append(t-delay)

            if len(test_var_dl) - ctr <= args.bs:
                break
            elif len(test_var_dl) - ctr <= 2*args.bs:
                ctr += args.bs
                step = len(test_var_dl) - ctr
            else:
                ctr += args.bs

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

        fig = plt.figure()
        fig, ax = plt.subplots(2, figsize=[18, 16], sharex=True)
        ax[0].plot(ts, test_var_dl)
        for cp in cps:
            ax[0].axvline(x=cp, color='g', alpha=0.6)

        ax[1].plot(ts, scores)
        for cp in preds:
            ax[1].axvline(x=ts[cp], color='g', alpha=0.6)
        plt.savefig(args.outfile + '/' + name + '.png')

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
    np.savez(npz_filename, rec=rec, FAR=FAR, prec=prec, f1score=f1score, f2score=f2score, dd=dd)



