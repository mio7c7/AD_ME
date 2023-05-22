import glob
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from Detector_mmd_real import Detector
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import rbf_kernel
import math
sys.path.append('./')
from evaluation import Evaluation_metrics
from ssa.btgym_ssa import SSA
import os

parser = argparse.ArgumentParser(description='Mstatistics evaluation on bottom 0.2 data')
parser.add_argument('--data', type=str, default='../data3/*.npz', help='directory of data')
parser.add_argument('--ssa_window', type=int, default=5, help='n_components for ssa preprocessing')
parser.add_argument('--bs', type=int, default=150, help='buffer size for ssa')
parser.add_argument('--ws', type=int, default=100, help='window size')
parser.add_argument('--min_requirement', type=int, default=300, help='window size')
parser.add_argument('--memory_size', type=int, default=500, help='memory size per distribution ')
parser.add_argument('--cp_range', type=int, default=5, help='range to determine cp')
parser.add_argument('--forgetting_factor', type=float, default=0.55, help='forgetting_factor')
parser.add_argument('--out_threshold', type=float, default=2, help='threshold for outlier filtering')
parser.add_argument('--threshold', type=float, default=1.5, help='threshold')
parser.add_argument('--quantile', type=float, default=0.95, help='quantile')
parser.add_argument('--fixed_outlier', type=float, default=1, help='preprocess outlier filter')
parser.add_argument('--outfile', type=str, default='mmdr1', help='name of file to save results')

args = parser.parse_args()
def preprocess(data, fixed_t):
    del_idx = []
    for i in range (data.shape[0]):
        if abs(data[i, 1]) > fixed_t:
            del_idx.append(i)
    return np.delete(data, del_idx, axis=0)

# Scale input data to range of -1 to 1
def scale_input(x):
    input_min = 0
    input_max = 1
    return (x - input_min) / (input_max - input_min)

def sliding_window(elements, window_size):
    if len(elements) <= window_size:
        return elements
    new = np.empty((0, window_size))
    for i in range(len(elements) - window_size + 1):
        new = np.vstack((new, elements[i:i+window_size]))
    return new

def maximum_mean_discrepancy(X, Y, kernel='rbf', gamma=None):
    K_XX = pairwise_kernels(X, metric=kernel, gamma=gamma)
    K_YY = pairwise_kernels(Y, metric=kernel, gamma=gamma)
    K_XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)
    mmd = np.mean(K_XX) - 2 * np.mean(K_XY) + np.mean(K_YY)
    return mmd

if __name__ == '__main__':
    folder = args.data
    fixed_threshold = 1.5

    error_margin = 1036800 # 12 days
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
        # if i !='../data3ata3ata3\\J496_T5bottom02.npz' and i !='../data3\\J802_T1bottom02.npz' and i !='../data3\\J023_T1bottom02.npz' and i !='../data3\\Q152_T2bottom02.npz':
        if i !='../data3\\Q152_T1bottom02.npz':
            continue
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
        test_var_dl = test_dl_1gal[:, 1]

        train_dl_2gal = train_dl[~np.isnan(train_dl).any(axis=1)]
        train_dl_2gal = preprocess(train_dl_2gal, fixed_threshold)

        # initialisation for preprocessing module
        X = train_dl_2gal[:, 1]
        ssa = SSA(window=args.ssa_window, max_length=len(X))
        X_pred = ssa.reset(X)
        X_pred = ssa.transform(X_pred, state=ssa.get_state())
        reconstructeds = X_pred.sum(axis=0)
        residuals = X - reconstructeds
        resmean = residuals.mean()
        M2 = ((residuals - resmean) ** 2).sum()

        # initialisation for feature extraction module
        reconstructeds = sliding_window(X, args.ws)
        class_no = 1
        memory = reconstructeds
        if len(reconstructeds) > args.memory_size:
            random_indices = np.random.choice(len(reconstructeds), size=args.memory_size, replace=False)
            memory = memory[random_indices]
        detector = Detector(args.ws, args)
        detector.addsample2memory(memory, class_no, len(memory))

        ctr = 0
        step = args.bs
        scores = [0]*(args.ws-1)
        mss = [0] * (args.ws - 1)
        outliers = []
        preds = []
        filtered = []
        sample = np.empty((0, args.ws))
        collection_period = 1000000000
        detected = False
        thresholds = [0]*(args.ws-1)
        gt_margin = []
        for tt in cps:
            closest_element = ts[ts < tt].max()
            idx = np.where(ts == closest_element)[0][0]
            gt_margin.append((ts[idx-96], tt+error_margin, tt))

        while ctr < test_var_dl.shape[0]:
            new = test_var_dl[ctr:ctr + step]
            updates = ssa.update(new)
            updates = ssa.transform(updates, state=ssa.get_state())[:, args.ssa_window-1:]
            reconstructed = updates.sum(axis=0)
            residual = new - reconstructed
            residuals = np.concatenate([residuals, residual])

            for i1 in range(len(new)):
                delta = residual[i1] - resmean
                resmean += delta / (ctr + i1 + len(train_dl_2gal))
                M2 += delta * (residual[i1] - resmean)
                stdev = math.sqrt(M2 / (ctr + i1 + len(train_dl_2gal) - 1))
                threshold_upper = resmean + args.out_threshold * stdev
                threshold_lower = resmean - args.out_threshold * stdev

                if residual[i1] > threshold_upper or residual[i1] < threshold_lower:
                    outliers.append(ctr + i1)
                    filtered.append(np.mean(filtered[-5:] if len(filtered)>5 else 0))
                else:
                    filtered.append(new[i1])

            # detection
            if collection_period > args.min_requirement:
                if ctr == 0:
                    window = np.array(filtered)
                else:
                    window = np.array(filtered[-args.ws - step + 1:])
                if len(window) <= args.ws:
                    break
                window = sliding_window(window, args.ws)
                for aa in range(len(window)):
                    score = maximum_mean_discrepancy(window[aa].reshape(-1, 1), detector.current_centroid.reshape(-1, 1))
                    scores.append(score)
                    thresholds.append(detector.memory_info[detector.current_index]['threshold'])
                    ms = np.median(scores[-args.cp_range:])
                    mss.append(ms)
                    if ms > detector.memory_info[detector.current_index]['threshold']:
                        min_dist = 100000
                        n = 1
                        dk = None
                        while n <= len(detector.memory):
                            distribution = detector.memory[n]['centroid']
                            cur_threshold = detector.memory_info[n]['threshold']
                            try:
                                mm = sliding_window(np.array(filtered[-args.ws-step+aa + 1-args.cp_range:-step+aa]), args.ws)
                                distance = np.median([maximum_mean_discrepancy(mm[i].reshape(-1, 1), detector.current_centroid.reshape(-1, 1)) for i in range(len(mm))])
                            except:
                                distance = maximum_mean_discrepancy(window[aa].reshape(-1, 1), detector.current_centroid.reshape(-1, 1))
                            if distance < cur_threshold:
                                if distance < min_dist:
                                    min_dist = distance
                                    dk = n
                            n += 1
                        if dk == len(detector.memory):
                            dk = None
                        if dk is None:
                            detector.N.append(ctr + aa)
                            detector.current_index = -1
                        else:
                            detector.R.append(ctr + aa)
                            detector.current_index = dk
                        collection_period = 0
                        detected = True
                        filtered = filtered[:-len(window) + aa + 1]
                        detector.newsample = []
                        break
                    else:
                        detector.newsample.append(window[aa])
                # update the rep and threshold for the current distribution
                if collection_period > args.min_requirement:
                    detector.updatememory()
            elif collection_period <= args.min_requirement:
                scores = scores + [0] * step
                mss = mss + [0] * step
                thresholds = thresholds + [0] * step
                if len(sample) == 0:
                    window = np.array(filtered[-step + 1:])
                else:
                    window = np.array(filtered[-args.ws - step + 1:])
                if len(window) <= args.ws:
                    break
                window = sliding_window(window, args.ws)
                if collection_period + len(window) <= args.min_requirement:
                    sample = np.concatenate((sample, window))
                    collection_period += len(window)
                else: #new
                    sample = np.concatenate((sample, window))
                    if detector.current_index == -1: # new cluster
                        class_no += 1
                        detector.addsample2memory(sample, class_no, len(sample))
                    else: # recurring
                        detector.updaterecur(sample)
                    collection_period = 1000000000
                    sample = np.empty((0, args.ws))
            if detected:
                ctr += aa + 1
                detected = False
            elif len(test_var_dl) - ctr <= args.bs:
                break
            elif len(test_var_dl) - ctr <= 2 * args.bs:
                ctr += args.bs
                step = len(test_var_dl) - ctr
            else:
                ctr += args.bs

        scores = scores + [0] * (len(ts) - len(scores))
        filtered = filtered + [0] * (len(ts) - len(filtered))
        thresholds = thresholds + [0] * (len(ts) - len(thresholds))
        fig = plt.figure()
        fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
        try:
            ax[0].plot(ts, test_var_dl)
            for cp in gt_margin:
                ax[0].axvline(x=cp[0], color='green', linestyle='--')
                ax[0].axvline(x=cp[1], color='green', linestyle='--')
            for cp in detector.N:
                ax[0].axvline(x=ts[cp], color='purple', alpha=0.6)
            for cp in detector.R:
                ax[0].axvline(x=ts[cp], color='r', alpha=0.6)
            ax[1].plot(ts, scores)
            ax[1].plot(ts, thresholds)
            ax[1].plot(ts, mss)
            ax[2].plot(ts, filtered)
            plt.savefig(args.outfile + '/' + name + '.png')
        except:
            print()

        preds = detector.N + detector.R
        no_CPs += len(cps)
        no_preds += len(preds)
        for j in preds:
            timestamp = ts[j]
            for l in gt_margin:
                if timestamp >= l[0] and timestamp <= l[1]:
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
    np.savez(npz_filename, rec=rec, FAR=FAR, prec=prec, f1score=f1score, f2score=f2score, dd=dd)