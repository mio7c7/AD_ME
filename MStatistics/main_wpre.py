from MS import Kernel
import numpy as np
import glob
import sys
import argparse
import matplotlib.pyplot as plt
import math
sys.path.append('./')
from evaluation import Evaluation_metrics
from ssa.btgym_ssa import SSA

parser = argparse.ArgumentParser(description='Mstatistics evaluation on bottom 0.2 data')
parser.add_argument('--data', type=str, default='../data3/*.npz', help='directory of data')
parser.add_argument('--ssa_window', type=int, default=5, help='n_components for ssa preprocessing')
parser.add_argument('--bo', type=int, default=20, help='reference block window size')
parser.add_argument('--N', type=int, default=20, help='number of reference blocks')
parser.add_argument('--bs', type=int, default=50, help='buffer size for ssa')
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
    bo = args.bo
    N = args.N
    threshold = args.threshold

    error_margin = 604800  # 7 days
    no_CPs = 0
    no_preds = 0
    no_TPS = 0
    delays = []
    step = args.bs
    ignored = ['../data3\\A043_T2bottom02.npz', '../data3\\A441_T2bottom02.npz',
               '../data3\\B402_T3bottom02.npz', '../data3\\B402_T4bottom02.npz',
               '../data3\\B402_T4bottom06.npz', '../data3\\F257_T2bottom02.npz',
               '../data3\\F257_T2bottom05.npz', '../data3\\F289_T4bottom02.npz', ]

    for i in glob.glob(folder):
        if i in ignored:
            continue
        data = np.load(i, allow_pickle=True)
        name = i[-19:-12]
        train_ts, train_dl, test_ts_1gal, test_dl_1gal, label = data['train_ts'], data['train_dl'], data[
            'test_ts_2gal'], data['test_dl_2gal'], data['label'].item()
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

        # initialisation
        preds = []
        outliers = []
        input = test_var_dl.copy()
        scores = np.zeros(len(test_var_dl))
        model = Kernel(b=threshold, bo=bo, N=N)
        history = X
        if len(history) >= args.N * args.bo:
            warm_up = False
        else:
            warm_up = True
        ctr = 0
        filtered = []
        detected = False

        while ctr <= len(test_var_dl):
            new = np.array(input[ctr:ctr + step])
            updates = ssa.update(new)
            updates = ssa.transform(updates, state=ssa.get_state())[:, -len(new):]
            reconstructed = updates.sum(axis=0)
            residual = new - reconstructed
            residuals = np.concatenate([residuals, residual])
            if warm_up:
                history = np.concatenate((history, new))
                if len(history) >= args.N * args.bo:
                    warm_up = False
                else:
                    ctr += args.bs
                    continue

            for k in range(len(new)):
                delta = residual[k] - resmean
                resmean += delta / (ctr + k + len(train_dl_2gal))
                M2 += delta * (residual[k] - resmean)
                stdev = math.sqrt(M2 / (ctr + k + + len(train_dl_2gal)- 1))
                threshold_upper = resmean + 2 * stdev
                threshold_lower = resmean - 2 * stdev
                if residual[k] > threshold_upper or residual[k] < threshold_lower:
                    outliers.append(ctr + k)
                    filtered.append(np.mean(filtered[-10:] if len(filtered)>10 else 0))
                    history = np.concatenate((history, [np.mean(filtered[-10:] if len(filtered)>10 else 0)]))
                else:
                    filtered.append(new[k])
                    history = np.concatenate((history, [new[k]]))

                if len(filtered) >= args.bo:
                    right_wnd = np.array(filtered[-args.bo:]).reshape(-1, 1)
                else:
                    continue

                reference_pool = history[:- args.bo]
                matrix_X = model.create_reference_blocks(reference_pool, args.N, args.bo)
                score = model.predict(right_wnd, matrix_X)
                scores[ctr + k] = score

                if score >= threshold:
                    preds.append(ctr + k)
                    ctr = ctr + k + 1
                    history = np.empty((0))
                    filtered = []
                    warm_up = True
                    detected = True
                    break

            if len(test_var_dl) - ctr <= args.bs:
                break
            elif detected:
                ctr = ctr + k + 1
                detected = False
            elif len(test_var_dl) - ctr <= 2 * args.bs:
                ctr += args.bs
                step = len(test_var_dl) - ctr
            else:
                ctr += args.bs

        fig = plt.figure()
        fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
        ax[0].plot(ts, test_var_dl)
        for cp in cps:
            ax[0].axvline(x=cp, color='g', alpha=0.6)

        ax[1].plot(ts, scores)
        for cp in preds:
            ax[1].axvline(x=ts[cp], color='g', alpha=0.6)

        # plt.show()
        # plt.savefig(name + '.png')
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