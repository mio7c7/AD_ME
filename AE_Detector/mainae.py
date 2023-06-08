import glob
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from ae import AE
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import *
import math
sys.path.append('../')
from evaluation import Evaluation_metrics
from ssa.btgym_ssa import SSA

parser = argparse.ArgumentParser(description='Mstatistics evaluation on bottom 0.2 data')
parser.add_argument('--data', type=str, default='../data3/*.npz', help='directory of data')
parser.add_argument('--ssa_window', type=int, default=5, help='n_components for ssa preprocessing')
parser.add_argument('--g_noise', type=float, default=0.01, help='white noise')
parser.add_argument('--buffer_ts', type=int, default=500, help='cold start period')
parser.add_argument('--bs', type=int, default=150, help='buffer size for ssa')
parser.add_argument('--ws', type=int, default=100, help='window size')
parser.add_argument('--dense_dim', type=int, default=4, help='no of neuron in dense')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--epoch', type=int, default=100, help='epoch')
parser.add_argument('--out_threshold', type=float, default=2, help='threshold for outlier filtering')
parser.add_argument('--threshold', type=float, default=1.25, help='threshold')
parser.add_argument('--fixed_outlier', type=float, default=1.5, help='preprocess outlier filter')
parser.add_argument('--outfile', type=str, default='ae', help='name of file to save results')
args = parser.parse_args()

def preprocess(data, fixed_t):
    del_idx = []
    for i in range (data.shape[0]):
        if abs(data[i, 1]) > fixed_t:
            del_idx.append(i)
    return np.delete(data, del_idx, axis=0)

def sliding_window(elements, window_size):
    if len(elements) < window_size:
        return elements
    new = np.empty((0, window_size))
    for i in range(len(elements) - window_size + 1):
        new = np.vstack((new, elements[i:i+window_size]))
    return new

if __name__ == '__main__':
    folder = args.data
    fixed_threshold = 1.5

    error_margin = 864000 # 7 days
    no_CPs = 0
    no_preds = 0
    no_TPS = 0
    delays = []

    if not os.path.exists(args.outfile):
        os.makedirs(args.outfile)

    for i in glob.glob(folder):
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
        residuals = X - reconstructeds
        resmean = residuals.mean()
        M2 = ((residuals - resmean) ** 2).sum()

        noise = np.random.normal(0, args.g_noise, size=(X.shape[0]))
        noisy_Data = X + noise
        noisy_Data = sliding_window(noisy_Data, args.ws)
        noisy_Data = np.expand_dims(noisy_Data, axis=-1)
        reconstructeds = sliding_window(X, args.ws)
        vae = AE(args.ws, 1, args.dense_dim, 'relu', args.dropout)
        es = EarlyStopping(patience=7, verbose=0, min_delta=0.0001, monitor='val_loss', mode='auto')
        optimis = RMSprop(learning_rate=0.001)
        vae.compile(loss=None, optimizer=optimis)
        vae.fit(noisy_Data, batch_size=args.batch_size, epochs=args.epoch, validation_split=0.3, callbacks=[es])
        z, pred = vae.predict(noisy_Data)
        MAE = np.mean(np.mean(np.abs(np.squeeze(pred, axis=-1) - reconstructeds), axis=1))
        threshold = args.threshold * MAE

        ctr = 0
        List_st = [i for i in range(ctr)]  # list of timestamps
        preds = []
        step = args.bs
        scores = np.zeros(test_var_dl.shape[0])
        outliers = []
        thresholds = []
        filtered = []
        gt_margin = []
        cp_ctr = []
        for tt in cps:
            closest_element = ts[ts < tt].max()
            idx = np.where(ts == closest_element)[0][0]
            gt_margin.append((ts[idx - 10], tt + error_margin, tt))
        initial = True
        collect = False

        while ctr < test_var_dl.shape[0]:
            new = test_var_dl[ctr:ctr + step]
            updates = ssa.update(new)
            updates = ssa.transform(updates, state=ssa.get_state())[:, args.ssa_window - 1:]
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
                    filtered.append(np.mean(new))
                else:
                    filtered.append(new[i1])
                thresholds.append(threshold)
                List_st.append(ctr+i1)

            if len(List_st) >= args.buffer_ts and collect:
                mm = np.array(filtered)
                X = mm[List_st]
                noise = np.random.normal(0, args.g_noise, size=(X.shape[0]))
                noisy_Data = X + noise
                noisy_Data = sliding_window(noisy_Data, args.ws)
                noisy_Data = np.expand_dims(noisy_Data, axis=-1)
                X = sliding_window(X, args.ws)
                # vae = AE(args.ws, 1, args.dense_dim, 'relu', args.dropout)
                # vae.compile(loss=None, optimizer=optimis)
                vae.fit(noisy_Data, batch_size=args.batch_size, epochs=args.epoch,
                        validation_split=0.3, callbacks=[es])
                z, pred = vae.predict(noisy_Data)
                MAE = np.mean(np.mean(np.abs(np.squeeze(pred, axis=-1) - X), axis=1))
                threshold = args.threshold * MAE
                initial = False
                collect = False
            elif len(List_st) > args.buffer_ts or initial:
                if len(filtered) < args.ws:
                    continue
                window = np.array(filtered[-args.ws - step + 1:])
                window = sliding_window(window, args.ws)
                _, pred = vae.predict(window)
                for aa in range(len(pred)):
                    score = np.mean(np.abs(np.squeeze(pred[aa], axis=-1) - window))
                    scores[ctr+aa] = score
                    if score > threshold:
                        preds.append(ctr+aa)
                        List_st = []
                        collect = True
                        initial = False
                        break

            if len(test_var_dl) - ctr <= args.bs:
                break
            elif len(test_var_dl) - ctr <= 2 * args.bs:
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

        filtered = filtered + [0] * (len(ts) - len(filtered))
        thresholds = thresholds + [0] * (len(ts) - len(thresholds))
        fig = plt.figure()
        fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
        try:
            ax[0].plot(ts, test_var_dl)
            for cp in gt_margin:
                ax[0].axvline(x=cp[0], color='green', linestyle='--')
                ax[0].axvline(x=cp[1], color='green', linestyle='--')
            for cp in preds:
                ax[0].axvline(x=ts[cp], color='purple', alpha=0.6)
            ax[1].plot(ts, scores)
            ax[1].plot(ts, thresholds)
            ax[2].plot(ts, filtered)
            plt.savefig(args.outfile + '/' + name + '.png')
        except:
            print()
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