from utils.Detector import Detector
from utils.Model import VAE
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import *
import numpy as np
import glob
import sys
import argparse
import matplotlib.pyplot as plt
import math
import tensorflow as tf
sys.path.append('./')
from evaluation import Evaluation_metrics
from ssa.btgym_ssa import SSA

parser = argparse.ArgumentParser(description='Mstatistics evaluation on bottom 0.2 data')
parser.add_argument('--data', type=str, default='../data3/*.npz', help='directory of data')
parser.add_argument('--ssa_window', type=int, default=5, help='n_components for ssa preprocessing')
parser.add_argument('--bs', type=int, default=24, help='buffer size for ssa')
parser.add_argument('--forgetting_factor', type=float, default=0.9, help='between 0.9 and 1')
parser.add_argument('--stabilisation_period', type=int, default=30, help='number of reference blocks')
parser.add_argument('--p', type=float, default=10, help='threshold')
parser.add_argument('--cs', type=float, default=1.5, help='c-separation')
parser.add_argument('--memory_size', type=int, default=1000, help='maximum memory size')
parser.add_argument('--fixed_outlier', type=float, default=1, help='preprocess outlier filter')
parser.add_argument('--outfile', type=str, default='firstchannel', help='name of file to save results')
args = parser.parse_args()

def preprocess(data, fixed_t):
    del_idx = []
    for i in range(data.shape[0]):
        if abs(data[i, 1]) > fixed_t:
            del_idx.append(i)
    return np.delete(data, del_idx, axis=0)

def sliding_window(seq, ws):
    i = 0
    tl = len(seq)-ws
    windows = np.empty((tl, ws, 1))
    while i < tl:
        windows[i] = seq[i:i+ws].reshape(-1,1)
        i += 1
    return windows

if __name__ == '__main__':
    folder = args.data
    fixed_threshold = 1.5
    forgetting_factor = args.forgetting_factor
    stabilisation_period = args.stabilisation_period
    p = args.p
    c = args.cs

    error_margin = 604800  # 7 days
    no_CPs = 0
    no_preds = 0
    no_TPS = 0
    delays = []

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
        M2 = ((residuals - resmean) ** 2).sum() * (len(residuals) - 1) * residuals.var()

        # initialisation for feature extraction module
        reconstructeds = sliding_window(reconstructeds, args.bs)
        feature_extracter = VAE(args.bs, 1, 4, 'elu', 1, 0.01)
        es = EarlyStopping(patience=5, verbose=1, min_delta=0.00001, monitor='val_loss', mode='auto',
                           restore_best_weights=True)
        optimis = RMSprop(learning_rate=0.001, momentum=0.9)
        feature_extracter.compile(loss=None, optimizer=optimis)
        feature_extracter.fit(reconstructeds, batch_size=16, epochs=50, validation_split=0.2, shuffle=True, callbacks=[es])
        feature_extracter.save_weights('experiment_log/' + args.outfile)

        preds = []
        outliers = []
        ctr = 0
        step = args.bs
        detector = Detector(forgetting_factor=forgetting_factor, stabilisation_period=args.stabilisation_period, p=p, c=c, memory_size=args.memory_size)
        detector.FeatureExtracter = feature_extracter
        detector.initialisation(reconstructeds)
        arrays = []

        while ctr < len(test_var_dl):
            new = test_var_dl[ctr:ctr + step]
            updates = ssa.update(new)
            updates = ssa.transform(updates, state=ssa.get_state())[:, -step:]
            reconstructed = updates.sum(axis=0)
            # reconstructed = updates[1,:]
            residual = new - reconstructed
            residuals = np.concatenate([residuals, residual])
            arrays.append(reconstructed)

            filtered = []
            for k in range(len(new)):
                delta = residual[k] - resmean
                resmean += delta / (ctr + k)
                M2 += delta * (residual[k] - resmean)

                stdev = math.sqrt(M2 / (ctr + k - 1))
                threshold_upper = resmean + 2 * stdev
                threshold_lower = resmean - 2 * stdev

                if residual[k] > threshold_upper or residual[k] < threshold_lower:
                    outliers.append(ctr + k)
                    filtered.append(0)
                    continue
                else:
                    filtered.append(reconstructed[k])

                if len(filtered) >= args.bs:
                    arr = np.array(filtered[-args.bs:])
                    arr = arr.reshape((1, len(arr), 1))
                    if detector.predict(arr, ctr + k):
                        preds.append(ctr + step)
                        feature_extracter.fit(detector.MemoryList, batch_size=16, epochs=20, validation_split=0.2,
                                              shuffle=True, callbacks=[es])
                        detector.FeatureExtracter = feature_extracter

            if len(test_var_dl) - ctr <= args.bs:
                break
            elif len(test_var_dl) - ctr <= 2*args.bs:
                ctr += args.bs
                step = len(test_var_dl) - ctr
                print(step)
            else:
                ctr += args.bs

        arrays = np.concatenate(arrays, axis=0)
        print(preds)
        fig = plt.figure()
        fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
        ax[0].plot(ts, test_var_dl)
        for cp in cps:
            ax[0].axvline(x=cp, color='g', alpha=0.6)

        ax[1].plot(ts, arrays)
        for cp in preds:
            ax[1].axvline(x=ts[cp], color='g', alpha=0.6)

        # ax[2].plot(ts, arrays)
        # plt.show()
        plt.savefig(name + '.png')
        no_CPs += len(cps)
        no_preds += len(preds)
        for j in preds:
            timestamp = ts[j]
            for l in cps:
                if timestamp >= l and timestamp <= l + error_margin:
                    no_TPS += 1
                    delays.append(timestamp - l)

    rec = Evaluation_metrics.recall(no_TPS, no_CPs)
    FAR = Evaluation_metrics.False_Alarm_Rate(no_preds, no_TPS)
    prec = Evaluation_metrics.precision(no_TPS, no_preds)
    f1score = Evaluation_metrics.F1_score(rec, prec)
    dd = Evaluation_metrics.detection_delay(delays)
    print('recall: ', rec)
    print('false alarm rate: ', FAR)
    print('precision: ', prec)
    print('F1 Score: ', f1score)
    print('detection delay: ', dd)

    npz_filename = args.outfile
    np.savez(npz_filename,
             rec=rec, FAR=FAR, prec=prec, f1score=f1score, dd=dd)