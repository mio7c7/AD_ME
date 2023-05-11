import glob
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Input, Dense, Lambda
from utils.Model import VAE
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import *
import math
sys.path.append('./')
from evaluation import Evaluation_metrics
from ssa.btgym_ssa import SSA
import os

parser = argparse.ArgumentParser(description='Mstatistics evaluation on bottom 0.2 data')
parser.add_argument('--data', type=str, default='../data3/*.npz', help='directory of data')
parser.add_argument('--ssa_window', type=int, default=5, help='n_components for ssa preprocessing')
parser.add_argument('--g_noise', type=float, default=0.01, help='gaussian noise')
parser.add_argument('--buffer_ts', type=int, default=500, help='Number of timestamps for initilisation')
parser.add_argument('--bs', type=int, default=48, help='buffer size for ssa')
parser.add_argument('--ws', type=int, default=40, help='window size')
parser.add_argument('--collection_period', type=int, default=400, help='preprocess outlier filter')
parser.add_argument('--memory_size', type=int, default=2000, help='preprocess outlier filter')
parser.add_argument('--dense_dim', type=int, default=2, help='no of neuron in dense')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
parser.add_argument('--kl_weight', type=float, default=1, help='kl_weight')
parser.add_argument('--latent_dim', type=int, default=1, help='latent_dim')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--epoch', type=int, default=300, help='epoch')
parser.add_argument('--out_threshold', type=float, default=2, help='threshold for outlier filtering')
parser.add_argument('--threshold', type=float, default=3, help='threshold')
parser.add_argument('--quantile', type=float, default=0.95, help='quantile')
parser.add_argument('--fixed_outlier', type=float, default=1, help='preprocess outlier filter')
parser.add_argument('--outfile', type=str, default='quantile1', help='name of file to save results')

args = parser.parse_args()
def preprocess(data, fixed_t):
    del_idx = []
    for i in range (data.shape[0]):
        if abs(data[i, 1]) > fixed_t:
            del_idx.append(i)
    return np.delete(data, del_idx, axis=0)

def sliding_window(elements, window_size):
    if len(elements) <= window_size:
        return elements
    new = np.empty((0, window_size))
    for i in range(len(elements) - window_size + 1):
        new = np.vstack((new, elements[i:i+window_size]))
    return new

def reservoir_sampling(memory, new_sample, class_no, seen):
    if args.memory_size // class_no >= args.collection_period:
        random_indices = np.random.choice(len(new_sample), size=args.collection_period, replace=True)
        random_samples = new_sample[random_indices]
        memory = np.vstack((memory, random_samples))
        seen += len(random_indices)
    else:
        for ss in new_sample:
            j = np.random.randint(0, seen)
            if j < args.memory_size:
                memory[j] = ss
            seen += 1
    return memory, seen

if __name__ == '__main__':
    folder = args.data
    fixed_threshold = 1.5

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
        M2 = ((residuals - resmean) ** 2).sum()

        # initialisation for feature extraction module
        reconstructeds = sliding_window(X, args.ws)
        reconstructeds = np.expand_dims(reconstructeds, axis=-1)

        feature_extracter = VAE(args.ws, 1, args.dense_dim, 'elu', args.latent_dim, args.kl_weight, args.dropout)
        es = EarlyStopping(patience=7, verbose=1, min_delta=0.0001, monitor='val_loss', mode='auto')
        # optimis = Adam(learning_rate=0.001)
        optimis = RMSprop(learning_rate=0.01)
        feature_extracter.compile(loss=None, optimizer=optimis)
        feature_extracter.fit(reconstructeds, batch_size=args.batch_size, epochs=args.epoch, validation_split=0.3, shuffle=True, callbacks=[es])
        # feature_extracter.save_weights('experiment_log/' + args.outfile)
        z_mean, z_log_sigma, z, pred = feature_extracter.predict(reconstructeds)
        MSE = [mean_squared_error(z_mean[i], z_mean[i + args.ws]) for i in range(len(reconstructeds) - args.ws)]
        mse_quantile = np.quantile(MSE, args.quantile)
        threshold = args.threshold*mse_quantile

        ctr = 0
        step = args.bs
        scores = [0]*(args.ws-1)
        outliers = []
        preds = []
        filtered = []
        memory = np.empty((0, reconstructeds.shape[1], reconstructeds.shape[2]))
        sample = np.empty((0, reconstructeds.shape[1], reconstructeds.shape[2]))
        collection_period = 1000000000
        class_no = 1
        seen = 0
        memory, seen = reservoir_sampling(memory, reconstructeds, class_no, seen)

        while ctr < test_var_dl.shape[0]:
            new = test_var_dl[ctr:ctr + step]
            updates = ssa.update(new)
            updates = ssa.transform(updates, state=ssa.get_state())[:, -step:]
            reconstructed = updates.sum(axis=0)
            residual = new - reconstructed
            residuals = np.concatenate([residuals, residual])
            # arrays.append(reconstructed)

            for k in range(len(new)):
                delta = residual[k] - resmean
                resmean += delta / (ctr + k + len(train_dl_2gal))
                M2 += delta * (residual[k] - resmean)
                stdev = math.sqrt(M2 / (ctr + k + len(train_dl_2gal) - 1))
                threshold_upper = resmean + args.out_threshold * stdev
                threshold_lower = resmean - args.out_threshold * stdev

                if residual[k] > threshold_upper or residual[k] < threshold_lower:
                    outliers.append(ctr + k)
                    filtered.append(np.mean(new))
                    continue
                else:
                    filtered.append(new[k])

                if collection_period < args.collection_period:
                    window = np.array(filtered[-args.ws:])
                    window = window.reshape((1, len(window), 1))
                    sample = np.vstack((sample, window))
                    collection_period += 1
                    continue
                elif collection_period == args.collection_period:
                    class_no += 1
                    memory, seen = reservoir_sampling(memory, sample, class_no, seen)
                    # feature_extracter = VAE(args.ws, 1, args.dense_dim, 'elu', args.latent_dim, args.kl_weight)
                    # feature_extracter.compile(loss=None, optimizer=optimis)
                    feature_extracter.fit(memory, batch_size=args.batch_size, epochs=args.epoch, validation_split=0.2,
                                          shuffle=True, callbacks=[es])
                    z_mean, z_log_sigma, z, pred = feature_extracter.predict(memory)
                    MSE = [mean_squared_error(z_mean[i], z_mean[i + args.ws]) for i in range(len(memory) - args.ws)]
                    mse_quantile = np.quantile(MSE, args.quantile)
                    threshold = args.threshold * mse_quantile
                    sample = np.empty((0, memory.shape[1], memory.shape[2]))
                    collection_period = 1000000000

            # detection
            if collection_period > args.collection_period:
                if ctr == 0:
                    window = np.array(filtered)
                else:
                    window = np.array(filtered[-args.ws * 2 - step + 1:])
                window = sliding_window(window, args.ws)
                z_mean, z_log_sigma, z, pred = feature_extracter.predict(window)
                score = [mean_squared_error(z_mean[i], z_mean[i + args.ws]) for i in range(len(window) - args.ws)]
                scores = scores + list(score)
                # MSE = MSE + score
                # threshold = np.mean(MSE) + args.threshold * np.std(MSE)
                for m in range(len(score)):
                    if score[m] > threshold:
                        preds.append(ctr + m)
                        sample = np.empty((0, reconstructeds.shape[1], reconstructeds.shape[2]))
                        collection_period = 0
                        break
                    else:
                        MSE.append(score[m])
                        mse_quantile = np.quantile(MSE, args.quantile)
                        threshold = args.threshold * mse_quantile
            else:
                scores = scores + [0] * step

            if len(test_var_dl) - ctr <= args.bs:
                break
            elif len(test_var_dl) - ctr <= 2 * args.bs:
                ctr += args.bs
                step = len(test_var_dl) - ctr
            else:
                ctr += args.bs

        no_CPs += len(cps)
        no_preds += len(preds)
        for j in preds:
            timestamp = ts[j]
            for l in cps:
                if timestamp >= l and timestamp <= l + error_margin:
                    no_TPS += 1
                    delays.append(timestamp - l)

        if not os.path.exists(args.outfile):
            os.makedirs(args.outfile)
        scores = scores + [0]*args.ws
        fig = plt.figure()
        fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
        ax[0].plot(ts, test_var_dl)
        for cp in cps:
            ax[0].axvline(x=cp, color='g', alpha=0.6)

        ax[1].plot(ts, filtered)
        for cp in preds:
            ax[1].axvline(x=ts[cp], color='g', alpha=0.6)

        ax[2].plot(ts, scores)
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
    np.savez(npz_filename,
             rec=rec, FAR=FAR, prec=prec, f1score=f1score, dd=dd)