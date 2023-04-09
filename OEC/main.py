from utils.Detector import Detector
import numpy as np
import glob
import sys
import argparse
sys.path.append('C:/Users/Administrator/Documents/GitHub/AD_ME/evaluation/')
import Evaluation_metrics

parser = argparse.ArgumentParser(description='Mstatistics evaluation on bottom 0.2 data')
parser.add_argument('--data', type=str, default='../data3/*.npz', help='directory of data')
parser.add_argument('--forgetting_factor', type=float, default=0.9, help='between 0.9 and 1')
parser.add_argument('--stabilisation_period', type=int, default=30, help='number of reference blocks')
parser.add_argument('--p', type=float, default=10, help='threshold')
parser.add_argument('--fixed_outlier', type=float, default=1, help='preprocess outlier filter')
parser.add_argument('--outfile', type=str, default='15IQRMED11WND100', help='name of file to save results')
args = parser.parse_args()


def preprocess(data, fixed_t):
    del_idx = []
    for i in range (data.shape[0]):
        if abs(data[i, 1]) > fixed_t:
            del_idx.append(i)
    return np.delete(data, del_idx, axis=0)

if __name__ == '__main__':
    folder = args.data
    fixed_threshold = 1.5
    forgetting_factor = args.forgetting_factor
    stabilisation_period = args.stabilisation_period
    p = args.p

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
        train_ht_dl = train_dl[:, 2]
        test_var_dl = test_dl_1gal[:, 1]
        test_ht_dl = test_dl_1gal[:, 2]
        multi_test = np.stack((test_var_dl, test_ht_dl), axis=1)
        test_var_dl = np.reshape(test_var_dl, (test_var_dl.shape[0], 1))

        # initialisation
        # initialsets = multi_test[:stabilisation_period]
        initialsets = test_var_dl[:stabilisation_period]
        detector = Detector(forgetting_factor=forgetting_factor, stabilisation_period=stabilisation_period,
                            p=p)
        detector.initialisation(initialsets)
        preds = []

        for ct, value in enumerate(test_var_dl[stabilisation_period:]):
            if detector.predict(value, ct):
                preds.append(stabilisation_period + ct)

        # print(preds)
        # fig = plt.figure()
        # fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
        # ax[0].plot(ts, multi_test[:, 0])
        # for cp in cps:
        #     ax[0].axvline(x=cp, color='g', alpha=0.6)
        #
        # ax[1].plot(ts[stabilisation_period:], detector.score)
        # for cp in preds:
        #     ax[1].axvline(x=ts[cp], color='g', alpha=0.6)
        #
        # ax[2].plot(ts, multi_test[:, 1])
        # # plt.show()
        # plt.savefig(name + '.png')
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