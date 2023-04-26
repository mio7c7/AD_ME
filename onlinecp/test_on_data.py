# -*- coding: utf-8 -*-
import argparse
import os
import time
import numpy as np
import glob
import onlinecp.algos as algos
# import onlinecp.utils.evaluation as ev
import onlinecp.utils.me_evaluation as ev
import onlinecp.utils.feature_functions as feat
import onlinecp.utils.gendata as gd
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('./')
from evaluation import Evaluation_metrics
from ssa.btgym_ssa import SSA
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data3/*.npz', help='directory of data')
parser.add_argument('--ssa_window', type=int, default=5, help='n_components for ssa preprocessing')
parser.add_argument('--algo', choices=['newmaRFF', 'newmaFF', 'newmaOPU', 'MA', 'ScanB'], default='newmaFF')
parser.add_argument('--outfile', type=str, default='test', help='name of file to save results')
parser.add_argument('--n', type=int, default=2000, help='number of samples for each distribution')
parser.add_argument('--nb', type=int, default=20, help='number of changes of distribution in the series')
parser.add_argument('--d', type=int, default=1, help='dimensionality of the samples in the time series')
parser.add_argument('--B', type=int, default=400, help='window size')
parser.add_argument('--seed', type=int, default=0, help='seed for PRNG')
parser.add_argument('--show', action='store_true', help='show performance metrics plots')
args = parser.parse_args()

def preprocess(data, fixed_t):
    del_idx = []
    for i in range (data.shape[0]):
        if abs(data[i, 1]) > fixed_t:
            del_idx.append(i)
    return np.delete(data, del_idx, axis=0)

if __name__ == '__main__':
    np.random.seed(args.seed)
    algo = args.algo
    d = args.d
    folder = args.data
    total_cp, total_nd, total_FA, total_DD = 0, np.zeros(20), np.zeros(20), np.zeros(20)
    ctr = 0
    fixed_threshold = 1.5

    error_margin = 50
    no_CPs = 0
    no_preds = 0
    no_TPS = 0
    delays = []

    for i in glob.glob(folder):
        data = np.load(i, allow_pickle=True)
        name = i[-19:-12]
        train_ts, train_dl, test_ts_1gal, test_dl_1gal, label = data['train_ts'], data['train_dl'], data['test_ts_2gal'], data['test_dl_2gal'], data['label'].item()
        test_dl_1gal = test_dl_1gal[~np.isnan(test_dl_1gal).any(axis=1)]
        test_dl_1gal = preprocess(test_dl_1gal, fixed_threshold)
        ts = test_dl_1gal[:, 0]
        cps = label['test_2gal']
        cps = np.array(list(cps)).astype(np.float64)
        test_var_dl = test_dl_1gal[:, 1]
        test_ht_dl = test_dl_1gal[:, 2]
        X = np.stack((test_var_dl, test_ht_dl), axis=1)
        # X = test_var_dl.reshape(-1,1)
        orig_x = X.copy()

        ground_truth = np.zeros(X.shape[0])
        indices = ts.searchsorted(cps)
        for ind in indices:
            ground_truth[ind] = 1

        # common config
        choice_sigma = 'median'
        numel = 100
        data_sigma_estimate = X[:numel]  # data for median trick to estimate sigma
        B = args.B  # window size

        # Scan-B config
        N = 3  # number of windows in scan-B

        # Newma and MA config
        big_Lambda, small_lambda = algos.select_optimal_parameters(B)  # forget factors chosen with heuristic in the paper
        thres_ff = small_lambda
        # number of random features is set automatically with this criterion
        m = int((1 / 4) / (small_lambda + big_Lambda) ** 2)
        m_OPU = 10 * m
        W, sigmasq = feat.generate_frequencies(m, d, data=data_sigma_estimate, choice_sigma=choice_sigma)

        if algo == 'ScanB':
            print('Start algo ', algo, '... (can be long !)')
            detector = algos.ScanB(X[0], kernel_func=lambda x, y: feat.gauss_kernel(x, y, np.sqrt(sigmasq)), window_size=B,
                                   nbr_windows=N, adapt_forget_factor=thres_ff)
            detector.apply_to_data(X)
        elif algo == 'MA':
            print('Start algo ', algo, '...')
            print('# RF: ', m)

            def feat_func(x):
                return feat.fourier_feat(x, W)

            detector = algos.MA(X[0], window_size=B, feat_func=feat_func, adapt_forget_factor=thres_ff)
            detector.apply_to_data(X)
        elif algo == 'newmaFF':
            print('Start algo ', algo, '...')
            print('# RF: ', m)
            import onlinecp.utils.fastfood as ff
            FF = ff.Fastfood(sigma=np.sqrt(sigmasq), n_components=m)
            FF.fit(X)
            X = FF.transform(X)

            detector = algos.NEWMA(X[0], forget_factor=big_Lambda, forget_factor2=small_lambda,
                                   adapt_forget_factor=thres_ff)
            detector.apply_to_data(X)
        elif algo == 'newmaRFF':  # newma RF
            print('Start algo ', algo, '...')
            print('# RF: ', m)

            def feat_func(x):
                return feat.fourier_feat(x, W)

            detector = algos.NEWMA(X[0], forget_factor=big_Lambda, forget_factor2=small_lambda, feat_func=feat_func,
                                   adapt_forget_factor=thres_ff)
            detector.apply_to_data(X)
        else:  # newmaOPU
            print('Start algo ', algo, '...')
            m_OPU = 34570
            m = m_OPU
            print('# RF: ', m)
            try:
                from lightonml.encoding.base import BinaryThresholdEncoder
                from lightonopu.opu import OPU
            except ImportError:
                raise Exception("Please get in touch to use LightOn OPU.")

            opu = OPU(n_components=m)
            opu.open()
            n_levels = 38
            Xencode = np.empty((X.shape[0], n_levels * X.shape[1]), dtype='uint8')
            t = time.time()
            mi, Ma = np.min(X), np.max(X)  # rescale to 0 255
            X = 255 * ((X - mi) / (Ma - mi))
            X = X.astype('uint8')

            for i in range(n_levels):
                Xencode[:, i * X.shape[1]:(i + 1) * X.shape[1]] = X > 65 + i * 5
            del X

            start = time.time()
            X = opu.transform(Xencode)
            print('Projections took:', time.time()-start)
            del Xencode
            opu.device.close()

            # convert to float online to avoid memory error
            mult = 1.5
            detector = algos.NEWMA(X[0], forget_factor=big_Lambda,
                                   feat_func=lambda x: x.astype('float32'),
                                   forget_factor2=small_lambda, adapt_forget_factor=thres_ff*mult,
                                   thresholding_quantile=0.95, dist_func=lambda z1, z2: np.linalg.norm(z1 - z2))
            detector.apply_to_data(X)

        # compute performance metrics
        detection_stat = np.array([i[0] for i in detector.stat_stored])  # padding
        online_th = np.array([i[1] for i in detector.stat_stored])

        # fig = plt.figure()
        # fig, ax = plt.subplots(3, figsize=[18, 16], sharex=True)
        # ax[0].plot(ts, orig_x[:, 0])
        # for i in cps:
        #     ax[0].axvline(x=i, color='g', alpha=0.6)
        # ax[1].plot(ts, orig_x[:, 0])
        # for i in cps:
        #     ax[1].axvline(x=i, color='g', alpha=0.6)
        # ax[2].plot(ts, detection_stat, color='r')
        # ax[2].plot(ts, online_th, color='b')
        # for i in cps:
        #     ax[2].axvline(x=i, color='g', alpha=0.6)
        # plt.savefig(name + '.jpg')

        # display perf
        EDD, FA, ND, CP = ev.compute_curves(ground_truth, detection_stat, num_points=20, start_coeff=0.6, end_coeff=0.9, error_margin=150)
        # EDDth, FAth, NDth = ev.compute_curves(ground_truth, detection_stat, num_points=1,
        #                                       thres_values=online_th, start_coeff=1, end_coeff=1, error_margin=80)
        total_cp += CP
        ctr += 1
        for i in range(20):
            total_DD[i] += EDD[i]
            total_FA[i] += FA[i]
            total_nd[i] += ND[i]
        pass


    # total_nd_ratio = total_nd/total_cp
    # total_FA_ratio = total_FA/ctr
    # total_DD_ratio = total_DD/ctr
    # print('ND: ', total_nd)
    # print('FA: ', total_FA)
    # print('DD: ', total_DD)
    # print(total_cp, ctr)
    # print('ND_ratio: ', total_nd_ratio)
    # print('FA_ratio: ', total_FA_ratio)
    # print('DD_ratio: ', total_DD_ratio)
    #
    # npz_filename = args.outfile
    # np.savez(npz_filename,
    #          EDD=total_DD_ratio, FA=total_FA_ratio, ND=total_nd_ratio)



        #
        # plt.figure()
        # plt.plot(FA, EDD, '-o', label='')
        # plt.plot(FAth, EDDth, 'o', markersize=20)
        # plt.xlabel('False Alarm')
        # plt.ylabel('Expected Detection Delay')
        # plt.show()
        #
        # plt.figure()
        # plt.plot(FA, ND, '-o')
        # plt.plot(FAth, NDth, 'o', markersize=20)
        # plt.xlabel('False Alarm')
        # plt.ylabel('Missed Detection')
        # plt.show()