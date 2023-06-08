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
parser.add_argument('--window_size', type=int, default=25, help='window_size')
parser.add_argument('--max_points', type=int, default=10, help='min blocks required in a distrib. before starting detection')
parser.add_argument('--min_batch_size', type=int, default=2, help='mini_batch_size')
parser.add_argument('--fixed_outlier', type=float, default=1, help='preprocess outlier filter')
parser.add_argument('--out_threshold', type=float, default=2, help='threshold for outlier filtering')
parser.add_argument('--epsilon', type=float, default=2, help='epsilon')
parser.add_argument('--outfile', type=str, default='25_400_20_15', help='name of file to save results')
args = parser.parse_args()

def generate_jumpingmean(window_size, stride=1, nr_cp=49, delta_t_cp=100, delta_t_cp_std=10, scale_min=-1, scale_max=1):
    """
    Generates one instance of a jumping mean time series, together with the corresponding windows and parameters
    """
    mu = np.zeros((nr_cp,))
    parameters_jumpingmean = []
    for n in range(1, nr_cp):
        mu[n] = mu[n - 1] + n / 16
    for n in range(nr_cp):
        nr = int(delta_t_cp + np.random.randn() * np.sqrt(delta_t_cp_std))
        parameters_jumpingmean.extend(mu[n] * np.ones((nr,)))

    parameters_jumpingmean = np.array([parameters_jumpingmean]).T

    ts_length = len(parameters_jumpingmean)
    timeseries = np.zeros((ts_length))
    for i in range(2, ts_length):
        # print(ar2(timeseries[i-1],timeseries[i-2], 0.6,-0.5, parameters_jumpingmean[i], 1.5))
        timeseries[i] = ar2(timeseries[i - 1], timeseries[i - 2], 0.6, -0.5, parameters_jumpingmean[i], 1.5)

    windows = ts_to_windows(timeseries, 0, window_size, stride)
    windows = minmaxscale(windows, scale_min, scale_max)
    return timeseries, windows, parameters_jumpingmean

def scale_input(x):
    input_min = 0
    input_max = 1
    return (x - input_min) / (input_max - input_min)

def sliding_window(elements, window_size, step):
    if len(elements) <= window_size:
        return elements
    new = np.empty((0, window_size))
    for i in range(0, len(elements) - window_size + 1, step):
        new = np.vstack((new, elements[i:i+window_size]))
    return new

def ar2(value1,value2,coef1,coef2,mu,sigma):
    """
    AR(2) model, cfr. paper
    """
    return coef1*value1+coef2*value2 + np.random.randn()*sigma+mu

def ts_to_windows(ts, onset, window_size, stride, normalization="timeseries"):
    """Transforms time series into list of windows"""
    windows = []
    len_ts = len(ts)
    onsets = range(onset, len_ts - window_size + 1, stride)

    if normalization == "timeseries":
        for timestamp in onsets:
            windows.append(ts[timestamp:timestamp + window_size])
    elif normalization == "window":
        for timestamp in onsets:
            windows.append(
                np.array(ts[timestamp:timestamp + window_size]) - np.mean(ts[timestamp:timestamp + window_size]))

    return np.array(windows)

def minmaxscale(data, a, b):
    """
    Scales data to the interval [a,b]
    """
    data_min = np.amin(data)
    data_max = np.amax(data)

    return (b - a) * (data - data_min) / (data_max - data_min) + a


def generate_gaussian(window_size, stride=1, nr_cp=49, delta_t_cp=100, delta_t_cp_std=10, scale_min=-1, scale_max=1):
    """
    Generates one instance of a Gaussian mixtures time series, together with the corresponding windows and parameters
    """
    mixturenumber = np.zeros((nr_cp,))
    parameters_gaussian = []
    for n in range(1, nr_cp - 1, 2):
        mixturenumber[n] = 1
    for n in range(nr_cp):
        nr = int(delta_t_cp + np.random.randn() * np.sqrt(delta_t_cp_std))
        parameters_gaussian.extend(mixturenumber[n] * np.ones((nr,)))

    parameters_gaussian = np.array([parameters_gaussian]).T

    ts_length = len(parameters_gaussian)
    timeseries = np.zeros((ts_length))
    for i in range(2, ts_length):
        # print(ar2(timeseries[i-1],timeseries[i-2], 0.6,-0.5, parameters_jumpingmean[i], 1.5))
        if parameters_gaussian[i] == 0:
            timeseries[i] = 0.5 * 0.5 * np.random.randn() + 0.5 * 0.5 * np.random.randn()
        else:
            timeseries[i] = -0.6 - 0.8 * 1 * np.random.randn() + 0.2 * 0.1 * np.random.randn()

    windows = ts_to_windows(timeseries, 0, window_size, stride)
    windows = minmaxscale(windows, scale_min, scale_max)

    return timeseries, windows, parameters_gaussian

def parameters_to_cps(parameters, window_size):
    length_ts = np.size(parameters, 0)

    index1 = range(window_size - 1, length_ts - window_size, 1)  # selects parameter at LAST time stamp of window
    index2 = range(window_size, length_ts - window_size + 1, 1)  # selects parameter at FIRST time stamp of next window

    diff_parameters = np.sqrt(np.sum(np.square(parameters[index1] - parameters[index2]), 1))

    max_diff = np.max(diff_parameters)

    return diff_parameters / max_diff

def cp_to_timestamps(changepoints, tolerance, length_ts):
    locations_cp = [idx for idx, val in enumerate(changepoints) if val > 0.0]

    output = []
    while len(locations_cp) > 0:
        k = 0
        for i in range(len(locations_cp) - 1):
            if locations_cp[i] + 1 == locations_cp[i + 1]:
                k += 1
            else:
                break

        output.append(
            list(range(max(locations_cp[0] - tolerance, 0), min(locations_cp[k] + 1 + tolerance, length_ts), 1)))
        del locations_cp[:k + 1]

    return output

if __name__ == '__main__':
    WINDOW_SIZE = 100
    np.random.seed(100)
    timeseries, windows_TD, parameters = generate_jumpingmean(WINDOW_SIZE)
    X_p = np.reshape(windows_TD, (windows_TD.shape[0], windows_TD.shape[1], 1))
    breakpoints = parameters_to_cps(parameters, WINDOW_SIZE)

    list_of_lists = cp_to_timestamps(breakpoints, 0, np.size(breakpoints))
    signal = timeseries
    ncp = len(list_of_lists)
    tprs, fprs = [0], [0]
    tol_distance = 15
    # initialisation for feature extraction module

    ws = args.window_size
    preds = []
    detector = Detector(ws, args.epsilon)
    ctr = 0
    scores, thresholds = np.zeros(len(timeseries)), np.zeros(len(timeseries))
    ini = timeseries[:50]

    mm = 0
    while mm < len(ini) - args.window_size:
        detector.current_distribution = np.append(detector.current_distribution,ini[mm:mm + args.window_size].reshape((1, -1)), axis=0)
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

    ctr = args.window_size
    X = timeseries[50:]
    while ctr < X.shape[0]:
        data = X[ctr:ctr + ws]
        Bi = np.array(data)
        if len(detector.current_distribution) < args.min_batch_size:
            detector.current_distribution = np.append(detector.current_distribution, [Bi], axis=0)  # 8
            if len(detector.current_distribution) >= args.min_batch_size:  # 9
                dis_threshold = detector.compute_threshold()  # 10
                detector.distribution_threshold.append(dis_threshold)  # 10
                detector.distribution_pool.append(detector.current_distribution)  # 11
                if detector.current_distribution_index is None:
                    detector.current_distribution_index = -1
        else:
            x = Bi
            y = detector.current_distribution
            distance = wasserstein_distance(x, y.reshape(-1))
            scores[ctr] = distance
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
                    detector.N.append(ctr + ws)
                    detector.current_distribution_index = -1
                    detector.current_distribution = np.empty((0, args.window_size))
                else:
                    detector.R.append(ctr + ws)
                    detector.current_distribution_index = m - 1
                    detector.current_distribution = detector.distribution_pool[m - 1]

            if len(detector.current_distribution) < args.max_points:
                detector.current_distribution = np.append(detector.current_distribution, [Bi], axis=0)
                dis_threshold = detector.compute_threshold()
                detector.distribution_threshold[detector.current_distribution_index] = dis_threshold

        ctr += args.window_size
        if len(X) - ctr <= ws:
            break

    fig, ax1 = plt.subplots()
    ax1.set_ylabel('pred', color='tab:red')
    ax1.plot(scores, color='tab:orange', label='filtered_zmean')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('raw', color=color)
    ax2.plot(timeseries, color=color, alpha=0.4)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    for i in list_of_lists:
        plt.axvline(x=i[0] + 25, color='g', alpha=0.6)

    for cp in detector.N:
        plt.axvline(x=cp + 50, color='r', alpha=0.6)
    for cp in detector.R:
        plt.axvline(x=cp + 50, color='b', alpha=0.6)

    plt.title('jumping mean')
    plt.show()
