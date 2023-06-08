import glob
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from Detector_mmd_real import Detector
from sklearn.metrics.pairwise import pairwise_kernels
import math
sys.path.append('./')
from evaluation import Evaluation_metrics
from ssa.btgym_ssa import SSA
import os

parser = argparse.ArgumentParser(description='Mstatistics evaluation on bottom 0.2 data')
parser.add_argument('--data', type=str, default='../data3/*.npz', help='directory of data')
parser.add_argument('--ssa_window', type=int, default=5, help='n_components for ssa preprocessing')
parser.add_argument('--bs', type=int, default=50, help='buffer size for ssa')
parser.add_argument('--ws', type=int, default=25, help='window size')
parser.add_argument('--step', type=int, default=5, help='step')
parser.add_argument('--min_requirement', type=int, default=25, help='window size')
parser.add_argument('--memory_size', type=int, default=4, help='memory size per distribution ')
parser.add_argument('--cp_range', type=int, default=5, help='range to determine cp')
parser.add_argument('--forgetting_factor', type=float, default=0.55, help='forgetting_factor')
parser.add_argument('--out_threshold', type=float, default=2, help='threshold for outlier filtering')
parser.add_argument('--threshold', type=float, default=2, help='threshold')
parser.add_argument('--quantile', type=float, default=0.975, help='quantile')
parser.add_argument('--fixed_outlier', type=float, default=1, help='preprocess outlier filter')
parser.add_argument('--outfile', type=str, default='mmd02', help='name of file to save results')

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

def maximum_mean_discrepancy(X, Y, kernel='rbf', gamma=0.01):
    K_XX = pairwise_kernels(X, metric=kernel, gamma=gamma)
    K_YY = pairwise_kernels(Y, metric=kernel, gamma=gamma)
    K_XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)
    mmd = np.mean(K_XX) - 2 * np.mean(K_XY) + np.mean(K_YY)
    return mmd

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

    reconstructeds = sliding_window(timeseries[:50], args.ws, args.step)
    class_no = 1
    memory = reconstructeds
    if len(reconstructeds) > args.memory_size:
        random_indices = np.random.choice(len(reconstructeds), size=args.memory_size, replace=False)
        memory = memory[random_indices]
    detector = Detector(args.ws, args)
    detector.addsample2memory(memory, class_no, len(memory))

    ctr = args.ws
    scores, thresholds = np.zeros(len(timeseries)), np.zeros(len(timeseries))
    X = timeseries[50:]
    collection_period = 1000000000
    sample = np.empty((0, args.ws))
    while ctr < X.shape[0]:
        new = X[ctr-args.ws:ctr]
        if collection_period > args.min_requirement:
            score = maximum_mean_discrepancy(new.reshape(-1, 1), detector.current_centroid.reshape(-1, 1))
            scores[ctr] = score
            thresholds[ctr] = detector.memory_info[detector.current_index]['threshold']
            if score > detector.memory_info[detector.current_index]['threshold']:
                min_dist = 100000
                n = 1
                dk = None
                while n <= len(detector.memory):
                    distribution = detector.memory[n]['centroid']
                    cur_threshold = detector.memory_info[n]['threshold']
                    distance = maximum_mean_discrepancy(new.reshape(-1, 1), distribution.reshape(-1, 1))
                    if distance < cur_threshold:
                        if distance < min_dist:
                            min_dist = distance
                            dk = n
                    n += 1
                if dk == len(detector.memory):
                    dk = None
                if dk is None:
                    detector.N.append(ctr)
                    detector.current_index = -1
                else:
                    detector.R.append(ctr)
                    detector.current_index = dk
                collection_period = 0
                detected = True
                detector.newsample = []
            else:
                detector.newsample.append(new)

            if collection_period > args.min_requirement:
                detector.updatememory()
        elif collection_period <= args.min_requirement:
            if collection_period + args.step <= args.min_requirement:
                sample = np.concatenate((sample, new.reshape(1,-1)))
                collection_period += args.step
            else:  # new
                sample = np.concatenate((sample, new.reshape(1,-1)))
                if detector.current_index == -1:  # new cluster
                    class_no += 1
                    detector.addsample2memory(sample, class_no, len(sample))
                else:  # recurring
                    detector.updaterecur(sample)
                collection_period = 1000000000
                sample = np.empty((0, args.ws))

        ctr += args.step

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
