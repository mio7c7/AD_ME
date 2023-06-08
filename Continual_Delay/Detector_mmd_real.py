import numpy as np
from sklearn.metrics import mean_squared_error
import random
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import pairwise_kernels

def maximum_mean_discrepancy(X, Y, kernel='rbf', gamma=0.01):
    K_XX = pairwise_kernels(X, metric=kernel, gamma=gamma)
    K_YY = pairwise_kernels(Y, metric=kernel, gamma=gamma)
    K_XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)
    mmd = np.mean(K_XX) - 2 * np.mean(K_XY) + np.mean(K_YY)
    return mmd

class Detector():
    '''
    https://github.com/mchenaghlou/OnCAD-PAKDD-2017/blob/1b91d2313cb4eee55ef4423d2731aabb3de2f50b/2.OnCAD/OnCAD.m
    '''

    def __init__(self, window_size, args):
        self.window_size = window_size
        self.memory = {}  # store existing distributions
        self.memory_info = {} # store the distribution corresponding thresholds
        self.current_index = None
        self.current_centroid = None
        self.N = []
        self.R = []
        self.newsample = []
        self.args = args
        self.n_components = 1

    def addsample2memory(self, sample, class_no, seen):
        self.memory[class_no] = {'sample': sample, 'centroid': np.mean(sample, axis=0)}
        self.current_index = class_no
        self.current_centroid = self.memory[class_no]['centroid']
        threshold = self.compute_threshold(sample, self.current_centroid, self.args.threshold + 1)
        self.memory_info[class_no] = {'size': len(sample), 'threshold': threshold, 'seen': seen}

    def resample(self, new_sample):
        org = self.memory[self.current_index]['sample']
        old = org
        if self.memory_info[self.current_index]['seen'] <= self.args.memory_size:
            forgetting_factor = 0.65
            threshold = self.args.threshold + 1
        elif self.memory_info[self.current_index]['seen'] <= 4000:
            forgetting_factor = 0.69444444444 - 0.4*self.memory_info[self.current_index]['seen']/3600
            if self.memory_info[self.current_index]['seen'] <= 1500:
                threshold = self.args.threshold + 0.75
            else:
                threshold = self.args.threshold + 0.25
        else:
            forgetting_factor = 0.25
            threshold = self.args.threshold
        if len(org) < self.args.memory_size:
            full = self.args.memory_size - len(org)
            org = np.vstack((org, new_sample[:full]))
            new_sample = new_sample[full:]
        for ss in new_sample:
            if random.random() < forgetting_factor:
                org = np.delete(org, 0, axis=0)
                org = np.concatenate((org, np.expand_dims(ss, axis=0)), axis=0)
        sam = org
        self.memory_info[self.current_index]['threshold'] = self.compute_threshold(old, self.current_centroid,threshold)
        self.memory[self.current_index]['centroid'] = np.mean(sam, axis=0)
        self.current_centroid = self.memory[self.current_index]['centroid']
        self.memory_info[self.current_index]['seen'] += len(new_sample)

    def updatememory(self):
        self.resample(self.newsample)
        self.newsample = []

    def compute_threshold(self, rep, centroid, threshold):
        MMD = [maximum_mean_discrepancy(rep[i].reshape(-1, 1), centroid.reshape(-1, 1)) for i in range(len(rep))]
        mse_quantile = np.quantile(MMD, self.args.quantile)
        threshold = threshold * mse_quantile
        return threshold
        # MSE = [maximum_mean_discrepancy(rep[i].reshape(-1, 1), centroid.reshape(-1, 1)) for i in range(len(rep))]
        # threshold = np.mean(MSE) + threshold * np.std(MSE)
        return threshold

    def updaterecur(self, new):
        org = self.memory[self.current_index]['sample']
        random_indices = np.random.choice(len(org) - 1, size=(self.args.memory_size - len(new)), replace=True)
        sample = np.concatenate((org[random_indices], new))
        self.memory[self.current_index]['sample'] = sample
        self.memory[self.current_index]['centroid'] = np.mean(sample, axis=0)
        self.current_centroid = self.memory[self.current_index]['centroid']
        threshold = self.compute_threshold(sample, self.current_centroid, self.args.threshold + 1)
        self.memory_info[self.current_index]['threshold'] = threshold
        self.memory_info[self.current_index]['seen'] = len(new)