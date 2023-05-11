from scipy.stats.distributions import chi2
import numpy as np
import numpy.linalg as la
from scipy.stats import wasserstein_distance
import pandas as pd
from sklearn.metrics import mean_squared_error

class Detector():
    '''
    https://github.com/mchenaghlou/OnCAD-PAKDD-2017/blob/1b91d2313cb4eee55ef4423d2731aabb3de2f50b/2.OnCAD/OnCAD.m
    '''

    def __init__(self, window_size, feature_extracter, args):
        self.window_size = window_size
        self.memory = {}  # store existing distributions
        self.memory_info = {} # store the distribution corresponding thresholds
        self.current_index = None
        self.current_centroid = None
        self.N = []
        self.R = []
        self.newsample = []
        self.feature_extracter = feature_extracter
        self.args = args

    def addsample2memory(self, sample, rep, seen, class_no):
        self.memory[class_no] = {'sample': sample, 'rep': rep, 'centroid': np.array([np.mean(rep)])}
        self.current_index = class_no
        self.current_centroid = self.memory[class_no]['centroid']
        threshold = self.compute_threshold(rep, self.current_centroid)
        self.memory_info[class_no] = {'size': len(sample), 'seen': seen, 'threshold': threshold}

    def resample(self, new_sample):
        new_sample = np.expand_dims(new_sample, axis=-1)
        org = self.memory[self.current_index]['sample']
        seen = self.memory_info[self.current_index]['seen']
        if len(org) < self.args.memory_size:
            full = self.args.memory_size - len(org)
            org = np.vstack((org, new_sample[:full]))
            seen += full
            new_sample = new_sample[full:]

        for ss in new_sample:
            j = np.random.randint(0, seen)
            if j < self.args.memory_size:
                org[j] = ss
            seen += 1
        self.memory[self.current_index]['sample'] = org
        z_mean, z_log_sigma, z, pred = self.feature_extracter.predict(org)
        self.memory[self.current_index]['rep'] = z_mean
        self.memory[self.current_index]['centroid'] = np.array([np.mean(z_mean)])
        self.current_centroid = self.memory[self.current_index]['centroid']
        self.memory_info[self.current_index]['seen'] = seen
        self.memory_info[self.current_index]['threshold'] = self.compute_threshold(z_mean, self.current_centroid)

    def updatememory(self):
        self.resample(self.newsample[:-self.args.ws])
        self.newsample = []

    def compute_threshold(self, rep, centroid):
        MSE = [mean_squared_error(rep[i], centroid) for i in range(len(rep))]
        mse_quantile = np.quantile(MSE, self.args.quantile)
        threshold = self.args.threshold * mse_quantile
        # threshold = np.mean(MSE) + self.args.threshold * np.std(MSE)
        return threshold

