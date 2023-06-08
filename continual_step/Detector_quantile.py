import numpy as np
from sklearn.metrics import mean_squared_error
import random

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

    def addsample2memory(self, sample, rep, class_no, seen):
        self.memory[class_no] = {'sample': sample, 'rep': rep, 'centroid': np.array([np.mean(rep)])}
        self.current_index = class_no
        self.current_centroid = self.memory[class_no]['centroid']
        threshold = self.compute_threshold(rep, self.current_centroid, self.args.threshold + 1)
        self.memory_info[class_no] = {'size': len(sample), 'threshold': threshold, 'seen': seen}

    def resample(self, new_sample):
        new_sample = np.expand_dims(new_sample, axis=-1)
        org = self.memory[self.current_index]['sample']
        if self.memory_info[self.current_index]['seen'] <= self.args.memory_size:
            forgetting_factor = 0.85
            threshold = self.args.threshold + 1
        elif self.memory_info[self.current_index]['seen'] <= 3000:
            forgetting_factor = 0.87 - 0.4*self.memory_info[self.current_index]['seen']/2900
            if self.memory_info[self.current_index]['seen'] <= 1500:
                threshold = self.args.threshold + 0.75
            else:
                threshold = self.args.threshold + 0.25
        else:
            forgetting_factor = 0.45
            threshold = self.args.threshold
        if len(org) < self.args.memory_size:
            full = self.args.memory_size - len(org)
            org = np.vstack((org, new_sample[:full]))
            new_sample = new_sample[full:]
        for ss in new_sample:
            if random.random() < forgetting_factor:
                org = np.delete(org, 0, axis=0)
                org = np.concatenate((org, np.expand_dims(ss, axis=0)), axis=0)
        self.memory_info[self.current_index]['threshold'] = self.compute_threshold(self.memory[self.current_index]['rep'],
                                                                                   self.current_centroid, threshold)
        self.memory[self.current_index]['sample'] = org
        z_mean, z_log_sigma, z, pred = self.feature_extracter.predict(org)
        self.memory[self.current_index]['rep'] = z_mean
        self.memory[self.current_index]['centroid'] = np.array([np.mean(z_mean)])
        self.current_centroid = self.memory[self.current_index]['centroid']
        self.memory_info[self.current_index]['seen'] += len(new_sample)

    def updatememory(self):
        self.resample(self.newsample)
        self.newsample = []

    def compute_threshold(self, rep, centroid, threshold):
        MSE = [np.sum((rep[i] - centroid) ** 2)/rep.shape[1] for i in range(len(rep))]
        mse_quantile = np.quantile(MSE, self.args.quantile)
        threshold = threshold * mse_quantile
        return threshold

    def updaterecur(self, new):
        org = self.memory[self.current_index]['sample']
        for ss in range(len(new)):
            if len(org) < self.args.memory_size:
                org = np.concatenate((org, np.expand_dims(new[ss], axis=0)), axis=0)
            else:
                org = np.delete(org, 0, axis=0)
                org = np.concatenate((org, np.expand_dims(new[ss], axis=0)), axis=0)
        z_mean, z_log_sigma, z, pred = self.feature_extracter.predict(org)
        self.memory[self.current_index]['sample'] = org
        self.memory[self.current_index]['rep'] = z_mean
        self.memory[self.current_index]['centroid'] = np.array([np.mean(self.memory[self.current_index]['rep'])])
        self.current_centroid = self.memory[self.current_index]['centroid']
        threshold = self.compute_threshold(z_mean, self.current_centroid, self.args.threshold + 1)
        self.memory_info[self.current_index]['threshold'] = threshold
        self.memory_info[self.current_index]['seen'] = len(new)