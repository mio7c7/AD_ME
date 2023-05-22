import numpy as np
from sklearn.metrics import mean_squared_error
import random

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

    def addsample2memory(self, sample, rep, class_no, seen):
        self.memory[class_no] = {'sample': sample, 'rep': rep, 'centroid': np.array([np.mean(rep)])}
        self.current_index = class_no
        self.current_centroid = self.memory[class_no]['centroid']
        threshold = self.compute_threshold(rep, self.current_centroid)
        self.memory_info[class_no] = {'size': len(sample), 'threshold': threshold, 'seen': seen}

    def resample(self, new_sample):
        org = self.memory[self.current_index]['sample']
        if self.memory_info[self.current_index]['seen'] <= self.args.memory_size:
            forgetting_factor = 0.75
        elif self.memory_info[self.current_index]['seen'] <= 4000:
            forgetting_factor = 0.783333332 - 0.00008333333*self.memory_info[self.current_index]['seen']
        else:
            forgetting_factor = 0.45
        if len(org) < self.args.memory_size:
            full = self.args.memory_size - len(org)
            org = np.vstack((org, new_sample[:full]))
            new_sample = new_sample[full:]
        for ss in new_sample:
            if random.random() < forgetting_factor:
                org = np.delete(org, 0, axis=0)
                org = np.concatenate((org, ss.reshape(1, -1)), axis=0)
        self.memory[self.current_index]['sample'] = org
        self.memory[self.current_index]['rep'] = np.mean(org, axis=1).reshape(-1, 1)
        self.memory[self.current_index]['centroid'] = np.array([np.mean(self.memory[self.current_index]['rep'])])
        self.current_centroid = self.memory[self.current_index]['centroid']
        self.memory_info[self.current_index]['threshold'] = self.compute_threshold(self.memory[self.current_index]['rep'], self.current_centroid)
        self.memory_info[self.current_index]['seen'] += len(new_sample)

    def updatememory(self):
        self.resample(self.newsample)
        self.newsample = []

    def compute_threshold(self, rep, centroid):
        MSE = [mean_squared_error(rep[i], centroid) for i in range(len(rep))]
        mse_quantile = np.quantile(MSE, self.args.quantile)
        threshold = self.args.threshold * mse_quantile
        return threshold

    def updaterecur(self, new, rep):
        if len(new) >= self.args.memory_size:
            random_indices = np.random.choice(len(new), size=self.args.memory_size, replace=False)
            self.memory[self.current_index] = {'sample': new[random_indices], 'rep': rep[random_indices], 'centroid': np.array([np.mean(rep[random_indices])])}
            self.current_centroid = self.memory[self.current_index]['centroid']
            threshold = self.compute_threshold(rep[random_indices], self.current_centroid)
            self.memory_info[self.current_index]['threshold'] = threshold
        else:
            org = self.memory[self.current_index]['sample']
            org_rep = self.memory[self.current_index]['rep']
            random_indices = np.random.choice(len(org), size=(self.args.memory_size - len(new)), replace=True)
            self.memory[self.current_index]['sample'] = np.concatenate((org[random_indices], new))
            self.memory[self.current_index]['rep'] = np.concatenate((org_rep[random_indices], rep))
            self.memory[self.current_index]['centroid'] = np.array([np.mean(self.memory[self.current_index]['rep'])])
            self.current_centroid = self.memory[self.current_index]['centroid']
            self.memory_info[self.current_index]['threshold'] = self.compute_threshold(self.memory[self.current_index]['rep'], self.current_centroid)