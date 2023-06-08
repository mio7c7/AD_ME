import numpy as np
from sklearn.metrics import mean_squared_error
import random
from scipy.spatial.distance import euclidean

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
        self.n_components = 1

    def addsample2memory(self, sample, rep, class_no, seen):
        self.memory[class_no] = {'sample': sample, 'rep': rep, 'centroid': np.mean(rep, axis=0)}
        self.current_index = class_no
        self.current_centroid = self.memory[class_no]['centroid']
        threshold = self.compute_threshold(rep, self.current_centroid, self.args.threshold)
        self.memory_info[class_no] = {'size': len(sample), 'threshold': threshold, 'seen': seen}

    def resample(self, new_sample, forgetting_factor):
        org = self.memory[self.current_index]['sample']
        # forgetting factor should be inverse with the seen size of the current distribution
        # want the forgetting factor to be smaller, when there is a lot of the sample in the current distribution
        # and there should be a floor value for the forgetting factor
        if self.memory_info[self.current_index]['seen'] <= self.args.memory_size:
            forgetting_factor = 0.65
            threshold = self.args.threshold
        elif self.memory_info[self.current_index]['seen'] <= 4000:
            forgetting_factor = 0.69444444444 - 0.4*self.memory_info[self.current_index]['seen']/3600
            if self.memory_info[self.current_index]['seen'] <= 1500:
                threshold = self.args.threshold - 0.5
            else:
                threshold = self.args.threshold - 1
        else:
            forgetting_factor = 0.25
            threshold = self.args.threshold - 1.25
        if len(org) < self.args.memory_size:
            full = self.args.memory_size - len(org)
            org = np.vstack((org, new_sample[:full]))
            new_sample = new_sample[full:]
        for ss in new_sample:
            if random.random() < forgetting_factor:
                org = np.delete(org, 0, axis=0)
                org = np.concatenate((org, np.expand_dims(ss, axis=0)), axis=0)
        sam = org
        kernel_matrix = self.feature_extracter.transform(sam)
        kernel_matrix = kernel_matrix - np.mean(kernel_matrix, axis=0)
        eigenvalues, eigenvectors = np.linalg.eig(kernel_matrix.T.dot(kernel_matrix))
        selected_eigenvectors = eigenvectors[:, :self.n_components]
        representation = kernel_matrix.dot(selected_eigenvectors)
        self.memory[self.current_index]['centroid'] = np.mean(representation, axis=0)
        self.current_centroid = self.memory[self.current_index]['centroid']
        self.memory_info[self.current_index]['threshold'] = self.compute_threshold(representation, self.current_centroid, threshold)
        self.memory_info[self.current_index]['seen'] += len(new_sample)

    def updatememory(self, forgetting_factor):
        self.resample(self.newsample, forgetting_factor)
        self.newsample = []

    def compute_threshold(self, rep, centroid, threshold):
        # MSE = [np.sum((rep[i] - centroid) ** 2)/rep.shape[1] for i in range(len(rep))]
        # mse_quantile = np.quantile(MSE, self.args.quantile)
        # threshold = threshold * mse_quantile
        # return threshold
        MSE = [euclidean(rep[i], centroid) for i in range(len(rep))]
        threshold = np.mean(MSE) + threshold * np.std(MSE)
        return threshold

    def updaterecur(self, new, rep):
        org = self.memory[self.current_index]['sample']
        org_rep = self.memory[self.current_index]['rep']
        random_indices = np.random.choice(len(org) - 1, size=(self.args.memory_size - len(new)), replace=True)
        sample = np.concatenate((org[random_indices], new))
        rep = np.concatenate((org_rep[random_indices], rep))
        self.memory[self.current_index]['sample'] = sample
        self.memory[self.current_index]['rep'] = rep
        self.memory[self.current_index]['centroid'] = np.mean(rep, axis=0)
        self.current_centroid = self.memory[self.current_index]['centroid']
        threshold = self.compute_threshold(rep, self.current_centroid, self.args.threshold)
        self.memory_info[self.current_index]['threshold'] = threshold
        self.memory_info[self.current_index]['seen'] = len(new)