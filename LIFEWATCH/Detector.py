from scipy.stats.distributions import chi2
import numpy as np
import numpy.linalg as la
from scipy.stats import wasserstein_distance

class Detector():
    '''
    https://github.com/mchenaghlou/OnCAD-PAKDD-2017/blob/1b91d2313cb4eee55ef4423d2731aabb3de2f50b/2.OnCAD/OnCAD.m
    '''

    def __init__(self, window_size, epsilon):
        self.window_size = window_size
        self.distribution_pool = []  # store existing distributions
        self.distribution_threshold = [] # store the distribution corresponding thresholds
        self.current_distribution_index = None
        self.current_distribution = np.empty((0, self.window_size))
        self.N = []
        self.R = []
        self.epsilon = epsilon

    def compute_threshold(self):
        threshold = -1
        i = 0
        while i < len(self.current_distribution)-1:
            x = self.current_distribution[i]
            y = self.current_distribution[i:]
            distance = self.epsilon*wasserstein_distance(x, y.reshape(-1))
            if distance > threshold:
                threshold = distance
            i += 1
        return threshold

