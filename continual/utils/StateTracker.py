import numpy as np
from scipy.spatial import distance

class StateTracker():
    '''
    https://github.com/mchenaghlou/OnCAD-PAKDD-2017/blob/1b91d2313cb4eee55ef4423d2731aabb3de2f50b/2.OnCAD/OnCAD.m
    '''

    def __init__(self, mk, inv_cov, last_update, forgetting_factor):
        self.mk = mk
        self.inv_cov = inv_cov
        # self.no_of_member = no_of_member
        # self.list_of_member = []
        self.last_update = last_update  # k
        self.forgetting_factor = forgetting_factor  # suggested range between 0.9 and 1

    def update_centroid(self, new_point):
        # equation 4.7
        self.mk = self.forgetting_factor * self.mk + (1 - self.forgetting_factor) * new_point

    def update_cov(self, new_point):
        # equation 4.8
        # (new_member - self.centroid)) * ((new_member - self.centroid) * self.inv_cov)
        enumerator = (new_point - self.mk) * (new_point - self.mk).T * self.inv_cov
        # denominator = (self.last_update - 1) / self.forgetting_factor + (
        #         (new_point - self.mk) * self.inv_cov * (new_point - self.mk))
        denominator = (self.last_update - 1) / self.forgetting_factor + distance.mahalanobis(new_point, self.mk, self.inv_cov) ** 2
        # identity_matrix = np.identity(self.inv_cov.shape[0])
        identity_matrix = np.identity(1)
        self.inv_cov = ((self.last_update * self.inv_cov) / (self.forgetting_factor * (self.last_update - 1))) * (
                identity_matrix - enumerator / denominator)
