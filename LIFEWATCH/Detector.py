from scipy.stats.distributions import chi2
import numpy as np
import numpy.linalg as la

class Detector():
    '''
    https://github.com/mchenaghlou/OnCAD-PAKDD-2017/blob/1b91d2313cb4eee55ef4423d2731aabb3de2f50b/2.OnCAD/OnCAD.m
    '''

    def __init__(self, forgetting_factor, stabilisation_period, p):
        self.forgetting_factor = forgetting_factor
        self.n_eff = 3 / (1 - self.forgetting_factor)
        self.stabilisation_period = stabilisation_period
        self.normal_boundary = 0.99
        self.guard_zone = 0.999
        self.p = 10
        self.score = []

        self.distribution_pool = []  # store existing distributions
        self.distribution_threshold = [] # store the distribution corresponding thresholds
        self.current_distribution = None

    def initialisation(self, inputs):
        pass

    def predict(self, new_member, ind):
        pass

    def new_cluster_detection(self, c=2):
        '''
        Compare the centroid of state tracker with current cluster to determine if a new cluster emerge
        :return: Boolean
        '''
        p = len(self.anomaly_buffer)
        ST_eigenvalues, _ = la.eig(np.reshape(1/self.StateTracker.inv_cov, (1,1)))
        C_eigenvalues, _ = la.eig(np.reshape(1/self.current_cluster.inv_cov, (1,1)))
        T1 = ST_eigenvalues[np.where(ST_eigenvalues == np.max(ST_eigenvalues))][0]
        T2 = C_eigenvalues[np.where(C_eigenvalues == np.max(C_eigenvalues))][0]
        # equation 4.9
        if la.norm(self.StateTracker.mk - self.current_cluster.centroid) >= c * np.sqrt(max(T1, T2)):
            return True
        return False


