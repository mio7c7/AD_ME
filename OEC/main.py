from utils.Cluster import EllipsoidalCluster
from utils.StateTracker import StateTracker
from utils.distance import MahalanobisDistance
from scipy.stats.distributions import chi2
import numpy as np

class Detector():
    '''
    https://github.com/mchenaghlou/OnCAD-PAKDD-2017/blob/1b91d2313cb4eee55ef4423d2731aabb3de2f50b/2.OnCAD/OnCAD.m

    '''
    def __init__(self):
        self.Clusters = []
        self.StateTracker = StateTracker()
        self.anomaly_buffer = []

    def predict(self, new_member):
        pass

    def determine_membership(self, new_member):
        mahal_dists = []
        member_clusters = []
        k = 0
        anomalies_update = []
        for cluster in self.Clusters:
            distance = MahalanobisDistance(new_member, cluster.inv_cov, cluster.centroid)
            if distance < chi2(cluster.alpha, cluster.dim): #normal data points in the cluster
                mahal_dists.append(distance)
                member_clusters.append(cluster)
                k += 1
            elif distance < chi2(cluster.beta, cluster.dim): #anomalies no update
                mahal_dists.append(distance)
                member_clusters.append(cluster)
                k += 1

        alld = [1/pow(i, 2) for i in mahal_dists]
        sum_alld = sum(alld)
        weights = [i/sum_alld for i in alld]

        [val, ind] = max(weights);
        ClusterIndexes(i) = {member_clusters(ind)};


