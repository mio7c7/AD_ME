from Cluster import EllipsoidalCluster
from StateTracker import StateTracker
from distance import MahalanobisDistance
from scipy.stats.distributions import chi2
import numpy as np

class Detector():
    '''
    https://github.com/mchenaghlou/OnCAD-PAKDD-2017/blob/1b91d2313cb4eee55ef4423d2731aabb3de2f50b/2.OnCAD/OnCAD.m
    '''
    def __init__(self):
        self.Clusters = [] # store existing clusters
        self.StateTracker = StateTracker()
        self.anomaly_buffer = []
        self.normal_boundary = 0.99
        self.guard_zone =0.999

    def predict(self, new_member):
        pass

    def determine_membership(self, new_member, ind):
        mahal_dists = []
        member_clusters = []
        anomalies_update = []
        for cluster in self.Clusters:
            distance = MahalanobisDistance(new_member, cluster.inv_cov, cluster.centroid)
            if distance < chi2(cluster.alpha, cluster.dim): #normal data points in the cluster
                mahal_dists.append(distance)
                member_clusters.append(cluster)
            elif distance < chi2(cluster.beta, cluster.dim): #anomalies no update
                mahal_dists.append(distance)
                member_clusters.append(cluster)

        alld = [1/pow(i, 2) for i in mahal_dists]
        sum_alld = sum(alld)
        weights = [i/sum_alld for i in alld]

        # [val, ind] = max(weights);
        # ClusterIndexes(i) = {member_clusters(ind)};

        k = 1
        while k <= len(member_clusters):
            cluster = member_clusters[k]
            weight = weights[k]
            cluster.update_centroid(new_member, weight)
            cluster.update_invcov(new_member, weight)

            # candidate_clusters{member_clusters(k)}.alpha = int64(candidate_clusters{member_clusters(k)}.alpha) + int64(weights(k));
            # candidate_clusters {member_clusters(k)}.beta = int64(candidate_clusters{member_clusters(k)}.beta) + int64(weights(k). ^ 2);

            cluster.last_update = ind
            cluster.no_of_member += 1
            k += 1

