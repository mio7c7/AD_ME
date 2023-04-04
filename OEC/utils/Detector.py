from .Cluster import EllipsoidalCluster
from .StateTracker import StateTracker
from .distance import MahalanobisDistance
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

        self.Clusters = []  # store existing clusters
        self.StateTracker = None
        self.anomaly_buffer = []  # to track change point with an emerging cluster

        self.change_counter = 0  # to track change point between existing clusters
        self.current_cluster = None
        self.potential_cp = None

    def initialisation(self, inputs):
        no = inputs.shape[0]
        centroid = np.mean(inputs, axis=0)
        inv_cov = np.linalg.inv(np.cov(inputs.T))
        cluster = EllipsoidalCluster(centroid=centroid, inv_cov=inv_cov, no_of_member=no,
                                     dim=inputs.shape[1], last_update=no, alpha=no, beta=no)
        self.Clusters.append(cluster)
        self.current_cluster = cluster
        self.StateTracker = StateTracker(mk=centroid, inv_cov=inv_cov,
                                         last_update=no, forgetting_factor=self.forgetting_factor)

    def predict(self, new_member, ind):
        if self.determine_membership(new_member, ind=ind):
            self.anomaly_buffer.append((ind, new_member))
        self.StateTracker.update_cov(new_member)
        self.StateTracker.update_centroid(new_member)

        if self.change_counter >= self.stabilisation_period:
            self.current_cluster = self.potential_cp
            return True

        if ind <= self.n_eff:
            self.StateTracker.last_update += 1
        if len(self.anomaly_buffer) >= self.p:
            if self.new_cluster_detection():
                # an emerging cluster should be formed
                new = np.empty((0,self.anomaly_buffer[0][1].shape[1]))
                for idd, value in self.anomaly_buffer:
                    new = new.vstack(value)
                no = new.shape[0]
                centroid = np.mean(new, axis=0)
                inv_cov = np.linalg.inv(np.cov(new))
                cluster = EllipsoidalCluster(centroid=centroid, inv_cov=inv_cov, no_of_member=no,
                                             dim=new.shape[1], last_update=no, alpha=no, beta=no)
                self.Clusters.append(cluster)
                self.current_cluster = cluster
                return True
            else:
                self.anomaly_buffer_cleanup(ind)
                return False

    def new_cluster_detection(self, c=2):
        '''
        Compare the centroid of state tracker with current cluster to determine if a new cluster emerge
        :return: Boolean
        '''
        p = len(self.anomaly_buffer)
        ST_eigenvalues, _ = la.eig(self.StateTracker.inv_cov)
        C_eigenvalues, _ = la.eig(self.current_cluster.inv_cov)
        T1 = ST_eigenvalues[np.where(ST_eigenvalues == np.max(ST_eigenvalues))][0]
        T2 = C_eigenvalues[np.where(C_eigenvalues == np.max(C_eigenvalues))][0]
        # equation 4.9
        if la.norm(self.StateTracker.mk - self.current_cluster.centroid) >= c * np.sqrt(p * np.max(T1, T2)):
            return True
        return False

    def anomaly_buffer_cleanup(self, ind):
        copy = self.anomaly_buffer
        for i in range(len(self.anomaly_buffer)):
            idd, value = self.anomaly_buffer[i]
            if ind - idd > self.stabilisation_period:
                copy = copy.pop(i)

    def determine_membership(self, new_member, ind):
        mahal_dists = []
        member_clusters = []
        anomalies_label = []
        for cluster in self.Clusters:
            distance = MahalanobisDistance(new_member, cluster.inv_cov, cluster.centroid)
            if distance < chi2.ppf(self.normal_boundary, cluster.dim):  # normal data points in the cluster
                mahal_dists.append(distance)
                member_clusters.append(cluster)
                anomalies_label.append(-1)
            elif distance < chi2.ppf(self.guard_zone, cluster.dim):  # anomalies no update
                mahal_dists.append(distance)
                member_clusters.append(cluster)
                anomalies_label.append(1)
        if len(member_clusters) == 0:
            return False
        else:
            alld = [1 / pow(i, 2) for i in mahal_dists]
            sum_alld = sum(alld)
            weights = [i / sum_alld for i in alld]
            # consider crisp cluster, which means that we will only consider the cluster with the highest weight
            sorted_weight_index = sorted(range(len(weights)), key=lambda k: weights[k], reverse=True)
            cluster = member_clusters[sorted_weight_index[0]]
            weight = weights[sorted_weight_index[0]]
            anomaly = anomalies_label[sorted_weight_index[0]]
            cluster.update_centroid(new_member, weight)
            cluster.update_invcov(new_member, weight)
            cluster.update_setting(weight)
            cluster.last_update = ind
            cluster.no_of_member += 1

            if cluster != self.current_cluster:
                self.change_counter += 1
                if self.potential_cp is None:
                    self.potential_cp = cluster
                elif self.potential_cp != cluster:
                    self.potential_cp = cluster
                    self.change_counter = 0
            else:
                self.change_counter = 0

            if anomaly == -1:
                return True
            else:
                return False
