from .Cluster import EllipsoidalCluster
from .StateTracker import StateTracker
# from .distance import MahalanobisDistance, mahalanobis
from scipy.stats.distributions import chi2
import numpy as np
import numpy.linalg as la
import logging
import pandas as pd
from scipy.spatial.distance import mahalanobis
logger = logging.getLogger()

class Detector():
    '''
    https://github.com/mchenaghlou/OnCAD-PAKDD-2017/blob/1b91d2313cb4eee55ef4423d2731aabb3de2f50b/2.OnCAD/OnCAD.m
    '''

    def __init__(self, forgetting_factor, stabilisation_period, p, c, memory_size, train_no):
        self.forgetting_factor = forgetting_factor
        self.n_eff = 3 / (1 - self.forgetting_factor)
        self.normal_boundary = 0.99
        self.guard_zone = 0.999
        self.p = p
        self.c = c # c-separation
        self.memory_size = memory_size
        self.already_mem_update = False
        self.seen = 0
        self.stabilisation_period = stabilisation_period
        self.DistributionPool = []  # store existing distributions
        self.StateTracker = None
        self.AnomalyBuffer = []  # to track change point with an emerging cluster
        self.AnomalyCounter = 0  # to track change point between existing clusters
        self.CurrentDistribution = None
        self.CPCandidates = None
        self.MemoryList = []
        self.FeatureExtracter = None
        self.StreamedList = []
        self.DistributionNo = len(self.DistributionPool)
        self.model = None
        self.EmbeddedArrays = None
        self.potential_cp = None
        self.train_no = train_no

        self.score = [] # for result evaluation

    def reservoir_sampling(self, samples):
        for sample in samples:
            if len(self.MemoryList) < self.memory_size:
                self.MemoryList += [sample]
            else:
                j = np.random.randint(0, self.seen)
                if j < self.memory_size:
                    self.MemoryList[j] = sample
            self.seen += 1

    def update_memory(self):
        if len(self.MemoryList) != 0:
            new = np.empty((0, self.StreamedList[0][2].shape[1], self.StreamedList[0][2].shape[2]))
            for idd, _, value in self.StreamedList:
                new = np.vstack((new, value))
            self.StreamedList = new
            candidates = np.concatenate((self.StreamedList, self.MemoryList), axis=0)
        else:
            candidates = self.StreamedList + self.MemoryList
        if len(candidates) <= self.memory_size:
            self.MemoryList = candidates
            self.seen = len(candidates)
            logger.warning("Candidates < Memory size")
        else:
            self.reservoir_sampling(self.StreamedList)

        assert len(self.MemoryList) <= self.memory_size
        logger.info("Memory statistic")

    def initialisation(self, inputs):
        _, _, embedded_vector = self.FeatureExtracter.encoder(inputs)
        no = embedded_vector.shape[0]
        embedded_vector = np.array(embedded_vector)
        self.EmbeddedArrays = embedded_vector
        embedded_vector = (embedded_vector - np.mean(embedded_vector, axis=0))/ np.linalg.norm(embedded_vector, axis=0)
        centroid = np.mean(embedded_vector, axis=0)
        inv_cov = np.array(1/np.cov(embedded_vector.T))
        distribution = EllipsoidalCluster(centroid=centroid, inv_cov=inv_cov, no_of_member=no,
                                     dim=embedded_vector.shape[1], last_update=no, alpha=no, beta=no)
        self.DistributionPool.append(distribution)
        self.CurrentDistribution = distribution
        self.StateTracker = StateTracker(mk=centroid, inv_cov=inv_cov,
                                         last_update=0, forgetting_factor=self.forgetting_factor)
        self.StreamedList = [np.array(x) for x in inputs.tolist()]
        self.update_memory()

    def predict(self, new_member, ind):
        _, _, embedded_vector = self.FeatureExtracter.encoder(new_member)
        embedded_vector = np.array(embedded_vector)
        self.EmbeddedArrays = np.concatenate((self.EmbeddedArrays, embedded_vector))
        embedded_vector = (embedded_vector - np.mean(self.EmbeddedArrays, axis=0))/np.linalg.norm(self.EmbeddedArrays, axis=0)

        if self.determine_membership(embedded_vector, ind=ind):
            self.AnomalyBuffer.append((ind, embedded_vector, new_member))
        self.StateTracker.update_cov(embedded_vector)
        self.StateTracker.update_centroid(embedded_vector)

        if self.AnomalyCounter >= self.stabilisation_period:
            self.CurrentDistribution = self.potential_cp
            return True

        if ind <= self.n_eff:
            self.StateTracker.last_update += 1

        if len(self.AnomalyBuffer) >= self.p:
            if self.new_cluster_detection():
                # an emerging cluster should be formed
                new = np.empty((0, self.AnomalyBuffer[0][1].shape[1]))
                for idd, value, _ in self.AnomalyBuffer:
                    new = np.vstack((new, value))
                no = new.shape[0]
                centroid = np.mean(new, axis=0)
                inv_cov = np.array(1/np.cov(new.T))
                distribution = EllipsoidalCluster(centroid=centroid, inv_cov=inv_cov, no_of_member=no,
                                             dim=new.shape[1], last_update=no, alpha=no, beta=no)
                self.DistributionPool.append(distribution)
                self.CurrentDistribution = distribution
                self.StreamedList = self.AnomalyBuffer
                self.AnomalyBuffer = []
                self.update_memory()
                return True
            else:
                self.anomaly_buffer_cleanup(ind)
        else:
            self.anomaly_buffer_cleanup(ind)
            return False

    def new_cluster_detection(self):
        '''
        Compare the centroid of state tracker with current cluster to determine if a new cluster emerge
        :return: Boolean
        '''
        # p = len(self.AnomalyBuffer)
        try:
            ST_eigenvalues, _ = la.eig(1/self.StateTracker.inv_cov)
        except np.linalg.LinAlgError:
            ST_eigenvalues, _ = la.eig(1 / (self.StateTracker.inv_cov + 1e+6 * (1 - np.eye(self.StateTracker.inv_cov.shape[0]))))
        C_eigenvalues, _ = la.eig(1/self.CurrentDistribution.inv_cov)
        T1 = ST_eigenvalues[np.where(ST_eigenvalues == np.max(ST_eigenvalues))][0]
        T2 = C_eigenvalues[np.where(C_eigenvalues == np.max(C_eigenvalues))][0]
        # equation 4.9
        if la.norm(self.StateTracker.mk - self.CurrentDistribution.centroid) >= self.c * np.sqrt(max(T1, T2)):
            return True
        return False

    def anomaly_buffer_cleanup(self, ind):
        buffer_copy = [(idd, value, _) for idd, value, _ in self.AnomalyBuffer if ind - idd < self.stabilisation_period]
        self.AnomalyBuffer = buffer_copy

    def determine_membership(self, new_member, ind):
        '''

        :param new_member: upcoming point
        :param ind: the index of the upcoming point
        :return: Bool (True if it is an anomaly)
        '''
        mahal_dists = []
        member_clusters = []
        anomalies_label = []
        min_dist = 100000000000
        for cluster in self.DistributionPool:
            distance = mahalanobis(new_member, cluster.centroid, cluster.inv_cov) ** 2
            if distance < min_dist:
                min_dist = distance
            if distance < chi2.ppf(self.normal_boundary, cluster.dim):  # normal data points in the cluster
                mahal_dists.append(distance)
                member_clusters.append(cluster)
                anomalies_label.append(-1)
            elif distance < chi2.ppf(self.guard_zone, cluster.dim):  # anomalies no update
                mahal_dists.append(distance)
                member_clusters.append(cluster)
                anomalies_label.append(1)

        self.score.append(min_dist)

        if len(member_clusters) == 0:
            return True
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
            cluster.last_update = ind + self.train_no
            cluster.no_of_member += 1

            if cluster != self.CurrentDistribution:
                self.AnomalyCounter += 1
                if self.potential_cp is None:
                    self.potential_cp = cluster
                elif self.potential_cp != cluster:
                    self.potential_cp = cluster
                    self.AnomalyCounter = 0
            else:
                self.AnomalyCounter = 0

            if anomaly == -1:
                return False
            else:
                return True
