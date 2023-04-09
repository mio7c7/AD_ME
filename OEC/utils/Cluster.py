from scipy.spatial import distance

class EllipsoidalCluster():
    '''
    https://github.com/mchenaghlou/OnCAD-PAKDD-2017/blob/1b91d2313cb4eee55ef4423d2731aabb3de2f50b/2.OnCAD/OnCAD.m
    '''
    def __init__(self, centroid, inv_cov, no_of_member, dim, last_update, alpha, beta):
        self.centroid = centroid #mean of the members mk
        self.inv_cov = inv_cov #covariance matrix Sk
        self.no_of_member = no_of_member
        self.alpha = alpha
        self.beta = beta
        self.dim = dim
        self.last_update = last_update

    def __eq__(self, other):
        if not isinstance(other, EllipsoidalCluster):
            return NotImplemented
        return (self.centroid == other.centroid).all()

    def update_setting(self, weight):
        # candidate_clusters{member_clusters(k)}.alpha = int64(candidate_clusters{member_clusters(k)}.alpha) + int64(weights(k));
        # candidate_clusters {member_clusters(k)}.beta = int64(candidate_clusters{member_clusters(k)}.beta) + int64(weights(k). ^ 2);
        self.beta = self.beta + weight
        self.alpha = self.alpha + weight ** 2

    def update_centroid(self, new_member, weight):
        # curr_cluster.mean = (curr_cluster.mean) + (double(w)/ double(curr_cluster.beta + w)) * (curr_ob - curr_cluster.mean);
        self.centroid = self.centroid + weight / (self.beta + weight) * (new_member - self.centroid)

    def update_invcov(self, new_member, weight):
        # https://github.com/mchenaghlou/OnCAD-PAKDD-2017/blob/1b91d2313cb4eee55ef4423d2731aabb3de2f50b/2.OnCAD/updateWeightedChrMat.m
        betaplus1 = self.beta + weight
        alphaplus1 = self.alpha + weight ** 2

        b = betaplus1 ** 2 - alphaplus1
        khi_enum = self.beta*b
        khi_denom = betaplus1 * (self.beta ** 2 - self.alpha)
        khiK = khi_enum/khi_denom

        delta_enum = betaplus1 * (self.beta ** 2 - self.alpha)
        delta_denom = self.beta * weight * (betaplus1 + weight - 2)
        delta = delta_enum / delta_denom

        enumerator = self.inv_cov * (new_member - self.centroid) * (new_member - self.centroid) * self.inv_cov
        # denominator = delta + (new_member - self.centroid) * self.inv_cov * (new_member - self.centroid)
        denominator = delta + distance.mahalanobis(new_member, self.centroid, self.inv_cov) ** 2
        new_sn_1 = khiK * (self.inv_cov - enumerator / denominator)
        self.inv_cov = new_sn_1



