
class EllipsoidalCluster():
    '''
    https://github.com/mchenaghlou/OnCAD-PAKDD-2017/blob/1b91d2313cb4eee55ef4423d2731aabb3de2f50b/2.OnCAD/OnCAD.m

    '''
    def __init__(self, centroid, inv_cov, no_of_member, dim, alpha=0.99, beta=0.999):
        self.centroid = centroid #mean of the members mk
        self.inv_cov = inv_cov #covariance matrix Sk
        self.no_of_member = no_of_member
        self.alpha = alpha #normal boundary
        self.beta = beta #guard zone
        self.boundary = 0
        self.dim = dim

    def update_centroid(self, new_member):
        pass




