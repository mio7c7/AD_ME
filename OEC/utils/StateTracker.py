
class StateTracker():
    '''
    https://github.com/mchenaghlou/OnCAD-PAKDD-2017/blob/1b91d2313cb4eee55ef4423d2731aabb3de2f50b/2.OnCAD/OnCAD.m

    '''
    def __init__(self, mk, inv_cov, no_of_member, forgetfactor):
        self.mk = mk
        self.inv_cov = inv_cov
        self.no_of_member = no_of_member
        self.list_of_member = []
        self.forgetfactor = forgetfactor

    def update_centroid(self, new_member):
        # equation 4.7
        self.mk = self.forgetfactor*self.mk + (1-self.forgetfactor)*new_member

    def update_cov(self, new_member):
        # equation 4.8
        pass




