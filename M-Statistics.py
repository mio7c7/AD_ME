import numpy as np

class Kernel():
    def __init__(self):
        self.alphabet = [-1, 0, 1]
        self.b = None # threshold
        self.Bo = 25 # block size
        self.m = None # alphabet size
        self.N = 10 # no. of blocks

    def set_alaphbet_size(self):
        self.m = len(self.alphabet)

    def create_reference_blocks(self, B, N):
        seed = np.random.randint(self.m, size=(B, N))
        blocks = self.alphabet(seed)
        return blocks

    def find_emp_pmf(self, X):
        emp_pmf = np.zeros((self.m, 1))
        B = len(X)
        for i in range(self.m):
            emp_pmf[i] = sum
        return emp_pmf

    def find_mmd_u_sq(self, X, Y):
        B = len(X)
        pmf_X = self.find_mmd_u_sq()

    def change_detection(self):
        pass
