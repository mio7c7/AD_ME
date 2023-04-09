import numpy as np
import math

# import torch
# U-Statistics MMD https://github.com/ruqizhang/discrete-langevin/blob/97bc36960676a5314016540b7027ae3f51ff4a80/BinaryBNN/GWG_release/mmd.py
def avg_hamming(x, y):
    diffs = (x[None, :] != y[:, None, :]).astype(float).mean(-1)
    return diffs

def exp_avg_hamming(x, y):
    diffs = avg_hamming(x, y)
    return np.exp(-diffs)

def compute_gram(x, y):
    """
    Compute Gram matrices:
        K: array((m+n, m+n))
        kxx: array((m, m))
        kyy: array((n, n))
        kxy: array((m, n))
    """
    (m, d1) = x.shape
    (n, d2) = y.shape
    assert d1 == d2

    xy = np.vstack([x, y])  # np.vstack([x, y])
    K = exp_avg_hamming(xy, xy)  # kxyxy
    # assert_shape(K, (m + n, m + n))
    # assert is_psd(K)

    kxx = K[:m, :m]
    # assert_shape(kxx, (m, m))
    # assert is_psd(kxx)
    # assert is_symmetric(kxx)

    kyy = K[m:, m:]
    # assert_shape(kyy, (n, n))
    # assert is_psd(kyy)
    # assert is_symmetric(kyy)

    kxy = K[:m, m:]
    # assert_shape(kxy, (m, n))

    return K, kxx, kyy, kxy

def compute_statistic(kxx, kyy, kxy):
    """
    Compute MMD test statistic.
    """
    m = kxx.shape[0]
    n = kyy.shape[0]

    # Compute U-statistics estimate
    term_xx = (kxx.sum() - np.diag(kxx).sum()) / (m * (m - 1))
    term_yy = (kyy.sum() - np.diag(kyy).sum()) / (n * (n - 1))
    term_xy = kxy.sum() / (m * n)
    res = term_xx + term_yy - 2 * term_xy
    return res

def compute_mmd(x, y):
    K, kxx, kyy, kxy = compute_gram(x, y)
    stat = compute_statistic(kxx, kyy, kxy)
    return stat

class Kernel():
    def __init__(self, b=5, bo=25, N=10):
        self.alphabet = [-1, 0, 1]
        self.threshold = b  # threshold
        self.Bo = bo  # block size
        self.m = None  # alphabet size
        self.N = N  # no. of blocks

    def set_alaphbet_size(self):
        self.m = len(self.alphabet)

    def create_reference_blocks(self, reference_pool, N, B):  # sample N reference blocks with size Bo from history data
        '''
        for simplicity, first divide X into blcoks with B size
        then sign probability to each block, more recent one with higher probability
        :param B:
        :param N:
        :return:
        '''
        residual = len(reference_pool) % B
        no_blocks = len(reference_pool) // B
        copy = reference_pool[residual:]
        copy = copy.reshape((no_blocks, B, copy.shape[1]))
        index = np.arange(no_blocks) + 1
        prob = index / np.sum(index)
        selected = np.random.choice(index, N, p=prob, replace=False)
        selected = selected - 1
        blocks = copy[selected]
        return blocks

    def find_mmd_u_sq(self, X, Y):
        return compute_mmd(X, Y)

    def change_detection(self, input):  # Y the whole input
        length = len(input)
        t = (self.N + 1) * self.Bo
        reference_pool = input[:t - self.Bo]
        matrix_X = self.create_reference_blocks(reference_pool, self.N, self.Bo)
        scores = [0] * ((self.N + 1) * self.Bo)
        while True:
            mmd_u_sq = np.zeros(self.N)
            tmp = input[t - self.Bo: t]
            for i in range(self.N):
                mmd_u_sq[i] = self.find_mmd_u_sq(matrix_X[i, :], tmp)

            stat = math.sqrt(self.Bo * (self.Bo - 1)) * np.mean(mmd_u_sq)
            scores.append(stat)
            if stat >= self.threshold:
                break
            elif t == length:
                return -1, scores[:-1]
            else:
                t = t + 1
                reference_pool = input[:t - self.Bo + 1]
                matrix_X = self.create_reference_blocks(reference_pool, self.N, self.Bo)

        stopping_time = t  # if CP detected stopping_time -> CP timestamp, if no detected, stopping_time is the end
        return stopping_time, scores
