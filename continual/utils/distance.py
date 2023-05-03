from scipy.spatial import distance
import numpy as np

def MahalanobisDistance(input, centre, matA):
    dd1 = (input - centre).dot(matA).dot((input - centre).T)
    dd = distance.mahalanobis(input, centre, matA) ** 2
    return dd1[0][0]

def _validate_vector(u, dtype=None):
    # XXX Is order='c' really necessary?
    u = np.asarray(u, dtype=dtype, order='c')
    if u.ndim == 1:
        return u
    raise ValueError("Input vector should be 1-D.")

def mahalanobis(u, v, VI):
    u = _validate_vector(u)
    v = _validate_vector(v)
    VI = np.atleast_2d(VI)
    delta = u - v
    m = np.dot(np.dot(delta, VI), delta)
    return m

