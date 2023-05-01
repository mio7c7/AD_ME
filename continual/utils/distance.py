from scipy.spatial import distance

def MahalanobisDistance(input, matA, centre):
    # dd1 = (input - centre).T.dot(matA).dot(input - centre)
    dd = distance.mahalanobis(input, centre, matA) ** 2
    return dd



