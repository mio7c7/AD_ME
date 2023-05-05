import pandas as pd
import numpy as np
from scipy.stats import chi2
from matplotlib import patches
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, euclidean, mahalanobis
from scipy.spatial import distance

def MahalanobisDistance(input, matA, centre):
    # dd1 = (input - centre).T.dot(matA).dot(input - centre)
    dd = distance.mahalanobis(input, centre, matA) ** 2
    return dd

