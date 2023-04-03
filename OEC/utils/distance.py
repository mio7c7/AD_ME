import pandas as pd
import numpy as np
from scipy.stats import chi2
from matplotlib import patches
import matplotlib.pyplot as plt

def MahalanobisDistance(input, matA, centre):
    distance = (input - centre).T.dot(matA).dot(input - centre)
    return distance

