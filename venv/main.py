import numpy as np
import glob
from itertools import islice

def window(seq, ws=2):
    it = iter(seq)
    result = tuple(islice(it, ws))
    if len(result) == ws:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

folder = './data/*.npz'
WINDOW = 10

for i in glob.glob(folder):
    data = np.load(i)
    train_ts, train_dl = data['train_ts'], data['train_dl']
    train_tsx = window(train_ts[:, 1], WINDOW)
    for expression in train_ts:
        print(expression)
    train_dl = window(train_dl[1], WINDOW)