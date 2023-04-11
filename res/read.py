import numpy as np
import glob
folder = '*.npz'
for i in glob.glob(folder):
    data = np.load(i)
    rec = data['rec']
    FAR = data['FAR']
    prec = data['prec']
    f1score = data['f1score']
    dd = data['dd']
    print(i ,':')
    print('recall: ', rec)
    print('false alarm rate: ', FAR)
    print('precision: ', prec)
    print('F1 Score: ', f1score)
    print('detection delay: ', dd)