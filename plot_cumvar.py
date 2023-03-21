import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
import numpy as np
import timeit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.io import savemat
import os
import plotly
import plotly.express as px
import plotly.graph_objects as go

file_folder = 'D:/calibrated_30min/pcs_csv/'

names = os.listdir(file_folder)
tanks = list(set([x.split('_')[0] for x in names]))
for tank in tanks:
    ms = [x for x in names if x.split('_')[0] == tank]
    fig = go.Figure()
    for i in ms:
        path = file_folder + i
        df = pd.read_csv(path, index_col=0).reset_index(drop=True)

        label = i.split('_')[1] + i.split('_')[2]
        if label == 'WOTAFWOSC.csv':
            color = 'green'
            name = 'w/o taf w/o strapping chart'
        elif label == 'WTAFWOSC.csv':
            color = 'red'
            name = 'w/ taf w/o strapping chart'
        elif label == 'WTAFWSC.csv':
            color = 'blue'
            name = 'w/ taf w/ strapping chart'
        elif label == 'WOTAFWSC.csv':
            color = 'orange'
            name = 'w/o taf w/ strapping chart'

        df['Cumsum_var'] = df['Var'].cumsum()
        df['Cumsum_vartc'] = df['Var_tc'].cumsum()

        fig.add_trace(go.Scatter(x=df['Time'], y=df['Cumsum_vartc'],
                                 mode='lines',
                                 fillcolor=color,
                                 opacity=0.7,
                                 name=name))
        # fig.add_trace(go.Scatter(x=df['ClosingStock_tc'], y=df['Cumsum_vartc'],
        #                          mode='markers',
        #                          fillcolor='green',
        #                          opacity=0.7,
        #                          name='no_filter'))
    fig.update_layout(
        title=i
    )
    # fig.show()
    plotly.offline.plot(fig, filename=tank + '.html', auto_open=False)