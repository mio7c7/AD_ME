import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
import numpy as np
import timeit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.io import savemat
import os
# import plotly.express as px
# import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def assign_period(df):
    # idle = 0, transaction = 1, delivery = 2
    conditions = [
        (df['Del_tc'] != 0),
        (df['Sales_Ini'] == 0),
        (df['Sales_Ini'] != 0) & (df['Del_tc'] == 0),
    ]

    # create a list of the values we want to assign for each condition
    values = [2, 0, 1]
    # create a new column and use np.select to assign values to it using our lists as arguments
    df['period'] = np.select(conditions, values)
    transactions = df.loc[df['period'] == 1]
    idles = df.loc[df['period'] == 0]
    deliveries = df.loc[df['period'] == 2]
    return transactions, idles, deliveries

# def generate_plots(transactions, idles, deliveries):
#     fig = go.Figure()
#     transactions['Cumsum_var'] = transactions['Var'].cumsum()
#     transactions['Cumsum_vartc'] = transactions['Var_tc'].cumsum()
#     idles['Cumsum_var'] = idles['Var'].cumsum()
#     idles['Cumsum_vartc'] = idles['Var_tc'].cumsum()
#     deliveries['Cumsum_var'] = deliveries['Var'].cumsum()
#     deliveries['Cumsum_vartc'] = deliveries['Var_tc'].cumsum()
#
#     fig.add_trace(go.Scatter(x=transactions['Time'], y=transactions['Cumsum_vartc'],
#                              mode='markers',
#                              fillcolor='green',
#                              opacity=0.7,
#                              name='transactions'))
#     fig.add_trace(go.Scatter(x=idles['Time'], y=idles['Cumsum_vartc'],
#                              mode='markers',
#                              fillcolor='blue',
#                              opacity=0.7,
#                              name='idles'))
#     fig.add_trace(go.Scatter(x=deliveries['Time'], y=deliveries['Cumsum_vartc'],
#                              mode='markers',
#                              fillcolor='red',
#                              opacity=0.7,
#                              name='deliveries'))
#     return fig
#
# file_folder = 'D:/calibrated_30min/temp/'
#
# names = os.listdir(file_folder)
# for tank in names:
#     path = file_folder + tank
#     df = pd.read_csv(path, index_col=0).reset_index(drop=True)
#
#
#     df['norm_var'] = StandardScaler().fit_transform(np.array(df['Var']).reshape(-1, 1))
#     df['norm_vartc'] = StandardScaler().fit_transform(np.array(df['Var_tc']).reshape(-1, 1))
#
#     transactions, idles, deliveries = assign_period(df)
#
#     fig = generate_plots(transactions, idles, deliveries)
#
#     # transactions['Cumsum_normvar'] = transactions['norm_var'].cumsum()
#     # transactions['Cumsum_normvartc'] = transactions['norm_vartc'].cumsum()
#     # idles['Cumsum_normvar'] = idles['norm_var'].cumsum()
#     # idles['Cumsum_normvartc'] = idles['norm_vartc'].cumsum()
#     # deliveries['Cumsum_normvar'] = deliveries['norm_var'].cumsum()
#     # deliveries['Cumsum_normvartc'] = deliveries['norm_vartc'].cumsum()
#
#     # fig.add_trace(go.Scatter(x=transactions['Time'], y=transactions['Cumsum_normvartc'],
#     #                          mode='markers',
#     #                          fillcolor='green',
#     #                          opacity=0.7,
#     #                          marker=dict(symbol="diamond"),
#     #                          name='transactions'))
#     # fig.add_trace(go.Scatter(x=idles['Time'], y=idles['Cumsum_normvartc'],
#     #                          mode='markers',
#     #                          fillcolor='blue',
#     #                          opacity=0.7,
#     #                          marker=dict(symbol="diamond"),
#     #                          name='idles'))
#     # fig.add_trace(go.Scatter(x=deliveries['Time'], y=deliveries['Cumsum_normvartc'],
#     #                          mode='markers',
#     #                          fillcolor='red',
#     #                          opacity=0.7,
#     #                          marker=dict(symbol="diamond"),
#     #                          name='deliveries'))
#
#     # fig.add_trace(go.Scatter(x=df['ClosingStock_tc'], y=df['Cumsum_vartc'],
#     #                          mode='markers',
#     #                          fillcolor='green',
#     #                          opacity=0.7,
#     #                          name='no_filter'))
#     fig.update_layout(
#         title=tank
#     )
#     # fig.show()
#     plotly.offline.plot(fig, filename=tank + 'periods.html', auto_open=False)