import pandas as pd
import os
import numpy as np
import glob

# simulate_info = pd.read_csv('D:/calibrated_30min/WTAF_WSC_csv/simulate_info.csv', index_col=0).reset_index(drop=True)
# folder = 'D:/calibrated_30min/WTAF_WSC_csv/normal/*.csv'
simulate_info = pd.read_csv('G:/RQ2 data/simulate_info.csv', index_col=0).reset_index(drop=True)
folder = 'G:/RQ2 data/normal/*.csv'
C = simulate_info.index[-1] + 1

for i in glob.glob(folder):
    Site = i[i.rfind('\\') + 1:i.rfind('\\') + 5]
    tank = i[i.rfind('\\') + 1:]
    tank = tank[:tank.find('_')]
    tank_no = int(tank[12])
    leak_rate, hr, hole_height, start_date, stop_date = 0, 0, 0, 0, 0
    file_name = i[i.rfind('\\') + 1:i.rfind('.')]
    simulate_info.loc[C] = [Site, tank_no, leak_rate, hr, hole_height, start_date, stop_date, file_name]
    C += 1

simulate_info.to_csv('simulate_info.csv')