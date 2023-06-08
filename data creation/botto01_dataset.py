import numpy as np
import pandas as pd
# from per_label import assign_period

# folder = 'D:/calibrated_30min/WTAF_WSC_csv/dataset/'
# simulate_info = pd.read_csv('D:/calibrated_30min/WTAF_WSC_csv/simulate_info.csv').reset_index(drop=True)
simulate_info = pd.read_csv('bottom005_info.csv').reset_index(drop=True)
#['Site', 'Tank', 'Leak_rate', 'Hole_range', 'Hole_height', 'Start_date', 'Stop_date', 'File_name']
TP = 720

# each tank share the same training data for different cases
# each tank as one npz file, ['train_transaction'], ['train_idle'], ['test_transaction'], ['test_idle'], ['label']
while len(simulate_info.index) != 0:
    row = simulate_info.loc[simulate_info.index[0]]
# for idx, row in simulate_info.iterrows():
    site, tank = row['Site'], row['Tank']
    tank_info = simulate_info[(simulate_info['Site'] == site) & (simulate_info['Tank'] == tank)]
    simulate_info = simulate_info.drop(index=tank_info.index)
    tank_info = tank_info.sort_values('Leak_rate', ascending=True).reset_index(drop=True)

    Flag = True
    np_name = site + '_T' + str(tank) + 'bottom005.npz'
    label = {}
    for idx1, row1 in tank_info.iterrows():
        filename = row1['File_name']
        df = pd.read_csv(filename + '.csv', index_col=0).reset_index(drop=True)

        transactions = df.loc[df['period'] == 1]
        idles = df.loc[df['period'] == 0]
        deliveries = df.loc[df['period'] == 2]

        train_ts = transactions.iloc[:TP][['Time_DN', 'Var_tc', 'ClosingHeight_readjusted']].to_numpy()
        train_dl = idles.iloc[:TP][['Time_DN', 'Var_tc', 'ClosingHeight_readjusted']].to_numpy()

        test_ts_1gal = transactions.iloc[TP:][['Time_DN', 'Var_tc_readjusted', 'ClosingHeight_readjusted']].to_numpy()
        test_dl_1gal = idles.iloc[TP:][['Time_DN', 'Var_tc_readjusted', 'ClosingHeight_readjusted']].to_numpy()
        label['test_1gal'] = {row1['Start_date'], row1['Stop_date']}

    np.savez(np_name, train_ts=train_ts, train_dl=train_dl,
             test_ts_1gal=test_ts_1gal, test_dl_1gal=test_dl_1gal,
             label = label)
