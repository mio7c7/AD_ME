import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

folder = 'D:/calibrated_30min/WTAF_WSC_csv/dataset/'
simulate_info = pd.read_csv('D:/calibrated_30min/WTAF_WSC_csv/simulate_info.csv', index_col=0).reset_index(drop=True)
#['Site', 'Tank', 'Leak_rate', 'Hole_range', 'Hole_height', 'Start_date', 'Stop_date', 'File_name']
TP = 720


def preprocess(data, fixed_t):
    del_idx = []
    for i in range (data.shape[0]):
        if abs(data[i, 1]) > fixed_t:
            del_idx.append(i)
    return np.delete(data, del_idx, axis=0)

# each tank share the same training data for different cases
# each tank as one npz file, ['train_transaction'], ['train_idle'], ['test_transaction'], ['test_idle'], ['label']
for idx, row in simulate_info.iterrows():
    site, tank = row['Site'], row['Tank']
    tank_info = simulate_info[(simulate_info['Site'] == site) & (simulate_info['Tank'] == tank)]
    simulate_info = simulate_info.drop(index=tank_info.index)
    tank_info = tank_info.sort_values('Leak_rate', ascending=True).reset_index(drop=True)

    Flag = True
    np_name = site + '_T' + str(tank) + '.npz'
    label = {}
    for idx1, row1 in tank_info.iterrows():
        filename = row1['File_name']
        df = pd.read_csv(folder + filename + '.csv', index_col=0).reset_index(drop=True)

        if row1['Leak_rate'] == 0:
            continue

        transactions = df.loc[df['period'] == 1]
        idles = df.loc[df['period'] == 0]
        deliveries = df.loc[df['period'] == 2]

        if Flag:
            train_ts = transactions.iloc[:TP][['Time_DN', 'Var_tc']].to_numpy()
            train_dl = idles.iloc[:TP][['Time_DN', 'Var_tc']].to_numpy()
            Flag = False

        if row1['Leak_rate'] == 0:
            test_ts_normal = transactions.iloc[TP:][['Time_DN', 'Var_tc']].to_numpy()
            test_dl_normal = idles.iloc[TP:][['Time_DN', 'Var_tc']].to_numpy()
        elif idx1 == 1:
            test_ts_05gal = transactions.iloc[TP:][['Time_DN', 'Var_tc_readjusted']].to_numpy()
            test_dl_05gal = idles.iloc[TP:][['Time_DN', 'Var_tc_readjusted']].to_numpy()
            label['test_05gal'] = {row['Start_date'], row['Stop_date']}
            test_dl_05gal = preprocess(test_dl_05gal, 4)
            plt.plot(test_dl_05gal[:, 0], test_dl_05gal[:, 1])
            plt.title(filename)
            plt.show()
        elif idx1 == 2:
            test_ts_1gal = transactions.iloc[TP:][['Time_DN', 'Var_tc_readjusted']].to_numpy()
            test_dl_1gal = idles.iloc[TP:][['Time_DN', 'Var_tc_readjusted']].to_numpy()
            label['test_1gal'] = {row['Start_date'], row['Stop_date']}
            test_dl_1gal = preprocess(test_dl_1gal, 4)
            plt.plot(test_dl_1gal[:,0], test_dl_1gal[:,1])
            plt.title(filename)
            plt.show()
        elif idx1 == 3:
            test_ts_2gal = transactions.iloc[TP:][['Time_DN', 'Var_tc_readjusted']].to_numpy()
            test_dl_2gal = idles.iloc[TP:][['Time_DN', 'Var_tc_readjusted']].to_numpy()
            label['test_2gal'] = {row['Start_date'], row['Stop_date']}
            test_dl_2gal = preprocess(test_dl_2gal, 4)
            plt.plot(test_dl_2gal[:, 0], test_dl_2gal[:, 1])
            plt.title(filename)
            plt.show()



    # np.savez(np_name, train_ts=train_ts, train_dl=train_dl,
    #          test_ts_normal=test_ts_normal, test_dl_normal=test_dl_normal,
    #          test_ts_05gal=test_ts_05gal, test_dl_05gal=test_dl_05gal,
    #          test_ts_1gal=test_ts_1gal, test_dl_1gal=test_dl_1gal,
    #          test_ts_2gal=test_ts_2gal, test_dl_2gal=test_dl_2gal,
    #          label = label)