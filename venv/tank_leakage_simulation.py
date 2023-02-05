import pandas as pd
import numpy as np
import glob
import math
import rpy2.robjects as robjects

# Initialisation, settings
GAL = 3.78541
AVG_leak_rate = {'GAL/2': (0.7*(GAL/2), 1.3*(GAL/2)), 'GAL': (0.7*(GAL), 1.3*(GAL)), 'GAL*2': (0.7*(GAL*2), 1.3*(GAL*2))}
hole_range = {'bottom': (0, 0), 'middle': (0.2, 0.5), 'top': (0.5, 0.75)}
folder = 'D:/calibrated_30min/WTAF_WSC_csv/normal/*.csv'
MONTH = 2629743

def adjust_volume(og, hole_height, leak_rate, start_date, stop_date, reversedSC):
    og['ClosingStock_tc_readjusted'] = og['ClosingStock_Caltc']
    og['OpeningStock_tc_readjusted'] = og['OpeningStock_Caltc']
    og['OpeningHeight_readjusted'] = og['OpeningHeight']
    og['ClosingHeight_readjusted'] = og['ClosingHeight']
    before = og[(og['Time_DN'] < start_date)].copy()

    leaking = og[(og['Time_DN'] >= start_date) & (og['Time_DN'] <= stop_date)].copy()
    # determine hmax, monthly update
    leaking['HMAX'] = 0
    monthlyflag, idx = 0, leaking.index[0]
    while idx <= leaking.index[-1]:
        monthlyflag = leaking.loc[idx, 'Time_DN']
        subset = leaking[(leaking['Time_DN'] >= monthlyflag) & (leaking['Time_DN'] <= monthlyflag + MONTH)]
        max_fil_height = max(subset['ClosingHeight'])
        leaking.loc[idx:subset.index[-1]+1, 'HMAX'] = max_fil_height
        idx = subset.index[-1]+1

    # update closing stock and starting stock during the leaking period
    cum_var = 0
    for idx, row in leaking.iterrows():
        if row.OpeningHeight_readjusted >= hole_height:
            leakage = - 0.5 * leak_rate * math.sqrt((row.OpeningHeight_readjusted - hole_height) / (row.HMAX - hole_height))
            cum_var += leakage

        leaking.at[idx, 'ClosingStock_tc_readjusted'] = row.ClosingStock_Caltc + cum_var
        temp = reversedSC['ChangePoint'].loc[lambda x: x >= row.ClosingStock_Caltc + cum_var]
        if len(temp.index) != 0:
            cp = temp.index[0]
        else:
            cp = reversedSC.index[-1]
        leaking.at[idx, 'ClosingHeight_readjusted'] = reversedSC.loc[cp, 'Intercept'] + reversedSC.loc[cp, 'B1']*leaking.loc[idx, 'ClosingStock_tc_readjusted']

        if idx == leaking.index[0]:
            leaking.at[idx, 'OpeningStock_tc_readjusted'] = og.loc[idx-1, 'ClosingStock_tc_readjusted']
            leaking.at[idx, 'OpeningHeight_readjusted'] = og.loc[idx-1, 'ClosingHeight_readjusted']
        else:
            leaking.at[idx, 'OpeningStock_tc_readjusted'] = leaking.loc[idx-1, 'ClosingStock_tc_readjusted']
            leaking.at[idx, 'OpeningHeight_readjusted'] = leaking.loc[idx-1, 'ClosingHeight_readjusted']

    after = og[og['Time_DN'] > stop_date].copy()
    df = pd.concat([before, leaking, after])
    return df


for i in glob.glob(folder):
    df = pd.read_csv(i, index_col=0).reset_index(drop=True)
    Site = i[i.rfind('\\')+1:i.rfind('\\')+5]
    tank = i[i.rfind('\\')+1:]
    tank = tank[:tank.find('_')]
    k = 'G:/Meter error/Pump Cal report/Data/'+ Site + '/' + Site + '_ACal.RDATA'
    robjects.r['load'](k)
    matrix = robjects.r['Cal_Output']
    ob = matrix.rx2(tank).rx2('ND_AMB').rx2('Strap').rx2('Coeff_MinErr')
    array = np.array(ob)
    sc = pd.DataFrame(data=array,
                      columns=['Intercept', 'B1', 'ChangePoint', 'Count'])
    reversedSC = sc.copy()
    for idx, row in sc.iterrows():
        reversedSC.loc[idx, 'ChangePoint'] = row.Intercept + row.B1*row.ChangePoint
        reversedSC.loc[idx, 'B1'] = 1/row.B1
        reversedSC.loc[idx, 'Intercept'] = -1*row.Intercept / row.B1

    duration = (df['Time_DN'].iloc[-1] - df['Time_DN'].iloc[0])/MONTH
    for _, alr in AVG_leak_rate.items():
        leak_rate = np.random.uniform(alr[0], alr[1])
        hr = np.random.choice(list(hole_range.keys()), 1)[0]
        hole_height = np.random.uniform(hole_range.get(hr)[0], hole_range.get(hr)[1])*
        if MONTH > 12:
            start_date = df['Time_DN'].iloc[0] + np.random.uniform(6*MONTH, 12*MONTH)
            stop_date = np.random.uniform(start_date+3*MONTH, start_date+6*MONTH)
        elif (MONTH < 12) and (MONTH > 6):
            start_date = df['Time_DN'].iloc[0] + np.random.uniform(2 * MONTH, 3 * MONTH)
            stop_date = np.random.uniform(start_date + 2 * MONTH, df['Time_DN'].iloc[-1] - 1 * MONTH)
        else:
            break
        start_date, stop_date = int(start_date), int(stop_date)

        induced_df = adjust_volume(df, hole_height, leak_rate, start_date, stop_date, reversedSC)
        induced_df.to_csv(tank + "_" + str(leak_rate) + '_' + str(hole_height) + '.csv')


