import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/Administrator/Downloads/RealWorldActivityData20111104/hasc/hasc-111018-165936-acc.csv', header=None)

header = ['ts', 'x', 'y', 'z']  # Replace with your desired column names
df.columns = header
df['L2_norm'] = np.linalg.norm(df[['x', 'y', 'z']].values, axis=1)

labels = [(5057.661,5091.26), (5098.502,5126.499), (5127.665,5143.411),
          (5154.309,5162.703), (5168.384,5209.209), (5210.934,5224.176), (5226.52,5237.975),
          (5239.063,5242.474), (5243.853,5271.633,), (5272.415,5273.939), (5274.893,5278.292),
          (5279.308,5281.812), (5282.791,5285.715), (5288.024,5293.109), (5296.491,5305.706),
          (5306.251,5312.854), (5316.488,5333.001), (5334.224,5336.147), (5337.78,5340.465),
          (5342.559,5369.625), (5371.264,5372.642), (5378.646,5389.953), (5392.041,5411.828), (5415.992,5422.95)]

for (start, stop) in labels:
    mask = (df['ts'] > start)
    filtered_data = df[mask]
    start_row = filtered_data['ts'].idxmin()
    mask = (df['ts'] > stop)
    filtered_data = df[mask]
    stop_row = filtered_data['ts'].idxmin()
    seg = df.iloc[start_row:stop_row+1]
    
