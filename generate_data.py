import pandas as pd
import numpy as np
import os

os.makedirs('sepsis_data/training', exist_ok=True)
cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'SepsisLabel']

for i in range(100):
    rows = np.random.randint(20, 50)
    data = np.random.randn(rows, len(cols))
    df = pd.DataFrame(data, columns=cols)
    df['SepsisLabel'] = 0
    if i < 10: # Make 10% of patients septic
        df.iloc[-5:, -1] = 1
    df.to_csv(f'sepsis_data/training/p{i:06d}.psv', sep='|', index=False)
print("Generated 100 synthetic patient records in PhysioNet format.")
