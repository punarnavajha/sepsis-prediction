import pandas as pd
import numpy as np
import glob
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc

# Look for the 100 synthetic files we just made
files = glob.glob('sepsis_data/training/*.psv')

def process_data(file_list):
    all_data = []
    if not file_list:
        return pd.DataFrame()
    for f in file_list:
        df = pd.read_csv(f, sep='|')
        # Physiological Feature Engineering (6-hour rolling mean)
        df['HR_6hr_mean'] = df['HR'].rolling(window=6, min_periods=1).mean()
        df = df.ffill().fillna(df.median())
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

print(f"Found {len(files)} synthetic patient records. Training Sepsis Model...")
data = process_data(files)

if not data.empty:
    X = data.drop(['SepsisLabel'], axis=1)
    y = data['SepsisLabel']
    # Fixing class imbalance (Sepsis is rare!)
    scale_weight = (y == 0).sum() / (y == 1).sum()
    model = XGBClassifier(scale_pos_weight=scale_weight, n_estimators=50)
    model.fit(X, y)
    
    probs = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, probs)
    print(f"\n--- SUCCESS ---")
    print(f"Model AUPRC Score: {auc(recall, precision):.4f}")
else:
    print("Error: Files found but data is empty. Check generate_data.py")
