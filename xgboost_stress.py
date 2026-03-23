import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

df = pd.read_csv('merged_df1.csv')

df['reported_stress'] = df.groupby(['participant_id', 'phase'])['reported_stress'].bfill()

df_labeled = df.dropna(subset=['reported_stress']).copy()

train_ids = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13']
test_ids = ['S14', 'S15', 'S16', 'S17', 'S18']

train_df = df_labeled[df_labeled['participant_id'].isin(train_ids)]
test_df = df_labeled[df_labeled['participant_id'].isin(test_ids)]

signals = ['ACC_x', 'ACC_y', 'ACC_z', 'BVP', 'EDA', 'TEMP', 'HR', 'IBI']
results = {}

for signal in signals:
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(train_df[[signal]], train_df['reported_stress'])
    
    preds = model.predict(test_df[[signal]])
    rmse = np.sqrt(mean_squared_error(test_df['reported_stress'], preds))
    results[signal] = rmse
    print(f"{signal} alone RMSE: {rmse:.4f}")

model_all = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model_all.fit(train_df[signals], train_df['reported_stress'])
rmse_all = np.sqrt(mean_squared_error(test_df['reported_stress'], model_all.predict(test_df[signals])))
results['ALL_SIGNALS'] = rmse_all

print(f"\nALL SIGNALS combined RMSE: {rmse_all:.4f}")