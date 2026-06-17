import pandas as pd
import numpy as np
from predict import classify

df = pd.read_csv('sample_validation.csv')
y_true = df['class_id'].values
X_raw = df.iloc[:, 3:].values.astype(np.float32)

print(f"{'IDX':>4}  {'TRUE':>5}  {'PRED':>5}  {'NAME':>5}  {'OK':>4}")
print("-" * 35)

correct = 0
for i in range(len(df)):
    pred_id, pred_name = classify(X_raw[i])
    ok = (pred_id == y_true[i])
    correct += ok
    print(f"{i:>4}  {y_true[i]:>5}  {pred_id:>5}  {pred_name:>5}  {'✓' if ok else '✗':>4}")

print("-" * 35)
print(f"Accuracy: {correct}/{len(df)} = {100*correct/len(df):.1f}%")