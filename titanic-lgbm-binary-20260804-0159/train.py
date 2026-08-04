import os
DATA_PATH = os.environ.get("DATA_PATH", "/tmp/data")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_parquet(DATA_PATH)

target_col = "Survived"
X = df.drop(columns=[target_col])
y = df[target_col]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
print(f"Class mapping: {class_mapping}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

import numpy as np
unique, counts = np.unique(y_encoded, return_counts=True)
class_dist = {int(k): int(v) for k, v in zip(unique, counts)}
print(f"Dataset summary: total samples={len(df)}, class distribution={class_dist}, train={len(X_train)}, val={len(X_val)}")

# TODO: implement model, training loop, and metric evaluation
# Metric to optimize: roc_auc
# Final output line must be: print(f"BEST_VAL_ROC_AUC: {value:.6f}")