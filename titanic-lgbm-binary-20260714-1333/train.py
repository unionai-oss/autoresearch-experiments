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

unique, counts = zip(*sorted(
    [(cls, int((y_encoded == idx).sum())) for cls, idx in class_mapping.items()],
    key=lambda x: x[1], reverse=True
))
class_dist_str = ", ".join(f"{cls}: {cnt}" for cls, cnt in zip(unique, counts))
print(f"[DATA] Samples: {len(df)}, Classes: {len(class_mapping)}, Distribution: {{{class_dist_str}}}, Train: {len(X_train)}, Val: {len(X_val)}")

# TODO: implement model, training loop, and metric evaluation
# Metric to optimize: roc_auc
# Final output line must be: print(f"BEST_VAL_ROC_AUC: {value:.6f}")