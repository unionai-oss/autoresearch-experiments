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

train_class_dist = {int(cls): int((y_train == cls).sum()) for cls in sorted(set(y_train))}
val_class_dist = {int(cls): int((y_val == cls).sum()) for cls in sorted(set(y_val))}

print(
    f"[DATA] Total samples: {len(df)}, "
    f"Train: {len(X_train)}, Val: {len(X_val)}, "
    f"Train class distribution: {train_class_dist}, "
    f"Val class distribution: {val_class_dist}"
)

# TODO: implement model, training loop, and metric evaluation
# Metric to optimize: roc_auc
# Final output line must be: print(f"BEST_VAL_ROC_AUC: {value:.6f}")