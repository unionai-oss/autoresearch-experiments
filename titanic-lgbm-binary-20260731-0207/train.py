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
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

unique, counts = zip(*sorted(
    zip(*[list(v) for v in [
        [int(c) for c in set(y_encoded)],
        [int((y_encoded == c).sum()) for c in sorted(set(y_encoded))]
    ]])
))

class_dist_train = {int(c): int((y_train == c).sum()) for c in sorted(set(y_train))}
class_dist_val = {int(c): int((y_val == c).sum()) for c in sorted(set(y_val))}

print(
    f"Dataset summary: total={len(df)} samples, "
    f"train={len(X_train)}, val={len(X_val)}, "
    f"classes={len(le.classes_)}, "
    f"class_dist_train={class_dist_train}, "
    f"class_dist_val={class_dist_val}"
)

# TODO: implement model, training loop, and metric evaluation
# Metric to optimize: roc_auc
# Final output line must be: print(f"BEST_VAL_ROC_AUC: {value:.6f}")