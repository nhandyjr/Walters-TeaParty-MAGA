import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

# Load data
out_path = r"C:\Users\Owner\Desktop\Data Science\Eastern\691\walters\ool_final.csv"
ool_final = pd.read_csv(out_path, index_col="CASEID")

# Features
features_cont = ['RateLatino_100', 'RateUnDoc_100']
features_cat = ['BWEqulOppty', 'BWEqulJust_7', 'Optmsm_Futr', 'HardToBetterParents']
target = 'TeaPartyMem_b'
y = (ool_final[target] == 1).astype(int)

X = pd.DataFrame(index=ool_final.index)

# Continuous: add missing indicator, but keep NaN (do NOT fill)
for col in features_cont:
    X[f'{col}_missing'] = ool_final[col].isna().astype(int)
    X[col] = ool_final[col]  # keep NaN as is

# Categorical: one‑hot encode (including missing)
for col in features_cat:
    dummies = pd.get_dummies(ool_final[col].astype(str), prefix=col, dummy_na=True)
    X = pd.concat([X, dummies], axis=1)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=15531
)

# XGBoost single tree
model = xgb.XGBClassifier(
    n_estimators=1,
    max_depth=10,
    scale_pos_weight=20,
    random_state=15531,
    missing=np.nan,          # important: treat NaN as missing
    tree_method='exact',
    objective='binary:logistic',
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# Evaluate
y_val_prob = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_val_prob)
print(f"XGBoost Validation AUC: {auc:.4f}")

y_val_pred = (y_val_prob >= 0.5).astype(int)
cm = confusion_matrix(y_val, y_val_pred)
print("Confusion Matrix:\n", cm)