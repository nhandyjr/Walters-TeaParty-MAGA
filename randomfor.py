import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------
out_path = r"C:\Users\Owner\Desktop\Data Science\Eastern\691\walters\ool_final.csv"
ool_final = pd.read_csv(out_path, index_col="CASEID")

# -----------------------------------------------------------------------------
# 2. Features – same as SAS tree (including Optmsm_Futr)
# -----------------------------------------------------------------------------
features_cont = ['RateLatino_100', 'RateUnDoc_100', 'RateBlk_100', 
                 'RateWhite_100', 'RateUnEe_100']
features_cat = ['BWEqulOppty', 'BWEqulJust_7', 'Optmsm_Futr']   # Optmsm_Futr included

target = 'TeaPartyMem_b'
y = (ool_final[target] == 1).astype(int)

# -----------------------------------------------------------------------------
# 3. Build feature matrix (matching SAS missing handling)
# -----------------------------------------------------------------------------
X = pd.DataFrame(index=ool_final.index)

# Continuous: add missing indicator, fill with sentinel -999
for col in features_cont:
    X[f'{col}_missing'] = ool_final[col].isna().astype(int)
    X[col] = ool_final[col].fillna(-999)

# Categorical: one‑hot encode (treat missing as its own category)
for col in features_cat:
    dummies = pd.get_dummies(ool_final[col].astype(str), prefix=col, dummy_na=True)
    X = pd.concat([X, dummies], axis=1)

# -----------------------------------------------------------------------------
# 4. Sample weights (34 for TeaParty=1, else 1) – matches SAS WEIGHT
# -----------------------------------------------------------------------------
weights = np.where(y == 1, 34, 1)

# -----------------------------------------------------------------------------
# 5. Train/validation split (stratified, seed 15531)
# -----------------------------------------------------------------------------
X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
    X, y, weights, test_size=0.3, stratify=y, random_state=15531
)

# -----------------------------------------------------------------------------
# 6. Random Forest with sample_weight (max_depth=5, same as tree)
# -----------------------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,               # same as SAS tree depth
    min_samples_leaf=5,        # same as SAS leavesize
    random_state=15531,
    n_jobs=-1,
    class_weight=None          # we use sample_weight instead
)
rf.fit(X_train, y_train, sample_weight=w_train)

# -----------------------------------------------------------------------------
# 7. Evaluate
# -----------------------------------------------------------------------------
y_proba = rf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_proba)
print(f"Random Forest Validation AUC: {auc:.4f}")

y_pred = (y_proba >= 0.5).astype(int)
cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:\n", cm)
tn, fp, fn, tp = cm.ravel()
print(f"Sensitivity: {tp/(tp+fn):.4f}, Specificity: {tn/(tn+fp):.4f}")