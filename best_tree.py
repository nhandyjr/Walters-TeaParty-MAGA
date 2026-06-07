import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
import joblib
import os
out_folder = r"C:\Users\Owner\Desktop\Data Science\Eastern\691\walters"

# -----------------------------------------------------------------------------
# 1. Load data set
# -----------------------------------------------------------------------------
out_path = r"C:\Users\Owner\Desktop\Data Science\Eastern\691\walters\surveys\ool_final.csv"
ool_final = pd.read_csv(out_path, index_col="CASEID")

#  Produce value counts for features
cols_to_check = ['TeaPartyMem_b', 'BWEqulOppty', 'BWEqulJust_7'
                ,'RateLatino_100','RateUnDoc_100'
                # ,'Optmsm_Futr'
]

for col in cols_to_check:
    print(f"\n{col} value counts:")
    print(ool_final[col].value_counts(dropna=False))
    print("-" * 50)

# -----------------------------------------------------------------------------
# 2. Define features and target
# -----------------------------------------------------------------------------
features_cont = ['RateLatino_100'
                 ,'RateUnDoc_100'
                 # ,'RateBlk_100'
                 # ,'RateWhite_100'
]

features_cat = ['BWEqulOppty'
                ,'BWEqulJust_7'
                # ,'Optmsm_Futr'
                # ,'HardToBetterParents'
                 
                ]
target = 'TeaPartyMem_b'

# Target as 0/1
y = (ool_final[target] == 1).astype(int)

# -----------------------------------------------------------------------------
# 3. Build feature matrix X
# -----------------------------------------------------------------------------
X = pd.DataFrame(index=ool_final.index)

# Continuous features: add missing indicator, fill missing with sentinel -999
for col in features_cont:
    X[f'{col}_missing'] = ool_final[col].isna().astype(int)
    X[col] = ool_final[col].fillna(-999)

# Categorical features: one‑hot encode (including missing as a separate dummy)
for col in features_cat:
    # Convert to string so that NaN becomes 'NaN' – creates a dummy column for missing
    dummies = pd.get_dummies(ool_final[col].astype(str), prefix=col, dummy_na=True)
    X = pd.concat([X, dummies], axis=1)

# -----------------------------------------------------------------------------
# 4. Class weights (34x for minority class, matching SAS weight_factor=34)
# -----------------------------------------------------------------------------
class_weight = {0: 1, 1: 34}

# -----------------------------------------------------------------------------
# 5. Train/validation split (stratified, same seed as SAS)
# -----------------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=15531
)

# -----------------------------------------------------------------------------
# 6. Train full tree (no depth limit) and get pruning path
# -----------------------------------------------------------------------------
full_tree = DecisionTreeClassifier(class_weight=class_weight, max_depth=10, random_state=15531)
full_tree.fit(X_train, y_train)
path = full_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# -----------------------------------------------------------------------------
# 7. Select best ccp_alpha by validation AUC
# -----------------------------------------------------------------------------
best_alpha = None
best_tree = None
best_val_auc = -np.inf

for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(class_weight=class_weight, ccp_alpha=alpha, random_state=15531)
    clf.fit(X_train, y_train)
    val_auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_alpha = alpha
        best_tree = clf

print(f"Best ccp_alpha: {best_alpha:.6f}, Validation AUC: {best_val_auc:.4f}")

# -----------------------------------------------------------------------------
# 8. Evaluate final tree on validation set
# -----------------------------------------------------------------------------
y_val_pred = best_tree.predict(X_val)
cm = confusion_matrix(y_val, y_val_pred)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0
specificity = tn / (tn + fp) if (tn+fp) > 0 else 0

print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print("Confusion Matrix:\n", cm)

joblib.dump(best_tree, os.path.join(out_folder, 'best_tree.pkl'))



