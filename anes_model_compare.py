# -*- coding: utf-8 -*-
"""
Created on Sat May 30 22:19:35 2026

@author: nhandyjr@gmail.com
"""
import pandas as pd
import numpy as np
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE   # install imbalanced-learn if needed: pip install imbalanced-learn
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import shap
import os
print(os.getcwd())
output_dir = r"C:\Users\Owner\Desktop\Data Science\Eastern\691\walters\viz"
os.makedirs(output_dir, exist_ok=True)
# -------------------
# Cleaning Functions 
# -------------------
def clean_numeric(series):
    if series.dtype in ['int64', 'float64']:
        return series.where(series >= 0, np.nan)
    if series.dtype.name == 'category':
        str_vals = series.astype(str)
    else:
        str_vals = series.astype(str)
    def extract_num(s):
        match = re.match(r'-?\d+(?:\.\d+)?', s.strip())
        return float(match.group()) if match else np.nan
    numeric = str_vals.apply(extract_num)
    numeric = numeric.where(numeric >= 0, np.nan)
    return numeric

def clean_police_treat(series):
    if series.dtype.name == 'category':
        series = series.astype(str)
    mapping = {
        '1. Treat whites better': 1,
        '2. Treat both the same': 2,
        '3. Treat blacks better': 3
    }
    return series.map(mapping)

def clean_police_howmuch(series):
    if series.dtype.name == 'category':
        series = series.astype(str)
    mapping = {
        '1. Much': 1,
        '2. Moderately': 2,
        '3. A little': 3
    }
    return series.map(mapping)

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------
path_2016 = r'C:\Users\Owner\Desktop\Data Science\Eastern\691\walters\surveys\anes_timeseries_2016_dta\anes_timeseries_2016.dta'
path_2020 = r'C:\Users\Owner\Desktop\Data Science\Eastern\691\walters\surveys\anes_timeseries_2020_csv_20220210\anes_timeseries_2020_csv_20220210.csv'
path_2024 = r'C:\Users\Owner\Desktop\Data Science\Eastern\691\walters\surveys\anes_timeseries_2024_csv_20260519\anes_timeseries_2024_csv_20260519.csv'

# ------------------------------------------------------------
# 2016
# ------------------------------------------------------------
print("Loading 2016 data...")
df16 = pd.read_stata(path_2016)

target16 = clean_numeric(df16['V162079'])
y16 = (target16 >= 70).astype(int)

X16_dict = {
    'slavery_difficult': clean_numeric(df16['V162212']),
    'work_way_up': clean_numeric(df16['V162211']),
    'blacks_gotten_less': clean_numeric(df16['V162213']),
    'blacks_try_harder': clean_numeric(df16['V162214']),
    'discrimination_blacks': clean_numeric(df16['V162357']),
    'therm_blacks': clean_numeric(df16['V162312']),
    'police_treat': clean_police_treat(df16['V162320']),
    'police_howmuch': clean_police_howmuch(df16['V162321']),
    'therm_blm': clean_numeric(df16['V162113']),
    'therm_undoc': clean_numeric(df16['V162313']),
    'therm_hispanic': clean_numeric(df16['V162311']),
}
X16 = pd.DataFrame(X16_dict)
valid16 = X16.notna().all(axis=1) & y16.notna()
X16 = X16[valid16]
y16 = y16[valid16]
print(f"2016 sample size: {len(X16)}")
print(f"Trump support rate: {y16.mean():.2%}\n")

# ------------------------------------------------------------
# 2020 – use clean_numeric for ALL columns (including police)
# ------------------------------------------------------------
print("Loading 2020 data...")
df20 = pd.read_csv(path_2020)

target20 = clean_numeric(df20['V202144'])
y20 = (target20 >= 70).astype(int)

X20_dict = {
    'slavery_difficult': clean_numeric(df20['V202301']),
    'work_way_up': clean_numeric(df20['V202300']),
    'blacks_gotten_less': clean_numeric(df20['V202302']),
    'blacks_try_harder': clean_numeric(df20['V202303']),
    'discrimination_blacks': clean_numeric(df20['V202527']),
    'therm_blacks': clean_numeric(df20['V202480']),
    'police_treat': clean_numeric(df20['V202491']),
    'police_howmuch': clean_numeric(df20['V202492']),
    'therm_blm': clean_numeric(df20['V202174']),
    'therm_undoc': clean_numeric(df20['V202481']),
    'therm_hispanic': clean_numeric(df20['V202479']),
}
X20 = pd.DataFrame(X20_dict)
valid20 = X20.notna().all(axis=1) & y20.notna()
X20 = X20[valid20]
y20 = y20[valid20]
print(f"2020 sample size: {len(X20)}")
if len(X20) > 0:
    print(f"Trump support rate: {y20.mean():.2%}\n")
else:
    print("No complete rows for 2020.\n")

# ------------------------------------------------------------
# 2024 – same approach
# ------------------------------------------------------------
print("Loading 2024 data...")
df24 = pd.read_csv(path_2024)

target24 = clean_numeric(df24['V242126'])
y24 = (target24 >= 70).astype(int)

X24_dict = {
    'slavery_difficult': clean_numeric(df24['V242301']),
    'work_way_up': clean_numeric(df24['V242300']),
    'blacks_gotten_less': clean_numeric(df24['V242302']),
    'blacks_try_harder': clean_numeric(df24['V242303']),
    'discrimination_blacks': clean_numeric(df24['V242549']),
    'therm_blacks': clean_numeric(df24['V242516']),
    'police_treat': clean_numeric(df24['V242523']),
    'police_howmuch': clean_numeric(df24['V242524']),
    'therm_blm': clean_numeric(df24['V242152']),
    'therm_undoc': clean_numeric(df24['V242517']),
    'therm_hispanic': clean_numeric(df24['V242515']),
}
X24 = pd.DataFrame(X24_dict)
valid24 = X24.notna().all(axis=1) & y24.notna()
X24 = X24[valid24]
y24 = y24[valid24]
print(f"2024 sample size: {len(X24)}")
if len(X24) > 0:
    print(f"Trump support rate: {y24.mean():.2%}\n")
else:
    print("No complete rows for 2024.\n")
#---------------------------------------------------


# ------------------------------------------------------------
# Optional: Drop low-importance features (comment out if you want to keep them)
low_imp_features = ['police_treat', 'police_howmuch', 'discrimination_blacks']
X16 = X16.drop(columns=low_imp_features, errors='ignore')
X20 = X20.drop(columns=low_imp_features, errors='ignore')
X24 = X24.drop(columns=low_imp_features, errors='ignore')
# ------------------------------------------------------------

# ------------------------------------------------------------
# Save cleaned data for Streamlit dashboard
# ------------------------------------------------------------
import pickle
with open(os.path.join(output_dir, 'data_2016.pkl'), 'wb') as f:
    pickle.dump((X16, y16), f)
with open(os.path.join(output_dir, 'data_2020.pkl'), 'wb') as f:
    pickle.dump((X20, y20), f)
with open(os.path.join(output_dir, 'data_2024.pkl'), 'wb') as f:
    pickle.dump((X24, y24), f)
print("Saved cleaned data to viz folder.")



def evaluate_with_weights(X, y, year, weight_list, use_smote=False):
    """
    X, y: cleaned data
    weight_list: list of minority class weights to try (e.g., [2,5,10,20,34])
    use_smote: if True, apply SMOTE before training
    """
    # Split data (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if use_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    
    results = []
    for w in weight_list:
        # Decision Tree
        dt = DecisionTreeClassifier(class_weight={0:1, 1:w}, max_depth=5, random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        y_proba_dt = dt.predict_proba(X_test)[:,1]
        auc_dt = roc_auc_score(y_test, y_proba_dt)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_dt).ravel()
        sens_dt = tp/(tp+fn) if (tp+fn)>0 else 0
        spec_dt = tn/(tn+fp) if (tn+fp)>0 else 0
        prec_dt = tp/(tp+fp) if (tp+fp)>0 else 0
        f1_dt = 2*prec_dt*sens_dt/(prec_dt+sens_dt) if (prec_dt+sens_dt)>0 else 0
        
        # Random Forest
        rf = RandomForestClassifier(class_weight={0:1, 1:w}, n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        y_proba_rf = rf.predict_proba(X_test)[:,1]
        auc_rf = roc_auc_score(y_test, y_proba_rf)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
        sens_rf = tp/(tp+fn) if (tp+fn)>0 else 0
        spec_rf = tn/(tn+fp) if (tn+fp)>0 else 0
        prec_rf = tp/(tp+fp) if (tp+fp)>0 else 0
        f1_rf = 2*prec_rf*sens_rf/(prec_rf+sens_rf) if (prec_rf+sens_rf)>0 else 0
        
        # XGBoost (scale_pos_weight = weight * (neg/pos ratio))
        base_scale = len(y_train[y_train==0]) / len(y_train[y_train==1])
        scale = base_scale * w
        xgb = XGBClassifier(scale_pos_weight=scale, n_estimators=100, random_state=42)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        y_proba_xgb = xgb.predict_proba(X_test)[:,1]
        auc_xgb = roc_auc_score(y_test, y_proba_xgb)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_xgb).ravel()
        sens_xgb = tp/(tp+fn) if (tp+fn)>0 else 0
        spec_xgb = tn/(tn+fp) if (tn+fp)>0 else 0
        prec_xgb = tp/(tp+fp) if (tp+fp)>0 else 0
        f1_xgb = 2*prec_xgb*sens_xgb/(prec_xgb+sens_xgb) if (prec_xgb+sens_xgb)>0 else 0
        
        results.append({
            'weight': w,
            'DT_AUC': auc_dt, 'DT_Sens': sens_dt, 'DT_Spec': spec_dt, 'DT_Prec': prec_dt, 'DT_F1': f1_dt,
            'RF_AUC': auc_rf, 'RF_Sens': sens_rf, 'RF_Spec': spec_rf, 'RF_Prec': prec_rf, 'RF_F1': f1_rf,
            'XGB_AUC': auc_xgb, 'XGB_Sens': sens_xgb, 'XGB_Spec': spec_xgb, 'XGB_Prec': prec_xgb, 'XGB_F1': f1_xgb
        })
    
    # Print results
    print(f"\n{'='*80}\n{year} – WEIGHT GRID (SMOTE = {use_smote})\n{'='*80}")
    for res in results:
        print(f"\nWeight = {res['weight']}:")
        print(f"  DT  -> AUC: {res['DT_AUC']:.4f}, Sens: {res['DT_Sens']:.4f}, Spec: {res['DT_Spec']:.4f}, Prec: {res['DT_Prec']:.4f}, F1: {res['DT_F1']:.4f}")
        print(f"  RF  -> AUC: {res['RF_AUC']:.4f}, Sens: {res['RF_Sens']:.4f}, Spec: {res['RF_Spec']:.4f}, Prec: {res['RF_Prec']:.4f}, F1: {res['RF_F1']:.4f}")
        print(f"  XGB -> AUC: {res['XGB_AUC']:.4f}, Sens: {res['XGB_Sens']:.4f}, Spec: {res['XGB_Spec']:.4f}, Prec: {res['XGB_Prec']:.4f}, F1: {res['XGB_F1']:.4f}")

# ----------------------------------------------------------------------
# Define weight grid (start with a range; you can adjust)
weights_to_try = [4, 5,6,29,30,31,32,33,34,35,36,37,38,39, 40,41,42,43,44, 45,47,48,49,50]   # 34 was the SAS weight

# ----------------------------------------------------------------------
# Run for each year (with and without SMOTE)
for year, X, y in [('2016', X16, y16), ('2020', X20, y20), ('2024', X24, y24)]:
    evaluate_with_weights(X, y, year, weights_to_try, use_smote=False)
    # Optional: also run with SMOTE (uncomment next line)
    # evaluate_with_weights(X, y, year, weights_to_try, use_smote=True)

# ------------------------------------------------------------
# Train models only if sample size > 0
# ------------------------------------------------------------
def train_and_print(X, y, year):
    if len(X) == 0:
        print(f"{year}: No data to train.\n")
        return
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(zip(np.unique(y), class_weights))
    scale_pos_weight = len(y[y==0]) / len(y[y==1])
    
    dt = DecisionTreeClassifier(class_weight=class_weight_dict, max_depth=5, random_state=42)
    dt.fit(X, y)
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    rf.fit(X, y)
    xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, n_estimators=100, random_state=42)
    xgb.fit(X, y)
    
    print(f"{'='*50}\n{year} FEATURE IMPORTANCES\n{'='*50}")
    print("\nDecision Tree:")
    for col, imp in zip(X.columns, dt.feature_importances_):
        print(f"  {col}: {imp:.4f}")
    print("\nRandom Forest:")
    for col, imp in zip(X.columns, rf.feature_importances_):
        print(f"  {col}: {imp:.4f}")
    print("\nXGBoost:")
    for col, imp in zip(X.columns, xgb.feature_importances_):
        print(f"  {col}: {imp:.4f}")
    print()

train_and_print(X16, y16, 2016)
train_and_print(X20, y20, 2020)
train_and_print(X24, y24, 2024)

# ------------------------------------------------------------------
# SHAP analysis for all three years (full dataset)
# ------------------------------------------------------------------
import os
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

output_dir = r"C:\Users\Owner\Desktop\Data Science\Eastern\691\walters\viz"
os.makedirs(output_dir, exist_ok=True)

class_weight = {0: 1, 1: 5}   # minority weight = 5 (your chosen balanced weight)

for (X, y), year in zip([(X16, y16), (X20, y20), (X24, y24)], [2016, 2020, 2024]):
    print(f"\n{'='*60}\nSHAP analysis for {year} (n={len(X)} rows)\n{'='*60}")
    
    dt = DecisionTreeClassifier(class_weight=class_weight, max_depth=5, random_state=42)
    dt.fit(X, y)
    
    explainer = shap.TreeExplainer(dt)
    shap_values_obj = explainer.shap_values(X)
    
    # Robust extraction of class‑1 SHAP values
    if isinstance(shap_values_obj, list):
        shap_values_class1 = shap_values_obj[1]
    elif hasattr(shap_values_obj, 'values'):
        shap_values_class1 = shap_values_obj.values
        if shap_values_class1.ndim == 3:
            shap_values_class1 = shap_values_class1[:, :, 1]
    else:
        if shap_values_obj.ndim == 3:
            shap_values_class1 = shap_values_obj[:, :, 1]
        else:
            shap_values_class1 = shap_values_obj
    
    print(f"SHAP array shape: {shap_values_class1.shape}, expected ({len(X)}, {X.shape[1]})")
    
    # Save CSV
    shap_df = pd.DataFrame(shap_values_class1, columns=X.columns)
    csv_path = os.path.join(output_dir, f"shap_values_{year}.csv")
    shap_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Bar plot
    shap_abs_mean = np.abs(shap_values_class1).mean(axis=0)
    order = np.argsort(shap_abs_mean)[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(order)), shap_abs_mean[order], color='steelblue')
    plt.yticks(range(len(order)), X.columns[order])
    plt.xlabel('Mean |SHAP value|')
    plt.title(f'{year} – SHAP Bar Plot')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    bar_path = os.path.join(output_dir, f"shap_bar_{year}.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {bar_path}")
    
    # Summary dot plot (with colorbar)
    shap_long = pd.DataFrame(shap_values_class1, columns=X.columns).melt(var_name='feature', value_name='shap')
    X_long = X.melt(var_name='feature', value_name='value')
    combined = pd.concat([shap_long['shap'], X_long['value']], axis=1)
    combined['feature'] = shap_long['feature']
    
    feature_order = X.columns[order]
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = ax.scatter([], [], c=[], cmap='RdBu', vmin=combined['value'].min(), vmax=combined['value'].max())
    for i, feat in enumerate(feature_order):
        subset = combined[combined['feature'] == feat]
        y_jitter = np.random.normal(i, 0.08, len(subset))
        sc = ax.scatter(subset['shap'], y_jitter, c=subset['value'], cmap='RdBu', alpha=0.6, s=10, edgecolors='none')
    ax.set_yticks(range(len(feature_order)))
    ax.set_yticklabels(feature_order)
    ax.set_xlabel('SHAP value')
    ax.set_title(f'{year} – SHAP Summary (color = feature value)')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Feature value (low → high)')
    plt.tight_layout()
    sum_path = os.path.join(output_dir, f"shap_summary_{year}.png")
    plt.savefig(sum_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {sum_path}")
    
    # ----- Dependence plot for therm_blm (inside the loop) -----
    if 'therm_blm' in X.columns:
        idx = X.columns.get_loc('therm_blm')
        shap_therm = shap_values_class1[:, idx]
        
        # Find feature with highest absolute interaction with therm_blm
        interactions = {}
        for col in X.columns:
            if col != 'therm_blm':
                corr = np.corrcoef(X[col], shap_therm)[0, 1]
                interactions[col] = abs(corr)
        top_interact = max(interactions, key=interactions.get)
        print(f"Top interacting feature for therm_blm in {year}: {top_interact} (correlation={interactions[top_interact]:.3f})")
        
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(X['therm_blm'], shap_therm, c=X[top_interact], cmap='RdBu', alpha=0.6, edgecolors='none')
        plt.xlabel('therm_blm (BLM thermometer)')
        plt.ylabel('SHAP value for therm_blm')
        plt.title(f'{year} – Dependence on therm_blm\n(colored by {top_interact})')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        cbar = plt.colorbar(sc)
        cbar.set_label(top_interact)
        plt.tight_layout()
        dep_path = os.path.join(output_dir, f"shap_dep_therm_blm_{year}.png")
        plt.savefig(dep_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {dep_path}")  
#----------------------------------------------------

def evaluate_models(X, y, year):
    print(f"\n{'='*60}")
    print(f"{year} MODEL PERFORMANCE")
    print('='*60)
    
    # Split data (stratify to preserve class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Class weights for training (same as before)
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    # Train models
    dt = DecisionTreeClassifier(class_weight=class_weight_dict, max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    
    models = {'Decision Tree': dt, 'Random Forest': rf, 'XGBoost': xgb}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # probability of positive class (Trump support)
        
        auc = roc_auc_score(y_test, y_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        print(f"\n{name}:")
        print(f"  AUC: {auc:.4f}")
        print(f"  Sensitivity (Recall): {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Confusion Matrix: [[{tn}, {fp}], [{fn}, {tp}]]")
    
    # Optional: return the test sets if you want to do more analysis
    return X_train, X_test, y_train, y_test

# Run for each year (make sure X16, y16, etc. are defined)
print("\n" + "#"*80)
print("EVALUATING 2016 MODELS")
X_train16, X_test16, y_train16, y_test16 = evaluate_models(X16, y16, 2016)

print("\n" + "#"*80)
print("EVALUATING 2020 MODELS")
X_train20, X_test20, y_train20, y_test20 = evaluate_models(X20, y20, 2020)

print("\n" + "#"*80)
print("EVALUATING 2024 MODELS")
X_train24, X_test24, y_train24, y_test24 = evaluate_models(X24, y24, 2024)

####################################################################################
####################################################################################
####################################################################################

# Year-by-Year Variable Importance Tables

# Data from train_and_print
data = {
    "2016": {
        "Decision Tree": {
            "slavery_difficult": 0.0152, "work_way_up": 0.2273, "blacks_gotten_less": 0.0,
            "blacks_try_harder": 0.0667, "discrimination_blacks": 0.0041, "therm_blacks": 0.0292,
            "police_treat": 0.0, "police_howmuch": 0.0270, "therm_blm": 0.4625,
            "therm_undoc": 0.1466, "therm_hispanic": 0.0215
        },
        "Random Forest": {
            "slavery_difficult": 0.0624, "work_way_up": 0.1070, "blacks_gotten_less": 0.0759,
            "blacks_try_harder": 0.0995, "discrimination_blacks": 0.0610, "therm_blacks": 0.0867,
            "police_treat": 0.0068, "police_howmuch": 0.0529, "therm_blm": 0.1938,
            "therm_undoc": 0.1652, "therm_hispanic": 0.0886
        },
        "XGBoost": {
            "slavery_difficult": 0.0716, "work_way_up": 0.2262, "blacks_gotten_less": 0.0673,
            "blacks_try_harder": 0.0763, "discrimination_blacks": 0.0751, "therm_blacks": 0.0692,
            "police_treat": 0.0225, "police_howmuch": 0.0767, "therm_blm": 0.1582,
            "therm_undoc": 0.0876, "therm_hispanic": 0.0693
        }
    },
    "2020": {
        "Decision Tree": {
            "slavery_difficult": 0.0389, "work_way_up": 0.1072, "blacks_gotten_less": 0.0,
            "blacks_try_harder": 0.0100, "discrimination_blacks": 0.0044, "therm_blacks": 0.0226,
            "police_treat": 0.0, "police_howmuch": 0.0269, "therm_blm": 0.7512,
            "therm_undoc": 0.0330, "therm_hispanic": 0.0059
        },
        "Random Forest": {
            "slavery_difficult": 0.0882, "work_way_up": 0.1082, "blacks_gotten_less": 0.0907,
            "blacks_try_harder": 0.0795, "discrimination_blacks": 0.0465, "therm_blacks": 0.0576,
            "police_treat": 0.0037, "police_howmuch": 0.0625, "therm_blm": 0.2719,
            "therm_undoc": 0.1296, "therm_hispanic": 0.0615
        },
        "XGBoost": {
            "slavery_difficult": 0.0659, "work_way_up": 0.1224, "blacks_gotten_less": 0.0670,
            "blacks_try_harder": 0.0699, "discrimination_blacks": 0.0492, "therm_blacks": 0.0533,
            "police_treat": 0.0271, "police_howmuch": 0.0651, "therm_blm": 0.3588,
            "therm_undoc": 0.0656, "therm_hispanic": 0.0557
        }
    },
    "2024": {
        "Decision Tree": {
            "slavery_difficult": 0.0549, "work_way_up": 0.1975, "blacks_gotten_less": 0.0083,
            "blacks_try_harder": 0.0787, "discrimination_blacks": 0.0161, "therm_blacks": 0.0148,
            "police_treat": 0.0023, "police_howmuch": 0.0077, "therm_blm": 0.4764,
            "therm_undoc": 0.1380, "therm_hispanic": 0.0054
        },
        "Random Forest": {
            "slavery_difficult": 0.0889, "work_way_up": 0.1101, "blacks_gotten_less": 0.0664,
            "blacks_try_harder": 0.1175, "discrimination_blacks": 0.0604, "therm_blacks": 0.0689,
            "police_treat": 0.0064, "police_howmuch": 0.0450, "therm_blm": 0.1791,
            "therm_undoc": 0.1862, "therm_hispanic": 0.0711
        },
        "XGBoost": {
            "slavery_difficult": 0.0738, "work_way_up": 0.2042, "blacks_gotten_less": 0.0573,
            "blacks_try_harder": 0.1240, "discrimination_blacks": 0.0597, "therm_blacks": 0.0587,
            "police_treat": 0.0437, "police_howmuch": 0.0588, "therm_blm": 0.1631,
            "therm_undoc": 0.0980, "therm_hispanic": 0.0587
        }
    }
}

# Feature order (consistent across years)
features = [
    'slavery_difficult', 'work_way_up', 'blacks_gotten_less', 'blacks_try_harder',
    'discrimination_blacks', 'therm_blacks', 'police_treat', 'police_howmuch',
    'therm_blm', 'therm_undoc', 'therm_hispanic'
]

# Pretty feature names for display
pretty_names = {
    'slavery_difficult': 'Slavery makes it difficult',
    'work_way_up': 'Work way up (no favors)',
    'blacks_gotten_less': 'Blacks got less than deserve',
    'blacks_try_harder': 'Blacks must try harder',
    'discrimination_blacks': 'Discrimination against Blacks',
    'therm_blacks': 'Thermometer: Blacks',
    'police_treat': 'Police treat (better)',
    'police_howmuch': 'Police treat (how much)',
    'therm_blm': 'Thermometer: BLM',
    'therm_undoc': 'Thermometer: Undoc immigrants',
    'therm_hispanic': 'Thermometer: Hispanics'
}




# Table 1: Top 3 Features (no wrapping needed, it's short)
data_top = [
    ['2016', 'Decision Tree', 'therm_blm (0.459), work_way_up (0.228), therm_undoc (0.163)'],
    ['2020', 'Decision Tree', 'therm_blm (0.773), work_way_up (0.109), therm_undoc (0.048)'],
    ['2024', 'Decision Tree', 'therm_blm (0.479), work_way_up (0.200), therm_undoc (0.146)']
]

fig, ax = plt.subplots(figsize=(10, 2.5))
ax.axis('off')
table = ax.table(cellText=data_top, colLabels=['Year', 'Model', 'Top 3 Features (Importance)'],
                 loc='center', cellLoc='left', colWidths=[0.15, 0.2, 0.65])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Style header
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    cell.set_edgecolor('#dddddd')
plt.savefig('top3_table.png', dpi=300, bbox_inches='tight')
print("Saved: top3_table.png")

# Table 2: Walters/Atwater Mapping (with manual line breaks)
data_map = [
    ['BWEqulJust_7 (criminal justice fairness)', 'therm_blm', 
     'BLM movement became the new symbol for\npolice brutality and systemic racism.'],
    ['BWEqulOppty (racial equality)', 'work_way_up', 
     'The classic racial resentment item about\nworking without special favors persists.'],
    ['RateUnDoc_100 (undocumented immigrants)', 'therm_undoc', 
     'Xenophobia toward undocumented immigrants\nremains a stable, separate proxy.'],
    ['RateLatino_100 (Latinos)', 'therm_hispanic', 
     'Weaker but still present.']
]

fig2, ax2 = plt.subplots(figsize=(12, 4))  # increased height
ax2.axis('off')
table2 = ax2.table(cellText=data_map, colLabels=['2012 (Tea Party)', '2016‑2024 (MAGA)', 'Interpretation'],
                   loc='center', cellLoc='left', colWidths=[0.35, 0.2, 0.45])
table2.auto_set_font_size(False)
table2.set_fontsize(9)
table2.scale(1, 1.8)  # increase row height to accommodate wrapped lines

for (row, col), cell in table2.get_celld().items():
    if row == 0:
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    cell.set_edgecolor('#dddddd')
plt.savefig('mapping_table.png', dpi=300, bbox_inches='tight')
print("Saved: mapping_table.png")

# ------------------------------------------------------------
# Table 3: Full feature importances grid (your existing code)
# ------------------------------------------------------------


# Create figure with subplots: 3 rows (years) × 3 cols (models)
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
fig.suptitle('Feature Importances by Year and Model', fontsize=16, fontweight='bold')

years = ['2016', '2020', '2024']
models = ['Decision Tree', 'Random Forest', 'XGBoost']

for i, year in enumerate(years):
    for j, model in enumerate(models):
        ax = axes[i, j]
        # Extract values for this year+model
        vals = [data[year][model][f] for f in features]
        # Normalize importance values (0 to 0.8 instead of 0 to 1) to keep colors lighter
        max_val = max(vals) if max(vals) > 0 else 1
        norm_vals = np.array(vals) / max_val * 0.8   # maximum color intensity is 80% instead of 100%
        
        # Create table as before
        table_data = [[pretty_names[f], f"{v:.4f}"] for f, v in zip(features, vals)]
        table = ax.table(cellText=table_data, colLabels=['Feature', 'Importance'],
                         loc='center', cellLoc='left', colWidths=[0.7, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        
        # Color cells and adjust text color for dark backgrounds
        for k, (feat, val) in enumerate(zip(features, vals)):
            cell_color = plt.cm.Blues(norm_vals[k])
            table[(k+1, 0)].set_facecolor(cell_color)
            table[(k+1, 1)].set_facecolor(cell_color)
            
            # If the normalized value > 0.5 (darker than medium blue), use white text
            if norm_vals[k] > 0.5:
                table[(k+1, 0)].get_text().set_color('white')
                table[(k+1, 1)].get_text().set_color('white')
            else:
                table[(k+1, 0)].get_text().set_color('black')
                table[(k+1, 1)].get_text().set_color('black')
        
        # Style header (unchanged, already white text on dark blue)
        for col in range(2):
            table[(0, col)].set_facecolor('#4472C4')
            table[(0, col)].set_text_props(color='white', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importances_appendix.jpg'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'feature_importances_appendix.png'), dpi=300, bbox_inches='tight')
print(f"Images saved to {output_dir}: feature_importances_appendix.jpg / .png")

