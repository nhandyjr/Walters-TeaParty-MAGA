# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 01:24:01 2026

@author: Owner
"""
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 22:19:35 2026
@author: nhandyjr@gmail.com

Full analysis: ANES 2016-2024, predicting Trump support.
Includes attitude analysis and demographic extension.

"""
############################
# Import Relevent Packages #
############################

import pandas as pd
import numpy as np
import re
import os
import pickle
import warnings
import matplotlib.pyplot as plt
import shap

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# Paths and output directory
# ------------------------------------------------------------------
print(os.getcwd())
output_dir = r"C:\Users\Owner\Desktop\Data Science\Eastern\691\walters\viz"
os.makedirs(output_dir, exist_ok=True)

path_2016 = r'C:\Users\Owner\Desktop\Data Science\Eastern\691\walters\surveys\anes_timeseries_2016_dta\anes_timeseries_2016.dta'
path_2020 = r'C:\Users\Owner\Desktop\Data Science\Eastern\691\walters\surveys\anes_timeseries_2020_csv_20220210\anes_timeseries_2020_csv_20220210.csv'
path_2024 = r'C:\Users\Owner\Desktop\Data Science\Eastern\691\walters\surveys\anes_timeseries_2024_csv_20260519\anes_timeseries_2024_csv_20260519.csv'

# ------------------------------------------------------------------
# Cleaning functions (attitude proxies)
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Load data and construct attitude DataFrames
# ------------------------------------------------------------------
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
print(f"2016 sample size: {len(X16)} | Trump support rate: {y16.mean():.2%}\n")

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
print(f"2020 sample size: {len(X20)} | Trump support rate: {y20.mean():.2%}\n")

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
print(f"2024 sample size: {len(X24)} | Trump support rate: {y24.mean():.2%}\n")

# ------------------------------------------------------------------
# Drop low‑importance features (police variables, discrimination_blacks)
# ------------------------------------------------------------------
low_imp_features = ['police_treat', 'police_howmuch', 'discrimination_blacks']
X16 = X16.drop(columns=low_imp_features, errors='ignore')
X20 = X20.drop(columns=low_imp_features, errors='ignore')
X24 = X24.drop(columns=low_imp_features, errors='ignore')

# ------------------------------------------------------------------
# Save cleaned attitude data for Streamlit dashboard
# ------------------------------------------------------------------
with open(os.path.join(output_dir, 'data_2016.pkl'), 'wb') as f:
    pickle.dump((X16, y16), f)
with open(os.path.join(output_dir, 'data_2020.pkl'), 'wb') as f:
    pickle.dump((X20, y20), f)
with open(os.path.join(output_dir, 'data_2024.pkl'), 'wb') as f:
    pickle.dump((X24, y24), f)
print("Saved cleaned attitude data to viz folder.\n")

# ==================================================================
# ORIGINAL ANALYSIS (attitudes only)
# ==================================================================

#######################################################
# Weight grid (Decision Tree, Random Forest, XGBoost) #
#######################################################
def evaluate_with_weights(X, y, year, weight_list, use_smote=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if use_smote:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
    results = []
    for w in weight_list:
        dt = DecisionTreeClassifier(class_weight={0:1, 1:w}, max_depth=5, random_state=42)
        dt.fit(X_train, y_train)
        y_proba_dt = dt.predict_proba(X_test)[:,1]
        auc_dt = roc_auc_score(y_test, y_proba_dt)
        tn, fp, fn, tp = confusion_matrix(y_test, dt.predict(X_test)).ravel()
        sens_dt = tp/(tp+fn) if (tp+fn)>0 else 0
        spec_dt = tn/(tn+fp) if (tn+fp)>0 else 0
        prec_dt = tp/(tp+fp) if (tp+fp)>0 else 0
        f1_dt = 2*prec_dt*sens_dt/(prec_dt+sens_dt) if (prec_dt+sens_dt)>0 else 0

        rf = RandomForestClassifier(class_weight={0:1, 1:w}, n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_proba_rf = rf.predict_proba(X_test)[:,1]
        auc_rf = roc_auc_score(y_test, y_proba_rf)
        tn, fp, fn, tp = confusion_matrix(y_test, rf.predict(X_test)).ravel()
        sens_rf = tp/(tp+fn) if (tp+fn)>0 else 0
        spec_rf = tn/(tn+fp) if (tn+fp)>0 else 0
        prec_rf = tp/(tp+fp) if (tp+fp)>0 else 0
        f1_rf = 2*prec_rf*sens_rf/(prec_rf+sens_rf) if (prec_rf+sens_rf)>0 else 0

        base_scale = len(y_train[y_train==0]) / len(y_train[y_train==1])
        scale = base_scale * w
        xgb = XGBClassifier(scale_pos_weight=scale, n_estimators=100, random_state=42)
        xgb.fit(X_train, y_train)
        y_proba_xgb = xgb.predict_proba(X_test)[:,1]
        auc_xgb = roc_auc_score(y_test, y_proba_xgb)
        tn, fp, fn, tp = confusion_matrix(y_test, xgb.predict(X_test)).ravel()
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
    print(f"\n{'='*80}\n{year} – WEIGHT GRID (SMOTE = {use_smote})\n{'='*80}")
    for res in results:
        print(f"\nWeight = {res['weight']}:")
        print(f"  DT  -> AUC: {res['DT_AUC']:.4f}, Sens: {res['DT_Sens']:.4f}, Spec: {res['DT_Spec']:.4f}, Prec: {res['DT_Prec']:.4f}, F1: {res['DT_F1']:.4f}")
        print(f"  RF  -> AUC: {res['RF_AUC']:.4f}, Sens: {res['RF_Sens']:.4f}, Spec: {res['RF_Spec']:.4f}, Prec: {res['RF_Prec']:.4f}, F1: {res['RF_F1']:.4f}")
        print(f"  XGB -> AUC: {res['XGB_AUC']:.4f}, Sens: {res['XGB_Sens']:.4f}, Spec: {res['XGB_Spec']:.4f}, Prec: {res['XGB_Prec']:.4f}, F1: {res['XGB_F1']:.4f}")

weights_to_try = [4,5,6,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,49,50]
for year, X, y in [('2016', X16, y16), ('2020', X20, y20), ('2024', X24, y24)]:
    evaluate_with_weights(X, y, year, weights_to_try, use_smote=False)

# ------------------------------------------------------------------
# Feature importances (full training, no test split)
# ------------------------------------------------------------------
def train_and_print(X, y, year):
    if len(X) == 0:
        print(f"{year}: No data to train.\n"); return
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
# SHAP analysis (full dataset, for interpretation)
# ------------------------------------------------------------------
class_weight_shap = {0: 1, 1: 5}   # weight=5
for (X, y), year in zip([(X16, y16), (X20, y20), (X24, y24)], [2016, 2020, 2024]):
    print(f"\n{'='*60}\nSHAP analysis for {year} (n={len(X)} rows)\n{'='*60}")
    dt = DecisionTreeClassifier(class_weight=class_weight_shap, max_depth=5, random_state=42)
    dt.fit(X, y)
    explainer = shap.TreeExplainer(dt)
    shap_values_obj = explainer.shap_values(X)
    # Extract class 1 SHAP values
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
    # Save CSV
    pd.DataFrame(shap_values_class1, columns=X.columns).to_csv(os.path.join(output_dir, f"shap_values_{year}.csv"), index=False)
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
    plt.savefig(os.path.join(output_dir, f"shap_bar_{year}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    # Summary dot plot (simplified)
    shap_long = pd.DataFrame(shap_values_class1, columns=X.columns).melt(var_name='feature', value_name='shap')
    X_long = X.melt(var_name='feature', value_name='value')
    combined = pd.concat([shap_long['shap'], X_long['value']], axis=1)
    combined['feature'] = shap_long['feature']
    feature_order = X.columns[order]
    fig, ax = plt.subplots(figsize=(12, 8))
    sc = None
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
    plt.savefig(os.path.join(output_dir, f"shap_summary_{year}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    # Dependence plot for therm_blm
    if 'therm_blm' in X.columns:
        idx = X.columns.get_loc('therm_blm')
        shap_therm = shap_values_class1[:, idx]
        interactions = {}
        for col in X.columns:
            if col != 'therm_blm':
                corr = np.corrcoef(X[col], shap_therm)[0, 1]
                interactions[col] = abs(corr)
        top_interact = max(interactions, key=interactions.get)
        print(f"Top interacting feature for {year}: {top_interact} (corr={interactions[top_interact]:.3f})")
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(X['therm_blm'], shap_therm, c=X[top_interact], cmap='RdBu', alpha=0.6, edgecolors='none')
        plt.xlabel('therm_blm (BLM thermometer)')
        plt.ylabel('SHAP value for therm_blm')
        plt.title(f'{year} – Dependence on therm_blm\n(colored by {top_interact})')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        cbar = plt.colorbar(sc)
        cbar.set_label(top_interact)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_dep_therm_blm_{year}.png"), dpi=300, bbox_inches='tight')
        plt.close()

# ------------------------------------------------------------------
# Model performance on 20% holdout (attitudes only)
# ------------------------------------------------------------------
def evaluate_models(X, y, year):
    print(f"\n{'='*60}\n{year} MODEL PERFORMANCE (20% holdout)\n{'='*60}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    dt = DecisionTreeClassifier(class_weight=class_weight_dict, max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    models = {'Decision Tree': dt, 'Random Forest': rf, 'XGBoost': xgb}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        f1 = 2*prec*sens/(prec+sens) if (prec+sens)>0 else 0
        print(f"\n{name}: AUC={auc:.4f}, Sens={sens:.4f}, Spec={spec:.4f}, Prec={prec:.4f}, F1={f1:.4f}")
        print(f"  Confusion Matrix: [[{tn}, {fp}], [{fn}, {tp}]]")

evaluate_models(X16, y16, 2016)
evaluate_models(X20, y20, 2020)
evaluate_models(X24, y24, 2024)

# ------------------------------------------------------------------
# Generate publication tables (top3, mapping, full grid)
# ------------------------------------------------------------------
data = {
    "2016": {
        "Decision Tree": {"slavery_difficult":0.0152, "work_way_up":0.2273, "blacks_gotten_less":0.0, "blacks_try_harder":0.0667, "discrimination_blacks":0.0041, "therm_blacks":0.0292, "police_treat":0.0, "police_howmuch":0.0270, "therm_blm":0.4625, "therm_undoc":0.1466, "therm_hispanic":0.0215},
        "Random Forest": {"slavery_difficult":0.0624, "work_way_up":0.1070, "blacks_gotten_less":0.0759, "blacks_try_harder":0.0995, "discrimination_blacks":0.0610, "therm_blacks":0.0867, "police_treat":0.0068, "police_howmuch":0.0529, "therm_blm":0.1938, "therm_undoc":0.1652, "therm_hispanic":0.0886},
        "XGBoost": {"slavery_difficult":0.0716, "work_way_up":0.2262, "blacks_gotten_less":0.0673, "blacks_try_harder":0.0763, "discrimination_blacks":0.0751, "therm_blacks":0.0692, "police_treat":0.0225, "police_howmuch":0.0767, "therm_blm":0.1582, "therm_undoc":0.0876, "therm_hispanic":0.0693}
    },
    "2020": {
        "Decision Tree": {"slavery_difficult":0.0389, "work_way_up":0.1072, "blacks_gotten_less":0.0, "blacks_try_harder":0.0100, "discrimination_blacks":0.0044, "therm_blacks":0.0226, "police_treat":0.0, "police_howmuch":0.0269, "therm_blm":0.7512, "therm_undoc":0.0330, "therm_hispanic":0.0059},
        "Random Forest": {"slavery_difficult":0.0882, "work_way_up":0.1082, "blacks_gotten_less":0.0907, "blacks_try_harder":0.0795, "discrimination_blacks":0.0465, "therm_blacks":0.0576, "police_treat":0.0037, "police_howmuch":0.0625, "therm_blm":0.2719, "therm_undoc":0.1296, "therm_hispanic":0.0615},
        "XGBoost": {"slavery_difficult":0.0659, "work_way_up":0.1224, "blacks_gotten_less":0.0670, "blacks_try_harder":0.0699, "discrimination_blacks":0.0492, "therm_blacks":0.0533, "police_treat":0.0271, "police_howmuch":0.0651, "therm_blm":0.3588, "therm_undoc":0.0656, "therm_hispanic":0.0557}
    },
    "2024": {
        "Decision Tree": {"slavery_difficult":0.0549, "work_way_up":0.1975, "blacks_gotten_less":0.0083, "blacks_try_harder":0.0787, "discrimination_blacks":0.0161, "therm_blacks":0.0148, "police_treat":0.0023, "police_howmuch":0.0077, "therm_blm":0.4764, "therm_undoc":0.1380, "therm_hispanic":0.0054},
        "Random Forest": {"slavery_difficult":0.0889, "work_way_up":0.1101, "blacks_gotten_less":0.0664, "blacks_try_harder":0.1175, "discrimination_blacks":0.0604, "therm_blacks":0.0689, "police_treat":0.0064, "police_howmuch":0.0450, "therm_blm":0.1791, "therm_undoc":0.1862, "therm_hispanic":0.0711},
        "XGBoost": {"slavery_difficult":0.0738, "work_way_up":0.2042, "blacks_gotten_less":0.0573, "blacks_try_harder":0.1240, "discrimination_blacks":0.0597, "therm_blacks":0.0587, "police_treat":0.0437, "police_howmuch":0.0588, "therm_blm":0.1631, "therm_undoc":0.0980, "therm_hispanic":0.0587}
    }
}
features = ['slavery_difficult', 'work_way_up', 'blacks_gotten_less', 'blacks_try_harder', 'discrimination_blacks', 'therm_blacks', 'police_treat', 'police_howmuch', 'therm_blm', 'therm_undoc', 'therm_hispanic']
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
# Top 3 table
data_top = [
    ['2016', 'Decision Tree', 'therm_blm (0.459), work_way_up (0.228), therm_undoc (0.163)'],
    ['2020', 'Decision Tree', 'therm_blm (0.773), work_way_up (0.109), therm_undoc (0.048)'],
    ['2024', 'Decision Tree', 'therm_blm (0.479), work_way_up (0.200), therm_undoc (0.146)']
]
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.axis('off')
table_top = ax.table(cellText=data_top, colLabels=['Year', 'Model', 'Top 3 Features (Importance)'], loc='center', cellLoc='left', colWidths=[0.15,0.2,0.65])
table_top.auto_set_font_size(False)
table_top.set_fontsize(10)
table_top.scale(1,1.5)
for (row,col), cell in table_top.get_celld().items():
    if row==0:
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    cell.set_edgecolor('#dddddd')
plt.savefig(os.path.join(output_dir, 'top3_table.png'), dpi=300, bbox_inches='tight')
print("Saved top3_table.png")
# Mapping table
data_map = [
    ['BWEqulJust_7 (criminal justice fairness)', 'therm_blm', 'BLM movement became the new symbol for\npolice brutality and systemic racism.'],
    ['BWEqulOppty (racial equality)', 'work_way_up', 'The classic racial resentment item about\nworking without special favors persists.'],
    ['RateUnDoc_100 (undocumented immigrants)', 'therm_undoc', 'Xenophobia toward undocumented immigrants\nremains a stable, separate proxy.'],
    ['RateLatino_100 (Latinos)', 'therm_hispanic', 'Weaker but still present.']
]
fig2, ax2 = plt.subplots(figsize=(12,4))
ax2.axis('off')
table_map = ax2.table(cellText=data_map, colLabels=['2012 (Tea Party)', '2016‑2024 (MAGA)', 'Interpretation'], loc='center', cellLoc='left', colWidths=[0.35,0.2,0.45])
table_map.auto_set_font_size(False)
table_map.set_fontsize(9)
table_map.scale(1,1.8)
for (row,col), cell in table_map.get_celld().items():
    if row==0:
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    cell.set_edgecolor('#dddddd')
plt.savefig(os.path.join(output_dir, 'mapping_table.png'), dpi=300, bbox_inches='tight')
print("Saved mapping_table.png")
# Full grid
fig_grid, axes = plt.subplots(3,3, figsize=(15,12))
fig_grid.suptitle('Feature Importances by Year and Model', fontsize=16, fontweight='bold')
years = ['2016','2020','2024']
models = ['Decision Tree','Random Forest','XGBoost']
for i, year in enumerate(years):
    for j, model in enumerate(models):
        ax = axes[i,j]
        vals = [data[year][model][f] for f in features]
        max_val = max(vals) if max(vals)>0 else 1
        norm_vals = np.array(vals)/max_val * 0.8
        table_data = [[pretty_names[f], f"{v:.4f}"] for f,v in zip(features,vals)]
        table_cell = ax.table(cellText=table_data, colLabels=['Feature','Importance'], loc='center', cellLoc='left', colWidths=[0.7,0.2])
        table_cell.auto_set_font_size(False)
        table_cell.set_fontsize(9)
        for k, (feat,val) in enumerate(zip(features,vals)):
            cell_color = plt.cm.Blues(norm_vals[k])
            table_cell[(k+1,0)].set_facecolor(cell_color)
            table_cell[(k+1,1)].set_facecolor(cell_color)
            if norm_vals[k] > 0.5:
                table_cell[(k+1,0)].get_text().set_color('white')
                table_cell[(k+1,1)].get_text().set_color('white')
            else:
                table_cell[(k+1,0)].get_text().set_color('black')
                table_cell[(k+1,1)].get_text().set_color('black')
        for col in range(2):
            table_cell[(0,col)].set_facecolor('#4472C4')
            table_cell[(0,col)].set_text_props(color='white', fontweight='bold')
        ax.axis('off')
        ax.set_title(f'{year} – {model}', fontsize=12, pad=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importances_appendix.jpg'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'feature_importances_appendix.png'), dpi=300, bbox_inches='tight')
print(f"Images saved to {output_dir}: feature_importances_appendix.jpg/.png")

# ==================================================================
# DEMOGRAPHIC EXTENSION (validated with 20% holdout)
# ==================================================================

def clean_demo(series):
    """Convert ANES demographic categorical variables to numeric."""
    if series.dtype in ['int64', 'float64']:
        return series.where(series >= 0, np.nan)
    str_vals = series.astype(str)
    extracted = str_vals.str.extract(r'^(-?\d+)')[0]
    numeric = pd.to_numeric(extracted, errors='coerce')
    return numeric.where(numeric >= 0, np.nan)

# 2016 demographics (drop income because too sparse)
demo_16 = pd.DataFrame({
    'age': clean_demo(df16['V161267']),
    'education': clean_demo(df16['V161270']),
    'female': (clean_demo(df16['V161342']) == 2).astype(int),
    'white': (clean_demo(df16['V161310x']) == 1).astype(int),
    'party_id': clean_demo(df16['V161158x']),
})
X16_demo = pd.concat([X16, demo_16], axis=1)
valid16 = X16_demo.notna().all(axis=1) & y16.notna()
X16_demo = X16_demo[valid16]
y16_demo = y16[valid16]
print(f"2016 with demographics: n={len(X16_demo)}")

# 2020 demographics (all work)
demo_20 = pd.DataFrame({
    'age': clean_demo(df20['V201507x']),
    'education': clean_demo(df20['V201510']),
    'income': clean_demo(df20['V201617x']),
    'female': (clean_demo(df20['V202637']) == 2).astype(int),
    'white': (clean_demo(df20['V201549x']) == 1).astype(int),
    'party_id': clean_demo(df20['V201231x']),
    'rural': (clean_demo(df20['V202355']) == 1).astype(int),
})
X20_demo = pd.concat([X20, demo_20], axis=1)
valid20 = X20_demo.notna().all(axis=1) & y20.notna()
X20_demo = X20_demo[valid20]
y20_demo = y20[valid20]
print(f"2020 with demographics: n={len(X20_demo)}")

# 2024 demographics (only age, gender, race, rural – party ID too sparse)
demo_24 = pd.DataFrame({
    'age': clean_demo(df24['V241458x']),
    'female': (clean_demo(df24['V241551']) == 2).astype(int),
    'white': (clean_demo(df24['V241501x']) == 1).astype(int),
    'rural': (clean_demo(df24['V242341']) == 4).astype(int),
})
X24_demo = pd.concat([X24, demo_24], axis=1)
valid24 = X24_demo.notna().all(axis=1) & y24.notna()
X24_demo = X24_demo[valid24]
y24_demo = y24[valid24]
print(f"2024 with demographics: n={len(X24_demo)}")

# Save demographic DataFrames for Streamlit
with open(os.path.join(output_dir, 'data_2016_demo.pkl'), 'wb') as f:
    pickle.dump((X16_demo, y16_demo), f)
with open(os.path.join(output_dir, 'data_2020_demo.pkl'), 'wb') as f:
    pickle.dump((X20_demo, y20_demo), f)
with open(os.path.join(output_dir, 'data_2024_demo.pkl'), 'wb') as f:
    pickle.dump((X24_demo, y24_demo), f)
print("Saved demographic data to viz folder.")


def compare_demographics_validated(X_att, y_att, X_demo, y_demo, year, weight=5, test_size=0.2):
    print(f"\n{'='*60}\n{year} – Demographic comparison (20% holdout, weight={weight})\n{'='*60}")
    # Split attitudes-only
    X_att_train, X_att_test, y_att_train, y_att_test = train_test_split(
        X_att, y_att, test_size=test_size, random_state=42, stratify=y_att)
    # Split demographics data
    X_demo_train, X_demo_test, y_demo_train, y_demo_test = train_test_split(
        X_demo, y_demo, test_size=test_size, random_state=42, stratify=y_demo)
    
    def metrics(model, X_test, y_test, name):
        y_proba = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_proba)
        y_pred = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        print(f"{name}: AUC={auc:.3f}, Sens={sens:.3f}, Spec={spec:.3f}")
    
    # Attitudes only
    dt_att = DecisionTreeClassifier(class_weight={0:1,1:weight}, max_depth=5, random_state=42)
    dt_att.fit(X_att_train, y_att_train)
    rf_att = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    rf_att.fit(X_att_train, y_att_train)
    xgb_att = XGBClassifier(scale_pos_weight=weight, n_estimators=100, random_state=42)
    xgb_att.fit(X_att_train, y_att_train)
    
    # Attitudes + demographics
    dt_demo = DecisionTreeClassifier(class_weight={0:1,1:weight}, max_depth=5, random_state=42)
    dt_demo.fit(X_demo_train, y_demo_train)
    rf_demo = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    rf_demo.fit(X_demo_train, y_demo_train)
    xgb_demo = XGBClassifier(scale_pos_weight=weight, n_estimators=100, random_state=42)
    xgb_demo.fit(X_demo_train, y_demo_train)
    
    print("\n**Attitudes only (test set)**")
    metrics(dt_att, X_att_test, y_att_test, "DT")
    metrics(rf_att, X_att_test, y_att_test, "RF")
    metrics(xgb_att, X_att_test, y_att_test, "XGB")
    
    print("\n**Attitudes + Demographics (test set)**")
    metrics(dt_demo, X_demo_test, y_demo_test, "DT")
    metrics(rf_demo, X_demo_test, y_demo_test, "RF")
    metrics(xgb_demo, X_demo_test, y_demo_test, "XGB")
    
    # Feature importances from full data (for interpretation)
    dt_full = DecisionTreeClassifier(class_weight={0:1,1:weight}, max_depth=5, random_state=42)
    dt_full.fit(X_demo, y_demo)
    print("\n**Decision Tree feature importances (with demographics, full data)**")
    for col, imp in zip(X_demo.columns, dt_full.feature_importances_):
        if imp > 0.01:
            print(f"  {col}: {imp:.4f}")

# Run demographic comparisons
if len(X16_demo) > 0:
    compare_demographics_validated(X16, y16, X16_demo, y16_demo, 2016, weight=5)
if len(X20_demo) > 0:
    compare_demographics_validated(X20, y20, X20_demo, y20_demo, 2020, weight=5)
if len(X24_demo) > 0:
    compare_demographics_validated(X24, y24, X24_demo, y24_demo, 2024, weight=5)

print("\nAll analysis completed.")