import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

# MUST be the first Streamlit command
st.set_page_config(page_title="Proxy Politics Dashboard", layout="wide")

# ------------------------------------------------------------
# Load data – absolute path to pickle files
# ------------------------------------------------------------
@st.cache_data
def load_data():
    base = r"C:\Users\Owner\Desktop\Data Science\Eastern\691\walters\code"
    with open(os.path.join(base, 'data_2016.pkl'), 'rb') as f:
        X16, y16 = pickle.load(f)
    with open(os.path.join(base, 'data_2020.pkl'), 'rb') as f:
        X20, y20 = pickle.load(f)
    with open(os.path.join(base, 'data_2024.pkl'), 'rb') as f:
        X24, y24 = pickle.load(f)
    return (X16, y16), (X20, y20), (X24, y24)

data_dict = {2016: load_data()[0], 2020: load_data()[1], 2024: load_data()[2]}

# ------------------------------------------------------------
# Helper: train and evaluate with a given weight
# ------------------------------------------------------------
def train_eval(X, y, weight, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    dt = DecisionTreeClassifier(class_weight={0:1, 1:weight}, max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    y_proba = dt.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    return dt, auc, sensitivity, specificity, precision, f1, X_test, y_test

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title("📊 Proxy Politics: From Tea Party to MAGA")
st.markdown("Explore how different minority class weights affect decision tree performance and feature importance.")

st.sidebar.header("⚙️ Model Parameters")
year = st.sidebar.selectbox("Select year", [2016, 2020, 2024])
weight = st.sidebar.slider("Minority class weight (Trump supporters)", min_value=1, max_value=50, value=5, step=1)

X, y = data_dict[year]
dt, auc, sens, spec, prec, f1, X_test, y_test = train_eval(X, y, weight)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("AUC", f"{auc:.3f}")
col2.metric("Sensitivity", f"{sens:.3f}")
col3.metric("Specificity", f"{spec:.3f}")
col4.metric("Precision", f"{prec:.3f}")
col5.metric("F1 Score", f"{f1:.3f}")

st.subheader(f"📈 Feature Importances (Decision Tree, weight={weight})")
importances = dt.feature_importances_
features = X.columns
order = np.argsort(importances)[::-1]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(range(len(order)), importances[order], color='steelblue')
ax.set_yticks(range(len(order)))
ax.set_yticklabels(features[order])
ax.set_xlabel('Importance')
ax.invert_yaxis()
st.pyplot(fig)

st.subheader("🌳 Decision Tree Rules (first 2 levels)")
def get_rules(dt, feature_names, max_depth=2):
    rules = []
    def recurse(node, depth, condition):
        if depth > max_depth:
            return
        if dt.tree_.children_left[node] == dt.tree_.children_right[node]:
            proba = dt.tree_.value[node][0][1] / dt.tree_.value[node][0].sum()
            rules.append(f"{condition} → P(Trump) = {proba:.2f}")
        else:
            feat = feature_names[dt.tree_.feature[node]]
            thr = dt.tree_.threshold[node]
            recurse(dt.tree_.children_left[node], depth+1, f"{condition} & ({feat} ≤ {thr:.2f})")
            recurse(dt.tree_.children_right[node], depth+1, f"{condition} & ({feat} > {thr:.2f})")
    recurse(0, 0, "Root")
    return rules
rules = get_rules(dt, features, max_depth=2)
st.text("\n".join(rules[:15]))

st.subheader("📊 Compare Feature Importances Across Years")
if st.button("Refresh comparison"):
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, yr in enumerate([2016, 2020, 2024]):
        X_yr, y_yr = data_dict[yr]
        dt_yr, _, _, _, _, _, _, _ = train_eval(X_yr, y_yr, weight)
        imp = dt_yr.feature_importances_
        order_yr = np.argsort(imp)[::-1]
        axes[i].barh(range(len(order_yr)), imp[order_yr], color='steelblue')
        axes[i].set_yticks(range(len(order_yr)))
        axes[i].set_yticklabels(X_yr.columns[order_yr])
        axes[i].set_title(str(yr))
        axes[i].invert_yaxis()
        axes[i].set_xlabel('Importance')
    plt.tight_layout()
    st.pyplot(fig2)

st.markdown("---")
st.caption("Data: ANES 2016‑2024 | Model: Decision Tree (max_depth=5) | Dashboard built with Streamlit")