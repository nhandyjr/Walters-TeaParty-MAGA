# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 01:43:38 2026

@author: Norman Handy 
        nhandyjr@gmail.com
"""

############################
# Import Relevent Packages #
############################
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

st.set_page_config(page_title="Proxy Politics Dashboard", layout="wide")
st.title("📊 Proxy Politics: From Tea Party to MAGA")
st.markdown("Explore how different minority class weights, features, and demographics affect Trump support predictions.")

# ------------------------------------------------------------
# Load data (attitudes only and with demographics)
# ------------------------------------------------------------
#output_dir = r"C:\Users\Owner\Desktop\Data Science\Eastern\691\walters\viz"

@st.cache_data
def load_data():
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Attitudes only (pickles are in the same folder as the script)
    with open(os.path.join(current_dir, 'data_2016.pkl'), 'rb') as f:
        X16_att, y16_att = pickle.load(f)
    with open(os.path.join(current_dir, 'data_2020.pkl'), 'rb') as f:
        X20_att, y20_att = pickle.load(f)
    with open(os.path.join(current_dir, 'data_2024.pkl'), 'rb') as f:
        X24_att, y24_att = pickle.load(f)
    
    # Demographics (if available)
    has_demo = False
    try:
        with open(os.path.join(current_dir, 'data_2016_demo.pkl'), 'rb') as f:
            X16_demo, y16_demo = pickle.load(f)
        with open(os.path.join(current_dir, 'data_2020_demo.pkl'), 'rb') as f:
            X20_demo, y20_demo = pickle.load(f)
        with open(os.path.join(current_dir, 'data_2024_demo.pkl'), 'rb') as f:
            X24_demo, y24_demo = pickle.load(f)
        has_demo = True
    except:
        # If demo pickles not found, use attitudes only
        X16_demo, y16_demo = X16_att, y16_att
        X20_demo, y20_demo = X20_att, y20_att
        X24_demo, y24_demo = X24_att, y24_att
    
    return (X16_att, y16_att), (X20_att, y20_att), (X24_att, y24_att), \
           (X16_demo, y16_demo), (X20_demo, y20_demo), (X24_demo, y24_demo), has_demo

data = load_data()
X16_att, y16_att = data[0]
X20_att, y20_att = data[1]
X24_att, y24_att = data[2]
X16_demo, y16_demo = data[3]
X20_demo, y20_demo = data[4]
X24_demo, y24_demo = data[5]
has_demo = data[6]

# ------------------------------------------------------------
# Helper: train model and predict for a single row
# ------------------------------------------------------------
def train_model_and_predict(X_train, y_train, X_pred, weight):
    dt = DecisionTreeClassifier(class_weight={0:1, 1:weight}, max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    proba = dt.predict_proba(X_pred)[0, 1]
    return proba, dt

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.header("⚙️ Model Parameters")
year = st.sidebar.selectbox("Select year", [2016, 2020, 2024])
weight = st.sidebar.slider("Minority class weight (Trump supporters)", min_value=1, max_value=50, value=5, step=1)

# Feature set choice
feature_set = st.sidebar.radio("Feature set", ["Attitudes only", "Attitudes + Demographics"])

if feature_set == "Attitudes only":
    X, y = {2016: X16_att, 2020: X20_att, 2024: X24_att}[year]
    demo_names = []
else:
    if not has_demo:
        st.sidebar.warning("Demographic data not available for this year. Using attitudes only.")
        X, y = {2016: X16_att, 2020: X20_att, 2024: X24_att}[year]
        demo_names = []
    else:
        X, y = {2016: X16_demo, 2020: X20_demo, 2024: X24_demo}[year]
        demo_names = [col for col in X.columns if col not in X16_att.columns]

# ------------------------------------------------------------
# User-defined demographics (if demographics are used)
# ------------------------------------------------------------
if feature_set == "Attitudes + Demographics" and has_demo and demo_names:
    st.sidebar.subheader("🔧 Set demographic profile")
    demo_values = {}
    for col in demo_names:
        if col == 'age':
            demo_values[col] = st.sidebar.slider("Age", 18, 90, 45)
        elif col == 'education':
            demo_values[col] = st.sidebar.slider("Education (1-16)", 1, 16, 12)
        elif col == 'income':
            # For simplicity, use the full range from data
            min_inc = int(X[col].min())
            max_inc = int(X[col].max())
            demo_values[col] = st.sidebar.slider("Income level", min_inc, max_inc, (min_inc+max_inc)//2)
        elif col == 'female':
            demo_values[col] = st.sidebar.selectbox("Gender", ["Male", "Female"]) == "Female"
        elif col == 'white':
            demo_values[col] = st.sidebar.selectbox("Race", ["White", "Non-White"]) == "White"
        elif col == 'party_id':
            demo_values[col] = st.sidebar.selectbox("Party ID (1=Strong Dem, 7=Strong Rep)", list(range(1,8)), index=3)
        elif col == 'rural':
            demo_values[col] = st.sidebar.selectbox("Rural/Urban", ["Urban", "Rural"]) == "Rural"
        else:
            # For any other demographic, use the median
            demo_values[col] = X[col].median()

# ------------------------------------------------------------
# Train model on full data and show metrics (as before)
# ------------------------------------------------------------
# (We'll reuse the existing train_eval function but with the selected X,y)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

def train_eval(X, y, weight, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    dt = DecisionTreeClassifier(class_weight={0:1, 1:weight}, max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    y_proba = dt.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    f1 = 2*prec*sens/(prec+sens) if (prec+sens)>0 else 0
    return dt, auc, sens, spec, prec, f1, X_test, y_test

dt, auc, sens, spec, prec, f1, X_test, y_test = train_eval(X, y, weight)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("AUC", f"{auc:.3f}")
col2.metric("Sensitivity", f"{sens:.3f}")
col3.metric("Specificity", f"{spec:.3f}")
col4.metric("Precision", f"{prec:.3f}")
col5.metric("F1 Score", f"{f1:.3f}")

# ------------------------------------------------------------
# Feature importance plot (same as before)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Predict for a specific demographic profile
# ------------------------------------------------------------
if feature_set == "Attitudes + Demographics" and has_demo and demo_names:
    st.subheader("👤 Predict for a specific demographic profile")
    
    # Build a single observation with the user-selected demographics
    # For attitude features, we need to set them to some typical values.
    # Let's take the median of each attitude feature from the dataset.
    att_features = [col for col in X.columns if col not in demo_names]
    median_values = {col: X[col].median() for col in att_features}
    
    # Combine with user demographics
    user_observation = {}
    for col in att_features:
        user_observation[col] = median_values[col]
    for col, val in demo_values.items():
        user_observation[col] = val
    
    user_df = pd.DataFrame([user_observation])
    
    # Predict probability
    proba = dt.predict_proba(user_df)[0, 1]
    
    st.write(f"**Predicted probability of Trump support for this profile:** {proba:.1%}")
    st.caption("Attitude features are set to their median values in the dataset. Adjust the weight slider above to see how class imbalance affects predictions.")

# ------------------------------------------------------------
# (Optional) Decision tree rules display (same as before)
# ------------------------------------------------------------
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

st.markdown("---")
st.caption("Data: ANES 2016‑2024 | Model: Decision Tree (max_depth=5) | Dashboard built with Streamlit")
