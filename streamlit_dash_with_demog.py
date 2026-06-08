import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

st.set_page_config(page_title="Proxy Politics Dashboard", layout="wide")
st.title("📊 Proxy Politics: From Tea Party to MAGA")
st.markdown("Explore how different minority class weights, features, and algorithms affect Trump support predictions.")

# ------------------------------------------------------------
# Load data – pickles are in the same directory as this script
# ------------------------------------------------------------
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Attitudes only
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
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.header("⚙️ Model Parameters")
year = st.sidebar.selectbox("Select year", [2016, 2020, 2024])
weight = st.sidebar.slider("Minority class weight (Trump supporters)", min_value=1, max_value=50, value=5, step=1)

feature_set = st.sidebar.radio("Feature set", ["Attitudes only", "Attitudes + Demographics"])
algorithm = st.sidebar.selectbox("Algorithm", ["Decision Tree", "Random Forest", "XGBoost"])

# Choose data based on feature_set
if feature_set == "Attitudes only":
    X_dict = {2016: X16_att, 2020: X20_att, 2024: X24_att}
    y_dict = {2016: y16_att, 2020: y20_att, 2024: y24_att}
    demo_cols = []
else:
    if not has_demo:
        st.sidebar.warning("Demographic data not available. Using attitudes only.")
        X_dict = {2016: X16_att, 2020: X20_att, 2024: X24_att}
        y_dict = {2016: y16_att, 2020: y20_att, 2024: y24_att}
        demo_cols = []
    else:
        X_dict = {2016: X16_demo, 2020: X20_demo, 2024: X24_demo}
        y_dict = {2016: y16_demo, 2020: y20_demo, 2024: y24_demo}
        att_cols = X16_att.columns.tolist()
        demo_cols = [col for col in X_dict[year].columns if col not in att_cols]

X = X_dict[year]
y = y_dict[year]

# ------------------------------------------------------------
# Cached train/test split (only recompute when X or y changes)
# ------------------------------------------------------------
@st.cache_data
def get_train_test_split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_test, y_train, y_test = get_train_test_split(X, y)

# ------------------------------------------------------------
# Cached model training (only recompute when algorithm, weight, or training set changes)
# ------------------------------------------------------------
@st.cache_resource
def get_cached_model(algorithm, weight, X_train_hash, y_train_hash):
    # The hash arguments are dummy; we pass the actual arrays but caching uses the hash.
    # To avoid large memory duplication, we use the identity of the arrays as proxy.
    # In practice, we'll just use X_train and y_train directly inside the function.
    # Better: use the actual X_train and y_train from the outer scope.
    pass

# But because st.cache_resource cannot directly hash large DataFrames efficiently,
# we will use a simple approach: wrap the model training in a function that takes
# the relevant parameters and the training data. Streamlit will cache based on the
# data's hash (which is fine for moderate size data).
@st.cache_resource
def train_cached_model(X_train, y_train, algorithm, weight):
    if algorithm == "Decision Tree":
        model = DecisionTreeClassifier(class_weight={0:1, 1:weight}, max_depth=5, random_state=42)
    elif algorithm == "Random Forest":
        model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    else:  # XGBoost – full n_estimators (cached)
        model = XGBClassifier(scale_pos_weight=weight, n_estimators=100, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    return model

model = train_cached_model(X_train, y_train, algorithm, weight)

# ------------------------------------------------------------
# Evaluate on test set
# ------------------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sens = tp/(tp+fn) if (tp+fn)>0 else 0
spec = tn/(tn+fp) if (tn+fp)>0 else 0
prec = tp/(tp+fp) if (tp+fp)>0 else 0
f1 = 2*prec*sens/(prec+sens) if (prec+sens)>0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("AUC", f"{auc:.3f}")
col2.metric("Sensitivity", f"{sens:.3f}")
col3.metric("Specificity", f"{spec:.3f}")
col4.metric("Precision", f"{prec:.3f}")
col5.metric("F1 Score", f"{f1:.3f}")

# ------------------------------------------------------------
# Feature importances (if available)
# ------------------------------------------------------------
if hasattr(model, 'feature_importances_'):
    st.subheader(f"📈 Feature Importances ({algorithm})")
    importances = model.feature_importances_
    features = X.columns
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(order)), importances[order], color='steelblue')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(features[order])
    ax.set_xlabel('Importance')
    ax.invert_yaxis()
    st.pyplot(fig)
else:
    st.info("Feature importances not available for this model.")

# ------------------------------------------------------------
# User-defined demographic profile (if demographics are used)
# ------------------------------------------------------------
user_obs = None
if feature_set == "Attitudes + Demographics" and has_demo and demo_cols:
    st.sidebar.subheader("🔧 Set demographic profile")
    demo_values = {}
    for col in demo_cols:
        if col == 'age':
            demo_values[col] = st.sidebar.slider("Age", 18, 90, 45)
        elif col == 'education':
            demo_values[col] = st.sidebar.slider("Education (1-16)", 1, 16, 12)
        elif col == 'income':
            min_inc = int(X[col].min()) if not np.isnan(X[col].min()) else 1
            max_inc = int(X[col].max()) if not np.isnan(X[col].max()) else 20
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
            demo_values[col] = X[col].median()
    
    # For attitude features, use medians
    att_cols_this = [c for c in X.columns if c not in demo_cols]
    median_vals = {c: X[c].median() for c in att_cols_this}
    user_row = {**median_vals, **demo_values}
    user_obs = pd.DataFrame([user_row])

if user_obs is not None:
    st.subheader("👤 Predict for your demographic profile")
    proba = model.predict_proba(user_obs)[0, 1]
    st.write(f"**Predicted probability of Trump support:** {proba:.1%}")
    st.caption("Attitude features are set to their median values in the dataset.")

# ------------------------------------------------------------
# Decision tree rules (only for Decision Tree)
# ------------------------------------------------------------
if algorithm == "Decision Tree":
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
    rules = get_rules(model, X.columns, max_depth=2)
    st.text("\n".join(rules[:15]))

st.markdown("---")
st.caption("Data: ANES 2016‑2024 | Model: Decision Tree (max_depth=5) | Dashboard built with Streamlit")