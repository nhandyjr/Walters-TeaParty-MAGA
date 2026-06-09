# Predicting Political Support Features Across Election Cycles (2012 -2024) 

**Predicting Trump support (2016‑2024) and Tea Party membership (2012) from survey proxy questions.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)  <!-- replace with your deployed URL -->

## 📌 Overview

This project tests **Ronald Walters' proxy theory** – the idea that racial resentment is expressed through seemingly neutral survey questions. Using machine learning on ANES (2016‑2024) and OOL (2012) data, we identify which questions most strongly predict (1) Strong Trump support (feeling thermometer ≥ 70); (2) Tea Party membership (2012, rare event: 2% of sample), and (3) stability of these features over election cycles.

**Key findings:**

- The **BLM thermometer** (`therm_blm`) becomes the dominant predictor in 2020 (Decision Tree importance = 0.75)
- `work_way_up` ("Blacks should work their way up without special favors") persists as the successor to `BWEqulOppty` (racial equality)
- `therm_undoc` (undocumented immigrant thermometer) remains strong across all years
- SHAP shows `blacks_gotten_less` is the top interacting feature with `therm_blm`
- Adding demographics (especially party ID) improves AUC, but proxies retain independent importance

## 🧠 Models

- **Decision Tree** – interpretable, high sensitivity (79‑98%)
- **Random Forest** – high specificity (96‑97%)
- **XGBoost** – balanced performance

## 📊 Interactive Dashboard

Explore the models live: **[Proxy Politics Dashboard](https://your-app-url.streamlit.app)**  
(adjust minority class weight, switch between attitudes/demographics, compare algorithms)

## 📁 Repository Structure
├── full_analysis.py # Main script: cleaning, training, SHAP, demographic controls
├── anes_streamlit_app.py # Streamlit dashboard (attitudes + demographics)
├── requirements.txt # Python dependencies
├── viz/ # Pickle files and all generated images (SHAP, tables)
│ ├── data_2016.pkl
│ ├── data_2020.pkl
│ ├── data_2024.pkl
│ ├── data_2016_demo.pkl
│ ├── data_2020_demo.pkl
│ ├── data_2024_demo.pkl
│ └── *.png
└── README.md
