"""
🏠 House Price Prediction - Streamlit App
Author: Pavan Perisetla
GitHub: https://github.com/perisetlapavan
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background: white; border-radius: 10px; padding: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .predict-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   border-radius: 15px; padding: 20px; text-align: center; color: white; margin: 10px 0; }
    .predict-box h1 { font-size: 2.5em; margin: 0; }
    .predict-box p  { font-size: 1.1em; margin: 5px 0 0 0; opacity: 0.9; }
    .info-card { background: white; border-radius: 10px; padding: 15px;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin: 8px 0; }
    .github-link { color: #764ba2; font-weight: bold; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path  = "model/house_price_model.pkl"
    scaler_path = "model/scaler.pkl"
    feat_path   = "model/feature_names.json"

    if not os.path.exists(model_path):
        st.error("⚠️ Model not found! Please run the Jupyter notebook first to train and save the model.")
        st.stop()

    model        = joblib.load(model_path)
    scaler       = joblib.load(scaler_path)
    with open(feat_path) as f:
        feature_names = json.load(f)
    return model, scaler, feature_names


model, scaler, feature_names = load_model()


# ── Header ─────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown("# 🏠")
with col_title:
    st.markdown("## House Price Prediction")
    st.markdown(
        "Predict California house prices using **XGBoost** · "
        "[<span class='github-link'>perisetlapavan</span>](https://github.com/perisetlapavan)",
        unsafe_allow_html=True,
    )

st.divider()

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Input Features")
    st.markdown("Adjust the sliders to set house characteristics:")
    st.markdown("---")

    med_inc       = st.slider("Median Income (×$10k)",     0.5, 15.0, 5.0, 0.1,
                               help="Median income in the block group")
    house_age     = st.slider("House Age (years)",          1,   52,  20,
                               help="Median age of houses in the block")
    ave_rooms     = st.slider("Average Rooms",              1.0, 15.0, 5.5, 0.1)
    ave_bedrms    = st.slider("Average Bedrooms",           0.5,  5.0, 1.1, 0.1)
    population    = st.slider("Block Population",          10, 5000, 1200)
    ave_occup     = st.slider("Average Occupants",          1.0,  6.0, 3.0, 0.1)
    latitude      = st.slider("Latitude",                  32.5, 42.0, 35.6, 0.1)
    longitude     = st.slider("Longitude",                -124.0, -114.0, -119.5, 0.1)

    st.markdown("---")
    st.markdown("**Project by:** [Pavan Perisetla](https://github.com/perisetlapavan)")


# ── Compute engineered features ─────────────────────────────────────────────────
rooms_per_person   = ave_rooms / (population / ave_occup + 1)
bedroom_ratio      = ave_bedrms / ave_rooms
population_density = population / ave_occup

# Build input in correct feature order
input_dict = {
    "MedInc":           med_inc,
    "HouseAge":         house_age,
    "AveRooms":         ave_rooms,
    "AveBedrms":        ave_bedrms,
    "Population":       population,
    "AveOccup":         ave_occup,
    "Latitude":         latitude,
    "Longitude":        longitude,
    "RoomsPerPerson":   rooms_per_person,
    "BedroomRatio":     bedroom_ratio,
    "PopulationDensity": population_density,
}

input_df     = pd.DataFrame([input_dict])[feature_names]
input_scaled = scaler.transform(input_df)
log_pred     = model.predict(input_scaled)[0]
predicted    = np.expm1(log_pred)


# ── Main Content ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Feature Analysis", "ℹ️ About Project"])

# ── Tab 1: Prediction ──────────────────────────────────────────────────────────
with tab1:
    st.markdown(f"""
    <div class="predict-box">
        <p>Predicted House Price</p>
        <h1>${predicted:,.0f}</h1>
        <p>Based on the features you selected</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📋 Input Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median Income", f"${med_inc*10:.0f}k")
    c2.metric("House Age",     f"{house_age} yrs")
    c3.metric("Avg Rooms",     f"{ave_rooms:.1f}")
    c4.metric("Population",    f"{population:,}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg Bedrooms",  f"{ave_bedrms:.1f}")
    c6.metric("Avg Occupants", f"{ave_occup:.1f}")
    c7.metric("Latitude",      f"{latitude:.2f}")
    c8.metric("Longitude",     f"{longitude:.2f}")

    st.markdown("#### 🔧 Engineered Features")
    e1, e2, e3 = st.columns(3)
    e1.metric("Rooms / Person",      f"{rooms_per_person:.3f}")
    e2.metric("Bedroom Ratio",       f"{bedroom_ratio:.3f}")
    e3.metric("Population Density",  f"{population_density:.1f}")

    # Price range indicator
    st.markdown("#### 📈 Price Range Context")
    price_ranges = {
        "Budget":     (0, 150_000),
        "Affordable": (150_000, 300_000),
        "Mid-Range":  (300_000, 500_000),
        "Premium":    (500_000, 750_000),
        "Luxury":     (750_000, 1_000_000),
        "Ultra":      (1_000_000, float("inf")),
    }
    label = next((k for k, (lo, hi) in price_ranges.items() if lo <= predicted < hi), "Ultra")
    st.info(f"🏷️ This property falls in the **{label}** price category (${predicted:,.0f})")


# ── Tab 2: Feature Analysis ────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🔍 How Each Feature Influences the Prediction")

    # Feature importance bar chart
    importances   = model.feature_importances_
    feat_imp_df   = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_imp_df   = feat_imp_df.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ["#764ba2" if f == feat_imp_df.iloc[-1]["Feature"] else "#667eea"
               for f in feat_imp_df["Feature"]]
    bars = ax.barh(feat_imp_df["Feature"], feat_imp_df["Importance"],
                   color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_title("Feature Importance (XGBoost)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### 📐 Price Sensitivity — Income vs Age")
    # Grid: income vs house age
    inc_vals  = np.linspace(1, 12, 30)
    age_vals  = np.linspace(5, 50, 30)
    INC, AGE  = np.meshgrid(inc_vals, age_vals)
    grid_rows = []

    for inc, age in zip(INC.flatten(), AGE.flatten()):
        rpp = ave_rooms / (population / ave_occup + 1)
        br  = ave_bedrms / ave_rooms
        pd_ = population / ave_occup
        row = {
            "MedInc": inc, "HouseAge": age,
            "AveRooms": ave_rooms, "AveBedrms": ave_bedrms,
            "Population": population, "AveOccup": ave_occup,
            "Latitude": latitude, "Longitude": longitude,
            "RoomsPerPerson": rpp, "BedroomRatio": br,
            "PopulationDensity": pd_,
        }
        grid_rows.append(row)

    grid_df     = pd.DataFrame(grid_rows)[feature_names]
    grid_scaled = scaler.transform(grid_df)
    grid_preds  = np.expm1(model.predict(grid_scaled)).reshape(INC.shape)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    contour = ax2.contourf(INC, AGE, grid_preds / 1000, levels=20, cmap="RdYlGn")
    cbar = plt.colorbar(contour, ax=ax2)
    cbar.set_label("Predicted Price ($k)")
    ax2.set_xlabel("Median Income (×$10k)")
    ax2.set_ylabel("House Age (years)")
    ax2.set_title("Price Heatmap: Income vs House Age", fontsize=13, fontweight="bold")
    # Mark current input
    ax2.scatter([med_inc], [house_age], color="white", s=120, zorder=5,
                edgecolors="black", linewidths=2, label="Your selection")
    ax2.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ── Tab 3: About ───────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📘 Project Overview")
    st.markdown("""
    <div class="info-card">
    <b>🎯 Objective:</b> Predict house prices using machine learning based on socioeconomic and geographic features.
    </div>
    <div class="info-card">
    <b>📦 Dataset:</b> California Housing Dataset — 20,640 samples, 8 original features + 3 engineered features.
    </div>
    <div class="info-card">
    <b>🤖 Models Compared:</b> Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting, XGBoost
    </div>
    <div class="info-card">
    <b>🏆 Best Model:</b> XGBoost with RandomizedSearchCV hyperparameter tuning — R² ≈ 0.84+
    </div>
    <div class="info-card">
    <b>🔧 Techniques:</b> EDA, Feature Engineering, StandardScaler, Log-Transform, Cross-Validation, Hyperparameter Tuning
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🗂️ Project Structure")
    st.code("""
house-price-prediction/
├── house_price_prediction.ipynb   # Full ML pipeline notebook
├── app.py                         # This Streamlit app
├── model/
│   ├── house_price_model.pkl      # Trained XGBoost model
│   ├── scaler.pkl                 # Feature scaler
│   └── feature_names.json         # Feature list
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
    """)

    st.markdown("### ▶️ How to Run")
    st.code("""
# 1. Clone repo
git clone https://github.com/perisetlapavan/house-price-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebook to train & save model
jupyter notebook house_price_prediction.ipynb

# 4. Launch Streamlit app
streamlit run app.py
    """, language="bash")

    st.markdown("---")
    st.markdown(
        "**Author:** [Pavan Perisetla](https://github.com/perisetlapavan) · "
        "Built as a Data Scientist Portfolio Project"
    )
