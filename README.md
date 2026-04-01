# 🏠 House Price Prediction

**Author:** Pavan Perisetla · [GitHub](https://github.com/perisetlapavan)

A complete end-to-end Machine Learning project that predicts house prices using the California Housing dataset — structured to mirror the Kaggle House Prices competition.

---

## 📌 Project Highlights

| Item | Detail |
|------|--------|
| Dataset | California Housing (20,640 rows) |
| Target | House Sale Price (USD) |
| Best Model | XGBoost (Tuned) |
| R² Score | ~0.84+ |
| Deployment | Streamlit Interactive App |

---

## 🗂️ Project Structure

```
house-price-prediction/
├── house_price_prediction.ipynb   # Full ML pipeline notebook
├── app.py                         # Streamlit app
├── model/
│   ├── house_price_model.pkl      # Trained XGBoost model
│   ├── scaler.pkl                 # StandardScaler
│   └── feature_names.json         # Feature list
├── requirements.txt
└── README.md
```

---

## 🔄 ML Pipeline

1. **EDA & Visualizations** — distributions, correlations, outliers
2. **Feature Engineering** — RoomsPerPerson, BedroomRatio, PopulationDensity
3. **Preprocessing** — StandardScaler, log-transform on target
4. **Model Comparison** — Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting, XGBoost
5. **Hyperparameter Tuning** — RandomizedSearchCV (30 iterations, 5-fold CV)
6. **Deployment** — Interactive Streamlit app

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/perisetlapavan/house-price-prediction
cd house-price-prediction

# Install dependencies
pip install -r requirements.txt

# Step 1: Run notebook to train and save the model
jupyter notebook house_price_prediction.ipynb

# Step 2: Launch the Streamlit app
streamlit run app.py
```

---

## 📊 Results

| Model | R² Score | MAE |
|-------|----------|-----|
| Linear Regression | ~0.61 | ~$55k |
| Ridge Regression | ~0.61 | ~$55k |
| Decision Tree | ~0.63 | ~$48k |
| Random Forest | ~0.81 | ~$35k |
| Gradient Boosting | ~0.80 | ~$36k |
| **XGBoost (Tuned)** | **~0.84** | **~$32k** |

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn` · `Streamlit` · `Joblib`

---

*Built as a Data Scientist portfolio project.*
