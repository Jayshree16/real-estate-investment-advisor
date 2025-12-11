# Real Estate Investment Advisor 🏠

A Streamlit-based web application that helps users evaluate whether a property is a **good investment** and estimates its **price after 5 years**, trained on Indian housing data.

---

## ✨ Features

- **Investment Prediction**
  - Classification: “Good Investment or Not?”
  - Shows confidence score (probability)
- **Price Forecast**
  - Regression model to estimate 5-year future price
- **Property Explorer**
  - Filter properties by **location, BHK, price range, area**
  - Download filtered results as CSV
- **Visual Insights**
  - Location-wise average price
  - Area vs price trend
  - BHK vs price distribution
- **Explainability**
  - Feature importance for the regression model
- **Model Info**
  - Shows model types and evaluation metrics (RMSE, R², Accuracy, F1)

---

## 🧱 Project Structure

```text
real_estate_investment_advisor/
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore
│
├── data/
│   ├── cleaned_real_estate.csv
│   ├── india_housing_prices.csv
│   ├── real_estate_realistic.csv
│   └── real_estate_realistic_with_labels.csv
│
├── models/
│   ├── reg_rf_pipeline.joblib       # Regression model
│   ├── clf_rf_pipeline.joblib       # Classification model
│   ├── preprocessor.joblib          # Preprocessor (if used)
│   └── train_test_splits.joblib     # Train/test splits for metrics
│
├── mlruns/                    # (Optional) MLflow experiment logs
└── assets/                    # Images / logos (optional)
