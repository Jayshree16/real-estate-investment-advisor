🏡 Real Estate Investment Advisor

Predict investment potential and 5-year future price of Indian properties using machine learning.

This application provides
✔️ Good / Not-a-Good investment classification
✔️ 5-year price prediction
✔️ Feature importance visualizations
✔️ Smart property insights
✔️ Dataset exploration & dashboards

Built using Python, scikit-learn, XGBoost, MLflow, Streamlit.


🚀 Live Features

Investment Prediction

5-year Price Estimation

Property Explorer

Visual Insights Dashboard

Model Explainability (Feature Importance)

Model Metrics & Documentation


📂 Project Structure
REALESTATE-CLEAN/
│
├── assets/
├── data/
│   └── cleaned_real_estate.csv        # Only dataset included
│
├── models/                            # Empty initially → populated after download
│
├── pages/                             # Streamlit multipage UI
│
├── download_models.py                 # Downloads joblib models from Google Drive
├── models_config.json                 # Contains drive URLs of trained models
│
├── app.py                             # Main Streamlit application
├── requirements.txt
├── README.md
└── .gitignore


📦 Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/Jayshree16/real-estate-investment-advisor.git
cd real-estate-investment-advisor

2️⃣ Create and activate a virtual environment
Windows:
python -m venv venv
venv\Scripts\activate

Mac/Linux:
python3 -m venv venv
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

⬇️ 4️⃣ Download the trained ML models

These models are NOT stored in GitHub (files are large).

Just run:
python download_models.py

This script will:

✔ Download the ML models from Google Drive
✔ Save them into the models/ folder
✔ Ensure the app can load all required pipelines


▶️ 5️⃣ Run the Streamlit App
streamlit run app.py
The dashboard will open in your browser:
👉 http://localhost:8501


🧠 ML Models Included

The following models are downloaded via Google Drive:

clf_rf_pipeline.joblib – Random Forest Classifier

reg_rf_pipeline.joblib – Random Forest Regressor

preprocessor.joblib – Preprocessing Pipeline

train_test_splits.joblib – Dataset splits for metrics


📊 Dataset

Only cleaned_real_estate.csv is included in the repo.
Other large intermediate CSVs are excluded to keep the repository lightweight.


⚙️ Tech Stack

Python, Pandas, NumPy

Scikit-Learn, Random Forest, XGBoost

MLflow for experiment tracking

Streamlit for UI

Joblib for model serialization

