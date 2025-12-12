# 🏠 Real Estate Investment Advisor

Predict investment potential and a 5-year future price for Indian properties using machine learning.  
This repo contains the Streamlit UI, a minimal dataset (cleaned CSV) and a downloader script that pulls the large trained ML models from Google Drive so the GitHub repo stays lightweight.

---

## 🚀 Live Features
- Investment Prediction (Good / Not-a-Good)
- 5-year Price Estimation
- Property Explorer (table)
- Visual Insights Dashboard
- Feature importance and model metrics

---

## 📁 Project Structure

```text
REALESTATE-CLEAN/
│── assets/                 # UI images/icons
│── data/
│   └── cleaned_real_estate.csv   # Small, included dataset
│
│── models/                 # Initially empty; populated after running download_models.py
│
│── mlruns/                 # Excluded from Git (experiment logs)
│
│── pages/                  # Streamlit multipage UI
│
│── download_models.py      # Script that downloads ML models from Google Drive
│── models_config.json      # Maps model names -> Google Drive URLs
│── app.py                  # Main Streamlit application
│── requirements.txt        # Project dependencies
│── README.md
│── .gitignore
```




---

## 🔧 Quickstart (local)
1. Clone:
```bash
git clone https://github.com/Jayshree16/real-estate-investment-advisor.git
cd real-estate-investment-advisor

2. Create & activate venv

-Windows:python -m venv venv
venv\Scripts\activate


-macOS / Linux:
python3 -m venv venv
source venv/bin/activate

3. Install dependencies:
pip install -r requirements.txt

4. Download trained ML models (models are not stored on GitHub):
python download_models.py
# This saves *.joblib files into models/.

5. Run the app:
streamlit run app.py
# then open http://localhost:8501


