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

## 📁 Project structure
REALESTATE-CLEAN/
├─ assets/ # UI images/icons
├─ data/
│ └─ cleaned_real_estate.csv # small, included dataset
├─ models/ # initially empty in repo; populated after running download_models.py
├─ mlruns/ # (excluded from git)
├─ pages/ # Streamlit multipage UI
├─ download_models.py # script that fetches joblib models from Google Drive
├─ models_config.json # maps model names -> Drive URLs (public)
├─ app.py # main Streamlit app
├─ requirements.txt
├─ README.md
└─ .gitignore



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
This saves *.joblib files into models/.

5. Run the app:
streamlit run app.py
# then open http://localhost:8501


