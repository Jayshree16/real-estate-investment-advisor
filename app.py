import os
import altair as alt
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.inspection import permutation_importance

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Material Icons + base CSS
st.markdown(
    """
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">

    <style>
    /* Global background tweak for a realistic real-estate dashboard feel */
    body {
        background-color: #f3f4f6;
    }

    .material-icons {
        font-size: 20px;
        vertical-align: middle;
        margin-right: 6px;
    }

    /* Top navigation bar with Material Icons */
    .navbar {
        display: flex;
        gap: 1.6rem;
        align-items: center;
        border-bottom: 1px solid #e5e7eb;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.35rem;
    }

    .nav-link {
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.95rem;
        color: #4b5563;
        padding: 0.35rem 0.1rem;
        border-bottom: 2px solid transparent;
        transition: all .15s ease;
    }

    .nav-link .material-icons {
        font-size: 18px;
        margin-right: 4px;
        color: #6b7280;
    }

    .nav-link:hover {
        color: #111827;
    }

    .nav-link:hover .material-icons {
        color: #111827;
    }

    .nav-link.active {
        color: #1d4ed8;
        border-bottom-color: #1d4ed8;
        font-weight: 600;
    }

    .nav-link.active .material-icons {
        color: #1d4ed8;
    }

    /* Metric cards - light, realistic, not overly glossy */
    .metric-card {
        background: #ffffff;
        padding: 1.1rem 1.4rem;
        border-radius: 1rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 10px 15px -3px rgba(15,23,42,0.06);
        min-height: 150px;
    # }
    .metric-label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: 0.15rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.1rem;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: .4rem;
        padding: .25rem .6rem;
        border-radius: 999px;
        font-size: .75rem;
        background: #eef2ff;
        color: #4f46e5;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .08em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# CONFIG – ADAPT TO YOUR FILES
# -------------------------------------------------
DATA_FILE = "data/real_estate_realistic_with_labels.csv"

REG_MODEL_FILE = "models/reg_rf_pipeline.joblib"      # regression: price after 5 years
CLF_MODEL_FILE = "models/clf_rf_pipeline.joblib"      # classification: good investment or not
SPLITS_FILE    = "models/train_test_splits.joblib"    # optional – for metrics

CLF_TARGET_COL = "Good_Investment"
REG_TARGET_COL = "Future_Price_5Y"


# -------------------------------------------------
# LOADERS (CACHED)
# -------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Dataset file '{path}' not found. Make sure the file exists in the /data folder.")
        st.stop()
    return pd.read_csv(path)


@st.cache_resource
def load_models(reg_path: str, clf_path: str):
    if not os.path.exists(reg_path):
        st.error(f"Regression model file '{reg_path}' not found in /models.")
        st.stop()
    if not os.path.exists(clf_path):
        st.error(f"Classification model file '{clf_path}' not found in /models.")
        st.stop()
    reg_model = joblib.load(reg_path)
    clf_model = joblib.load(clf_path)
    return reg_model, clf_model


@st.cache_data
def load_splits(path: str):
    """Load train/test splits for metrics & permutation importance (optional)."""
    if not os.path.exists(path):
        return None
    try:
        obj = joblib.load(path)
        if isinstance(obj, (tuple, list)) and len(obj) >= 8:
            (
                X_train,
                X_test,
                y_train_class,
                y_test_class,
                X_train_reg,
                X_test_reg,
                y_train_reg,
                y_test_reg,
                *_,
            ) = obj + (None,) * (8 - len(obj))
            return {
                "X_train": X_train,
                "X_test": X_test,
                "y_train_class": y_train_class,
                "y_test_class": y_test_class,
                "X_train_reg": X_train_reg,
                "X_test_reg": X_test_reg,
                "y_train_reg": y_train_reg,
                "y_test_reg": y_test_reg,
            }
        return None
    except Exception:
        return None


df = load_data(DATA_FILE)
reg_pipeline, clf_pipeline = load_models(REG_MODEL_FILE, CLF_MODEL_FILE)
splits = load_splits(SPLITS_FILE)

# -------------------------------------------------
# EXPLICIT COLUMN MAPPING FOR YOUR DATASET
# -------------------------------------------------
location_col = "City"
bhk_col      = "BHK"
area_col     = "Size_in_SqFt"
price_col    = "Price_in_Lakhs"
age_col      = "Age_of_Property"

# Label columns used for training (if present)
label_cols = set()
if CLF_TARGET_COL and CLF_TARGET_COL in df.columns:
    label_cols.add(CLF_TARGET_COL)
if REG_TARGET_COL and REG_TARGET_COL in df.columns:
    label_cols.add(REG_TARGET_COL)

# All features = everything except labels
feature_cols = [c for c in df.columns if c not in label_cols]

# -------------------------------------------------
# SIDEBAR FILTERS (USED BY EXPLORER & INSIGHTS)
# -------------------------------------------------
st.sidebar.title("🔎 Filters (Dataset View)")

filtered_df = df.copy()

if location_col in df.columns:
    locations = sorted(df[location_col].dropna().unique().tolist())
    default_locs = locations[: min(5, len(locations))]
    selected_locations = st.sidebar.multiselect(
        "Location",
        options=locations,
        default=default_locs,
    )
    if selected_locations:
        filtered_df = filtered_df[filtered_df[location_col].isin(selected_locations)]

if bhk_col in df.columns:
    min_bhk, max_bhk = int(filtered_df[bhk_col].min()), int(filtered_df[bhk_col].max())
    bhk_range = st.sidebar.slider("BHK", min_bhk, max_bhk, (min_bhk, max_bhk))
    filtered_df = filtered_df[
        (filtered_df[bhk_col] >= bhk_range[0]) & (filtered_df[bhk_col] <= bhk_range[1])
    ]

if price_col in df.columns:
    min_price = float(filtered_df[price_col].min())
    max_price = float(filtered_df[price_col].max())
    price_range = st.sidebar.slider(
        "Price range",
        float(min_price),
        float(max_price),
        (float(min_price), float(max_price)),
    )
    filtered_df = filtered_df[
        (filtered_df[price_col] >= price_range[0]) & (filtered_df[price_col] <= price_range[1])
    ]

if area_col in df.columns:
    min_area = float(filtered_df[area_col].min())
    max_area = float(filtered_df[area_col].max())
    area_range = st.sidebar.slider(
        "Area range",
        float(min_area),
        float(max_area),
        (float(min_area), float(max_area)),
    )
    filtered_df = filtered_df[
        (filtered_df[area_col] >= area_range[0]) & (filtered_df[area_col] <= area_range[1])
    ]

st.sidebar.markdown(f"**Filtered rows:** {len(filtered_df)}")

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 0;'>🏠 Real Estate Investment Advisor</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #9CA3AF;'>Predict investment potential and 5-year price for Indian properties.</p>",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# CUSTOM NAV BAR WITH MATERIAL ICONS
# -------------------------------------------------
params = st.query_params
default_tab = "home"
# st.query_params returns a STRING, so no [0] here
active_tab = params.get("tab", default_tab)

tabs = [
    ("home", "home", "Home"),
    ("predict", "trending_up", "Investment Prediction"),
    ("explorer", "folder_open", "Property Explorer"),
    ("insights", "bar_chart", "Visual Insights"),
    ("importance", "insights", "Feature Importance"),
    ("model", "settings", "Model Info"),
    ("about", "info", "About"),
]

nav_html = "<div class='navbar'>"
for key, icon, label in tabs:
    active_class = " active" if key == active_tab else ""
    nav_html += (
        f"<a class='nav-link{active_class}' href='?tab={key}' target='_self'>"
        f"<span class='material-icons'>{icon}</span>"
        f"<span class='nav-label'>{label}</span>"
        "</a>"
    )
nav_html += "</div>"

st.markdown(nav_html, unsafe_allow_html=True)

# Keep in session_state as well (optional)
st.session_state["active_tab"] = active_tab


# Pastel palette used across charts
PASTEL_COLORS = ["#bfdbfe", "#93c5fd", "#60a5fa", "#38bdf8", "#0ea5e9", "#6366f1"]

# =================================================
# TAB 1 — HOME / OVERVIEW
# =================================================
if active_tab == "home":
    # Hero section
    hero_left, hero_right = st.columns([3, 1.5])

    with hero_left:
        st.markdown(
            """
            <div class="hero-badge">
                📈 Smart Property Insights
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h2 style='margin-top: .6rem;'>Welcome to Real Estate Investment Advisor</h2>",
            unsafe_allow_html=True,
        )
        st.write(
            """
            This dashboard helps you understand whether a property is a **good investment** and
            estimate its **price after 5 years** using machine-learning models trained on
            an all-India real estate dataset.
            """
        )
        st.write(
            """
            Use the navigation above to:
            - run **what-if predictions** for any property,  
            - **explore** the dataset with filters,  
            - view **visual insights** and **feature importance**, and  
            - inspect **model performance & metrics**.
            """
        )

    with hero_right:
        # Snapshot cards in a 2x2 grid
        total_props = len(df)
        num_locs = df[location_col].nunique() if location_col in df.columns else None
        avg_price = df[price_col].mean() if price_col in df.columns else None
        avg_area  = df[area_col].mean() if area_col in df.columns else None

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Properties</div>
                    <div class="metric-value">{total_props:,}</div>
                    <div class="metric-sub">Rows in modelling dataset</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Locations</div>
                    <div class="metric-value">{num_locs}</div>
                    <div class="metric-sub">Cities covered across India</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        c3, c4 = st.columns(2)
        with c3:
            if avg_price is not None:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Avg Price (Lakhs)</div>
                        <div class="metric-value">{avg_price:,.1f}</div>
                        <div class="metric-sub">Mean price across all properties</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with c4:
            if avg_area is not None:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Avg Size (SqFt)</div>
                        <div class="metric-value">{avg_area:,.0f}</div>
                        <div class="metric-sub">Average carpet area</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # Dataset preview
    st.subheader("Dataset Preview")
    st.caption(
        "First 10 rows of the modelling dataset. Apply filters in the sidebar and use the "
        "**Property Explorer** view to see the full filtered table."
    )
    st.dataframe(df.head(10))

# =================================================
# TAB 2 — INVESTMENT PREDICTION 
# =================================================
elif active_tab == "predict":
    st.header("🔮 Investment Prediction")
    st.write("Fill in the property details to predict investment decision and 5-year price.")

    col_form, col_results = st.columns([3, 2])

    # ---------------------------------------------
    # 1️⃣ CHOOSE WHICH FEATURES THE USER ACTUALLY ENTERS
    # ---------------------------------------------
    # These are realistic property attributes a user can know.
    # We will ask only for these in the UI.
    desired_user_features = [
        "State",
        "City",
        "Locality",
        "Property_Type",
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Year_Built",
        "Total_Floors",
        "Floor_No",
        "Furnished_Status",
        "Parking_Space",
        "Security",
        "Amenities",
        "Facing",
        "Owner_Type",
        "Availability_Status",
        "Nearby_Schools",
        "Nearby_Hospitals",
        "Public_Transport_Accessibility",
    ]

    # Only keep those that actually exist in your dataset & features
    user_feature_cols = [
        c for c in desired_user_features
        if c in df.columns and c in feature_cols
    ]

    # ---------------------------------------------
    # 2️⃣ LEFT SIDE — PROPERTY DETAILS FORM
    # ---------------------------------------------
    with col_form:
        st.subheader("Property Details")

        with st.form("prediction_form"):
            input_values = {}
            cols_row = st.columns(3)  # 3-column grid

            for idx, col in enumerate(user_feature_cols):
                series = df[col]
                col_streamlit = cols_row[idx % 3]

                with col_streamlit:
                    if pd.api.types.is_numeric_dtype(series):
                        min_val = float(series.min())
                        max_val = float(series.max())
                        default_val = float(series.median())
                        step = (max_val - min_val) / 100 if max_val != min_val else 1.0
                        input_values[col] = st.number_input(
                            label=col.replace("_", " "),
                            value=default_val,
                            min_value=min_val,
                            max_value=max_val,
                            step=step,
                            key=f"num_{col}",
                        )
                    else:
                        options = sorted(series.dropna().unique().tolist())
                        default_opt = options[0] if options else ""
                        input_values[col] = st.selectbox(
                            col.replace("_", " "),
                            options if options else [""],
                            index=options.index(default_opt) if options else 0,
                            key=f"sel_{col}",
                        )

            submitted = st.form_submit_button("Predict")

    # ---------------------------------------------
    # 3️⃣ BUILD FULL FEATURE VECTOR FOR THE MODEL
    # ---------------------------------------------
    with col_results:
        st.subheader("Prediction Results")
        if submitted:
            # Start with user-entered values
            model_input = {}

            for col in feature_cols:
                if col in input_values:
                    model_input[col] = input_values[col]
                else:
                    # For features we don't show in the form, fill with sensible defaults
                    series = df[col]

                    if pd.api.types.is_numeric_dtype(series):
                        # Use median for numeric features
                        val = float(series.median())
                    else:
                        # Use most frequent category for categorical features
                        if series.dropna().empty:
                            val = ""
                        else:
                            val = series.mode().iloc[0]
                    model_input[col] = val

            input_df = pd.DataFrame([model_input])

            # -----------------------------------------
            # 4️⃣ RUN CLASSIFICATION + REGRESSION MODELS
            # -----------------------------------------
            inv_label = "N/A"
            inv_conf = None
            price_pred = None

            # Classification
            try:
                if hasattr(clf_pipeline, "predict_proba"):
                    prob = clf_pipeline.predict_proba(input_df)[0, 1]
                    inv_conf = float(prob)
                pred_class = clf_pipeline.predict(input_df)[0]
                inv_label = "✅ GOOD INVESTMENT" if pred_class == 1 else "⚠️ NOT A GOOD INVESTMENT"
            except Exception as e:
                st.error(f"Classification prediction error: {e}")

            # Regression
            try:
                price_pred = float(reg_pipeline.predict(input_df)[0])
            except Exception as e:
                st.error(f"Regression prediction error: {e}")

            # -----------------------------------------
            # 5️⃣ DISPLAY RESULTS
            # -----------------------------------------
            card_col1, card_col2 = st.columns(2)

            with card_col1:
                st.markdown("**Investment Decision**")
                st.write(inv_label)
                if inv_conf is not None:
                    st.write(f"Confidence: {inv_conf * 100:.1f}%")
                    st.progress(inv_conf)

            with card_col2:
                st.markdown("**Estimated Price After 5 Years**")
                if price_pred is not None:
                    st.write(f"₹ {price_pred:,.2f}")
                else:
                    st.write("Prediction unavailable.")

            st.markdown("---")
            st.caption(
                "Predictions are made using your trained models based on the cleaned & engineered dataset. "
                "Fields not shown in the form are filled with typical values (median/mode) from the dataset."
            )
        else:
            st.info("Submit the form to see predictions.")

# =================================================
# TAB 3 — PROPERTY EXPLORER
# =================================================
elif active_tab == "explorer":
    st.header("📂 Property Explorer")
    st.write("Use the sidebar filters to refine properties, then explore them in the table below.")

    st.write(f"Showing **{len(filtered_df)}** filtered properties.")
    st.dataframe(filtered_df)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered as CSV",
        data=csv_data,
        file_name="filtered_properties.csv",
        mime="text/csv",
    )

# =================================================
# TAB 4 — VISUAL INSIGHTS
# =================================================
elif active_tab == "insights":
    st.header("📊 Visual Insights Dashboard")
    st.caption("Insights are computed on the **filtered dataset** from the sidebar.")

    df_vis = filtered_df.copy()

    if df_vis.empty:
        st.warning("No rows after applying filters. Please relax the filters to see insights.")
    else:
        kpi_col, chart_col = st.columns([1, 3])

        # KPIs (same card style as home)
        with kpi_col:
            st.subheader("Key Metrics")

            total_props = len(df_vis)
            n_cities = df_vis[location_col].nunique() if location_col in df_vis.columns else None
            avg_price = df_vis[price_col].mean() if price_col in df_vis.columns else None
            avg_size = df_vis[area_col].mean() if area_col in df_vis.columns else None
            avg_bhk  = df_vis[bhk_col].mean() if bhk_col in df_vis.columns else None

            k1, k2 = st.columns(2)
            k3, k4 = st.columns(2)

            # 🔥 Fixed-height KPI card template (all 4 same size)
            card_tpl = """
                <div class="metric-card"
                     style="height:170px; display:flex; flex-direction:column;
                            justify-content:space-between;">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-sub">{subtitle}</div>
                </div>
            """

            k1.markdown(
                card_tpl.format(
                    label="Total Properties",
                    value=f"{total_props:,.0f}",
                    subtitle="Rows in filtered dataset",
                ),
                unsafe_allow_html=True,
            )

            k2.markdown(
                card_tpl.format(
                    label="Locations",
                    value=n_cities if n_cities is not None else "–",
                    subtitle="Cities covered",
                ),
                unsafe_allow_html=True,
            )

            k3.markdown(
                card_tpl.format(
                    label="Avg Price (₹ Lakhs)",
                    value=f"{avg_price:,.1f}" if avg_price is not None else "–",
                    subtitle="Mean property price",
                ),
                unsafe_allow_html=True,
            )

            k4.markdown(
                card_tpl.format(
                    label="Avg Size (SqFt)",
                    value=f"{avg_size:,.0f}" if avg_size is not None else "–",
                    subtitle="Average carpet area",
                ),
                unsafe_allow_html=True,
            )

        # Charts (2x2 grid)  ✅ unchanged, all insights still here
        with chart_col:
            top_row = st.columns(2)
            bottom_row = st.columns(2)

            # 1) City-wise average price
            with top_row[0]:
                st.subheader("City-wise Avg Price")
                if {location_col, price_col}.issubset(df_vis.columns):
                    city_price = (
                        df_vis.groupby(location_col)[price_col]
                        .mean()
                        .reset_index()
                        .sort_values(price_col, ascending=False)
                        .head(10)
                    )
                    fig_city = px.bar(
                        city_price,
                        x=price_col,
                        y=location_col,
                        orientation="h",
                        color=location_col,
                        color_discrete_sequence=PASTEL_COLORS,
                        labels={price_col: "Avg Price (Lakhs)", location_col: "City"},
                        template="plotly_white",
                    )
                    fig_city.update_layout(
                        showlegend=False,
                        height=320,
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig_city, use_container_width=True)
                else:
                    st.info("`City` or `Price_in_Lakhs` column not found; cannot plot city-wise average price.")

            # 2) BHK distribution
            with top_row[1]:
                st.subheader("BHK Distribution")
                if bhk_col in df_vis.columns:
                    bhk_counts = (
                        df_vis[bhk_col]
                        .value_counts()
                        .sort_index()
                        .reset_index()
                    )
                    bhk_counts.columns = [bhk_col, "Count"]

                    fig_bhk = px.bar(
                        bhk_counts,
                        x=bhk_col,
                        y="Count",
                        color=bhk_col,
                        color_discrete_sequence=PASTEL_COLORS,
                        labels={bhk_col: "BHK", "Count": "Number of Properties"},
                        template="plotly_white",
                    )
                    fig_bhk.update_layout(
                        showlegend=False,
                        height=320,
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig_bhk, use_container_width=True)
                else:
                    st.info("`BHK` column not found; cannot plot BHK distribution.")

            # 3) Size vs Price
            with bottom_row[0]:
                st.subheader("Size vs Price")
                if {area_col, price_col}.issubset(df_vis.columns):
                    fig_scatter = px.scatter(
                        df_vis,
                        x=area_col,
                        y=price_col,
                        color=bhk_col if bhk_col in df_vis.columns else None,
                        color_discrete_sequence=PASTEL_COLORS,
                        opacity=0.6,
                        labels={area_col: "Size (SqFt)", price_col: "Price (Lakhs)"},
                        template="plotly_white",
                    )
                    fig_scatter.update_layout(
                        height=320,
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Area or price column not found; cannot plot size vs price.")

            # 4) Property Age vs Price
            with bottom_row[1]:
                st.subheader("Property Age vs Price")
                if age_col in df_vis.columns and price_col in df_vis.columns:
                    age_df = df_vis[[age_col, price_col]].dropna().copy()
                    age_df["Age_Bucket"] = pd.cut(
                        age_df[age_col],
                        bins=[0, 5, 10, 20, 30, 50],
                        labels=["0-5", "5-10", "10-20", "20-30", "30+"],
                        include_lowest=True,
                    )
                    fig_age = px.box(
                        age_df,
                        x="Age_Bucket",
                        y=price_col,
                        color="Age_Bucket",
                        color_discrete_sequence=PASTEL_COLORS,
                        labels={
                            "Age_Bucket": "Age of Property (years)",
                            price_col: "Price (Lakhs)",
                        },
                        template="plotly_white",
                    )
                    fig_age.update_layout(
                        showlegend=False,
                        height=320,
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig_age, use_container_width=True)
                else:
                    st.info("`Age_of_Property` or price column not found; cannot plot age vs price.")

# =================================================
# TAB 5 — FEATURE IMPORTANCE 
# =================================================
elif active_tab == "importance":
    st.header("🧠 Feature Importance (Regression Model)")
    st.caption(
        "We compute feature importance for the 5-year price prediction. "
        "Two views: (A) full model (may include Price_in_Lakhs) and (B) model importance *excluding* Price_in_Lakhs to inspect other drivers."
    )

    reg_steps = getattr(reg_pipeline, "named_steps", {})
    reg_inner = reg_steps.get("reg", reg_pipeline)  # note your pipeline may use 'reg' or 'model'
    preproc = reg_steps.get("preproc", reg_steps.get("preprocessor", None))

    # helper to compute permutation importance (safe)
    def compute_perm_importance(pipeline, X_base, y_base, max_rows=1000):
        try:
            sample_n = min(max_rows, len(X_base))
            X_sample = X_base.sample(sample_n, random_state=42)
            y_sample = y_base.loc[X_sample.index]
            result = permutation_importance(
                pipeline, X_sample, y_sample, n_repeats=5, random_state=42, n_jobs=-1
            )
            importances = np.maximum(result.importances_mean, 0)
            return importances, X_sample.columns
        except Exception as e:
            st.error(f"Permutation importance failed: {e}")
            return None, None

    # Try to use saved test split
    if splits is not None and "X_test_reg" in splits and "y_test_reg" in splits:
        X_base = splits["X_test_reg"]
        y_base = splits["y_test_reg"]
        if not isinstance(X_base, pd.DataFrame):
            X_base = pd.DataFrame(X_base, columns=feature_cols)
    else:
        X_base = df[feature_cols].copy()
        y_base = df[REG_TARGET_COL] if REG_TARGET_COL in df.columns else None

    if y_base is None:
        st.error("Regression target unavailable. Cannot compute importances.")
    else:
        colA, colB = st.columns(2)

        # A) Full model importances (like before)
        with colA:
            st.subheader("A — Full Model (includes Price_in_Lakhs)")
            imp_full, feat_names = compute_perm_importance(reg_pipeline, X_base, y_base)
            if imp_full is not None and feat_names is not None:
                imp_df = pd.DataFrame({"feature": feat_names, "importance": imp_full})
                imp_df["importance_pct"] = 100 * imp_df["importance"] / imp_df["importance"].sum()
                imp_df = imp_df.sort_values("importance_pct", ascending=False).reset_index(drop=True)
                st.dataframe(imp_df.head(20), use_container_width=True)
                # simple bar chart (top 15)
                top = imp_df.head(15)
                fig_full = px.bar(top.iloc[::-1], x="importance_pct", y="feature", orientation="h",
                                  labels={"importance_pct": "Importance (%)", "feature": "Feature"},
                                  template="plotly_white")
                fig_full.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_full, use_container_width=True)
            else:
                st.info("Could not compute full-model permutation importances.")

        # B) Importance with Price removed
        with colB:
            st.subheader("B — Without Price_in_Lakhs (inspect other drivers)")
            if price_col in X_base.columns:
                X_no_price = X_base.drop(columns=[price_col])
                imp_no_price, feat_names_no = compute_perm_importance(reg_pipeline, X_no_price, y_base)
                if imp_no_price is not None and feat_names_no is not None:
                    imp_df2 = pd.DataFrame({"feature": feat_names_no, "importance": imp_no_price})
                    imp_df2["importance_pct"] = 100 * imp_df2["importance"] / imp_df2["importance"].sum()
                    imp_df2 = imp_df2.sort_values("importance_pct", ascending=False).reset_index(drop=True)
                    st.dataframe(imp_df2.head(20), use_container_width=True)
                    top2 = imp_df2.head(15)
                    fig_no_price = px.bar(top2.iloc[::-1], x="importance_pct", y="feature", orientation="h",
                                          labels={"importance_pct": "Importance (%)", "feature": "Feature"},
                                          template="plotly_white")
                    fig_no_price.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_no_price, use_container_width=True)
                else:
                    st.info("Could not compute importance without price.")
            else:
                st.info(f"Column {price_col} not present; showing full importances only.")

        st.markdown("---")
        st.caption(
            "Note: Price_in_Lakhs is naturally highly predictive of future price. View (B) to understand other important features "
            "(infrastructure, appreciation rate, city, availability) without the price signal dominating."
        )


# =================================================
# TAB 6 — MODEL INFO
# =================================================
elif active_tab == "model":
    st.header("⚙️ Model Information & Diagnostics")

    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.subheader("Regression Model")
        inner_reg = getattr(reg_pipeline, "named_steps", {}).get("reg", reg_pipeline)
        st.write(type(inner_reg).__name__)
        # print some useful attributes (if available)
        for attr in ("n_estimators", "max_depth", "alpha"):
            if hasattr(inner_reg, attr):
                st.write(f"{attr}: {getattr(inner_reg, attr)}")

    with col_m2:
        st.subheader("Classification Model")
        inner_clf = getattr(clf_pipeline, "named_steps", {}).get("clf", clf_pipeline)
        st.write(type(inner_clf).__name__)
        for attr in ("n_estimators", "max_depth", "class_weight"):
            if hasattr(inner_clf, attr):
                st.write(f"{attr}: {getattr(inner_clf, attr)}")

    with col_m3:
        st.subheader("Dataset Shape")
        st.write(f"Full dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
        st.write(f"Filtered dataset: {filtered_df.shape[0]:,} rows")

    st.markdown("---")
    st.subheader("Metrics (Test Split, if available)")

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix, classification_report

    rmse = r2 = acc = f1 = None

    # Regression metrics
    if splits is not None and "X_test_reg" in splits and "y_test_reg" in splits:
        try:
            X_test_reg = splits["X_test_reg"]
            y_test_reg = splits["y_test_reg"]
            y_pred_reg = reg_pipeline.predict(X_test_reg)
            mse = mean_squared_error(y_test_reg, y_pred_reg)
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test_reg, y_pred_reg))
        except Exception:
            rmse, r2 = None, None

    # Classification metrics + confusion matrix
    cm = None
    class_report = None
    if splits is not None and "X_test" in splits and "y_test_class" in splits:
        try:
            X_test_cls = splits["X_test"]
            y_test_cls = splits["y_test_class"]
            y_pred_cls = clf_pipeline.predict(X_test_cls)
            acc = float(accuracy_score(y_test_cls, y_pred_cls))
            f1 = float(f1_score(y_test_cls, y_pred_cls, average="binary"))
            cm = confusion_matrix(y_test_cls, y_pred_cls)
            class_report = classification_report(y_test_cls, y_pred_cls, output_dict=True)
        except Exception:
            acc, f1, cm, class_report = None, None, None, None

    # Display numeric metrics
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("RMSE (Regression)", f"{rmse:.3f}" if rmse is not None else "–")
    with mcol2:
        st.metric("R² (Regression)", f"{r2:.3f}" if r2 is not None else "–")
    with mcol3:
        st.metric("Accuracy (Classification)", f"{acc * 100:.1f}%" if acc is not None else "–")
    with mcol4:
        st.metric("F1-score (Classification)", f"{f1:.3f}" if f1 is not None else "–")

    # Confusion matrix heatmap
    if cm is not None:
        st.markdown("#### Confusion Matrix (Classification)")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    # Show classification report
    if class_report is not None:
        st.markdown("#### Classification Report")
        # convert to table for nice display
        rep_df = pd.DataFrame(class_report).transpose()
        st.dataframe(rep_df, use_container_width=True)

    # Short plain-English summary
    st.markdown("---")
    st.subheader("Quick Summary")
    summary_msgs = []
    if acc is not None and f1 is not None:
        summary_msgs.append(f"The classifier has **{acc*100:.1f}% accuracy** and **F1 {f1:.3f}** on the saved test split.")
    if rmse is not None and r2 is not None:
        summary_msgs.append(f"The regressor has **RMSE {rmse:.3f}** and **R² {r2:.3f}**, indicating very tight predictions (check for leakage if R² too close to 1).")
    if cm is not None:
        summary_msgs.append("Confusion matrix shows how many examples were correctly/incorrectly classified by class.")
    if not summary_msgs:
        st.write("Test split metrics are not available in the saved splits.")

    for s in summary_msgs:
        st.write("- " + s)

# =================================================
# TAB 7 — ABOUT
# =================================================
elif active_tab == "about":
    st.header("ℹ️ About This App")
    st.write(
        "This Real Estate Investment Advisor was built as a part of an academic project to explore "
        "end-to-end machine learning, experiment tracking with MLflow, and deployment with Streamlit."
    )

    st.subheader("Project Goal")
    st.write(
        "To help users understand whether a given property is a good investment and estimate its "
        "future price based on historical and engineered features."
    )

    st.subheader("Tech Stack")
    st.write("- Python, pandas, numpy")
    st.write("- scikit-learn (Random Forest / other models)")
    st.write("- MLflow for experiment tracking (during training)")
    st.write("- Streamlit for interactive web UI")
    st.write("- joblib for model persistence")

    st.subheader("How to Use")
    st.write(
        "1. Go to the **Investment Prediction** view and enter property details.\n"
        "2. Check if it is a good investment and the 5-year price estimate.\n"
        "3. Use **Property Explorer** and **Visual Insights** to explore the dataset.\n"
        "4. Inspect **Feature Importance** and **Model Info** for deeper understanding."
    )

    st.subheader("Credits")
    st.write("Developed by: *[Jayshree Pawar]*")
 
