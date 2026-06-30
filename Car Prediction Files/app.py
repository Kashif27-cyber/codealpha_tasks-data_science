"""
Car Price Prediction — Interactive Streamlit Web App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title="Car Price Predictor 🚗",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-header { text-align: center; color: #555; margin-bottom: 2rem; }
    .metric-card {
        background: #f0f4ff; border-radius: 12px; padding: 1rem;
        border-left: 4px solid #1a73e8; margin: 0.5rem 0;
    }
    .price-display {
        font-size: 2rem; font-weight: 800; color: #1a73e8; text-align: center;
        padding: 1.5rem; background: #e8f0fe; border-radius: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚗 Car Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning powered price estimation for used cars</div>',
            unsafe_allow_html=True)

# ─── Load model ───
MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')

@st.cache_resource
def load_model():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'car_data.csv'))

try:
    model, scaler = load_model()
    df = load_data()
    model_loaded = True
except Exception as e:
    st.error(f"Model not found. Run `python data/train_model.py` first.\n{e}")
    model_loaded = False

# ─── Sidebar: Input ───
st.sidebar.header("🔧 Car Configuration")

goodwill_map = {
    'Maruti': 0.95, 'Hyundai': 0.90, 'Honda': 0.88, 'Toyota': 0.92,
    'Ford': 0.82, 'Volkswagen': 0.85, 'Skoda': 0.80, 'Audi': 0.75,
    'BMW': 0.72, 'Mercedes': 0.70, 'Mahindra': 0.78, 'Tata': 0.76,
    'Renault': 0.79, 'Datsun': 0.77, 'Chevrolet': 0.74, 'Fiat': 0.73,
    'Jaguar': 0.68, 'Land Rover': 0.65, 'Jeep': 0.71, 'Nissan': 0.80,
}

car_name     = st.sidebar.selectbox("Brand", sorted(goodwill_map.keys()))
year         = st.sidebar.slider("Year of Manufacture", 2000, 2023, 2016)
present_price= st.sidebar.number_input("Showroom Price (₹ Lakhs)", 1.0, 100.0, 8.0, 0.5)
kms_driven   = st.sidebar.number_input("Kilometres Driven", 500, 600000, 45000, 1000)
fuel_type    = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'Electric'])
seller_type  = st.sidebar.selectbox("Seller Type", ['Individual', 'Dealer'])
transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
owner        = st.sidebar.selectbox("Previous Owners", [0, 1, 2, 3])
mileage      = st.sidebar.slider("Mileage (kmpl)", 5.0, 40.0, 18.5, 0.5)
engine       = st.sidebar.selectbox("Engine CC",
    [800, 998, 1197, 1248, 1368, 1498, 1598, 1794, 1968, 2199, 2993, 3498])
max_power    = st.sidebar.number_input("Max Power (bhp)", 30.0, 450.0, 82.0, 1.0)
seats        = st.sidebar.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10], index=2)

# ─── Feature Engineering ───
fuel_enc    = {'Petrol': 2, 'Diesel': 1, 'CNG': 0, 'Electric': 3}[fuel_type]
seller_enc  = {'Individual': 1, 'Dealer': 0}[seller_type]
trans_enc   = {'Manual': 1, 'Automatic': 0}[transmission]
car_age     = 2020 - year
kms_per_yr  = kms_driven / (car_age + 1)
power_per_cc= max_power / engine
brand_good  = goodwill_map.get(car_name, 0.75)

features_input = pd.DataFrame([{
    'Present_Price': present_price, 'Kms_Driven': kms_driven,
    'Car_Age': car_age, 'Owner': owner, 'Fuel_Type_enc': fuel_enc,
    'Seller_Type_enc': seller_enc, 'Transmission_enc': trans_enc,
    'Mileage': mileage, 'Engine': engine, 'Max_Power': max_power,
    'Seats': seats, 'Brand_Goodwill': brand_good,
    'Kms_Per_Year': kms_per_yr, 'Power_Per_CC': power_per_cc,
}])

# ─── Main Area ───
if model_loaded:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎯 Price Prediction")
        if st.button("💰 Predict Selling Price", use_container_width=True, type="primary"):
            pred = model.predict(features_input)[0]
            pred = max(0.1, pred)
            savings = present_price - pred
            depreciation = (savings / present_price) * 100

            st.markdown(f"""
            <div class="price-display">
                Estimated Price: ₹ {pred:.2f} Lakhs
            </div>""", unsafe_allow_html=True)

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("Showroom Price", f"₹{present_price:.1f}L")
            c2.metric("Predicted Price", f"₹{pred:.2f}L")
            c3.metric("Depreciation", f"{depreciation:.1f}%", delta=f"-₹{savings:.2f}L")

            # Confidence breakdown
            st.markdown("#### 📊 Key Factors")
            factor_data = {
                'Brand Goodwill': brand_good,
                'Age Factor': max(0, 1 - car_age * 0.07),
                'Mileage Score': mileage / 40,
                'Power Score': max_power / 400,
                'KMs Factor': max(0, 1 - kms_driven / 500000),
            }
            fig, ax = plt.subplots(figsize=(6, 3))
            bars = ax.barh(list(factor_data.keys()),
                           list(factor_data.values()),
                           color=sns.color_palette('Blues_r', len(factor_data)))
            ax.set_xlim(0, 1.1)
            ax.set_title("Factor Scores (0–1)")
            for bar, v in zip(bars, factor_data.values()):
                ax.text(v + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{v:.2f}', va='center')
            st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.subheader("📈 Market Insights")

        # Price distribution
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df['Selling_Price'], bins=25, color='steelblue',
                edgecolor='white', alpha=0.7, label='All Cars')
        ax.axvline(present_price, color='orange', linestyle='--',
                   lw=2, label=f'Showroom: ₹{present_price}L')
        ax.set_xlabel("Price (₹ Lakhs)"); ax.set_ylabel("Count")
        ax.set_title("Price Distribution in Dataset"); ax.legend()
        st.pyplot(fig, use_container_width=True); plt.close()

        # Depreciation by age
        df2 = df.copy()
        df2['Car_Age'] = 2020 - df2['Year']
        age_price = df2.groupby('Car_Age')['Selling_Price'].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.plot(age_price['Car_Age'], age_price['Selling_Price'],
                 'o-', color='coral', lw=2)
        ax2.axvline(car_age, color='navy', linestyle='--', lw=1.5,
                    label=f'Your car age: {car_age}y')
        ax2.set_xlabel("Car Age (years)"); ax2.set_ylabel("Avg Price (₹ Lakhs)")
        ax2.set_title("Average Price by Car Age"); ax2.legend()
        st.pyplot(fig2, use_container_width=True); plt.close()

    # ─── Dataset Explorer ───
    st.markdown("---")
    st.subheader("🗂️ Dataset Explorer")
    col_a, col_b = st.columns(2)
    with col_a:
        brand_filter = st.multiselect("Filter by Brand",
            sorted(df['Car_Name'].unique()), default=[])
    with col_b:
        fuel_filter = st.multiselect("Filter by Fuel Type",
            df['Fuel_Type'].unique().tolist(), default=[])

    filtered = df.copy()
    if brand_filter: filtered = filtered[filtered['Car_Name'].isin(brand_filter)]
    if fuel_filter:  filtered = filtered[filtered['Fuel_Type'].isin(fuel_filter)]
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=250)
    st.caption(f"Showing {len(filtered)} of {len(df)} records")

# ─── Footer ───
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem;'>
    🚗 Car Price Prediction ML Project &nbsp;|&nbsp;
    Built with Python · Scikit-learn · Streamlit &nbsp;|&nbsp;
    Dataset: Kaggle — vijayaadithyanvg/car-price-predictionused-cars
</div>
""", unsafe_allow_html=True)
