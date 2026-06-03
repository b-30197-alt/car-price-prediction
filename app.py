import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration Setup
st.set_page_config(
    page_title="Car Price Prediction Dashboard",
    page_icon="🚗",
    layout="centered"
)

# Custom premium styling for dashboard cards
st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; text-align: center; color: #2c3e50; margin-bottom: 5px; }
    .subtitle { font-size: 16px; text-align: center; color: #7f8c8d; margin-bottom: 30px; }
    .metric-box { background-color: #e8f8f5; border-left: 6px solid #27ae60; padding: 20px; border-radius: 8px; text-align: center; margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Car Valuation ML Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter the vehicle configurations below to approximate real-time market value benchmarks.</div>', unsafe_allow_html=True)

# 2. Secure Model Binary Loader
@st.cache_resource
def load_ml_pipeline():
    return joblib.load("car_price_best_model (1).pkl")

try:
    pipeline = load_ml_pipeline()
except Exception as e:
    st.error("⚠️ Error: Unable to locate or serialize 'car_price_best_model (1).pkl'. Make sure it is saved in the same directory.")
# st.write(f"The minimum value is {np.min(selling_price)}")
# st.write(f"The maximum value is {np.max(selling_price)}")
# st.write(f"The average value is {np.mean(selling_price)}")
st.subheader("Vehicle Configuration Specifications")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand / Manufacturer", [
        'Maruti', 'Hyundai', 'Honda', 'Toyota', 'Mahindra', 'Ford', 'Tata', 
        'Chevrolet', 'Skoda', 'BMW', 'Audi', 'Volkswagen', 'Nissan', 'Renault', 
        'Fiat', 'Datsun', 'Jaguar', 'Mercedes-Benz', 'Land', 'Mitsubishi', 'Volvo', 
        'Jeep', 'Ambassador', 'MG', 'Kia', 'Lexus', 'Force', 'Isuzu'
    ])
    
    car_age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=40, value=5, step=1)
    
    fuel = st.selectbox("Fuel Category Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])

with col2:
    km_driven = st.number_input("Total Kilometers Driven", min_value=0, max_value=500000, value=45000, step=1000)
    
    transmission = st.selectbox("Transmission System", ['Manual', 'Automatic'])
    
    seller_type = st.selectbox("Seller Classification", ['Individual', 'Dealer', 'Trustmark Dealer'])

owner = st.selectbox("Ownership Ledger Status", [
    'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'
])

# 4. Pipeline Valuation Prediction
if st.button("Calculate Estimated Value", type="primary", use_container_width=True):
    input_dataframe = pd.DataFrame([{
        'km_driven': km_driven,
        'car_age': car_age,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner,
        'brand': brand
    }])
    
    try:
        predicted_array = pipeline.predict(input_dataframe)
        raw_prediction = predicted_array[0]
        final_price = max(15000, raw_prediction)
        formatted_price = f"₹{final_price:,.2f}"
        
        st.markdown(f"""
            <div class="metric-box">
                <h3 style="color: #1e8449; margin-bottom: 5px; font-size:14px; text-transform: uppercase; letter-spacing:0.5px;">Estimated Valuation Result</h3>
                <div style="font-size: 32px; font-weight: bold; color: #2c3e50;">{formatted_price}</div>
            </div>
        """, unsafe_allow_html=True)
    except Exception as err:
        st.error(f"Execution Error inside transformation matrices: {err}")
