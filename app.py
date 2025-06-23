import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import json

# Paths
MODEL_DIR = 'models'
PIPELINE_FILE = os.path.join(MODEL_DIR, 'best_house_price_pipeline.pkl')
FEATURES_FILE = os.path.join(MODEL_DIR, 'feature_columns.json')

# Load trained model pipeline
@st.cache_resource
def load_model():
    if not os.path.exists(PIPELINE_FILE):
        st.error(f"Model file not found at {PIPELINE_FILE}.")
        return None
    try:
        pipeline = joblib.load(PIPELINE_FILE)
        return pipeline
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load expected feature list
@st.cache_data
def load_feature_list():
    if not os.path.exists(FEATURES_FILE):
        st.error(f"Feature list not found at {FEATURES_FILE}.")
        return None
    try:
        with open(FEATURES_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading feature list: {e}")
        return None

pipeline = load_model()
expected_features = load_feature_list()

st.title('🏠 House Price Prediction')

if pipeline is not None and expected_features is not None:
    st.sidebar.header("🏗️ Input House Features")

    # Input fields
    bedrooms = st.sidebar.number_input('Bedrooms', min_value=0, max_value=30, value=3)
    bathrooms = st.sidebar.number_input('Bathrooms', min_value=0.0, max_value=10.0, value=2.25, step=0.25)
    sqft_living = st.sidebar.number_input('Sqft Living', min_value=0, value=2000)
    sqft_lot = st.sidebar.number_input('Sqft Lot', min_value=0, value=5000)
    floors = st.sidebar.number_input('Floors', min_value=1.0, max_value=5.0, value=1.0, step=0.5)
    waterfront_input = st.sidebar.selectbox('Waterfront (1=Yes, 0=No)', options=[0, 1])
    view = st.sidebar.number_input('View (0-4)', min_value=0, max_value=4, value=0)
    condition = st.sidebar.number_input('Condition (1-5)', min_value=1, max_value=5, value=3)
    grade = st.sidebar.number_input('Grade (1-13)', min_value=1, max_value=13, value=7)
    sqft_above = st.sidebar.number_input('Sqft Above Ground', min_value=0, value=2000)
    sqft_basement = st.sidebar.number_input('Sqft Basement', min_value=0, value=0)
    zipcode = st.sidebar.number_input('Zipcode', min_value=98001, max_value=98199, value=98001)
    lat = st.sidebar.number_input('Latitude', format="%.6f", value=47.5)
    long = st.sidebar.number_input('Longitude', format="%.6f", value=-122.2)
    sqft_living15 = st.sidebar.number_input('Sqft Living (2015)', min_value=0, value=2000)
    sqft_lot15 = st.sidebar.number_input('Sqft Lot (2015)', min_value=0, value=5000)
    yr_built = st.sidebar.number_input('Year Built', min_value=1900, max_value=2024, value=2000)
    yr_renovated = st.sidebar.number_input('Year Renovated (0 if not renovated)', min_value=0, max_value=2024, value=0)
    sale_year = st.sidebar.number_input('Sale Year', min_value=2014, max_value=2024, value=2023)
    sale_month = st.sidebar.number_input('Sale Month', min_value=1, max_value=12, value=1)

    input_data_dict = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront_input,
        'view': view,
        'condition': condition,
        'grade': grade,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement,
        'zipcode': zipcode,
        'lat': lat,
        'long': long,
        'sqft_living15': sqft_living15,
        'sqft_lot15': sqft_lot15,
        'yr_built': yr_built,
        'yr_renovated': yr_renovated,
        'sale_year': sale_year,
        'sale_month': sale_month
    }

    input_df = pd.DataFrame([input_data_dict])

    def apply_streamlit_feature_engineering(df_input):
        df_input['house_age'] = df_input['sale_year'] - df_input['yr_built']
        df_input['years_since_renov'] = np.where(df_input['yr_renovated'] == 0,
                                                 df_input['house_age'],
                                                 df_input['sale_year'] - df_input['yr_renovated'])
        df_input['total_area'] = df_input['sqft_living'] + df_input['sqft_basement']
        df_input['lot_ratio'] = df_input['sqft_living'] / df_input['sqft_lot']
        df_input['lot_ratio'] = df_input['lot_ratio'].replace([np.inf, -np.inf], 0)
        df_input['living_ratio'] = df_input['sqft_living'] / df_input['sqft_living15']
        df_input['living_ratio'] = df_input['living_ratio'].replace([np.inf, -np.inf], 0)
        df_input['bed_bath_ratio'] = df_input['bedrooms'] / df_input['bathrooms']
        df_input['bed_bath_ratio'] = df_input['bed_bath_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
        df_input['rooms'] = df_input['bedrooms'] + df_input['bathrooms']
        df_input['is_waterfront'] = df_input['waterfront'].apply(lambda x: 1 if x > 0 else 0)
        df_input['has_view'] = df_input['view'].apply(lambda x: 1 if x > 0 else 0)

        # Drop columns
        df_input = df_input.drop(columns=['yr_built', 'yr_renovated'])

        # Add dummy price_per_sqft if required
        df_input['price_per_sqft'] = df_input['sqft_living'] * 0.5  # Dummy constant or average logic

        return df_input

    input_df_engineered = apply_streamlit_feature_engineering(input_df)

    # Ensure all required features exist
    for col in expected_features:
        if col not in input_df_engineered.columns:
            input_df_engineered[col] = 0  # or mean of training set if known

    input_df_final = input_df_engineered[expected_features]  # align order

    if st.button('🔍 Predict House Price'):
        try:
            prediction = pipeline.predict(input_df_final)[0]
            st.success(f'🏡 Estimated House Price: **${prediction:,.2f}**')
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.warning("❗ Model or feature list not loaded.")
