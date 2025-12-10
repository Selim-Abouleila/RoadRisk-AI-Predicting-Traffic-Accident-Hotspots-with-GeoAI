import streamlit as st
import pandas as pd
import pickle
import folium
from streamlit_folium import st_folium
import numpy as np

# Page config
st.set_page_config(
    page_title="Paris Accident Severity Predictor",
    page_icon="🚗",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    with open('models/xgboost_accident_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model_pkg = load_model()
    model = model_pkg['model']
    cat_mappings = model_pkg['categorical_mappings']
    feature_names = model_pkg['feature_names']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Title and Description
st.title("🚗 Paris Accident Severity Predictor")
st.markdown("""
This application predicts the severity of a traffic accident in Paris based on various factors.
**Select the location on the map** and adjust the parameters in the sidebar.
""")

# Sidebar Inputs
st.sidebar.header("📝 Accident Details")

# Dynamic inputs based on model mappings
inputs = {}

# Year (Annee)
inputs['annee'] = st.sidebar.slider("Year", 2019, 2025, 2024)

# Age
inputs['age'] = st.sidebar.slider("Age of Victim", 0, 100, 30)

# Categorical Inputs
# Define friendly names for columns
friendly_names = {
    'victime_type': "Victim's Transport Mode (What were they using?)",
    'sexe_victime': "Victim's Gender",
    'categorie': "Victim's Status (Driver, Passenger, or Pedestrian)",
    'milieu': "Accident Environment (Urban or Rural)"
}

for col in ['victime_type', 'sexe_victime', 'categorie', 'milieu']:
    options = cat_mappings.get(col, [])
    label = friendly_names.get(col, col.replace('_', ' ').title())
    inputs[col] = st.sidebar.selectbox(label, options)

# Map Section
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 Location Selection")
    
    # Default to Paris center
    default_lat, default_lon = 48.8566, 2.3522
    
    m = folium.Map(location=[default_lat, default_lon], zoom_start=12)
    m.add_child(folium.LatLngPopup())
    
    # Display map and get output
    map_output = st_folium(m, height=500, width="100%")

    # Get coordinates from click
    if map_output['last_clicked']:
        lat = map_output['last_clicked']['lat']
        lon = map_output['last_clicked']['lng']
    else:
        lat, lon = default_lat, default_lon

    st.info(f"Selected Coordinates: Lat: **{lat:.6f}**, Lon: **{lon:.6f}**")
    inputs['latitude'] = lat
    inputs['longitude'] = lon

# Prediction Logic
with col2:
    st.subheader("🔮 Prediction")
    
    if st.button("Predict Severity", type="primary", use_container_width=True):
        # Prepare input dataframe
        input_df = pd.DataFrame([inputs])
        
        # Ensure correct column order
        input_df = input_df[feature_names]
        
        # Convert categorical to category dtype with EXPLICIT categories
        # This fixes the issue where single-row inputs get code 0
        for col in ['victime_type', 'sexe_victime', 'categorie', 'milieu']:
            cat_type = pd.CategoricalDtype(categories=cat_mappings[col], ordered=False)
            input_df[col] = input_df[col].astype(cat_type)
            
        # Scale Age (Model expects normalized age)
        # Exact stats from paris_accidents_cleaned.parquet
        MEAN_AGE = 37.7751
        STD_AGE = 17.2978
        input_df['age'] = (input_df['age'] - MEAN_AGE) / STD_AGE
            
        # Run prediction
        try:
            # Get probabilities
            probs = model.predict_proba(input_df)[0]
            prob_minor = probs[0]
            prob_serious = probs[1]
            
            # Threshold (using the tuned one)
            THRESHOLD = 0.645
            is_serious = prob_serious >= THRESHOLD
            
            # Display Result
            st.markdown("---")
            if is_serious:
                st.error("### ⚠️ Prediction: Serious/Fatal")
                st.metric("Probability of Serious Crash", f"{prob_serious:.1%}")
            else:
                st.success("### ✅ Prediction: Minor")
                st.metric("Probability of Serious Crash", f"{prob_serious:.1%}")
                
            # Detailed breakdown
            st.markdown("#### Risk Analysis")
            st.progress(float(prob_serious), text="Risk Level")
            
            if prob_serious > 0.8:
                st.warning("Extreme risk detected! Please exercise extreme caution.")
            elif prob_serious > 0.5:
                st.warning("Moderate to High risk.")
            else:
                st.info("Low risk scenario.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# Footer
st.markdown("---")
st.caption("Powered by XGBoost & RoadRisk AI")
