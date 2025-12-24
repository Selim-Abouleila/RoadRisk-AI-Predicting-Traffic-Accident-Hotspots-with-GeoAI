import streamlit as st
import pandas as pd
import pickle
import folium
from streamlit_folium import st_folium
import numpy as np

# Page config
st.set_page_config(
    page_title="Paris Crash Analytics",
    page_icon="traffic_light", # Standard Streamlit icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern Formal UI ---
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0f172a; /* Slate 900 - Solid professional background */
        color: #e2e8f0;
        font-family: 'Segoe UI', 'Roboto', Helvetica, Arial, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px;
    }
    
    h1 {
        margin-bottom: 1.5rem !important;
        border-bottom: 1px solid #334155;
        padding-bottom: 1rem;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e293b; /* Slate 800 */
        border-right: 1px solid #334155;
    }
    
    .sidebar-header {
        font-size: 1rem;
        font-weight: 700;
        color: #cbd5e1;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Cards & Containers */
    .metric-card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Input Elements */
    .stSelectbox label, .stSlider label {
        color: #cbd5e1 !important;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #2563eb; /* Blue 600 */
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-weight: 600;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8; /* Blue 700 */
        border: none;
        color: white;
    }
    
    /* info boxes */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Hide the default Streamlit header decoration (the colored band) */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    
    /* If specifically referring to the top decoration bar */
    .stApp > header {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

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

# Title and Context
st.title("Paris Crash Analytics")
st.markdown("### Severity Prediction Dashboard")
st.markdown("""
<div style='color: #94a3b8; margin-bottom: 2rem; font-size: 0.95rem;'>
    Use this dashboard to assess the risk of serious injury based on accident parameters and location. 
    Select a coordinate on the map to begin.
</div>
""", unsafe_allow_html=True)

# Sidebar Inputs
with st.sidebar:
    st.markdown('<div class="sidebar-header">Demographics & Time</div>', unsafe_allow_html=True)
    
    inputs = {}
    inputs['annee'] = st.slider("Year", 2019, 2025, 2024)
    inputs['age'] = st.slider("Age of Victim", 0, 100, 30)
    
    st.markdown('<div class="sidebar-header">Accident Conditions</div>', unsafe_allow_html=True)
    
    # Categorical Inputs
    friendly_names = {
        'victime_type': "Transport Mode",
        'sexe_victime': "Gender",
        'categorie': "Victim Status",
        'milieu': "Environment"
    }
    
    for col in ['victime_type', 'sexe_victime', 'categorie', 'milieu']:
        options = cat_mappings.get(col, [])
        # Plain text labels for formal look
        label = friendly_names.get(col, col.replace('_', ' ').title())
        inputs[col] = st.selectbox(label, options)

# Main Layout
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("#### Location Selector", unsafe_allow_html=True)
    
    default_lat, default_lon = 48.8566, 2.3522
    # Switch to 'CartoDB positron' for a clean "White" map
    m = folium.Map(location=[default_lat, default_lon], zoom_start=12, tiles="CartoDB positron")
    m.add_child(folium.LatLngPopup())
    
    # Map
    map_output = st_folium(m, height=550, width="100%")

    if map_output['last_clicked']:
        lat = map_output['last_clicked']['lat']
        lon = map_output['last_clicked']['lng']
    else:
        lat, lon = default_lat, default_lon

    st.markdown(f"""
    <div style='background-color: #1e293b; padding: 0.75rem; border-radius: 6px; border: 1px solid #334155; margin-top: 1rem; font-family: monospace; color: #7dd3fc;'>
        LAT: {lat:.5f} | LON: {lon:.5f}
    </div>
    """, unsafe_allow_html=True)
    
    inputs['latitude'] = lat
    inputs['longitude'] = lon

with col2:
    st.markdown("#### Risk Analysis", unsafe_allow_html=True)
    
    # Spacer
    st.write("")
    
    if st.button("Calculate Risk", type="primary", use_container_width=True):
        input_df = pd.DataFrame([inputs])
        input_df = input_df[feature_names]
        
        for col in ['victime_type', 'sexe_victime', 'categorie', 'milieu']:
            cat_type = pd.CategoricalDtype(categories=cat_mappings[col], ordered=False)
            input_df[col] = input_df[col].astype(cat_type)
            
        MEAN_AGE = 37.7751
        STD_AGE = 17.2978
        input_df['age'] = (input_df['age'] - MEAN_AGE) / STD_AGE
            
        try:
            probs = model.predict_proba(input_df)[0]
            prob_serious = probs[1]
            THRESHOLD = 0.645
            is_serious = prob_serious >= THRESHOLD
            
            # Formal Status Colors
            if prob_serious > 0.8:
                color = "#ef4444" # Red
                status = "CRITICAL"
                desc = "High probability of severe outcome"
            elif prob_serious > 0.5:
                color = "#f59e0b" # Amber
                status = "ELEVATED"
                desc = "Moderate risk factor detected"
            else:
                color = "#10b981" # Emerald
                status = "LOW"
                desc = "Standard risk profile"
            
            # Simple, Professional Card
            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {color}; text-align: left; padding: 1.25rem;">
                <div style="color: {color}; font-weight: 700; font-size: 0.85rem; letter-spacing: 1px; margin-bottom: 0.25rem;">RISK LEVEL</div>
                <div class="metric-value" style="font-size: 1.75rem; margin-top: 0;">{status}</div>
                <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 0.5rem;">{desc}</div>
                <div style="margin-top: 1.5rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.25rem;">
                        <span>Probability</span>
                        <span>{prob_serious:.1%}</span>
                    </div>
                    <div style="width: 100%; height: 6px; background: #334155; border-radius: 3px;">
                        <div style="width: {prob_serious*100}%; height: 100%; background-color: {color}; border-radius: 3px;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if is_serious:
                st.warning("Prediction indicates potential for serious injury.")
            else:
                st.success("Prediction indicates likelihood of minor injury.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

st.markdown("---")
st.caption("RoadRisk AI System | v2.0")
