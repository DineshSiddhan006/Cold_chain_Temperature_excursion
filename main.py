import os
import joblib
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin

# Set up a clean, wide web page layout
st.set_page_config(
    page_title="Cold Chain Temperature Excursion Prediction",
    layout="wide"
)

# ==============================================================================
# BACKGROUND MODEL REQUIREMENTS (DO NOT EDIT)
# ==============================================================================
class AdaptiveProductionPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, date_format='%d:%m:%Y, %H:%M'):
        self.date_format = date_format
        self.datetime_cols_ = []
        self.numeric_cols_ = []
        self.categorical_cols_ = []
        self.scaler_ = None
        self.encoder_ = None
        
    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        output_parts = []
        if self.numeric_cols_ and self.scaler_ is not None:
            output_parts.append(self.scaler_.transform(X_df[self.numeric_cols_]))
        if self.categorical_cols_ and self.encoder_ is not None:
            output_parts.append(self.encoder_.transform(X_df[self.categorical_cols_]))
        for col in self.datetime_cols_:
            dt_series = pd.to_datetime(X_df[col], format=self.date_format)
            h_sin = np.sin(2 * np.pi * dt_series.dt.hour / 24.0).values.reshape(-1, 1)
            h_cos = np.cos(2 * np.pi * dt_series.dt.hour / 24.0).values.reshape(-1, 1)
            m_sin = np.sin(2 * np.pi * dt_series.dt.month / 12.0).values.reshape(-1, 1)
            m_cos = np.cos(2 * np.pi * dt_series.dt.month / 12.0).values.reshape(-1, 1)
            output_parts.extend([h_sin, h_cos, m_sin, m_cos])
        return np.hstack(output_parts)

MODEL_PATH = "final_cold_chain_pipeline.pkl"
DATA_PATH = "cold_chain_excursion_data.csv"

@st.cache_resource
def load_model_file(path):
    if not os.path.exists(path):
        st.error(f"Error: Required model file '{path}' was not found.")
        st.stop()
    return joblib.load(path)

@st.cache_data
def load_data_file(path):
    if not os.path.exists(path):
        st.error(f"Error: Required data file '{path}' was not found.")
        st.stop()
    df = pd.read_csv(path)
    df['parsed_time'] = pd.to_datetime(df['timestamp'], format='%d:%m:%Y, %H:%M')
    return df

# Load the backend tools
pipeline = load_model_file(MODEL_PATH)
preprocessor = pipeline["preprocessor"]
model = pipeline["model"]
historical_data = load_data_file(DATA_PATH)

# Standard safety limits for each cargo type
safety_limits = {
    "PIL": {"min_temp": 2.0, "max_temp": 8.0},
    "COL": {"min_temp": 2.0, "max_temp": 8.0},
    "FRO": {"min_temp": -30.0, "max_temp": -18.0},
    "CRT": {"min_temp": 15.0, "max_temp": 25.0}
}

# App Title
st.title("Cold Chain Temperature Excursion Prediction")
st.markdown("---")

# Simple Navigation Tabs
tab1, tab2 = st.tabs([
    "Prediction", 
    "Analytics Dashboard"
])

# ==============================================================================
# TAB 1: PREDICTION PAGE
# ==============================================================================
with tab1:
    st.markdown("### Enter Shipment Details Below")
    
    # Row 1: Time and Cargo type
    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1:
        date_input = st.date_input("Date", datetime.date(2026, 6, 1))
    with r1_c2:
        time_input = st.time_input("Time", datetime.time(12, 0))
    with r1_c3:
        shc_code = st.selectbox("Cargo Type (SHC Code)", options=["PIL", "COL", "FRO", "CRT"])
        limits = safety_limits[shc_code]

    # Row 2: Temperature Limits and Current Temperature
    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1:
        target_temp_min = st.number_input("Minimum Safe Temp (°C)", value=float(limits["min_temp"]), step=0.1)
    with r2_c2:
        target_temp_max = st.number_input("Maximum Safe Temp (°C)", value=float(limits["max_temp"]), step=0.1)
    with r2_c3:
        current_internal_temp = st.number_input(
            "Current Container Temp (°C)", 
            value=float(np.mean([limits["min_temp"], limits["max_temp"]])), 
            step=0.1
        )

    # Row 3: Doors and Outside Temperature
    r3_c1, r3_c2, r3_c3 = st.columns(3)
    with r3_c1:
        door_open_count = st.number_input("Number of Times Door Opened (Last 1 Hour)", min_value=0, value=9, step=1)
    with r3_c2:
        door_open_duration = st.number_input("Total Seconds Door Stayed Open (Last 1 Hour)", min_value=0, value=283, step=10)
    with r3_c3:
        ambient_temp = st.number_input("Outside Air Temp (°C)", value=30.60, step=0.5)

    # Row 4: Humidity, Weather, and Equipment Age
    r4_c1, r4_c2, r4_c3 = st.columns(3)
    with r4_c1:
        external_humidity = st.slider("Outside Humidity (%)", min_value=0.0, max_value=100.0, value=45.30, step=0.5)
    with r4_c2:
        weather_condition = st.selectbox("Outside Weather", options=["Clear", "Cloudy", "Rainy", "Sandstorm"])
    with r4_c3:
        equipment_age = st.number_input("Cooling System Age (Years)", min_value=0.0, value=5.70, step=0.5)

    # Row 5: Compressor, Defrost, and Container Space
    r5_c1, r5_c2, r5_c3 = st.columns(3)
    with r5_c1:
        compressor_duty_cycle = st.slider("Compressor Workload Ratio", min_value=0.0, max_value=1.0, value=0.98, step=0.01)
    with r5_c2:
        defrost_cycle_active = st.selectbox("Is Defrost Cycle Running Right Now?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with r5_c3:
        cargo_utilization = st.slider("Container Space Filled (%)", min_value=0.0, max_value=1.0, value=0.63, step=0.01)

    st.markdown("---")
    
    if st.button("Calculate Risk Now", use_container_width=True):
        time_string = f"{date_input.strftime('%d:%m:%Y')}, {time_input.strftime('%H:%M')}"
        
        if current_internal_temp < target_temp_min or current_internal_temp > target_temp_max:
            st.error("TEMPERATURE BREAK DETECTED")
            st.metric(label="Calculated Risk Score", value="1.00 (Maximum Risk)")
            st.warning("Action Required: The temperature inside the container has already gone past the safe limits.")
        else:
            input_row = pd.DataFrame({
                'timestamp':                        [time_string],
                'SHC_code':                        [shc_code],
                'target_temp_min':                 [target_temp_min],
                'target_temp_max':                 [target_temp_max],
                'current_internal_temp':           [current_internal_temp],
                'door_open_count_1h':              [door_open_count],
                'cumulative_door_open_duration_1h': [door_open_duration],
                'ambient_dry_bulb_temp':           [ambient_temp],
                'external_relative_humidity':       [external_humidity],
                'weather_condition':                [weather_condition],
                'equipment_age_years':             [equipment_age],
                'compressor_duty_cycle_ratio_2h':  [compressor_duty_cycle],
                'defrost_cycle_active':            [defrost_cycle_active],
                'cargo_volume_utilization_ratio':  [cargo_utilization]
            })
            
            try:
                processed_row = preprocessor.transform(input_row)
                prediction = model.predict(processed_row)[0]
                final_risk = np.clip(prediction, 0.00, 1.00)
                
                st.metric(label="Calculated Risk Score", value=f"{final_risk:.2f}")
                
                if final_risk >= 0.80:
                    st.error("HIGH RISK: This container is highly likely to experience a temperature breach soon. Check cooling settings immediately.")
                elif final_risk >= 0.40:
                    st.warning("MODERATE RISK: Risk is elevated. Avoid opening container doors and monitor closely.")
                else:
                    st.success("SAFE: This container is running smoothly and under safe operational conditions.")
            except Exception as error_msg:
                st.error(f"Calculation Error: Could not compute risk score. Details: {str(error_msg)}")

# ==============================================================================
# TAB 2: ANALYTICS DASHBOARD - STRICT HIGH-RISK ISOLATION OVERRIDE (>80%)
# ==============================================================================
with tab2:
    st.subheader("High-Risk Shipment Diagnostics (Risk > 80%)")
    
    # Get dataset date ranges for the filters
    dataset_start_date = historical_data['parsed_time'].min().date()
    dataset_end_date = historical_data['parsed_time'].max().date()
    
    # ------------------------------------------------------------
    # FILTERS (Row 1 of Analytics)
    # ------------------------------------------------------------
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        from_date = st.date_input(
            "From Date",
            value=dataset_start_date,
            min_value=dataset_start_date,
            max_value=dataset_end_date
        )

    with filter_col2:
        to_date = st.date_input(
            "To Date",
            value=dataset_end_date,
            min_value=dataset_start_date,
            max_value=dataset_end_date
        )

    with filter_col3:
        quick_filter = st.selectbox(
            "Quick Filter", [
                "All Time",
                "Last 30 Days",
                "Last 90 Days",
                "Last 6 Months"
            ]
        )
        
    # ------------------------------------------------------------
    # SUB-FILTERS FOR SUB-SEGMENTS
    # ------------------------------------------------------------
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        shc_options = sorted(historical_data['SHC_code'].unique().tolist())
        selected_shc = st.multiselect("Filter Cargo Types", options=shc_options, default=shc_options)
    with sub_col2:
        weather_options = sorted(historical_data['weather_condition'].unique().tolist())
        selected_weather = st.multiselect("Filter Weather Conditions", options=weather_options, default=weather_options)

    # Apply Quick Filter Date Calculations
    max_date = historical_data['parsed_time'].max()
    if quick_filter == "Last 30 Days":
        from_date = (max_date - pd.Timedelta(days=30)).date()
    elif quick_filter == "Last 90 Days":
        from_date = (max_date - pd.Timedelta(days=90)).date()
    elif quick_filter == "Last 6 Months":
        from_date = (max_date - pd.Timedelta(days=180)).date()

    # Apply base filters and force strict isolation to ONLY records with risk > 0.80
    filtered_data = historical_data[
        (historical_data['parsed_time'].dt.date >= from_date) & 
        (historical_data['parsed_time'].dt.date <= to_date) & 
        (historical_data['SHC_code'].isin(selected_shc)) & 
        (historical_data['weather_condition'].isin(selected_weather)) &
        (historical_data['excursion_risk_score'] > 0.80)
    ]

    # ------------------------------------------------------------
    # SUMMARY KPI CARDS
    # ------------------------------------------------------------
    st.markdown("---")
    stat1, stat2, stat3, stat4 = st.columns(4)
    with stat1:
        st.metric("Total High-Risk Records Found", value=f"{len(filtered_data):,}")
    with stat2:
        avg_risk = filtered_data['excursion_risk_score'].mean() if not filtered_data.empty else 0.0
        st.metric("Average High-Risk Level", value=f"{avg_risk:.4f}")
    with stat3:
        avg_temp = filtered_data['current_internal_temp'].mean() if not filtered_data.empty else 0.0
        st.metric("Avg Container Temp during Alert", value=f"{avg_temp:.1f} °C")
    with stat4:
        st.metric("Target Scope Minimum Limit", value="> 0.80")
        
    st.markdown("---")
    
    # ------------------------------------------------------------
    # INTERACTIVE CHARTS SECTION - NOW PLOTTING ONLY SCOPE (>0.80)
    # ------------------------------------------------------------
    if not filtered_data.empty:
        chart_row1_left, chart_row1_right = st.columns(2)
        
        with chart_row1_left:
            st.markdown("### Total High-Risk Incidents by Equipment Age")
            # Aggregating incident volume by hardware age profile
            age_summary = filtered_data.groupby('equipment_age_years').size().reset_index(name='Incident Count')
            fig_age = px.bar(
                age_summary,
                x="equipment_age_years",
                y="Incident Count",
                text_auto=True,
                labels={"equipment_age_years": "Equipment Age (Years)", "Incident Count": "Number of High-Risk Incidents"},
                color_discrete_sequence=["#984ea3"]
            )
            fig_age.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)
            
        with chart_row1_right:
            st.markdown("### Total High-Risk Incidents by Cargo Type")
            shc_summary = filtered_data.groupby('SHC_code').size().reset_index(name='Incident Count')
            fig_shc = px.bar(
                shc_summary, 
                x="SHC_code", 
                y="Incident Count",
                text_auto=True,
                labels={"SHC_code": "Cargo Type", "Incident Count": "Number of High-Risk Incidents"},
                color_discrete_sequence=["#0b66c3"]
            )
            fig_shc.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_shc, use_container_width=True)

        st.markdown("---")
        chart_row2_left, chart_row2_right = st.columns(2)

        with chart_row2_left:
            st.markdown("### Total High-Risk Incidents by Weather Condition")
            weather_summary = filtered_data.groupby('weather_condition').size().reset_index(name='Incident Count')
            fig_weather = px.bar(
                weather_summary, 
                x="weather_condition", 
                y="Incident Count",
                text_auto=True,
                labels={"weather_condition": "Weather Condition", "Incident Count": "Number of High-Risk Incidents"},
                color_discrete_sequence=["#ff7f0e"]
            )
            fig_weather.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_weather, use_container_width=True)

        with chart_row2_right:
            st.markdown("### Share of Critical Incidents by Weather Status")
            fig_pie = px.pie(
                filtered_data,
                names="weather_condition",
                values="excursion_risk_score",
                labels={"weather_condition": "Weather Status"},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)
            
    else:
        st.info("No shipments fall into the critical category (> 80% Risk) matching the selected filters.")