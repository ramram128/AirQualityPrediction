import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import os
from data_processor import load_and_preprocess_data, split_data
from model import train_model, evaluate_model, predict_future
from visualization import (
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_aqi_time_series,
    plot_actual_vs_predicted
)
from utils import calculate_aqi, display_aqi_description
from data_fetcher import get_historical_and_current_data, get_cities_list

# Set page configuration
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌬️",
    layout="wide"
)

# App title
st.title("Air Quality Index (AQI) Prediction System")

# Top navigation with custom styling
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #2E86C1;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        margin: 0px 10px;
        text-align: center;
        text-decoration: none;
        font-size: 16px;
    }
    div.stButton > button:hover {
        background-color: #1A5276;
    }
    div.stButton > button:focus {
        background-color: #1A5276;
    }
    .nav-container {
        border: 2px solid #2E86C1;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 25px;
        background-color: #EBF5FB;
    }
    .content-container {
        margin-top: 30px;
    }
    </style>
    
    <div class="nav-container">
    """, 
    unsafe_allow_html=True
)

# Initialize session state if the page key doesn't exist
if 'page' not in st.session_state:
    st.session_state.page = "Current Air Quality"

# Create top navigation with buttons
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Current Air Quality"):
        st.session_state.page = "Current Air Quality"
with col2:
    if st.button("Historical Data"):
        st.session_state.page = "Historical Data"
with col3:
    if st.button("AQI Prediction"):
        st.session_state.page = "AQI Prediction"
with col4:
    if st.button("About"):
        st.session_state.page = "About"

# Close the navigation container
st.markdown("</div>", unsafe_allow_html=True)

# Start the content container
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# Get the current page from session state
page = st.session_state.page

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = "Delhi"
if 'future_data' not in st.session_state:
    st.session_state.future_data = None
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None

# City selection and app description in two columns
col1, col2 = st.columns([1, 3])

with col1:
    # City selection at the top
    cities = get_cities_list()
    selected_city = st.selectbox("Select City", cities, index=cities.index(st.session_state.selected_city))
    
with col2:
    # App description
    st.markdown("""
    This application shows current air quality and predicts the Air Quality Index (AQI) for the next 24 hours 
    using real-time pollution data. The model leverages supervised machine learning techniques to make accurate 
    forecasts and support smart urban planning and public health decisions.
    """)

# Update session state if city changed
if selected_city != st.session_state.selected_city:
    st.session_state.selected_city = selected_city
    st.session_state.data = None  # Reset data when city changes
    st.session_state.model = None
    st.session_state.future_data = None
    
# Fetch data if not already in session state or if it's time to refresh
if st.session_state.data is None or st.session_state.last_update_time is None or \
   (datetime.now() - st.session_state.last_update_time).total_seconds() > 1800:  # Refresh every 30 minutes
    with st.spinner(f'Fetching air quality data for {selected_city}...'):
        try:
            st.session_state.data = get_historical_and_current_data(city=selected_city, days=7)
            st.session_state.last_update_time = datetime.now()
            
            # Train model automatically with the new data
            if 'PM2.5' in st.session_state.data.columns:
                features = [col for col in st.session_state.data.columns if col not in 
                           ['date', 'city', 'location', 'PM2.5']]
                
                X_train, X_test, y_train, y_test = split_data(
                    st.session_state.data, features, 'PM2.5', test_size=0.2, random_state=42
                )
                
                model, feature_importance = train_model(X_train, y_train)
                st.session_state.model = model
                
                # Generate future predictions
                if st.session_state.model is not None:
                    future_data = predict_future(model, st.session_state.data, hours=24)
                    st.session_state.future_data = future_data
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")

# Current Air Quality page
if page == "Current Air Quality":
    st.header(f"Current Air Quality in {selected_city}")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Get most recent data
        latest_data = data.sort_values('date', ascending=False).iloc[0]
        latest_date = latest_data['date']
        
        # Display current time and last update
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Current Date and Time")
            st.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        with col2:
            st.subheader("Last Data Update")
            st.write(latest_date.strftime("%Y-%m-%d %H:%M:%S"))
        
        # Display current AQI
        pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
        available_pollutants = [p for p in pollutants if p in data.columns]
        
        if 'PM2.5' in available_pollutants:
            current_pm25 = latest_data['PM2.5']
            current_aqi = calculate_aqi(current_pm25, 'PM2.5')
            category, color, message = display_aqi_description(current_aqi)
            
            # Display AQI with styling
            st.markdown(f"""
            <div style='background-color: {color}; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h2 style='color: white; text-align: center;'>Current AQI: {current_aqi:.0f}</h2>
                <h3 style='color: white; text-align: center;'>{category}</h3>
                <p style='color: white; text-align: center;'>{message}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display current pollutant levels
        st.subheader("Current Pollutant Levels")
        
        # Create metrics in multiple columns
        cols = st.columns(len(available_pollutants))
        for i, pollutant in enumerate(available_pollutants):
            if pollutant in latest_data:
                value = latest_data[pollutant]
                aqi = calculate_aqi(value, pollutant)
                category, color, _ = display_aqi_description(aqi)
                
                cols[i].metric(
                    label=f"{pollutant}",
                    value=f"{value:.1f} μg/m³",
                    delta=f"AQI: {aqi:.0f} ({category})"
                )
        
        # Display weather conditions if available
        weather_cols = ['Temperature', 'Humidity', 'Wind_speed']
        available_weather = [col for col in weather_cols if col in data.columns]
        
        if available_weather:
            st.subheader("Current Weather Conditions")
            weather_cols = st.columns(len(available_weather))
            
            for i, col in enumerate(available_weather):
                if col in latest_data:
                    value = latest_data[col]
                    unit = "°C" if col == "Temperature" else "%" if col == "Humidity" else "m/s"
                    weather_cols[i].metric(label=col, value=f"{value:.1f} {unit}")
        
        # Show latest data in a table
        st.subheader("Latest Measurements")
        display_cols = [col for col in data.columns if col not in ['city', 'location']]
        st.dataframe(data[display_cols].head(24))
        
        # Plot today's trend
        today = datetime.now().date()
        today_data = data[data['date'].dt.date == today]
        
        if not today_data.empty:
            st.subheader("Today's Air Quality Trend")
            
            # Allow user to select pollutant to view
            selected_pollutant = st.selectbox("Select Pollutant to View", available_pollutants, index=0)
            
            fig = px.line(
                today_data, 
                x='date', 
                y=selected_pollutant,
                title=f"{selected_pollutant} Levels Today",
                labels={'date': 'Time', selected_pollutant: f"{selected_pollutant} (μg/m³)"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show AQI categories reference
        with st.expander("AQI Categories Reference"):
            aqi_categories = [
                {"Category": "Good", "AQI Range": "0-50", "Description": "Air quality is satisfactory, and air pollution poses little or no risk."},
                {"Category": "Moderate", "AQI Range": "51-100", "Description": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."},
                {"Category": "Unhealthy for Sensitive Groups", "AQI Range": "101-150", "Description": "Members of sensitive groups may experience health effects. The general public is less likely to be affected."},
                {"Category": "Unhealthy", "AQI Range": "151-200", "Description": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."},
                {"Category": "Very Unhealthy", "AQI Range": "201-300", "Description": "Health alert: The risk of health effects is increased for everyone."},
                {"Category": "Hazardous", "AQI Range": "301+", "Description": "Health warning of emergency conditions: everyone is more likely to be affected."}
            ]
            
            st.table(pd.DataFrame(aqi_categories))
            
    else:
        st.info("Fetching air quality data... Please wait.")

# Historical Data page
elif page == "Historical Data":
    st.header(f"Historical Air Quality Data for {selected_city}")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Date range selector
        st.subheader("Select Date Range")
        col1, col2 = st.columns(2)
        with col1:
            min_date = data['date'].min().date()
            max_date = data['date'].max().date()
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Filter data by date range
        filtered_data = data[(data['date'].dt.date >= start_date) & (data['date'].dt.date <= end_date)]
        
        if not filtered_data.empty:
            # Get available pollutants
            pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
            available_pollutants = [p for p in pollutants if p in data.columns]
            
            # Pollutant selection
            selected_pollutant = st.selectbox("Select Pollutant", available_pollutants, index=0)
            
            # Display time series plot
            st.subheader(f"{selected_pollutant} Trend")
            fig = px.line(
                filtered_data, 
                x='date', 
                y=selected_pollutant,
                title=f"{selected_pollutant} Levels ({start_date} to {end_date})",
                labels={'date': 'Date', selected_pollutant: f"{selected_pollutant} (μg/m³)"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Daily patterns
            st.subheader("Daily Patterns")
            hourly_avg = filtered_data.groupby('hour')[selected_pollutant].mean().reset_index()
            
            fig = px.bar(
                hourly_avg, 
                x='hour', 
                y=selected_pollutant,
                title=f"Average {selected_pollutant} by Hour of Day",
                labels={'hour': 'Hour of Day', selected_pollutant: f"Average {selected_pollutant} (μg/m³)"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Weekly patterns
            st.subheader("Weekly Patterns")
            # Map 0-6 to day names
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            filtered_data['day_name'] = filtered_data['dayofweek'].apply(lambda x: day_names[x])
            
            daily_avg = filtered_data.groupby('day_name')[selected_pollutant].mean().reindex(day_names).reset_index()
            
            fig = px.bar(
                daily_avg, 
                x='day_name', 
                y=selected_pollutant,
                title=f"Average {selected_pollutant} by Day of Week",
                labels={'day_name': 'Day of Week', selected_pollutant: f"Average {selected_pollutant} (μg/m³)"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a section for air quality insights
            st.subheader("Air Quality Insights")
            
            # Calculate average AQI for the selected period
            if 'PM2.5' in filtered_data.columns:
                avg_pm25 = filtered_data['PM2.5'].mean()
                avg_aqi = calculate_aqi(avg_pm25, 'PM2.5')
                category, color, message = display_aqi_description(avg_aqi)
                
                st.markdown(f"""
                <div style='background-color: {color}; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
                    <h3 style='color: white; text-align: center;'>Average AQI: {avg_aqi:.0f}</h3>
                    <h4 style='color: white; text-align: center;'>{category}</h4>
                    <p style='color: white; text-align: center;'>{message}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate peak pollution hours
                hourly_pollution = filtered_data.groupby('hour')['PM2.5'].mean()
                peak_hour = hourly_pollution.idxmax()
                lowest_hour = hourly_pollution.idxmin()
                
                st.markdown(f"""
                ### Peak Pollution Times
                - **Highest pollution typically occurs at:** {peak_hour}:00 hours
                - **Lowest pollution typically occurs at:** {lowest_hour}:00 hours
                
                ### Air Quality Recommendations
                - Consider planning outdoor activities around {lowest_hour}:00 when air quality is better
                - If possible, limit outdoor exposure at {peak_hour}:00 when pollution peaks
                """)
            
        else:
            st.warning("No data available for the selected date range.")
            
    else:
        st.info("Fetching air quality data... Please wait.")

# AQI Prediction page
elif page == "AQI Prediction":
    st.header(f"AQI Prediction for {selected_city}")
    
    if st.session_state.model is not None and st.session_state.future_data is not None:
        # Display predictions
        st.subheader("AQI Predictions for Next 24 Hours")
        
        # Plot predictions
        fig = plot_aqi_time_series(st.session_state.future_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display predicted values table
        with st.expander("View Detailed Prediction Values"):
            st.dataframe(st.session_state.future_data)
        
        # Display current and future AQI status
        current_aqi = st.session_state.future_data.iloc[0]['predicted_value']
        current_category, current_color, current_message = display_aqi_description(current_aqi)
        
        # Max AQI in the next 24 hours
        max_aqi = st.session_state.future_data['predicted_value'].max()
        max_category, max_color, max_message = display_aqi_description(max_aqi)
        
        # Display current and max AQI with styling
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style='background-color: {current_color}; padding: 20px; border-radius: 10px;'>
                <h3 style='color: white; text-align: center;'>Current Predicted AQI</h3>
                <h2 style='color: white; text-align: center;'>{current_aqi:.0f}</h2>
                <h4 style='color: white; text-align: center;'>{current_category}</h4>
                <p style='color: white; text-align: center;'>{current_message}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div style='background-color: {max_color}; padding: 20px; border-radius: 10px;'>
                <h3 style='color: white; text-align: center;'>Max Predicted AQI (Next 24h)</h3>
                <h2 style='color: white; text-align: center;'>{max_aqi:.0f}</h2>
                <h4 style='color: white; text-align: center;'>{max_category}</h4>
                <p style='color: white; text-align: center;'>{max_message}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Health recommendations based on predicted AQI
        st.subheader("Health Recommendations")
        
        if max_aqi <= 50:
            st.success("✅ Air quality is good. It's a great time for outdoor activities!")
        elif max_aqi <= 100:
            st.info("ℹ️ Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion.")
        elif max_aqi <= 150:
            st.warning("⚠️ Members of sensitive groups may experience health effects. Consider reducing prolonged or heavy outdoor exertion.")
        elif max_aqi <= 200:
            st.warning("⚠️ Everyone may begin to experience health effects. Sensitive groups should reduce prolonged or heavy outdoor exertion.")
        elif max_aqi <= 300:
            st.error("🚨 Health alert: everyone may experience more serious health effects. Avoid prolonged or heavy exertion.")
        else:
            st.error("🚨 Health warnings of emergency conditions. Everyone should avoid all outdoor exertion.")
        
        # Create table of all AQI categories
        with st.expander("AQI Categories Reference"):
            aqi_categories = [
                {"Category": "Good", "AQI Range": "0-50", "Description": "Air quality is satisfactory, and air pollution poses little or no risk."},
                {"Category": "Moderate", "AQI Range": "51-100", "Description": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."},
                {"Category": "Unhealthy for Sensitive Groups", "AQI Range": "101-150", "Description": "Members of sensitive groups may experience health effects. The general public is less likely to be affected."},
                {"Category": "Unhealthy", "AQI Range": "151-200", "Description": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."},
                {"Category": "Very Unhealthy", "AQI Range": "201-300", "Description": "Health alert: The risk of health effects is increased for everyone."},
                {"Category": "Hazardous", "AQI Range": "301+", "Description": "Health warning of emergency conditions: everyone is more likely to be affected."}
            ]
            
            st.table(pd.DataFrame(aqi_categories))
            
    else:
        st.info("Preparing prediction model... Please wait.")

# About page
else:
    st.header("About")
    st.markdown("""
    ## Air Quality Index (AQI) Prediction System
    
    This project provides real-time air quality monitoring and prediction of the Air Quality Index (AQI) for the next 24 hours. 
    The model leverages supervised machine learning techniques to make accurate forecasts and support smart urban planning and public health decisions.
    
    ### Objectives
    - To provide real-time air quality monitoring for major cities
    - To predict future air quality conditions with high accuracy
    - To analyze the impact of pollution and weather patterns on urban environments
    - To offer health recommendations based on current and predicted air quality
    
    ### Technologies Used
    - Python 3.10+
    - pandas, numpy – Data manipulation
    - matplotlib, seaborn – Visualization
    - scikit-learn – Preprocessing and modeling
    - xgboost – Advanced regression modeling
    - Streamlit – Interactive web application
    
    ### Data Sources
    This application uses real-time air quality data from public APIs and monitoring stations, including:
    - PM2.5, PM10, NO2, CO (Pollutants)
    - Temperature, Humidity, Wind Speed (Weather)
    
    ### Real-Life Impact
    This model mirrors real-world AI systems used in major cities to forecast AQI and issue public warnings. By enabling early responses to high pollution days, AI can help:
    - Reduce outdoor exposure during dangerous air quality periods
    - Improve city traffic management during pollution events
    - Support sustainable urban planning for cleaner air
    """)

