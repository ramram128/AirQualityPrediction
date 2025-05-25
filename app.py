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

# Set page configuration
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌬️",
    layout="wide"
)

# App title and description
st.title("Air Quality Index (AQI) Prediction System")
st.markdown("""
This application predicts the Air Quality Index (AQI) for the next 24 hours using historical air pollution, 
weather, and traffic data. The model leverages supervised machine learning techniques to make accurate forecasts 
and support smart urban planning and public health decisions.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Model Training", "AQI Prediction", "About"])

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None

# Data Overview page
if page == "Data Overview":
    st.header("Data Overview")
    
    # Data upload section
    uploaded_file = st.file_uploader("Upload Air Quality Dataset (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load and preprocess data
            with st.spinner('Loading and preprocessing data...'):
                data = load_and_preprocess_data(uploaded_file)
                st.session_state.data = data
                
            # Display sample data
            st.subheader("Sample Data")
            st.dataframe(data.head())
            
            # Display data info
            st.subheader("Data Information")
            buffer = pd.io.StringIO()
            data.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            
            # Display descriptive statistics
            st.subheader("Descriptive Statistics")
            st.dataframe(data.describe())
            
            # Visualizations
            st.subheader("Data Visualizations")
            
            # Distribution of pollutants
            pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
            available_pollutants = [p for p in pollutants if p in data.columns]
            
            if available_pollutants:
                selected_pollutant = st.selectbox("Select Pollutant", available_pollutants)
                
                # Distribution plot
                st.write(f"Distribution of {selected_pollutant}")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data[selected_pollutant].dropna(), kde=True, ax=ax)
                st.pyplot(fig)
                
                # Time series plot
                if 'date' in data.columns:
                    st.write(f"{selected_pollutant} over Time")
                    fig = px.line(data, x='date', y=selected_pollutant, title=f"{selected_pollutant} Trend Over Time")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Correlation Heatmap")
            fig = plot_correlation_heatmap(data)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
    else:
        st.info("Please upload a CSV file to get started.")
        
# Model Training page
elif page == "Model Training":
    st.header("Model Training")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Model training parameters
        st.subheader("Training Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random State", 0, 100, 42, 1)
        
        with col2:
            target_column = st.selectbox("Target Variable", 
                                        [col for col in data.columns if col not in ['date', 'hour', 'day', 'month', 'year']], 
                                        index=0)
            
        # Feature selection
        st.subheader("Feature Selection")
        all_features = [col for col in data.columns if col not in ['date', target_column]]
        selected_features = st.multiselect("Select Features for Training", all_features, default=all_features[:10])
        
        if len(selected_features) == 0:
            st.warning("Please select at least one feature for training.")
        else:
            # Train model button
            if st.button("Train Model"):
                with st.spinner('Training model...'):
                    try:
                        # Split data
                        X_train, X_test, y_train, y_test = split_data(
                            data, selected_features, target_column, test_size, random_state
                        )
                        
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        
                        # Train model
                        model, feature_importance = train_model(X_train, y_train)
                        st.session_state.model = model
                        st.session_state.feature_importance = feature_importance
                        
                        # Evaluate model
                        predictions, metrics = evaluate_model(model, X_test, y_test)
                        st.session_state.predictions = predictions
                        st.session_state.metrics = metrics
                        
                        st.success("Model trained successfully!")
                        
                        # Display metrics
                        st.subheader("Model Performance")
                        metrics_df = pd.DataFrame([metrics])
                        st.dataframe(metrics_df)
                        
                        # Plot feature importance
                        st.subheader("Feature Importance")
                        fig = plot_feature_importance(feature_importance, selected_features)
                        st.pyplot(fig)
                        
                        # Plot actual vs predicted
                        st.subheader("Actual vs Predicted Values")
                        fig = plot_actual_vs_predicted(y_test, predictions)
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
            
            # Display previous results if available
            if st.session_state.metrics is not None:
                st.subheader("Previous Model Performance")
                metrics_df = pd.DataFrame([st.session_state.metrics])
                st.dataframe(metrics_df)
                
                if st.session_state.feature_importance is not None:
                    st.subheader("Feature Importance")
                    fig = plot_feature_importance(st.session_state.feature_importance, selected_features)
                    st.pyplot(fig)
                
                if st.session_state.predictions is not None:
                    st.subheader("Actual vs Predicted Values")
                    fig = plot_actual_vs_predicted(st.session_state.y_test, st.session_state.predictions)
                    st.pyplot(fig)
    else:
        st.info("Please upload and process data in the 'Data Overview' page first.")

# AQI Prediction page
elif page == "AQI Prediction":
    st.header("AQI Prediction for Next 24 Hours")
    
    if st.session_state.model is not None and st.session_state.data is not None:
        data = st.session_state.data
        model = st.session_state.model
        
        # Get latest data for prediction
        latest_date = data['date'].max()
        st.write(f"Latest data date: {latest_date}")
        
        # Predict next 24 hours
        future_data = predict_future(model, data, hours=24)
        
        # Display predictions
        st.subheader("AQI Predictions for Next 24 Hours")
        
        # Plot predictions
        fig = plot_aqi_time_series(future_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display predicted values table
        st.subheader("Predicted AQI Values")
        st.dataframe(future_data)
        
        # Display AQI categories
        st.subheader("AQI Categories")
        
        # Get current (latest) AQI
        current_aqi = future_data.iloc[0]['predicted_value']
        current_category, current_color, current_message = display_aqi_description(current_aqi)
        
        # Display current AQI with styling
        st.markdown(f"""
        <div style='background-color: {current_color}; padding: 10px; border-radius: 5px;'>
            <h3 style='color: white;'>Current AQI: {current_aqi:.2f}</h3>
            <h4 style='color: white;'>{current_category}</h4>
            <p style='color: white;'>{current_message}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create table of all AQI categories
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
        st.info("Please train a model in the 'Model Training' page first.")

# About page
else:
    st.header("About")
    st.markdown("""
    ## Air Quality Index (AQI) Prediction System
    
    This project aims to predict the Air Quality Index (AQI) for the next 24 hours using historical air pollution, weather, and traffic data. 
    The model leverages supervised machine learning techniques to make accurate forecasts and support smart urban planning and public health decisions.
    
    ### Objectives
    - To understand the key factors affecting air quality
    - To develop a machine learning model that can predict AQI with high accuracy
    - To analyze the impact of pollution and weather patterns on urban environments
    - To demonstrate how AI can support early-warning systems for public safety
    
    ### Technologies Used
    - Python 3.10+
    - pandas, numpy – Data manipulation
    - matplotlib, seaborn – Visualization
    - scikit-learn – Preprocessing and modeling
    - xgboost – Advanced regression modeling
    - Streamlit – Interactive web application
    
    ### Dataset
    This application works with the Delhi Air Quality Dataset, which contains data from 2015 to 2020, including:
    - PM2.5, PM10, NO2, CO (Pollutants)
    - Temperature, Humidity, Wind Speed (Weather)
    - Timestamp for time-series modeling
    
    ### Real-Life Impact
    This model mimics the real-world AI system used in Delhi to forecast AQI and issue public warnings. By enabling early responses to high pollution days, AI can help:
    - Reduce outdoor exposure
    - Improve city traffic management
    - Support sustainable urban planning
    """)

