import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_correlation_heatmap(data, figsize=(12, 10)):
    """
    Create a correlation heatmap for numeric features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data containing numeric columns
    figsize : tuple, default=(12, 10)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with correlation heatmap
    """
    # Select numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Calculate correlation
    corr = numeric_data.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        square=True, 
        linewidths=0.5, 
        ax=ax
    )
    
    plt.title('Correlation Heatmap of Numeric Features', fontsize=16)
    plt.tight_layout()
    
    return fig

def plot_feature_importance(feature_importance, feature_names, figsize=(12, 8)):
    """
    Plot feature importance from a trained model.
    
    Parameters:
    -----------
    feature_importance : array-like
        Feature importance values from model
    feature_names : list
        Names of features
    figsize : tuple, default=(12, 8)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with feature importance plot
    """
    # Create dataframe for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    
    plt.title('Feature Importance', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    return fig

def plot_actual_vs_predicted(y_true, y_pred, figsize=(12, 8)):
    """
    Plot actual vs predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    figsize : tuple, default=(12, 8)
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with actual vs predicted plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Plot perfect prediction line
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Actual vs Predicted Values', fontsize=16)
    plt.xlabel('Actual', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_aqi_time_series(future_data):
    """
    Create a time series plot of predicted AQI values.
    
    Parameters:
    -----------
    future_data : pd.DataFrame
        Data with 'date' and 'predicted_value' columns
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive time series plot
    """
    # Create figure with plotly
    fig = px.line(
        future_data, 
        x='date', 
        y='predicted_value',
        title='Predicted AQI for Next 24 Hours',
        labels={'date': 'Date and Time', 'predicted_value': 'Predicted AQI'},
        markers=True
    )
    
    # Add thresholds for AQI categories
    aqi_ranges = [
        (0, 50, "Good", "green"),
        (51, 100, "Moderate", "yellow"),
        (101, 150, "Unhealthy for Sensitive Groups", "orange"),
        (151, 200, "Unhealthy", "red"),
        (201, 300, "Very Unhealthy", "purple"),
        (301, 500, "Hazardous", "maroon")
    ]
    
    # Add colored regions for AQI categories
    for i, (low, high, label, color) in enumerate(aqi_ranges):
        fig.add_shape(
            type="rect",
            x0=future_data['date'].min(),
            x1=future_data['date'].max(),
            y0=low,
            y1=high,
            fillcolor=color,
            opacity=0.1,
            layer="below",
            line_width=0,
        )
    
    # Add a legend for AQI categories
    for low, high, label, color in aqi_ranges:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                showlegend=True,
                name=f"{label} ({low}-{high})"
            )
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Date and Time",
        yaxis_title="AQI Value",
        legend_title="AQI Categories",
        height=600,
        hovermode="x unified"
    )
    
    return fig
