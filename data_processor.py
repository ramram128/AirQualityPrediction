import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the air quality dataset.
    
    Parameters:
    -----------
    file_path : str or file-like object
        Path to the CSV file or file-like object containing the data
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed data
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # Check if date column exists, if not create it
    if 'date' not in data.columns:
        # Look for date-like columns
        date_cols = [col for col in data.columns if any(kw in col.lower() for kw in ['date', 'time', 'timestamp'])]
        
        if date_cols:
            # Rename the first date-like column to 'date'
            data = data.rename(columns={date_cols[0]: 'date'})
        else:
            # No date column found, raise error
            raise ValueError("No date or timestamp column found in the dataset")
    
    # Convert date to datetime
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    
    # Filter out rows with invalid dates
    data = data.dropna(subset=['date'])
    
    # Extract time features
    data['hour'] = data['date'].dt.hour
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data['dayofweek'] = data['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    # Handle missing values
    # Strategy: Forward fill for time series data, then backward fill, then median for any remaining NAs
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    # Apply forward fill, then backward fill for time series data
    data[numeric_cols] = data[numeric_cols].fillna(method='ffill')
    data[numeric_cols] = data[numeric_cols].fillna(method='bfill')
    
    # For any remaining NAs, fill with median
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())
    
    # Remove rows with all remaining NAs
    data = data.dropna(how='all')
    
    # Sort by date
    data = data.sort_values(by='date')
    
    return data

def split_data(data, features, target, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Preprocessed data
    features : list
        List of feature column names
    target : str
        Target column name
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple of pd.DataFrame/Series
        Split datasets for training and testing
    """
    # Select features and target
    X = data[features]
    y = data[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def prepare_future_data(data, hours=24):
    """
    Prepare data for future prediction.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Preprocessed data
    hours : int, default=24
        Number of hours to predict into the future
        
    Returns:
    --------
    pd.DataFrame
        Data prepared for future prediction
    """
    # Get latest date in the dataset
    latest_date = data['date'].max()
    
    # Create future dates
    future_dates = [latest_date + timedelta(hours=i+1) for i in range(hours)]
    
    # Create a future dataframe with dates
    future_data = pd.DataFrame({'date': future_dates})
    
    # Extract time features
    future_data['hour'] = future_data['date'].dt.hour
    future_data['day'] = future_data['date'].dt.day
    future_data['month'] = future_data['date'].dt.month
    future_data['year'] = future_data['date'].dt.year
    future_data['dayofweek'] = future_data['date'].dt.dayofweek
    
    # For each numeric column in the original data, use the last value
    # This is a simple approach; more sophisticated methods could be used
    last_row = data.iloc[-1].copy()
    for col in data.select_dtypes(include=[np.number]).columns:
        if col not in ['hour', 'day', 'month', 'year', 'dayofweek']:
            future_data[col] = last_row[col]
    
    return future_data
