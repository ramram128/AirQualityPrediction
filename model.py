import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from data_processor import prepare_future_data

def train_model(X_train, y_train, params=None):
    """
    Train an XGBoost regression model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target values
    params : dict, optional
        Parameters for XGBoost model
        
    Returns:
    --------
    tuple
        (trained model, feature importance)
    """
    # Default parameters if none provided
    if params is None:
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'random_state': 42
        }
    
    # Initialize and train model
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Get feature importance
    feature_importance = model.feature_importances_
    
    return model, feature_importance

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    
    Parameters:
    -----------
    model : XGBRegressor
        Trained model
    X_test : pd.DataFrame
        Testing features
    y_test : pd.Series
        Testing target values
        
    Returns:
    --------
    tuple
        (predictions, evaluation metrics)
    """
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Metrics dictionary
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    return predictions, metrics

def predict_future(model, data, hours=24):
    """
    Predict future AQI values.
    
    Parameters:
    -----------
    model : XGBRegressor
        Trained model
    data : pd.DataFrame
        Historical data
    hours : int, default=24
        Number of hours to predict into the future
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with future dates and predicted AQI values
    """
    # Prepare future data
    future_data = prepare_future_data(data, hours)
    
    # Extract features used by the model
    model_features = model.get_booster().feature_names
    
    # Check if all features are available
    missing_features = [f for f in model_features if f not in future_data.columns]
    if missing_features:
        raise ValueError(f"Missing features for prediction: {missing_features}")
    
    # Select only the features used by the model
    X_future = future_data[model_features]
    
    # Scale features
    scaler = StandardScaler()
    X_future_scaled = pd.DataFrame(
        scaler.fit_transform(X_future),
        columns=X_future.columns,
        index=X_future.index
    )
    
    # Make predictions
    predictions = model.predict(X_future_scaled)
    
    # Add predictions to future data
    future_data['predicted_value'] = predictions
    
    return future_data[['date', 'predicted_value']]
