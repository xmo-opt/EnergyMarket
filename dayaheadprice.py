import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# 1. Load and preprocess data
def load_data(file_path):
    """Load and preprocess the electricity data."""
    df = pd.read_csv(file_path)
    df['times'] = pd.to_datetime(df['times'])
    df = df.sort_values('times')
    
    # Extract time features
    df['hour'] = df['times'].dt.hour
    df['day'] = df['times'].dt.day
    df['month'] = df['times'].dt.month
    df['day_of_week'] = df['times'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Aggregate weather features across cities
    cities = ['济南', '潍坊', '临沂', '德州', '滨州', '泰安', '烟台', '青岛']
    weather_features = ['temp_2m', 'wind_speed_10m', 'solar_radiation_ghi', 'humidity_2m']
    
    for feature in weather_features:
        cols = [f'{city}_{feature}' for city in cities if f'{city}_{feature}' in df.columns]
        if cols:
            df[f'avg_{feature}'] = df[cols].mean(axis=1)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
    df = df.dropna(subset=['da_clearing_price'])
    
    return df

# 2. Feature engineering with strict time boundary awareness
def engineer_features(df, prediction_start_time):
    """Engineer features with strict time boundary awareness for day-ahead forecasting."""
    combined_data = df.copy()

    # Add day-of-year features
    combined_data['day_of_year'] = combined_data['times'].dt.dayofyear
    combined_data['day_of_year_sin'] = np.sin(combined_data['day_of_year'] / 365.25 * 2 * np.pi)
    combined_data['day_of_year_cos'] = np.cos(combined_data['day_of_year'] / 365.25 * 2 * np.pi)

    # Create lag features with proper time boundaries
    # For 15-minute data: 96 points = 1 day, 192 points = 2 days, 672 points = 1 week
    # Ensure minimum lag of 192 (2 days) for price data
    
    # Price lags - using historical data only
    price_lags = [192, 288, 384, 672, 960]  # 2, 3, 4, 7, 10 days
    for lag in price_lags:
        combined_data[f'da_clearing_price_lag_{lag}'] = combined_data['da_clearing_price'].shift(lag)
        combined_data[f'rt_clearing_price_lag_{lag}'] = combined_data['rt_clearing_price'].shift(lag)
    
    # Load and generation lags
    load_gen_cols = ['load_actual', 'wind_actual', 'solar_actual']
    load_gen_lags = [96, 192, 288, 672]  # 1, 2, 3, 7 days
    
    for col in load_gen_cols:
        for lag in load_gen_lags:
            combined_data[f'{col}_lag_{lag}'] = combined_data[col].shift(lag)
    
    # Create rolling window features with appropriate lookback
    window_sizes = [96, 192, 336, 672]  # 1, 2, 3.5, 7 days
    
    for col in ['da_clearing_price', 'load_actual']:
        for window in window_sizes:
            combined_data[f'{col}_rolling_mean_{window}'] = combined_data[col].rolling(
                window=window, min_periods=1).mean().shift(192)  # Shift by at least 2 days
            
            combined_data[f'{col}_rolling_std_{window}'] = combined_data[col].rolling(
                window=window, min_periods=1).std().shift(192)  # Shift by at least 2 days

    # Add interaction features
    combined_data['hour_x_temp'] = combined_data['hour'] * combined_data['avg_temp_2m']
    if 'load_actual_lag_192' in combined_data.columns:
        combined_data['hour_x_load_lag'] = combined_data['hour'] * combined_data['load_actual_lag_192']
    
    # Add day-of-week and hour-of-day effects (based on historical averages)
    # These are calculated using data from before the prediction start time
    train_data = combined_data[combined_data['times'] < prediction_start_time].copy()
    
    # Calculate average price by hour and day of week (using only training data)
    hour_avg = train_data.groupby('hour')['da_clearing_price'].mean().to_dict()
    dow_avg = train_data.groupby('day_of_week')['da_clearing_price'].mean().to_dict()
    
    # Apply these averages to the entire dataset
    combined_data['hour_price_avg'] = combined_data['hour'].map(hour_avg)
    combined_data['dow_price_avg'] = combined_data['day_of_week'].map(dow_avg)
    
    # Split data into train and test
    train_data = combined_data[combined_data['times'] < prediction_start_time].copy()
    test_data = combined_data[combined_data['times'] >= prediction_start_time].copy()
    
    # For test data, ensure we're only using forecast data that would be available
    # at prediction time, and not any actual values
    
    # Select features that respect time boundaries
    feature_columns = [
        # Time features
        'hour', 'day', 'month', 'day_of_week', 'is_weekend',
        'day_of_year_sin', 'day_of_year_cos', # Added day of year features
        'hour_price_avg', 'dow_price_avg',
        
        # Price lags (all shifted by at least 192 points)
        'da_clearing_price_lag_192', 'da_clearing_price_lag_288', 
        'da_clearing_price_lag_384', 'da_clearing_price_lag_672',
        'rt_clearing_price_lag_192', 'rt_clearing_price_lag_384',
        
        # Rolling statistics (all shifted by at least 192 points)
        'da_clearing_price_rolling_mean_192', 'da_clearing_price_rolling_std_192',
        'da_clearing_price_rolling_mean_672', 'da_clearing_price_rolling_std_672',
        'load_actual_rolling_mean_192', 'load_actual_rolling_std_192',
        
        # Load and generation lags
        'load_actual_lag_192', 'load_actual_lag_384',
        'wind_actual_lag_192', 'solar_actual_lag_192',
        
        # Forecasts
        'load_forecast', 'wind_forecast', 'solar_forecast',
        
        # Weather
        'avg_temp_2m', 'avg_wind_speed_10m', 'avg_solar_radiation_ghi', 'avg_humidity_2m',

        # Interaction features
        'hour_x_temp'
    ]
    if 'hour_x_load_lag' in combined_data.columns:
        feature_columns.append('hour_x_load_lag')
    
    # Filter to include only columns that exist
    feature_columns = [col for col in feature_columns if col in combined_data.columns]
    
    # Prepare datasets
    X_train = train_data[feature_columns]
    y_train = train_data['da_clearing_price']
    
    X_test = test_data[feature_columns]
    y_test = test_data['da_clearing_price']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, y_test, feature_columns, test_data['times']

# 3. Model training and prediction
def train_and_predict(X_train, y_train, X_test):
    """Train XGBoost model with hyperparameter tuning and make predictions."""
    
    param_grid = {
        'n_estimators': [100, 200], # Reduced for faster execution
        'max_depth': [3, 5],        # Reduced for faster execution
        'learning_rate': [0.05, 0.1], # Reduced for faster execution
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'min_child_weight': [1, 2]
    }
    
    # Ensure random_state is passed to XGBRegressor for reproducibility
    xgb_model = XGBRegressor(random_state=42, gamma=0.1, reg_alpha=0.1, reg_lambda=1) 
    
    # TimeSeriesSplit for cross-validation
    # Adjust n_splits based on data size and computational limits
    # Using a small number of splits for this example
    tscv = TimeSeriesSplit(n_splits=2) 
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=tscv,
        n_jobs=-1, # Use all available cores
        verbose=1 # Add verbosity to see progress
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    
    return best_model, y_pred

# 4. Evaluate and save results
def evaluate_and_save(model, y_test, y_pred, test_times, feature_columns, forecast_horizon=96):
    """Evaluate model and save the 96-point forecast."""
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    
    # Get feature importance
    feature_imp = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_imp.head(10))
    
    # Save forecasts for the specified horizon
    n_points = min(forecast_horizon, len(y_pred))
    forecast_df = pd.DataFrame({
        'timestamp': test_times[:n_points],
        'actual_price': y_test[:n_points],
        'forecast_price': y_pred[:n_points],
        'absolute_error': abs(y_test[:n_points] - y_pred[:n_points]),
        'percentage_error': 100 * abs(y_test[:n_points] - y_pred[:n_points]) / (y_test[:n_points] + 1e-10)
    })
    
    forecast_df.to_csv('day_ahead_price_forecast.csv', index=False)
    print(f"\nForecasts for {n_points} points saved to 'day_ahead_price_forecast.csv'")
    
    # Print average metrics by hour of day
    forecast_df['hour'] = pd.to_datetime(forecast_df['timestamp']).dt.hour
    hour_metrics = forecast_df.groupby('hour')['absolute_error'].mean().reset_index()
    hour_metrics.columns = ['Hour', 'MAE']
    print("\nMAE by Hour of Day:")
    print(hour_metrics.sort_values('MAE', ascending=False).head(5))
    
    return forecast_df

# 5. Main function
def main():
    # Load data
    df = load_data('xytest.csv')
    print(f"Data loaded: {df.shape[0]} rows from {df['times'].min()} to {df['times'].max()}")
    
    # Define prediction start time as April 5, 2025
    prediction_start_time = pd.to_datetime('2025-04-05')
    print(f"Prediction start time: {prediction_start_time}")
    
    # Engineer features
    X_train, y_train, X_test, y_test, feature_columns, test_times = engineer_features(df, prediction_start_time)
    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
    print(f"Using {len(feature_columns)} features")
    
    # Train model and predict
    model, y_pred = train_and_predict(X_train, y_train, X_test)
    
    # Evaluate and save results
    forecast_df = evaluate_and_save(model, y_test, y_pred, test_times, feature_columns)
    
    # Print any features with importance over 0.05
    feature_imp = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    significant_features = feature_imp[feature_imp['Importance'] > 0.05]
    if len(significant_features) > 0:
        print("\nSignificant features (importance > 0.05):")
        print(significant_features)

if __name__ == "__main__":
    main()