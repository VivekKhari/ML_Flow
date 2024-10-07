# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import mlflow
import mlflow.sklearn

# Load the dataset
df = pd.read_csv('autompg.csv')

# Handle missing values in 'horsepower'
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())

# Selecting relevant features
dflr = df[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']]

# Features and target variable
X = dflr[['displacement', 'horsepower', 'weight', 'acceleration']].values
Y = dflr['mpg'].values

# Standardizing the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize MLflow experiment
mlflow.set_experiment("autompg_")  # Experiment name from the data file name

# Variable to track the best model
best_model = None
best_r2 = -np.inf

# Logging for Random Forest Regressor
with mlflow.start_run(run_name="RandomForestRegressor"):
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, Y_train)
    
    # Predictions
    Y_pred_rf = rf_regressor.predict(X_test)
    
    # Metrics
    r2_rf = r2_score(Y_test, Y_pred_rf)
    mse_rf = mean_squared_error(Y_test, Y_pred_rf)
    
    # Log metrics
    mlflow.log_metric("r2_score", r2_rf)
    mlflow.log_metric("mean_squared_error", mse_rf)
    
    # Log model name as a parameter
    mlflow.log_param("model_name", "RandomForestRegressor")
    
    # Log model
    mlflow.sklearn.log_model(rf_regressor, "RandomForestRegressor_model")
    
    print(f"Random Forest Regressor -> R2: {r2_rf}, MSE: {mse_rf}")
    
    # Update the best model if Random Forest has higher R2
    if r2_rf > best_r2:
        best_model = rf_regressor
        best_r2 = r2_rf
        best_model_name = "RandomForestRegressor"

# Logging for Linear Regression
with mlflow.start_run(run_name="LinearRegression"):
    lr_model = LinearRegression()
    lr_model.fit(X_train, Y_train)
    
    # Predictions
    Y_pred_lr = lr_model.predict(X_test)
    
    # Metrics
    r2_lr = r2_score(Y_test, Y_pred_lr)
    mse_lr = mean_squared_error(Y_test, Y_pred_lr)
    
    # Log metrics
    mlflow.log_metric("r2_score", r2_lr)
    mlflow.log_metric("mean_squared_error", mse_lr)
    
    # Log model name as a parameter
    mlflow.log_param("model_name", "LinearRegression")
    
    # Log model
    mlflow.sklearn.log_model(lr_model, "LinearRegression_model")
    
    print(f"Linear Regression -> R2: {r2_lr}, MSE: {mse_lr}")
    
    # Update the best model if Linear Regression has higher R2
    if r2_lr > best_r2:
        best_model = lr_model
        best_r2 = r2_lr
        best_model_name = "LinearRegression"

# Log the best-performing model in MLflow’s Model Registry
if best_model is not None:
    with mlflow.start_run(run_name="BestModel"):
        mlflow.log_param("Best_Model_Name", best_model_name)
        mlflow.log_metric("Best_r2_score", best_r2)
        
        # Log the best model
        mlflow.sklearn.log_model(best_model, "Best_Model")
        print(f"Best model saved: {best_model_name} with R2: {best_r2}")

