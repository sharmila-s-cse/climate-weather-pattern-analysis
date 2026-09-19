"""
Regression Modeling Module: Climate & Weather Pattern Analysis
--------------------------------------------------------------
Applies Scikit-learn Linear Regression models to quantify the explanatory power
of meteorological factors on consumer purchasing behavior.
Focuses on clear, interpretable statistical modeling, computing coefficients,
intercepts, R², MAE, and RMSE metrics.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Any

def run_single_linear_regression(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    model_name: str
) -> Dict[str, Any]:
    """
    Fits an Ordinary Least Squares (OLS) Linear Regression model for 1 feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data subset containing feature and target.
    feature_col : str
        Predictor variable name (e.g., 'Temperature').
    target_col : str
        Target variable name (e.g., 'Purchase_Amount').
    model_name : str
        Identifier name for the model.
        
    Returns:
    --------
    Dict[str, Any] : Fitted model, predictions, coefficients, and goodness-of-fit metrics.
    """
    X = df[[feature_col]].values
    y = df[target_col].values
    
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    slope = float(reg.coef_[0])
    intercept = float(reg.intercept_)
    
    return {
        "model_name": model_name,
        "feature": feature_col,
        "target": target_col,
        "model": reg,
        "slope": slope,
        "intercept": intercept,
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "formula": f"{target_col} = {slope:.2f} * {feature_col} + {intercept:.2f}",
        "X": X,
        "y": y,
        "y_pred": y_pred
    }

def run_multivariable_regression(
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Fits a multivariable linear regression model predicting daily total store sales
    from all primary meteorological drivers: Temperature, Precipitation, Humidity.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset.
        
    Returns:
    --------
    Dict[str, Any] : Evaluation metrics and coefficients.
    """
    # Aggregate to daily total store level
    daily_df = df.groupby('Date').agg({
        'Temperature': 'first',
        'Precipitation': 'first',
        'Humidity': 'first',
        'Purchase_Amount': 'sum',
        'Number_of_Purchases': 'sum'
    }).reset_index()
    
    feature_cols = ['Temperature', 'Precipitation', 'Humidity']
    X = daily_df[feature_cols].values
    y = daily_df['Purchase_Amount'].values
    
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    coeff_dict = {col: float(coef) for col, coef in zip(feature_cols, reg.coef_)}
    
    return {
        "model_name": "Multivariable Daily Total Sales Model",
        "features": feature_cols,
        "target": "Daily_Total_Purchase_Amount",
        "coefficients": coeff_dict,
        "intercept": float(reg.intercept_),
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "y": y,
        "y_pred": y_pred
    }

def run_all_regressions(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Executes the suite of domain-specific linear regression analyses.
    
    Returns:
    --------
    Dict[str, Dict[str, Any]] : Dictionary of results for each regression model.
    """
    results = {}
    
    # 1. Cold Beverages & Ice Cream vs Temperature
    cold_df = df[df['Product_Category'] == 'Cold Beverages & Ice Cream']
    results['cold_beverages'] = run_single_linear_regression(
        cold_df, 'Temperature', 'Purchase_Amount',
        'Cold Beverages Spend vs Temperature'
    )
    
    # 2. Rainwear & Umbrellas vs Precipitation
    rain_df = df[df['Product_Category'] == 'Rainwear & Umbrellas']
    results['rainwear'] = run_single_linear_regression(
        rain_df, 'Precipitation', 'Purchase_Amount',
        'Rainwear Spend vs Precipitation'
    )
    
    # 3. Heating & Winter Apparel vs Temperature
    winter_df = df[df['Product_Category'] == 'Heating & Winter Apparel']
    results['winter_apparel'] = run_single_linear_regression(
        winter_df, 'Temperature', 'Purchase_Amount',
        'Winter Apparel Spend vs Temperature'
    )
    
    # 4. Multivariable model for overall store sales
    results['multivariable'] = run_multivariable_regression(df)
    
    return results

if __name__ == "__main__":
    from data_preprocessing import process_and_save_data
    df = process_and_save_data()
    results = run_all_regressions(df)
    for k, res in results.items():
        print(f"\n--- {res['model_name']} ---")
        if 'formula' in res:
            print(f"Formula: {res['formula']}")
        print(f"R² Score: {res['r2']:.4f} | MAE: {res['mae']:.2f} | RMSE: {res['rmse']:.2f}")
