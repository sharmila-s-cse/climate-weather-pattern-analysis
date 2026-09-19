"""
Data Preprocessing Module: Climate & Weather Pattern Analysis
-------------------------------------------------------------
Handles data loading, structural inspection, missing value detection and imputation,
duplicate record removal, type conversion, and temporal feature engineering.
"""

import os
import pandas as pd
import numpy as np

def load_and_inspect_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw CSV data and performs initial quality inspection.
    
    Parameters:
    -----------
    filepath : str
        Path to the raw CSV file.
        
    Returns:
    --------
    pd.DataFrame : The loaded raw dataframe.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at path: {filepath}")
        
    df = pd.read_csv(filepath)
    print("=" * 60)
    print(f"Data Loaded Successfully from: {filepath}")
    print(f"Initial Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("=" * 60)
    print("\nFirst 5 Rows:")
    print(df.head())
    print("\nData Info:")
    df.info()
    return df

def check_quality_issues(df: pd.DataFrame) -> dict:
    """
    Checks for missing values and duplicate records.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe.
        
    Returns:
    --------
    dict : Summary dictionary of quality issues found.
    """
    missing_series = df.isnull().sum()
    missing_counts = missing_series[missing_series > 0].to_dict()
    duplicate_count = int(df.duplicated().sum())
    
    print("\n" + "=" * 60)
    print("DATA QUALITY AUDIT REPORT")
    print("=" * 60)
    print(f"Total Duplicate Rows Found: {duplicate_count}")
    if missing_counts:
        print("Missing Values Per Column:")
        for col, cnt in missing_counts.items():
            pct = (cnt / len(df)) * 100
            print(f"  - {col}: {cnt} missing ({pct:.2f}%)")
    else:
        print("Missing Values: None found (0)")
        
    return {
        "duplicates": duplicate_count,
        "missing_counts": missing_counts
    }

def clean_and_transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes full data cleaning pipeline:
    1. Deduplication: Removes exact duplicate rows.
    2. Missing Value Imputation:
       - Humidity: Median imputation conditioned on Weather_Condition.
       - Precipitation: Filled with 0.0 or median conditioned on Weather_Condition.
    3. Type Conversion:
       - Date string -> pd.to_datetime
       - Categorical conversions for Weather_Condition and Product_Category
    4. Feature Engineering:
       - Year, Month, Day, Day_of_Week, Is_Weekend
       - Season (Winter: Dec-Feb, Spring: Mar-May, Summer: Jun-Aug, Fall: Sep-Nov)
       
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe.
        
    Returns:
    --------
    pd.DataFrame : Cleaned and feature-enriched dataframe.
    """
    cleaned_df = df.copy()
    
    # 1. Remove duplicates
    initial_rows = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    dropped_dups = initial_rows - len(cleaned_df)
    print(f"\n[Step 1] Deduplication: Removed {dropped_dups} duplicate records.")
    
    # 2. Impute missing values
    if cleaned_df['Precipitation'].isnull().sum() > 0:
        # Group by weather condition to impute sensible precipitation
        precip_medians = cleaned_df.groupby('Weather_Condition')['Precipitation'].transform('median')
        cleaned_df['Precipitation'] = cleaned_df['Precipitation'].fillna(precip_medians).fillna(0.0)
        print("[Step 2a] Imputation: Handled missing Precipitation values using category-conditioned median.")
        
    if cleaned_df['Humidity'].isnull().sum() > 0:
        # Group by weather condition to impute sensible humidity
        humidity_medians = cleaned_df.groupby('Weather_Condition')['Humidity'].transform('median')
        cleaned_df['Humidity'] = cleaned_df['Humidity'].fillna(humidity_medians).fillna(cleaned_df['Humidity'].median())
        print("[Step 2b] Imputation: Handled missing Humidity values using category-conditioned median.")
        
    # 3. Type Conversion
    cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])
    cleaned_df['Weather_Condition'] = cleaned_df['Weather_Condition'].astype('category')
    cleaned_df['Product_Category'] = cleaned_df['Product_Category'].astype('category')
    cleaned_df['Purchase_Amount'] = cleaned_df['Purchase_Amount'].astype(float)
    cleaned_df['Number_of_Purchases'] = cleaned_df['Number_of_Purchases'].astype(int)
    print("[Step 3] Type Casting: Successfully converted Date to datetime and categorical types.")
    
    # 4. Feature Engineering
    cleaned_df['Year'] = cleaned_df['Date'].dt.year
    cleaned_df['Month'] = cleaned_df['Date'].dt.month
    cleaned_df['Month_Name'] = cleaned_df['Date'].dt.strftime('%b')
    cleaned_df['Day'] = cleaned_df['Date'].dt.day
    cleaned_df['Day_of_Week'] = cleaned_df['Date'].dt.day_name()
    cleaned_df['Is_Weekend'] = cleaned_df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Define meteorological seasons
    def assign_season(month: int) -> str:
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
            
    cleaned_df['Season'] = cleaned_df['Month'].apply(assign_season)
    cleaned_df['Season'] = pd.Categorical(cleaned_df['Season'], categories=['Winter', 'Spring', 'Summer', 'Fall'], ordered=True)
    
    # Sort chronologically by Date and Product_Category
    cleaned_df = cleaned_df.sort_values(by=['Date', 'Product_Category']).reset_index(drop=True)
    print(f"[Step 4] Feature Engineering: Created Year, Month, Season, Day_of_Week, Is_Weekend.")
    print(f"Final Cleaned Dataset Shape: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns.\n")
    
    return cleaned_df

def process_and_save_data(
    raw_path: str = "data/raw/climate_weather_purchasing_raw.csv",
    processed_dir: str = "data/processed"
) -> pd.DataFrame:
    """
    End-to-end preprocessing runner that loads, cleans, and exports the dataset.
    """
    os.makedirs(processed_dir, exist_ok=True)
    df_raw = load_and_inspect_data(raw_path)
    check_quality_issues(df_raw)
    df_clean = clean_and_transform_data(df_raw)
    
    # Verify zero missing values remaining
    assert df_clean.isnull().sum().sum() == 0, "Warning: Missing values remain after cleaning!"
    assert df_clean.duplicated().sum() == 0, "Warning: Duplicate rows remain after cleaning!"
    
    out_path = os.path.join(processed_dir, "climate_weather_purchasing_clean.csv")
    df_clean.to_csv(out_path, index=False)
    print(f"Cleaned dataset exported to: {out_path}")
    return df_clean

if __name__ == "__main__":
    process_and_save_data()
