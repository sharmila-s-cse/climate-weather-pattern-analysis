"""
Data Generation Module: Climate & Weather Pattern Analysis
-----------------------------------------------------------
Generates a realistic, synthetic multi-year daily dataset modeling the
interaction between meteorological variables (Temperature, Precipitation,
Humidity, Weather Condition) and consumer retail purchasing behaviors
across distinct product categories.
"""

import os
import numpy as np
import pandas as pd

def generate_climate_purchasing_dataset(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    random_seed: int = 42,
    include_imperfections: bool = True
) -> pd.DataFrame:
    """
    Generate synthetic daily meteorological and retail purchasing dataset.

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    random_seed : int
        Seed for reproducibility.
    include_imperfections : bool
        Whether to introduce minor missing values and duplicates for cleaning demonstration.

    Returns:
    --------
    pd.DataFrame : The generated raw dataset.
    """
    np.random.seed(random_seed)
    
    # 1. Generate continuous date sequence
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    num_days = len(date_range)
    day_of_year = date_range.dayofyear.values
    
    # 2. Meteorological Simulation
    # Temperature: Annual seasonal sinusoidal wave with stochastic noise
    # Seasonal peak around mid-July (day ~195) and trough around mid-January (day ~15)
    temp_seasonal_cycle = 18.0 - 13.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
    temp_noise = np.random.normal(loc=0.0, scale=2.8, size=num_days)
    temperature = np.round(temp_seasonal_cycle + temp_noise, 1)
    
    # Precipitation: Seasonal rain probability + Gamma distributed rainfall depth
    # Higher rain probability in Spring (March-May) and Autumn (Sep-Nov)
    rain_prob_seasonal = 0.28 + 0.15 * np.sin(2 * np.pi * (day_of_year - 60) / 365.25)**2
    is_rainy_day = np.random.binomial(n=1, p=rain_prob_seasonal, size=num_days)
    
    # Gamma distribution for rainfall amounts (zero-inflated)
    rain_amounts = np.random.gamma(shape=2.5, scale=5.0, size=num_days)
    precipitation = np.where(is_rainy_day == 1, rain_amounts, 0.0)
    
    # Introduce occasional extreme precipitation anomalies (extreme storms)
    anomaly_indices = np.random.choice(num_days, size=14, replace=False)
    precipitation[anomaly_indices] += np.random.uniform(45.0, 95.0, size=14)
    precipitation = np.round(precipitation, 1)
    
    # Humidity: Positively driven by precipitation, inverse relationship with extreme dry heat
    base_humidity = 55.0 + 0.65 * precipitation - 0.4 * (temperature - 18.0)
    humidity_noise = np.random.normal(loc=0.0, scale=6.5, size=num_days)
    humidity = np.clip(np.round(base_humidity + humidity_noise, 1), 22.0, 98.5)
    
    # Weather Condition Categorization
    weather_conditions = []
    for t, p, h in zip(temperature, precipitation, humidity):
        if p >= 35.0:
            weather_conditions.append('Stormy')
        elif p > 1.0 and t <= 1.0:
            weather_conditions.append('Snowy')
        elif p >= 0.5:
            weather_conditions.append('Rainy')
        elif h >= 72.0:
            weather_conditions.append('Cloudy')
        else:
            weather_conditions.append('Sunny')
            
    daily_weather_df = pd.DataFrame({
        'Date': date_range.strftime('%Y-%m-%d'),
        'Temperature': temperature,
        'Precipitation': precipitation,
        'Humidity': humidity,
        'Weather_Condition': weather_conditions
    })
    
    # 3. Product Categories & Retail Demand Dynamics
    categories = [
        'Cold Beverages & Ice Cream',
        'Hot Beverages & Soups',
        'Rainwear & Umbrellas',
        'Outdoor & Camping Gear',
        'Heating & Winter Apparel'
    ]
    
    records = []
    for _, row in daily_weather_df.iterrows():
        d_str = row['Date']
        t = row['Temperature']
        p = row['Precipitation']
        h = row['Humidity']
        w = row['Weather_Condition']
        
        # Day of week multiplier (weekend shopping bump)
        dt = pd.to_datetime(d_str)
        is_weekend = dt.weekday() >= 5
        weekend_mult = 1.25 if is_weekend else 1.0
        
        for cat in categories:
            # Baseline parameters
            base_purchases = 120
            avg_unit_price = 25.0
            
            if cat == 'Cold Beverages & Ice Cream':
                # Strongly driven by high temperature, depressed by rain/cold
                temp_factor = max(0.2, (t - 5.0) / 20.0) ** 1.6
                rain_penalty = 0.65 if p > 5.0 else 1.0
                num_purchases = int(base_purchases * temp_factor * rain_penalty * weekend_mult + np.random.normal(0, 10))
                avg_unit_price = 8.5
                
            elif cat == 'Hot Beverages & Soups':
                # Driven by cold temperature and rain
                temp_factor = max(0.3, (32.0 - t) / 20.0) ** 1.3
                rain_boost = 1.35 if p > 3.0 else 1.0
                num_purchases = int(base_purchases * temp_factor * rain_boost * weekend_mult + np.random.normal(0, 8))
                avg_unit_price = 12.0
                
            elif cat == 'Rainwear & Umbrellas':
                # Highly non-linear sensitivity to precipitation
                rain_surge = 1.0 + (p ** 1.25) * 2.8
                num_purchases = int(25 * rain_surge * (1.1 if is_weekend else 1.0) + np.random.normal(0, 6))
                avg_unit_price = 28.0
                
            elif cat == 'Outdoor & Camping Gear':
                # Flourishes in mild/warm sunny weather, collapses during rain/storms
                temp_suitability = np.exp(-((t - 22.0) ** 2) / (2 * 7.5 ** 2))
                rain_deterrent = max(0.15, 1.0 - (p / 15.0))
                num_purchases = int(70 * temp_suitability * rain_deterrent * (1.4 if is_weekend else 1.0) + np.random.normal(0, 7))
                avg_unit_price = 85.0
                
            elif cat == 'Heating & Winter Apparel':
                # Driven by cold temperatures (< 10°C)
                cold_factor = max(0.1, (18.0 - t) / 12.0) ** 2.0 if t < 18.0 else 0.08
                num_purchases = int(55 * cold_factor * weekend_mult + np.random.normal(0, 5))
                avg_unit_price = 65.0
            
            num_purchases = max(5, int(num_purchases))
            # Purchase amount with realistic basket value noise
            spend_noise = np.random.uniform(0.92, 1.08)
            purchase_amount = np.round(num_purchases * avg_unit_price * spend_noise, 2)
            
            records.append({
                'Date': d_str,
                'Temperature': t,
                'Precipitation': p,
                'Humidity': h,
                'Weather_Condition': w,
                'Product_Category': cat,
                'Purchase_Amount': purchase_amount,
                'Number_of_Purchases': num_purchases
            })
            
    df = pd.DataFrame(records)
    
    # 4. Introduce minor real-world imperfections if requested
    if include_imperfections:
        # A few missing values in Humidity (~0.5%) and Precipitation (~0.3%)
        missing_h_indices = np.random.choice(len(df), size=int(len(df) * 0.005), replace=False)
        missing_p_indices = np.random.choice(len(df), size=int(len(df) * 0.003), replace=False)
        df.loc[missing_h_indices, 'Humidity'] = np.nan
        df.loc[missing_p_indices, 'Precipitation'] = np.nan
        
        # A few duplicate rows (e.g., 12 duplicates) to demonstrate deduplication
        dup_indices = np.random.choice(len(df), size=12, replace=False)
        duplicates = df.iloc[dup_indices].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
        # Shuffle slightly to make duplicates realistic
        df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        
    return df

def save_raw_dataset(output_dir: str = "data/raw"):
    """Generate and save the raw dataset."""
    os.makedirs(output_dir, exist_ok=True)
    df = generate_climate_purchasing_dataset(random_seed=42)
    filepath = os.path.join(output_dir, "climate_weather_purchasing_raw.csv")
    df.to_csv(filepath, index=False)
    print(f"Dataset successfully created and saved to: {filepath}")
    print(f"Total Rows: {len(df):,}, Total Columns: {df.shape[1]}")
    return filepath, df

if __name__ == "__main__":
    save_raw_dataset()
