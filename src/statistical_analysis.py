"""
Statistical Analysis Module: Climate & Weather Pattern Analysis
--------------------------------------------------------------
Performs descriptive statistics, seasonal aggregations, precipitation anomaly
detection via z-score metrics, correlation analysis, and formal hypothesis
testing (Pearson/Spearman correlation tests, ANOVA, Independent t-tests) using SciPy.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple

def compute_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary descriptive statistics for numerical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset.
        
    Returns:
    --------
    pd.DataFrame : Descriptive statistics including mean, median, std, IQR, skewness.
    """
    num_cols = ['Temperature', 'Precipitation', 'Humidity', 'Purchase_Amount', 'Number_of_Purchases']
    stats_df = df[num_cols].describe().T
    
    # Add median, IQR, skewness, kurtosis
    stats_df['median'] = df[num_cols].median()
    stats_df['iqr'] = stats_df['75%'] - stats_df['25%']
    stats_df['skewness'] = df[num_cols].skew()
    stats_df['kurtosis'] = df[num_cols].kurtosis()
    
    return stats_df[['count', 'mean', 'std', 'min', '25%', 'median', '75%', 'max', 'iqr', 'skewness', 'kurtosis']]

def analyze_monthly_and_seasonal_temperatures(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates monthly and seasonal temperature distributions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset.
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame] : (Monthly summary, Seasonal summary)
    """
    # Group at the daily level first to avoid multi-counting across categories
    daily_weather = df[['Date', 'Month', 'Month_Name', 'Season', 'Temperature']].drop_duplicates()
    
    monthly_temp = daily_weather.groupby(['Month', 'Month_Name'], observed=True)['Temperature'].agg(
        Mean_Temp='mean',
        Std_Temp='std',
        Min_Temp='min',
        Max_Temp='max'
    ).reset_index().sort_values('Month')
    
    seasonal_temp = daily_weather.groupby('Season', observed=True)['Temperature'].agg(
        Mean_Temp='mean',
        Std_Temp='std',
        Min_Temp='min',
        Max_Temp='max'
    ).reset_index()
    
    return monthly_temp, seasonal_temp

def detect_precipitation_anomalies(
    df: pd.DataFrame,
    z_threshold: float = 2.5
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Identifies precipitation anomalies using z-score thresholding on daily rainfall.
    
    Formula:
        z = (Precipitation - Mean) / StdDev
        Anomaly = |z| > z_threshold
        
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset.
    z_threshold : float
        Z-score threshold for extreme precipitation event (default 2.5).
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, Dict] :
        - daily_precip: Daily precipitation timeline with z-scores and anomaly flags.
        - anomaly_records: Dataframe containing only the anomaly days.
        - metrics: Dictionary of summary anomaly statistics.
    """
    daily_precip = df[['Date', 'Month', 'Season', 'Precipitation', 'Weather_Condition']].drop_duplicates().copy()
    daily_precip = daily_precip.sort_values('Date').reset_index(drop=True)
    
    mean_p = daily_precip['Precipitation'].mean()
    std_p = daily_precip['Precipitation'].std()
    
    daily_precip['Z_Score'] = (daily_precip['Precipitation'] - mean_p) / std_p
    daily_precip['Is_Anomaly'] = daily_precip['Z_Score'] > z_threshold
    
    # Also calculate 30-day rolling mean and anomaly baseline
    daily_precip['Rolling_Mean_30d'] = daily_precip['Precipitation'].rolling(window=30, min_periods=1, center=True).mean()
    daily_precip['Anomaly_Deviation'] = daily_precip['Precipitation'] - daily_precip['Rolling_Mean_30d']
    
    anomaly_days = daily_precip[daily_precip['Is_Anomaly']].copy()
    
    # Purchasing impact on anomaly days vs normal days
    anomaly_dates = set(anomaly_days['Date'])
    df_flagged = df.copy()
    df_flagged['Is_Rain_Anomaly'] = df_flagged['Date'].isin(anomaly_dates)
    
    purchases_on_anomaly = df_flagged.groupby(['Is_Rain_Anomaly', 'Product_Category'], observed=True)['Purchase_Amount'].agg(
        Mean_Spend='mean',
        Total_Spend='sum'
    ).reset_index()
    
    metrics = {
        "mean_precipitation": float(mean_p),
        "std_precipitation": float(std_p),
        "total_days": len(daily_precip),
        "total_rainy_days": int((daily_precip['Precipitation'] > 0).sum()),
        "anomaly_count": int(daily_precip['Is_Anomaly'].sum()),
        "anomaly_threshold_mm": float(mean_p + z_threshold * std_p),
        "max_precipitation_mm": float(daily_precip['Precipitation'].max())
    }
    
    return daily_precip, anomaly_days, metrics

def analyze_product_category_purchases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes retail purchases grouped by Product_Category and Weather_Condition.
    """
    cat_summary = df.groupby(['Product_Category', 'Weather_Condition'], observed=True).agg(
        Avg_Purchase_Amount=('Purchase_Amount', 'mean'),
        Total_Purchase_Amount=('Purchase_Amount', 'sum'),
        Avg_Number_of_Purchases=('Number_of_Purchases', 'mean'),
        Total_Transactions=('Number_of_Purchases', 'sum'),
        Observations=('Date', 'count')
    ).reset_index()
    return cat_summary

def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Pearson correlation matrix across numerical variables.
    """
    num_cols = ['Temperature', 'Precipitation', 'Humidity', 'Purchase_Amount', 'Number_of_Purchases']
    return df[num_cols].corr(method='pearson')

def perform_statistical_hypothesis_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Conducts formal statistical significance tests using SciPy:
    1. Pearson & Spearman correlation tests with p-values for weather vs purchases.
    2. Category-specific correlation tests (e.g. Temp vs Cold Beverages, Rain vs Rainwear).
    3. One-Way ANOVA across Weather Conditions for total daily purchase spend.
    4. Two-sample independent t-test comparing spending on rainy vs non-rainy days.
    
    Returns:
    --------
    Dict[str, Any] : Detailed dictionary of test statistics and p-values.
    """
    results = {}
    
    # 1. Pearson and Spearman tests (Overall)
    r_temp_amount, p_temp_amount = stats.pearsonr(df['Temperature'], df['Purchase_Amount'])
    rho_temp_amount, p_rho_temp_amount = stats.spearmanr(df['Temperature'], df['Purchase_Amount'])
    
    r_precip_amount, p_precip_amount = stats.pearsonr(df['Precipitation'], df['Purchase_Amount'])
    rho_precip_amount, p_rho_precip_amount = stats.spearmanr(df['Precipitation'], df['Purchase_Amount'])
    
    results['overall_correlations'] = {
        'temp_vs_amount_pearson': {'r': float(r_temp_amount), 'p_value': float(p_temp_amount)},
        'temp_vs_amount_spearman': {'rho': float(rho_temp_amount), 'p_value': float(p_rho_temp_amount)},
        'precip_vs_amount_pearson': {'r': float(r_precip_amount), 'p_value': float(p_precip_amount)},
        'precip_vs_amount_spearman': {'rho': float(rho_precip_amount), 'p_value': float(p_rho_precip_amount)}
    }
    
    # 2. Key Category-Specific Hypotheses
    # Hypothesis A: Temperature significantly drives 'Cold Beverages & Ice Cream' purchases
    cold_bev = df[df['Product_Category'] == 'Cold Beverages & Ice Cream']
    r_cold, p_cold = stats.pearsonr(cold_bev['Temperature'], cold_bev['Purchase_Amount'])
    
    # Hypothesis B: Precipitation significantly drives 'Rainwear & Umbrellas' purchases
    rainwear = df[df['Product_Category'] == 'Rainwear & Umbrellas']
    r_rain, p_rain = stats.pearsonr(rainwear['Precipitation'], rainwear['Purchase_Amount'])
    
    # Hypothesis C: Temperature negatively drives 'Heating & Winter Apparel' purchases
    winter = df[df['Product_Category'] == 'Heating & Winter Apparel']
    r_winter, p_winter = stats.pearsonr(winter['Temperature'], winter['Purchase_Amount'])
    
    results['category_hypotheses'] = {
        'cold_beverages_temp': {'pearson_r': float(r_cold), 'p_value': float(p_cold), 'significant': p_cold < 0.05},
        'rainwear_precipitation': {'pearson_r': float(r_rain), 'p_value': float(p_rain), 'significant': p_rain < 0.05},
        'winter_apparel_temp': {'pearson_r': float(r_winter), 'p_value': float(p_winter), 'significant': p_winter < 0.05}
    }
    
    # 3. One-Way ANOVA: Does mean purchase spend differ across Weather Conditions?
    # Daily aggregate spend across weather conditions
    daily_spend = df.groupby(['Date', 'Weather_Condition'], observed=True)['Purchase_Amount'].sum().reset_index()
    groups = [group['Purchase_Amount'].values for _, group in daily_spend.groupby('Weather_Condition', observed=True) if len(group) > 0]
    
    f_stat, anova_p = stats.f_oneway(*groups)
    results['anova_weather_condition'] = {
        'f_statistic': float(f_stat),
        'p_value': float(anova_p),
        'significant': anova_p < 0.05
    }
    
    # 4. Independent Two-Sample T-test: Rainy vs Non-Rainy Day Purchasing
    daily_spend_all = df.groupby('Date').agg({
        'Purchase_Amount': 'sum',
        'Precipitation': 'first'
    }).reset_index()
    
    rainy_spend = daily_spend_all[daily_spend_all['Precipitation'] > 0.5]['Purchase_Amount']
    dry_spend = daily_spend_all[daily_spend_all['Precipitation'] <= 0.5]['Purchase_Amount']
    
    t_stat, t_pval = stats.ttest_ind(rainy_spend, dry_spend, equal_var=False)
    results['ttest_rainy_vs_dry'] = {
        't_statistic': float(t_stat),
        'p_value': float(t_pval),
        'rainy_mean_spend': float(rainy_spend.mean()),
        'dry_mean_spend': float(dry_spend.mean()),
        'significant': t_pval < 0.05
    }
    
    return results

if __name__ == "__main__":
    from data_preprocessing import process_and_save_data
    df = process_and_save_data()
    stats_df = compute_descriptive_statistics(df)
    print("Descriptive Statistics:\n", stats_df)
    daily_p, anom, p_metrics = detect_precipitation_anomalies(df)
    print(f"\nPrecipitation Anomalies Detected: {p_metrics['anomaly_count']} days (Threshold: {p_metrics['anomaly_threshold_mm']:.1f} mm)")
    test_results = perform_statistical_hypothesis_tests(df)
    print("\nHypothesis Testing Results:\n", test_results)
