"""
Main Pipeline Runner: Climate & Weather Pattern Analysis
=========================================================
Executes the end-to-end analytics workflow:
1. Synthetic data generation with realistic meteorology & retail demand
2. Data quality audit, missing-value imputation, deduplication, feature engineering
3. Descriptive statistics, seasonal analysis, precipitation anomaly detection
4. Correlation analysis and SciPy statistical hypothesis testing
5. Scikit-learn Linear Regression modeling and evaluation
6. Publication-grade visualization export
"""

import os
import sys

# Ensure src/ is on Python search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from generate_data import save_raw_dataset
from data_preprocessing import process_and_save_data
from statistical_analysis import (
    compute_descriptive_statistics,
    analyze_monthly_and_seasonal_temperatures,
    detect_precipitation_anomalies,
    analyze_product_category_purchases,
    compute_correlation_matrix,
    perform_statistical_hypothesis_tests
)
from regression_modeling import run_all_regressions
from visualization import generate_all_visualizations

def run_project_pipeline():
    print("=" * 80)
    print("      CLIMATE & WEATHER PATTERN ANALYSIS: END-TO-END PIPELINE")
    print("=" * 80)
    
    # 1. Dataset Generation
    raw_csv = "data/raw/climate_weather_purchasing_raw.csv"
    if not os.path.exists(raw_csv):
        print("\n>>> STEP 1: Generating Realistic Climate & Retail Dataset...")
        save_raw_dataset()
    else:
        print(f"\n>>> STEP 1: Found existing raw dataset at {raw_csv}")
        
    # 2. Data Preprocessing & Cleaning
    print("\n>>> STEP 2: Executing Data Preprocessing & Quality Checks...")
    df = process_and_save_data(raw_path=raw_csv, processed_dir="data/processed")
    
    # 3. Statistical Analysis
    print("\n>>> STEP 3: Computing Statistical Metrics & Trends...")
    descriptive_stats = compute_descriptive_statistics(df)
    print("\n--- Descriptive Statistics Table ---")
    print(descriptive_stats.to_string())
    
    monthly_temp, seasonal_temp = analyze_monthly_and_seasonal_temperatures(df)
    print("\n--- Monthly Temperature Climatology (°C) ---")
    print(monthly_temp[['Month_Name', 'Mean_Temp', 'Std_Temp', 'Min_Temp', 'Max_Temp']].to_string(index=False))
    
    print("\n--- Seasonal Temperature Summary (°C) ---")
    print(seasonal_temp.to_string(index=False))
    
    # Precipitation Anomaly Detection
    daily_precip, anomaly_days, anom_metrics = detect_precipitation_anomalies(df, z_threshold=2.5)
    print(f"\n--- Precipitation Anomaly Detection (Z-Score > 2.5) ---")
    print(f"Total Observation Days: {anom_metrics['total_days']}")
    print(f"Total Rainy Days: {anom_metrics['total_rainy_days']} ({(anom_metrics['total_rainy_days']/anom_metrics['total_days'])*100:.1f}%)")
    print(f"Precipitation Mean: {anom_metrics['mean_precipitation']:.2f} mm | Std Dev: {anom_metrics['std_precipitation']:.2f} mm")
    print(f"Anomaly Threshold: {anom_metrics['anomaly_threshold_mm']:.2f} mm")
    print(f"Detected Anomaly Events: {anom_metrics['anomaly_count']} days")
    print(f"Peak Rainfall Record: {anom_metrics['max_precipitation_mm']:.2f} mm")
    
    # Category Purchasing Breakdown
    print("\n--- Product Category Purchasing Summary ---")
    cat_summary = analyze_product_category_purchases(df)
    cat_pivot = cat_summary.pivot(index='Product_Category', columns='Weather_Condition', values='Avg_Purchase_Amount')
    print(cat_pivot.round(2).to_string())
    
    # Correlation & Hypothesis Testing
    print("\n>>> STEP 4: Correlation & Formal Hypothesis Testing (SciPy)...")
    corr_mat = compute_correlation_matrix(df)
    print("\n--- Pearson Correlation Matrix ---")
    print(corr_mat.round(3).to_string())
    
    test_results = perform_statistical_hypothesis_tests(df)
    print("\n--- Hypothesis Test Findings ---")
    for cat_name, cat_res in test_results['category_hypotheses'].items():
        sig_str = "Statistically Significant (p < 0.05)" if cat_res['significant'] else "Not Significant"
        print(f"  • {cat_name}: Pearson r = {cat_res['pearson_r']:.4f}, p-value = {cat_res['p_value']:.4e} -> {sig_str}")
        
    anova_res = test_results['anova_weather_condition']
    print(f"  • One-Way ANOVA across Weather Conditions: F = {anova_res['f_statistic']:.4f}, p = {anova_res['p_value']:.4e}")
    
    ttest_res = test_results['ttest_rainy_vs_dry']
    print(f"  • T-Test (Rainy vs Dry Day Total Spend): t = {ttest_res['t_statistic']:.4f}, p = {ttest_res['p_value']:.4e}")
    print(f"    Rainy Day Mean Spend: ${ttest_res['rainy_mean_spend']:.2f} vs Dry Day Mean Spend: ${ttest_res['dry_mean_spend']:.2f}")
    
    # 4. Linear Regression Analysis
    print("\n>>> STEP 5: Scikit-learn Linear Regression Modeling...")
    regressions = run_all_regressions(df)
    for model_key, model_data in regressions.items():
        print(f"\nModel: {model_data['model_name']}")
        if 'formula' in model_data:
            print(f"  Fitted Line: {model_data['formula']}")
        if 'coefficients' in model_data:
            print(f"  Coefficients: {model_data['coefficients']}")
            print(f"  Intercept: {model_data['intercept']:.2f}")
        print(f"  R² Score: {model_data['r2']:.4f} | MAE: ${model_data['mae']:.2f} | RMSE: ${model_data['rmse']:.2f}")
        
    # 5. Visualizations
    print("\n>>> STEP 6: Generating Publication Visualizations...")
    generate_all_visualizations(df, regressions, output_dir="visualizations")
    
    print("\n" + "=" * 80)
    print("SUCCESS: Pipeline completed flawlessly! All data, metrics, and charts ready.")
    print("=" * 80)
    
    return {
        "df": df,
        "descriptive_stats": descriptive_stats,
        "monthly_temp": monthly_temp,
        "seasonal_temp": seasonal_temp,
        "anomaly_metrics": anom_metrics,
        "test_results": test_results,
        "regressions": regressions
    }

if __name__ == "__main__":
    run_project_pipeline()
