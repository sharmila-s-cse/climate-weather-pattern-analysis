"""
Visualization Module: Climate & Weather Pattern Analysis
---------------------------------------------------------
Generates publication-ready visualizations using Matplotlib and Seaborn:
1. Temperature Trend Over Time (with 30-day moving average)
2. Monthly Average Temperature
3. Precipitation Trend Timeline
4. Precipitation Anomaly Chart (highlighting extreme events > threshold)
5. Purchase Amount Trend Over Time
6. Product-Category Purchase Comparison (across Weather Conditions)
7. Correlation Heatmap (meteorological vs retail purchasing metrics)
8. Linear Regression Plots (with fitted line, confidence interval, equations)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

# Set modern, publication-ready style aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def plot_temperature_trend(df: pd.DataFrame, output_dir: Optional[str] = None):
    """1. Temperature trend over time with rolling 30-day mean."""
    daily = df[['Date', 'Temperature']].drop_duplicates().sort_values('Date').copy()
    daily['Temp_30d_MA'] = daily['Temperature'].rolling(window=30, min_periods=1, center=True).mean()
    
    fig, ax = plt.subplots(figsize=(13, 5), dpi=300)
    ax.plot(daily['Date'], daily['Temperature'], color='#90caf9', alpha=0.55, linewidth=1.0, label='Daily Temperature (°C)')
    ax.plot(daily['Date'], daily['Temp_30d_MA'], color='#d32f2f', linewidth=2.4, label='30-Day Moving Average')
    
    ax.set_title('Daily Temperature Trend & Seasonal Oscillations (2022 - 2024)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    
    # Format x-ticks to display quarter markers cleanly
    step = max(1, len(daily) // 10)
    ax.set_xticks(daily['Date'][::step])
    ax.set_xticklabels([d.strftime('%b %Y') for d in pd.to_datetime(daily['Date'][::step])], rotation=35, ha='right')
    
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, '01_temperature_trend.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig, ax

def plot_monthly_avg_temperature(df: pd.DataFrame, output_dir: Optional[str] = None):
    """2. Monthly average temperature bar chart with error bars and seasonal styling."""
    daily = df[['Date', 'Month', 'Month_Name', 'Temperature']].drop_duplicates()
    monthly = daily.groupby(['Month', 'Month_Name'], observed=True)['Temperature'].agg(['mean', 'std']).reset_index()
    monthly = monthly.sort_values('Month')
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    palette = sns.color_palette("coolwarm", n_colors=12)
    bars = ax.bar(monthly['Month_Name'], monthly['mean'], yerr=monthly['std'], capsize=4, 
                  color=palette, edgecolor='#333333', linewidth=0.8, alpha=0.85)
    
    # Add numerical labels on top of bars
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h / 2, f"{h:.1f}°C",
                ha='center', va='center', color='black' if h > 15 else '#111111', fontsize=9, fontweight='bold')
                
    ax.set_title('Monthly Average Temperature Climatology (±1 Std Dev)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Temperature (°C)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, '02_monthly_avg_temperature.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig, ax

def plot_precipitation_trend(df: pd.DataFrame, output_dir: Optional[str] = None):
    """3. Precipitation trend timeline."""
    daily = df[['Date', 'Precipitation']].drop_duplicates().sort_values('Date').copy()
    daily['Precip_30d_MA'] = daily['Precipitation'].rolling(window=30, min_periods=1, center=True).mean()
    
    fig, ax = plt.subplots(figsize=(13, 5), dpi=300)
    ax.bar(daily['Date'], daily['Precipitation'], color='#0288d1', alpha=0.6, width=1.0, label='Daily Precipitation (mm)')
    ax.plot(daily['Date'], daily['Precip_30d_MA'], color='#01579b', linewidth=2.0, label='30-Day Moving Average')
    
    ax.set_title('Precipitation Trend & Daily Rainfall Events (2022 - 2024)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precipitation (mm)', fontsize=11, fontweight='bold')
    
    step = max(1, len(daily) // 10)
    ax.set_xticks(daily['Date'][::step])
    ax.set_xticklabels([d.strftime('%b %Y') for d in pd.to_datetime(daily['Date'][::step])], rotation=35, ha='right')
    
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, '03_precipitation_trend.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig, ax

def plot_precipitation_anomalies(df: pd.DataFrame, z_threshold: float = 2.5, output_dir: Optional[str] = None):
    """4. Precipitation anomaly chart highlighting extreme events."""
    daily = df[['Date', 'Precipitation']].drop_duplicates().sort_values('Date').copy()
    mean_p = daily['Precipitation'].mean()
    std_p = daily['Precipitation'].std()
    threshold_val = mean_p + z_threshold * std_p
    
    daily['Z_Score'] = (daily['Precipitation'] - mean_p) / std_p
    anomalies = daily[daily['Z_Score'] > z_threshold]
    normal = daily[daily['Z_Score'] <= z_threshold]
    
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=300)
    ax.scatter(normal['Date'], normal['Precipitation'], color='#81d4fa', alpha=0.5, s=16, label='Normal Precipitation')
    ax.scatter(anomalies['Date'], anomalies['Precipitation'], color='#d50000', s=70, edgecolor='black', 
               linewidth=1.2, zorder=5, label=f'Extreme Anomaly (Z > {z_threshold})')
    
    ax.axhline(threshold_val, color='#d50000', linestyle='--', linewidth=1.5, 
               label=f'Anomaly Threshold ({threshold_val:.1f} mm, μ+{z_threshold}σ)')
    ax.axhline(mean_p, color='#0288d1', linestyle=':', linewidth=1.2, label=f'Mean Rainfall ({mean_p:.1f} mm)')
    
    # Annotate top 3 extreme anomaly days
    top_anomalies = anomalies.nlargest(3, 'Precipitation')
    for _, row in top_anomalies.iterrows():
        ax.annotate(f"{pd.to_datetime(row['Date']).strftime('%b %d, %Y')}\n{row['Precipitation']:.1f} mm",
                    xy=(row['Date'], row['Precipitation']),
                    xytext=(10, -5), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.2),
                    fontsize=8.5, fontweight='bold', backgroundcolor='#ffffffcc')
                    
    ax.set_title(f'Precipitation Anomaly Identification (Z-Score > {z_threshold})', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precipitation (mm)', fontsize=11, fontweight='bold')
    
    step = max(1, len(daily) // 10)
    ax.set_xticks(daily['Date'][::step])
    ax.set_xticklabels([d.strftime('%b %Y') for d in pd.to_datetime(daily['Date'][::step])], rotation=35, ha='right')
    
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, '04_precipitation_anomalies.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig, ax

def plot_purchase_amount_trend(df: pd.DataFrame, output_dir: Optional[str] = None):
    """5. Total store daily purchase amount trend."""
    daily_spend = df.groupby('Date')['Purchase_Amount'].sum().reset_index().sort_values('Date')
    daily_spend['Spend_30d_MA'] = daily_spend['Purchase_Amount'].rolling(window=30, min_periods=1, center=True).mean()
    
    fig, ax = plt.subplots(figsize=(13, 5), dpi=300)
    ax.plot(daily_spend['Date'], daily_spend['Purchase_Amount'], color='#a5d6a7', alpha=0.6, linewidth=1.0, label='Daily Spend ($)')
    ax.plot(daily_spend['Date'], daily_spend['Spend_30d_MA'], color='#2e7d32', linewidth=2.4, label='30-Day Moving Average')
    
    ax.set_title('Daily Total Consumer Purchase Amount Trend (2022 - 2024)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Purchase Amount ($)', fontsize=11, fontweight='bold')
    
    step = max(1, len(daily_spend) // 10)
    ax.set_xticks(daily_spend['Date'][::step])
    ax.set_xticklabels([d.strftime('%b %Y') for d in pd.to_datetime(daily_spend['Date'][::step])], rotation=35, ha='right')
    
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, '05_purchase_amount_trend.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig, ax

def plot_product_category_comparison(df: pd.DataFrame, output_dir: Optional[str] = None):
    """6. Product-category purchase comparison across weather conditions."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    order_cond = ['Sunny', 'Cloudy', 'Rainy', 'Stormy', 'Snowy']
    existing_cond = [c for c in order_cond if c in df['Weather_Condition'].values]
    
    sns.barplot(
        data=df,
        x='Product_Category',
        y='Purchase_Amount',
        hue='Weather_Condition',
        hue_order=existing_cond,
        palette='magma',
        ax=ax,
        edgecolor='#333333',
        linewidth=0.7,
        errorbar=None
    )
    
    ax.set_title('Average Daily Purchase Amount by Product Category across Weather Conditions', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Product Category', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Daily Purchase Amount ($)', fontsize=11, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right', fontsize=9.5)
    ax.legend(title='Weather Condition', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, '06_product_category_comparison.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig, ax

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Optional[str] = None):
    """7. Correlation heatmap between meteorological variables and sales behavior."""
    num_cols = ['Temperature', 'Precipitation', 'Humidity', 'Purchase_Amount', 'Number_of_Purchases']
    corr_matrix = df[num_cols].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='vlag',
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=1.2,
        linecolor='white',
        cbar_kws={'label': 'Pearson Correlation Coefficient (r)'},
        ax=ax
    )
    
    ax.set_title('Correlation Matrix: Climate Factors vs Purchasing Behavior', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, '07_correlation_heatmap.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig, ax

def plot_regression_analysis(df: pd.DataFrame, regressions: Dict[str, Any], output_dir: Optional[str] = None):
    """8. Linear regression plots for key meteorological drivers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Subplot 1: Temperature vs Cold Beverages
    cold_df = df[df['Product_Category'] == 'Cold Beverages & Ice Cream']
    cold_res = regressions.get('cold_beverages', {})
    
    sns.regplot(
        data=cold_df,
        x='Temperature',
        y='Purchase_Amount',
        scatter_kws={'alpha': 0.35, 'color': '#ff9800', 's': 20},
        line_kws={'color': '#d84315', 'linewidth': 2.5},
        ax=axes[0]
    )
    r2_cold = cold_res.get('r2', 0)
    eq_cold = cold_res.get('formula', '')
    axes[0].set_title(f"Cold Beverages Spend vs Temperature\n($R^2 = {r2_cold:.3f}$ | {eq_cold})", fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Temperature (°C)', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Purchase Amount ($)', fontsize=10, fontweight='bold')
    
    # Subplot 2: Precipitation vs Rainwear & Umbrellas
    rain_df = df[df['Product_Category'] == 'Rainwear & Umbrellas']
    rain_res = regressions.get('rainwear', {})
    
    sns.regplot(
        data=rain_df,
        x='Precipitation',
        y='Purchase_Amount',
        scatter_kws={'alpha': 0.35, 'color': '#00bcd4', 's': 20},
        line_kws={'color': '#006064', 'linewidth': 2.5},
        ax=axes[1]
    )
    r2_rain = rain_res.get('r2', 0)
    eq_rain = rain_res.get('formula', '')
    axes[1].set_title(f"Rainwear Spend vs Precipitation\n($R^2 = {r2_rain:.3f}$ | {eq_rain})", fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Precipitation (mm)', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Purchase Amount ($)', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, '08_regression_plots.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: {path}")
    return fig, axes

def generate_all_visualizations(df: pd.DataFrame, regressions: Dict[str, Any], output_dir: str = "visualizations"):
    """Generates and saves all 8 required visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print(f"Generating and exporting plots to: {output_dir}")
    print("=" * 60)
    plot_temperature_trend(df, output_dir)
    plot_monthly_avg_temperature(df, output_dir)
    plot_precipitation_trend(df, output_dir)
    plot_precipitation_anomalies(df, z_threshold=2.5, output_dir=output_dir)
    plot_purchase_amount_trend(df, output_dir)
    plot_product_category_comparison(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_regression_analysis(df, regressions, output_dir)
    plt.close('all')
    print("All 8 visualizations successfully generated and saved!")
