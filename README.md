# Climate & Weather Pattern Analysis

> **Web Application Platform:** **Climate Intelligence**  
> *Tagline:* **Understand climate patterns through data.**

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.15-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.10+-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PWA](https://img.shields.io/badge/PWA-Ready-5A0FC8?logo=pwa&logoColor=white)](https://web.dev/progressive-web-apps/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end meteorological and econometric data analytics platform investigating the empirical relationships between **temperature oscillations**, **precipitation anomalies**, and **consumer purchasing trends** across multiple retail product categories.

---

## 🌟 Key Empirical Results (Calculated from Dataset)

- **Dataset Scope**: **5,480** cleaned daily records across **1,103** continuous observation days (2022–2024).
- **Temperature Range**: Mean temperature of **18.09°C** (Annual range: **-0.9°C** to **41.5°C**).
- **Precipitation Baseline**: Daily mean of **5.40 mm** ($\sigma = 11.33\text{ mm}$), with **403 rainy days (36.5%)**.
- **Extreme Rainfall Anomalies**: **22 anomaly events** isolated using $Z$-Score thresholding ($Z > 2.5\sigma$, threshold: **33.73 mm**, peak record: **110.90 mm**).
- **Cold Beverage Elasticity**: Pearson $r = 0.9089$ ($p < 0.0001$), Linear Regression $R^2 = 0.8262$ ($\text{Spend} = 55.77 \times T - 355.25$).
- **Rainwear Surge Elasticity**: Pearson $r = 0.9822$ ($p < 0.0001$), Linear Regression $R^2 = 0.9647$ ($\text{Spend} = 5361.39 \times P - 4518.83$).
- **Winter Apparel Sensitivity**: Pearson $r = -0.7584$ ($p = 1.42 \times 10^{-205}$), Linear Regression $R^2 = 0.5751$.
- **Multivariable Econometric Predictability**: Scikit-Learn regression achieves **$R^2 = 0.9586$** ($95.86\%$ of daily aggregate store spend variance is explained by weather drivers).
- **One-Way ANOVA**: Statistically significant consumer behavior variance across weather conditions ($F = 1104.96, p < 0.0001$).
- **Two-Sample $t$-Test**: Mean rainy day spend ($\$71,176.98$) significantly exceeds dry day spend ($\$7,457.64$) ($t = 14.25, p = 1.91 \times 10^{-37}$).

---

## 🏛️ Project Architecture

```
Raw Dataset Ingestion (5,492 rows)
       │
       ▼
Data Quality Audit (12 Duplicates, 43 Missing Values)
       │
       ▼
Data Cleaning & Imputation (Category-conditioned Medians)
       │
       ▼
Feature Engineering (Year, Month, Season, DayOfWeek, IsWeekend)
       │
       ▼
Exploratory Climatology & Z-Score Anomaly Detection (Z > 2.5σ)
       │
       ▼
Inferential Statistics (SciPy: Pearson, Spearman, ANOVA, t-tests)
       │
       ▼
Predictive Modeling (Scikit-Learn OLS Linear Regression, R² = 0.9586)
       │
       ▼
Publication Visualizations (Matplotlib & Seaborn 300 DPI)
       │
       ▼
Interactive Web Platform ("Climate Intelligence" SPA & PWA)
```

---

## 🚀 How to Run the Project

### Option 1: Launch the Interactive Web Application (Recommended)

Run the built-in server with Python:

```bash
python app.py
```

- **Localhost URL**: `http://localhost:8000/`
- **Mobile / Local Network**: `http://<your-local-ip>:8000/`
- Click the **"Mobile QR"** button in the header to display a QR code and preview the application instantly on your smartphone!

### Option 2: Execute the Python Analytics Pipeline

Run the master command-line pipeline:

```bash
python main.py
```

This runs:
1. `src/generate_data.py`: Raw dataset generation with meteorological physics.
2. `src/data_preprocessing.py`: Quality audit, deduplication, and imputation.
3. `src/statistical_analysis.py`: Descriptive stats, seasonal cycles, anomaly detection, SciPy tests.
4. `src/regression_modeling.py`: Scikit-Learn regression models ($R^2$, MAE, RMSE).
5. `src/visualization.py`: 8 publication charts saved to `visualizations/`.

### Option 3: Explore in Jupyter Notebook

Open and execute `climate_weather_analysis.ipynb` in VS Code or JupyterLab:

```bash
jupyter notebook climate_weather_analysis.ipynb
```

---

## 📊 Visualizations

### 1. Temperature Trend
Continuous timeline tracking seasonal sinusoidal cycles with 30-day moving average.
![Temperature Trend](visualizations/01_temperature_trend.png)

### 2. Monthly Average Temperature
Monthly climatology showing seasonal shifts and standard deviation error bounds.
![Monthly Average Temperature](visualizations/02_monthly_avg_temperature.png)

### 3. Precipitation Trend
Timeline of daily rainfall depth (mm) and rolling baseline precipitation.
![Precipitation Trend](visualizations/03_precipitation_trend.png)

### 4. Precipitation Anomalies
Z-score anomaly identification highlighting 22 extreme storm events exceeding $Z > 2.5\sigma$ ($> 33.73\text{ mm}$).
![Precipitation Anomalies](visualizations/04_precipitation_anomalies.png)

### 5. Purchase Amount Trend
Total daily retail purchasing spend timeline across the 3-year observation window.
![Purchase Amount Trend](visualizations/05_purchase_amount_trend.png)

### 6. Product Category Comparison
Mean daily sales by product category across Sunny, Cloudy, Rainy, Stormy, and Snowy conditions.
![Product Category Comparison](visualizations/06_product_category_comparison.png)

### 7. Correlation Heatmap
Annotated Pearson correlation matrix across weather factors and purchasing metrics.
![Correlation Heatmap](visualizations/07_correlation_heatmap.png)

### 8. Regression Analysis
Scikit-Learn Ordinary Least Squares regression lines with confidence bands and formula annotations.
![Regression Analysis](visualizations/08_regression_plots.png)

---

## 📱 Progressive Web App (PWA) & Public Deployment

The application includes:
- `manifest.json`: Web App Manifest for mobile installation ("Add to Home Screen").
- `sw.js`: Service Worker for offline asset caching.
- Zero-Build Static Deployment: Deploy directly to **Vercel**, **Netlify**, or **GitHub Pages** by pushing the repository.

---

## 📂 Repository Structure

```
climate_weather_pattern_analysis/
├── data/
│   ├── raw/
│   │   └── climate_weather_purchasing_raw.csv        # Raw generated data (5,492 rows)
│   └── processed/
│       ├── climate_weather_purchasing_clean.csv      # Cleaned dataset (5,480 rows)
│       ├── climate_analytics_summary.json            # Web analytics JSON payload
│       └── climate_records_sample.json               # Data Explorer records
├── src/
│   ├── __init__.py
│   ├── generate_data.py                              # Meteorological & demand generator
│   ├── data_preprocessing.py                         # Cleaning, imputation, deduplication
│   ├── statistical_analysis.py                       # SciPy hypothesis tests & anomalies
│   ├── regression_modeling.py                        # Scikit-Learn linear regressions
│   ├── visualization.py                              # Matplotlib & Seaborn 300 DPI plots
│   ├── export_web_data.py                            # Web JSON export pipeline
│   └── create_notebook.py                            # Jupyter notebook generator
├── css/
│   └── styles.css                                    # Dribbble-inspired SaaS design system
├── js/
│   └── app.js                                        # SPA routing, Chart.js, filters, modals
├── icons/
│   └── icon.svg                                      # Vector app mark & PWA icon
├── visualizations/                                   # 8 publication-quality PNG charts
├── climate_weather_analysis.ipynb                    # Complete Jupyter Notebook
├── main.py                                           # CLI master pipeline runner
├── app.py                                            # Local dev server & network host
├── index.html                                        # 14-view interactive web application
├── index_backup.html                                 # Safe backup of previous preview
├── manifest.json                                     # Progressive Web App manifest
├── sw.js                                             # PWA service worker
├── requirements.txt                                  # Dependencies specification
├── .gitignore
└── README.md                                         # Project documentation
```

---

## 🛠️ Technology Stack

- **Data Processing**: Python 3.12 / 3.15, Pandas, NumPy
- **Statistics**: SciPy (`scipy.stats`)
- **Machine Learning**: Scikit-Learn (`LinearRegression`, `r2_score`, `mean_squared_error`)
- **Static Visuals**: Matplotlib, Seaborn
- **Frontend Web UI**: HTML5, Modern CSS Design System, Vanilla JavaScript (ES6+), Chart.js 4.x, Lucide Icons
- **PWA**: Service Worker API, Web App Manifest
