# M5 Forecasting - Accuracy Competition

## 📌 Overview

The [M5 Forecasting Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy), hosted on Kaggle, challenged participants to forecast daily unit sales for Walmart products over a 28-day horizon.

The competition involved **42,840 hierarchical time series** organized across multiple aggregation levels: item, department, category, store, and state.

---

## 🎯 Objective

Generate accurate forecasts for the next 28 days of unit sales for all product-store combinations.

The main challenge was to create **hierarchical forecasts** that respect the aggregation structure of Walmart’s retail operations and perform well across all levels of the hierarchy.

---

## 📂 Dataset Description

The dataset included the following files:

- `sales_train_evaluation.csv`: Daily historical unit sales data for each item-store combination.
- `calendar.csv`: Contains date metadata, including event and SNAP (Supplemental Nutrition Assistance Program) indicators.
- `sell_prices.csv`: Item prices by store and date.
- `sample_submission.csv`: Template for forecast submission (28 days × 42,840 series).

---

## 📏 Evaluation Metric: WRMSSE

The competition was evaluated using the **Weighted Root Mean Squared Scaled Error (WRMSSE)**. This metric extends RMSSE by incorporating weights based on each series' revenue contribution, penalizing errors on high-revenue series more heavily.

### WRMSSE Formula

WRMSSE = ∑ (w_i * RMSSE_i)
Where:

w_i = weight of time series i based on its recent sales revenue

RMSSE_i = Root Mean Squared Scaled Error for time series i

WRMSSE is computed across 12 hierarchical aggregation levels, from total sales to individual item-store combinations.

🛠️ What Was Done
1. Data Preparation
Merged sales_train_evaluation, calendar, and sell_prices using item_id, store_id, and date.

Converted long-format sales data to wide format using pivot tables.

Handled missing values in price and calendar data with forward/backward fills and zero-imputation where appropriate.

2. Feature Engineering
Lag Features: Created rolling mean, rolling std, and lag values for 7, 14, 28, and 56-day windows.

Price Features: Calculated price momentum, price volatility, and normalized price levels.

Date Features: Extracted day, week, month, year, event name, SNAP flag, and weekend indicators.

Demand Trends: Included categorical interactions and shift-based features to capture demand shifts.

3. Modeling
Used LightGBM for gradient boosting with categorical encoding.

Built global models across all series instead of local models per time series.

Performed per-day prediction (multi-output regression treated as 28 single-output tasks).

Used hierarchical aggregation constraints by optimizing forecasts at bottom level and reconciling with upper levels post-prediction.

4. Validation Strategy
Employed rolling-window cross-validation using the last 28 days of the training data as validation.

Custom scoring function implemented to simulate WRMSSE locally before final submission.

5. Ensembling & Tuning
Blended multiple LightGBM models with different hyperparameters and feature sets.

Applied Bayesian Optimization for hyperparameter tuning.

Some top solutions included DeepAR, XGBoost, and simple exponential smoothing as part of model ensembles.

💻 Tools & Libraries
Python, Pandas, NumPy

LightGBM, XGBoost, CatBoost (optional)

Scikit-learn

Matplotlib, Seaborn

tqdm, joblib for processing and parallelization

Kaggle notebooks and Docker (for submission validation)

