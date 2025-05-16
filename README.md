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

```math
WRMSSE = ∑ (w_i * RMSSE_i)

Where:

w_i = weight of time series i based on its recent sales revenue

RMSSE_i = Root Mean Squared Scaled Error for time series i

WRMSSE is computed across 12 hierarchical aggregation levels, from total sales to individual item-store combinations.
