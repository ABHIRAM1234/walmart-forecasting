# main.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import gc
from google.cloud import storage, bigquery
import functions_framework

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_TABLE = os.environ.get("BQ_TABLE")

MODEL_PATH_GCS = "model_artifacts/m5_stable_model.txt"
RAW_DATA_PATH = f"gs://{GCS_BUCKET_NAME}/data"
LOCAL_MODEL_PATH = "/tmp/m5_stable_model.txt"

# This decorator tells the Functions Framework that this is the function to run.
@functions_framework.cloud_event
def run_batch_forecast(cloud_event):
    """
    The main entry point for the Cloud Function/Run service.
    Triggered by a Pub/Sub message.
    """
    print(f"--- Batch Forecast Started for Project: {PROJECT_ID} ---")
    
    # --- 1. Download Model from GCS ---
    print(f"Downloading model from gs://{GCS_BUCKET_NAME}/{MODEL_PATH_GCS}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(MODEL_PATH_GCS)
    blob.download_to_filename(LOCAL_MODEL_PATH)
    print("Model downloaded successfully.")
    
    model = lgb.Booster(model_file=LOCAL_MODEL_PATH)
    features = model.feature_name()

    # --- 2. Load and Prepare Data ---
    print("Loading supporting data...")
    sales_train = pd.read_csv(os.path.join(RAW_DATA_PATH, 'sales_train_validation.csv'))
    calendar = pd.read_csv(os.path.join(RAW_DATA_PATH, 'calendar.csv'))
    prices = pd.read_csv(os.path.join(RAW_DATA_PATH, 'sell_prices.csv'))
    calendar['d'] = calendar['d'].str.extract(r'(\d+)').astype('int16')
    
    print("Preparing historical data for feature engineering...")
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    start_day_for_history = 1914 - 100
    day_cols = [f'd_{i}' for i in range(start_day_for_history, 1914)]
    data = pd.melt(sales_train, id_vars=id_cols, value_vars=day_cols, var_name='d', value_name='sales')
    data['d'] = data['d'].str.extract(r'(\d+)').astype('int16')

    # --- 3. Recursive Forecasting ---
    print("Starting recursive forecast for the next 28 days...")
    all_predictions = []
    for day_to_predict in range(1914, 1942):
        print(f"  > Predicting for day: {day_to_predict}")
        
        pred_template = data[id_cols].drop_duplicates()
        pred_template['d'] = day_to_predict
        
        pred_template = pd.merge(pred_template, calendar, on='d', how='left')
        pred_template = pd.merge(pred_template, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        
        for lag in [28, 35, 42, 49, 56]:
            lag_data = data[data['d'] == day_to_predict - lag][['id', 'sales']].rename(columns={'sales': f'sales_lag_{lag}'})
            pred_template = pd.merge(pred_template, lag_data, on='id', how='left')
        for window in [7, 14, 28]:
            start, end = day_to_predict - 28 - window + 1, day_to_predict - 28
            roll_data = data[(data['d'] >= start) & (data['d'] <= end)]
            roll_mean = roll_data.groupby('id')['sales'].mean().reset_index().rename(columns={'sales': f'sales_rolling_mean_{window}'})
            roll_std = roll_data.groupby('id')['sales'].std().reset_index().rename(columns={'sales': f'sales_rolling_std_{window}'})
            pred_template = pd.merge(pred_template, roll_mean, on='id', how='left')
            pred_template = pd.merge(pred_template, roll_std, on='id', how='left')
        
        X_pred = pred_template[features]
        for col in X_pred.select_dtypes(include=['category', 'object']).columns:
            X_pred[col] = X_pred[col].astype('category').cat.codes.astype('int16')
        for col in X_pred.select_dtypes(include=['float']).columns:
            X_pred[col] = X_pred[col].fillna(-1).astype('float32')
        
        predictions = model.predict(X_pred)
        predictions[predictions < 0] = 0
        pred_template['forecast_sales'] = predictions
        all_predictions.append(pred_template[['id', 'd', 'forecast_sales']])
        
        pred_template['sales'] = predictions
        data = pd.concat([data, pred_template[['id', 'sales', 'd']]], ignore_index=True)

    # --- 4. Save to BigQuery ---
    print("Forecast complete. Saving results to BigQuery...")
    final_output = pd.concat(all_predictions, ignore_index=True)
    final_output['forecast_timestamp'] = pd.Timestamp.now(tz='UTC')

    bq_client = bigquery.Client()
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
    
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = bq_client.load_table_from_dataframe(final_output, table_id, job_config=job_config)
    job.result()
    
    print(f"Successfully loaded {len(final_output)} rows into {table_id}.")
    print("--- Batch Forecast Finished Successfully ---")
    
    return "OK"