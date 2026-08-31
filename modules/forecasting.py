"""
ETS Forecasting Module for ENERGYSCAPE.
Implements Exponential Smoothing (ETS) using statsmodels to generate point forecasts 
and forecast bounds for monthly electricity bills.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Dict, Any, Optional, Tuple

from modules.validation import calculate_mape, calculate_rmse, calculate_mae

def fit_ets_forecast(df: pd.DataFrame, 
                     school_name: str, 
                     forecast_horizon: int = 12,
                     holdout_steps: int = 6) -> Dict[str, Any]:
    """
    Fit Exponential Smoothing (ETS) model to historical electricity bills.
    Performs train/validation split for backtesting, then forecasts future steps.
    """
    school_df = df[df['school'] == school_name].copy()
    school_df = school_df.dropna(subset=['bill_php']).sort_values(by='date_dt').reset_index(drop=True)
    
    series = school_df['bill_php'].values
    dates = school_df['date_dt'].values
    
    if len(series) < 12:
        # Too short for complex seasonal decomposition, use simple exponential smoothing
        model = ExponentialSmoothing(series, trend='add', seasonal=None)
        fitted_model = model.fit()
        forecast_vals = fitted_model.forecast(forecast_horizon)
        
        # Validation on last holdout_steps
        train_series = series[:-holdout_steps] if len(series) > holdout_steps else series
        val_series = series[-holdout_steps:] if len(series) > holdout_steps else series
        val_fitted = ExponentialSmoothing(train_series, trend='add', seasonal=None).fit()
        val_preds = val_fitted.forecast(len(val_series))
    else:
        # Try multiplicative/additive seasonal ETS model with seasonal_periods=10 or 12
        # Note: Academic school year dataset has 10 months per SY (Jun-Mar) or 12 calendar months.
        try:
            model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=10)
            fitted_model = model.fit()
        except Exception:
            model = ExponentialSmoothing(series, trend='add', seasonal=None)
            fitted_model = model.fit()
            
        forecast_vals = fitted_model.forecast(forecast_horizon)
        
        # Backtest validation split
        if len(series) > holdout_steps:
            train_series = series[:-holdout_steps]
            val_series = series[-holdout_steps:]
            try:
                val_model = ExponentialSmoothing(train_series, trend='add', seasonal='add', seasonal_periods=10)
                val_fitted = val_model.fit()
            except Exception:
                val_model = ExponentialSmoothing(train_series, trend='add', seasonal=None)
                val_fitted = val_model.fit()
            val_preds = val_fitted.forecast(len(val_series))
        else:
            val_series = series
            val_preds = fitted_model.fittedvalues
            
    # Calculate std residual for simple confidence bounds
    residuals = fitted_model.resid
    std_resid = np.std(residuals) if len(residuals) > 0 else 0.05 * np.mean(series)
    
    # Generate future dates
    last_date = pd.to_datetime(dates[-1])
    future_dates = []
    curr_date = last_date
    for i in range(forecast_horizon):
        # Move to next month
        curr_date = curr_date + pd.DateOffset(months=1)
        future_dates.append(curr_date)
        
    lower_bound = np.maximum(0, forecast_vals - 1.96 * std_resid)
    upper_bound = forecast_vals + 1.96 * std_resid
    
    forecast_df = pd.DataFrame({
        "school": school_name,
        "date_dt": future_dates,
        "date_str": [d.strftime("%Y-%m") for d in future_dates],
        "month": [d.strftime("%B") for d in future_dates],
        "forecast_bill": forecast_vals,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    })
    
    val_mape_val = calculate_mape(val_series, val_preds)
    val_rmse_val = calculate_rmse(val_series, val_preds)
    val_mae_val = calculate_mae(val_series, val_preds)
    
    return {
        "school": school_name,
        "fitted_model": fitted_model,
        "historical_series": series,
        "historical_dates": dates,
        "val_actuals": val_series,
        "val_predictions": val_preds,
        "val_mape": val_mape_val,
        "val_rmse": val_rmse_val,
        "val_mae": val_mae_val,
        "forecast_df": forecast_df
    }
