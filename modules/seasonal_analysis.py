"""
Seasonal Analysis Module for ENERGYSCAPE.
Classifies energy consumption by Dry vs Wet seasons and computes seasonal indices.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

DEFAULT_DRY_MONTHS = ["January", "February", "March", "April", "May", "December"]
DEFAULT_WET_MONTHS = ["June", "July", "August", "September", "October", "November"]

def classify_season(month: str, dry_months: List[str] = DEFAULT_DRY_MONTHS) -> str:
    """Classify month into Dry or Wet season based on mapping."""
    return "Dry" if month in dry_months else "Wet"

def calculate_seasonal_metrics(df: pd.DataFrame, 
                               dry_months: List[str] = DEFAULT_DRY_MONTHS, 
                               wet_months: List[str] = DEFAULT_WET_MONTHS,
                               school_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate seasonal averages, differences, and indices for energy consumption.
    Accepts dataframe with 'month' and 'consumption_kwh' (or 'bill_php').
    """
    data = df.copy()
    if school_name and school_name != "Both":
        data = data[data['school'] == school_name]
        
    val_col = 'consumption_kwh' if 'consumption_kwh' in data.columns else 'bill_php'
    data = data.dropna(subset=[val_col])
    
    if data.empty:
        return {}
        
    data['season'] = data['month'].apply(lambda m: classify_season(m, dry_months))
    
    # Monthly averages
    monthly_avg = data.groupby('month')[val_col].mean()
    overall_avg = data[val_col].mean()
    
    # Seasonal averages
    seasonal_summary = data.groupby('season')[val_col].agg(['mean', 'count', 'std']).rename(
        columns={'mean': 'average', 'count': 'count', 'std': 'std_dev'}
    )
    
    dry_avg = float(seasonal_summary.loc['Dry', 'average']) if 'Dry' in seasonal_summary.index else 0.0
    wet_avg = float(seasonal_summary.loc['Wet', 'average']) if 'Wet' in seasonal_summary.index else 0.0
    
    seasonal_diff = dry_avg - wet_avg
    pct_diff = (seasonal_diff / wet_avg * 100.0) if wet_avg > 0 else 0.0
    
    # Seasonal Index: Monthly Avg / Overall Avg
    seasonal_indices = (monthly_avg / overall_avg).to_dict()
    
    return {
        "school": school_name if school_name else "All Selected",
        "val_col": val_col,
        "overall_avg": float(overall_avg),
        "dry_avg": dry_avg,
        "wet_avg": wet_avg,
        "seasonal_difference": float(seasonal_diff),
        "percentage_difference": float(pct_diff),
        "seasonal_summary": seasonal_summary,
        "monthly_averages": monthly_avg.to_dict(),
        "seasonal_indices": seasonal_indices
    }
