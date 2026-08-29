"""
Data Processing and Validation Module for ENERGYSCAPE.
Handles loading, schema validation, data cleaning, and preprocessing for:
- Historical electricity bills (in ₱)
- Appliance electrical loads (in Watts, Hours, Days)
- Seasonal consumption records (in kWh)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"

def load_historical_bills(file_path_or_buffer: Optional[Any] = None) -> pd.DataFrame:
    """
    Load and preprocess historical electricity billing data.
    Ensures missing/TBF entries are handled as NaN without dropping date structures.
    """
    if file_path_or_buffer is None:
        file_path_or_buffer = DEFAULT_DATA_DIR / "historical_bills.csv"
    
    df = pd.read_csv(file_path_or_buffer)
    
    required_cols = {"school", "date", "school_year", "month", "bill_php"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Historical bills CSV missing required columns: {required_cols - set(df.columns)}")
    
    df['bill_php'] = pd.to_numeric(df['bill_php'], errors='coerce')
    df['date_dt'] = pd.to_datetime(df['date'], format='%Y-%m', errors='coerce')
    df = df.sort_values(by=['school', 'date_dt']).reset_index(drop=True)
    return df

def load_appliance_loads(file_path_or_buffer: Optional[Any] = None) -> pd.DataFrame:
    """
    Load and preprocess appliance electrical load data.
    """
    if file_path_or_buffer is None:
        file_path_or_buffer = DEFAULT_DATA_DIR / "appliance_loads.csv"
        
    df = pd.read_csv(file_path_or_buffer)
    
    required_cols = {"school", "appliance", "quantity", "power_watts", "hours_per_day", "operating_days"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Appliance loads CSV missing required columns: {required_cols - set(df.columns)}")
        
    numeric_cols = ["quantity", "power_watts", "hours_per_day", "operating_days"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

def load_seasonal_data(file_path_or_buffer: Optional[Any] = None) -> pd.DataFrame:
    """
    Load and preprocess seasonal consumption dataset.
    """
    if file_path_or_buffer is None:
        file_path_or_buffer = DEFAULT_DATA_DIR / "seasonal_data.csv"
        
    df = pd.read_csv(file_path_or_buffer)
    
    required_cols = {"school", "month", "season", "consumption_kwh"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Seasonal data CSV missing required columns: {required_cols - set(df.columns)}")
        
    df['consumption_kwh'] = pd.to_numeric(df['consumption_kwh'], errors='coerce')
    return df

def validate_dataset(df: pd.DataFrame, dataset_type: str) -> Dict[str, Any]:
    """
    Generate diagnostic health report for uploaded or loaded datasets.
    Checks missing values, invalid values, duplicates, and TBF records.
    """
    report = {
        "status": "PASS",
        "total_rows": len(df),
        "missing_records": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "messages": [],
        "warnings": []
    }
    
    if dataset_type == "historical":
        missing_bills = int(df['bill_php'].isna().sum())
        report["tbf_missing_count"] = missing_bills
        if missing_bills > 0:
            report["warnings"].append(f"Found {missing_bills} unavailable (TBF) historical bill entries. These are correctly preserved as missing/NaN.")
        negative_bills = int((df['bill_php'] < 0).sum())
        if negative_bills > 0:
            report["status"] = "FAIL"
            report["messages"].append(f"Found {negative_bills} invalid negative bill amounts.")
            
    elif dataset_type == "appliance":
        invalid_qty = int((df['quantity'] <= 0).sum())
        invalid_power = int((df['power_watts'] <= 0).sum())
        invalid_hours = int(((df['hours_per_day'] <= 0) | (df['hours_per_day'] > 24)).sum())
        invalid_days = int(((df['operating_days'] <= 0) | (df['operating_days'] > 31)).sum())
        
        if invalid_qty or invalid_power or invalid_hours or invalid_days:
            report["status"] = "FAIL"
            report["messages"].append(f"Invalid parameters detected: qty ({invalid_qty}), power ({invalid_power}), hours ({invalid_hours}), days ({invalid_days}).")
            
    elif dataset_type == "seasonal":
        invalid_kwh = int((df['consumption_kwh'] < 0).sum())
        if invalid_kwh > 0:
            report["status"] = "FAIL"
            report["messages"].append(f"Found {invalid_kwh} invalid negative consumption entries.")
            
    return report
