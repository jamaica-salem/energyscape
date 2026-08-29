"""
Electrical Load Quantification Module for ENERGYSCAPE.
Calculates appliance-level kWh consumption, financial costs, load shares, and rankings.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

DEFAULT_ELECTRICITY_RATE = 11.00  # ₱/kWh

def calculate_appliance_loads(df: pd.DataFrame, 
                              electricity_rate: float = DEFAULT_ELECTRICITY_RATE,
                              school_name: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate monthly energy consumption (kWh), monthly cost (₱), percentage share, 
    and priority rank for appliances in the dataset.
    """
    data = df.copy()
    if school_name and school_name != "Both":
        data = data[data['school'] == school_name]
        
    if data.empty:
        return pd.DataFrame()
        
    # Formula: (Power * Qty * Hours * Days) / 1000
    data['monthly_kwh'] = (data['power_watts'] * data['quantity'] * data['hours_per_day'] * data['operating_days']) / 1000.0
    data['monthly_cost_php'] = data['monthly_kwh'] * electricity_rate
    
    total_kwh = data['monthly_kwh'].sum()
    data['percentage_share'] = (data['monthly_kwh'] / total_kwh * 100.0) if total_kwh > 0 else 0.0
    
    # Sort by monthly_kwh descending
    data = data.sort_values(by='monthly_kwh', ascending=False).reset_index(drop=True)
    data['rank'] = data.index + 1
    
    # Priority assignment based on share and rank
    def assign_priority(row):
        if row['rank'] <= 2 or row['percentage_share'] >= 20.0:
            return "Very High"
        elif row['percentage_share'] >= 10.0:
            return "High"
        elif row['percentage_share'] >= 5.0:
            return "Moderate"
        else:
            return "Low"
            
    data['priority'] = data.apply(assign_priority, axis=1)
    return data

def get_load_summary(appliance_df: pd.DataFrame, electricity_rate: float = DEFAULT_ELECTRICITY_RATE) -> Dict[str, Any]:
    """
    Summarize total electrical load, top loads, and concentration share.
    """
    if appliance_df.empty:
        return {}
        
    total_kwh = float(appliance_df['monthly_kwh'].sum())
    total_cost = float(appliance_df['monthly_cost_php'].sum())
    total_quantity = int(appliance_df['quantity'].sum())
    
    top_appliance = appliance_df.iloc[0]['appliance']
    top_kwh = float(appliance_df.iloc[0]['monthly_kwh'])
    top_share = float(appliance_df.iloc[0]['percentage_share'])
    
    second_appliance = appliance_df.iloc[1]['appliance'] if len(appliance_df) > 1 else None
    second_kwh = float(appliance_df.iloc[1]['monthly_kwh']) if len(appliance_df) > 1 else 0.0
    second_share = float(appliance_df.iloc[1]['percentage_share']) if len(appliance_df) > 1 else 0.0
    
    top2_share = top_share + second_share
    
    return {
        "total_kwh": total_kwh,
        "annual_kwh": total_kwh * 12.0,
        "total_cost_php": total_cost,
        "annual_cost_php": total_cost * 12.0,
        "total_quantity": total_quantity,
        "top_appliance": top_appliance,
        "top_kwh": top_kwh,
        "top_share": top_share,
        "second_appliance": second_appliance,
        "second_kwh": second_kwh,
        "second_share": second_share,
        "top2_combined_share": top2_share,
        "electricity_rate": electricity_rate
    }
