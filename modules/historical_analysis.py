"""
Historical Electricity Bill Analysis Module for ENERGYSCAPE.
Calculates statistical summaries and aggregate trends for school electricity bills.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def calculate_historical_metrics(df: pd.DataFrame, school_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive historical bill statistics for a specific school or combined dataset.
    """
    filtered_df = df.copy()
    if school_name and school_name != "Both":
        filtered_df = filtered_df[filtered_df['school'] == school_name]
        
    valid_bills = filtered_df['bill_php'].dropna()
    
    if valid_bills.empty:
        return {}
        
    total_bill = valid_bills.sum()
    avg_bill = valid_bills.mean()
    median_bill = valid_bills.median()
    std_bill = valid_bills.std()
    min_bill = valid_bills.min()
    max_bill = valid_bills.max()
    bill_range = max_bill - min_bill
    
    # Yearly summary
    yearly_summary = filtered_df.groupby('school_year')['bill_php'].agg(['sum', 'mean', 'count']).rename(
        columns={'sum': 'total_bill', 'mean': 'average_bill', 'count': 'months_recorded'}
    )
    
    # Monthly averages
    month_order = ['June', 'July', 'August', 'September', 'October', 'November', 'December', 'January', 'February', 'March']
    filtered_df['month_cat'] = pd.Categorical(filtered_df['month'], categories=month_order, ordered=True)
    monthly_summary = filtered_df.groupby('month_cat', observed=False)['bill_php'].agg(['mean', 'min', 'max']).rename(
        columns={'mean': 'average_bill'}
    )
    
    highest_month = monthly_summary['average_bill'].idxmax() if not monthly_summary['average_bill'].isna().all() else None
    lowest_month = monthly_summary['average_bill'].idxmin() if not monthly_summary['average_bill'].isna().all() else None
    
    return {
        "school": school_name if school_name else "All Selected",
        "total_bill": float(total_bill),
        "avg_bill": float(avg_bill),
        "median_bill": float(median_bill),
        "std_bill": float(std_bill) if not np.isnan(std_bill) else 0.0,
        "min_bill": float(min_bill),
        "max_bill": float(max_bill),
        "range_bill": float(bill_range),
        "count": len(valid_bills),
        "yearly_summary": yearly_summary,
        "monthly_summary": monthly_summary,
        "highest_month": str(highest_month),
        "lowest_month": str(lowest_month)
    }
