"""
Data-Driven Recommendation Engine for ENERGYSCAPE.
Generates explainable, evidence-based energy management priorities based on calculated model outputs.
"""

import pandas as pd
from typing import List, Dict, Any

APPLIANCE_ACTION_MAP = {
    "Air Conditioner": "Reduce unnecessary operating hours; improve scheduling; prevent simultaneous high-load operation",
    "Computers": "Turn off when not in use; implement automatic sleep modes; optimize administrative schedules",
    "Lighting Fixtures": "Turn off unused lights; maximize daylighting; replace with energy-efficient LEDs",
    "Electric Fans": "Schedule fan operation according to occupancy; turn off during vacant periods",
    "Refrigerators": "Maintain efficient operating conditions; verify door seals and thermostat settings",
    "Projectors": "Avoid unnecessary standby operation; turn off immediately after lectures/presentations",
    "Printers": "Turn off printers when not in active use; consolidate print tasks"
}

def generate_appliance_recommendations(appliance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate data-driven energy management priorities ranked by monthly load contribution.
    """
    if appliance_df.empty:
        return pd.DataFrame()
        
    sorted_df = appliance_df.sort_values(by='monthly_kwh', ascending=False).reset_index(drop=True)
    
    rows = []
    for idx, row in sorted_df.iterrows():
        app_name = row['appliance']
        kwh = row['monthly_kwh']
        action = APPLIANCE_ACTION_MAP.get(app_name, "Optimize operating hours and usage schedule")
        
        rows.append({
            "Priority": idx + 1,
            "Target Load": app_name,
            "Recommended Management Action": action,
            "Basis (Monthly Load)": f"{kwh:,.2f} kWh/month ({row['percentage_share']:.2f}%)"
        })
        
    return pd.DataFrame(rows)

def generate_executive_summary_recommendation(top_appliance: str, 
                                               top_share: float, 
                                               optimized_target_kwh: float = 0.0,
                                               dry_vs_wet_diff_pct: float = 0.0,
                                               annual_kwh_savings: float = None,
                                               annual_cost_savings_php: float = None,
                                               annual_avoided_co2_kg: float = None) -> str:
    """
    Generate executive-level analytical summary statement based on calculated outputs.
    """
    savings_parts = []
    if annual_kwh_savings is not None:
        savings_parts.append(f"<strong>{annual_kwh_savings:,.2f} kWh</strong>")
    if annual_cost_savings_php is not None:
        savings_parts.append(f"<strong>₱{annual_cost_savings_php:,.2f}</strong>")
    if annual_avoided_co2_kg is not None:
        savings_parts.append(f"avoiding <strong>{annual_avoided_co2_kg:,.2f} kg CO₂e</strong>")

    if savings_parts:
        savings_sentence = f"delivering an estimated annual reduction of {', '.join(savings_parts)}."
    else:
        savings_sentence = "with annual savings calculated from the active scenario results."

    summary = (
        f"Analytical results identify <strong>{top_appliance}</strong> as the primary energy load, contributing <strong>{top_share:.2f}%</strong> "
        f"of total estimated appliance consumption. "
        f"Seasonal analysis shows dry season consumption is <strong>{dry_vs_wet_diff_pct:.2f}% higher</strong>, indicating peak demand during warmer months. "
        f"The evaluated 15% Moderate Conservation Scenario establishes an optimized target of <strong>{optimized_target_kwh:,.2f} kWh/month</strong>, "
        f"{savings_sentence}"
    )
    return summary
