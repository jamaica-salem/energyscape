"""
Business-as-Usual (BAU) Baseline and Conservation Scenarios Module for ENERGYSCAPE.
Simulates 5%, 10%, and 15% energy reduction scenarios against the BAU baseline.
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from modules.carbon import calculate_carbon_emissions

DEFAULT_SCENARIOS = [0.05, 0.10, 0.15]

def calculate_bau_baseline(bau_monthly_kwh: float = 2289.10, 
                           electricity_rate: float = 11.00,
                           emission_factor: float = 0.70) -> Dict[str, Any]:
    """
    Compute Business-as-Usual (BAU) baseline metrics.
    """
    bau_annual_kwh = bau_monthly_kwh * 12.0
    bau_monthly_cost = bau_monthly_kwh * electricity_rate
    bau_annual_cost = bau_annual_kwh * electricity_rate
    bau_monthly_co2 = calculate_carbon_emissions(bau_monthly_kwh, emission_factor)
    bau_annual_co2 = calculate_carbon_emissions(bau_annual_kwh, emission_factor)
    
    return {
        "monthly_kwh": bau_monthly_kwh,
        "annual_kwh": bau_annual_kwh,
        "electricity_rate": electricity_rate,
        "emission_factor": emission_factor,
        "monthly_cost_php": bau_monthly_cost,
        "annual_cost_php": bau_annual_cost,
        "monthly_co2_kg": bau_monthly_co2,
        "annual_co2_kg": bau_annual_co2
    }

def simulate_conservation_scenarios(bau_baseline: Dict[str, Any], 
                                     reduction_rates: List[float] = DEFAULT_SCENARIOS) -> pd.DataFrame:
    """
    Simulate mathematical conservation scenarios (5%, 10%, 15%).
    Calculates projected consumption, energy saved, cost savings, and avoided CO2e.
    """
    bau_kwh = bau_baseline["monthly_kwh"]
    rate = bau_baseline["electricity_rate"]
    ef = bau_baseline["emission_factor"]
    
    rows = []
    # Include BAU row first for baseline comparison
    rows.append({
        "Scenario": "BAU Baseline",
        "Reduction %": 0.0,
        "Reduction Rate": 0.0,
        "Projected Monthly kWh": bau_kwh,
        "Monthly Energy Saved (kWh)": 0.0,
        "Projected Monthly Cost (₱)": bau_kwh * rate,
        "Monthly Cost Savings (₱)": 0.0,
        "Projected Annual kWh": bau_kwh * 12.0,
        "Annual Energy Saved (kWh)": 0.0,
        "Projected Annual Cost (₱)": bau_kwh * 12.0 * rate,
        "Annual Cost Savings (₱)": 0.0,
        "Projected Monthly CO₂e (kg)": calculate_carbon_emissions(bau_kwh, ef),
        "Monthly Avoided CO₂e (kg)": 0.0,
        "Projected Annual CO₂e (kg)": calculate_carbon_emissions(bau_kwh * 12.0, ef),
        "Annual Avoided CO₂e (kg)": 0.0
    })
    
    for r in reduction_rates:
        pct_label = f"{int(r * 100)}% Reduction"
        proj_monthly_kwh = bau_kwh * (1.0 - r)
        monthly_saved_kwh = bau_kwh * r
        proj_monthly_cost = proj_monthly_kwh * rate
        monthly_saved_cost = monthly_saved_kwh * rate
        
        proj_annual_kwh = proj_monthly_kwh * 12.0
        annual_saved_kwh = monthly_saved_kwh * 12.0
        proj_annual_cost = proj_annual_kwh * rate
        annual_saved_cost = annual_saved_kwh * rate
        
        proj_monthly_co2 = calculate_carbon_emissions(proj_monthly_kwh, ef)
        monthly_avoided_co2 = calculate_carbon_emissions(monthly_saved_kwh, ef)
        proj_annual_co2 = calculate_carbon_emissions(proj_annual_kwh, ef)
        annual_avoided_co2 = calculate_carbon_emissions(annual_saved_kwh, ef)
        
        rows.append({
            "Scenario": pct_label,
            "Reduction %": r * 100.0,
            "Reduction Rate": r,
            "Projected Monthly kWh": proj_monthly_kwh,
            "Monthly Energy Saved (kWh)": monthly_saved_kwh,
            "Projected Monthly Cost (₱)": proj_monthly_cost,
            "Monthly Cost Savings (₱)": monthly_saved_cost,
            "Projected Annual kWh": proj_annual_kwh,
            "Annual Energy Saved (kWh)": annual_saved_kwh,
            "Projected Annual Cost (₱)": proj_annual_cost,
            "Annual Cost Savings (₱)": annual_saved_cost,
            "Projected Monthly CO₂e (kg)": proj_monthly_co2,
            "Monthly Avoided CO₂e (kg)": monthly_avoided_co2,
            "Projected Annual CO₂e (kg)": proj_annual_co2,
            "Annual Avoided CO₂e (kg)": annual_avoided_co2
        })
        
    return pd.DataFrame(rows)
