"""
Carbon Emission Model Module for ENERGYSCAPE.
Calculates carbon footprint (CO2e kg) from electricity consumption (kWh).
"""

import pandas as pd
from typing import Dict, Any, Optional

DEFAULT_EMISSION_FACTOR = 0.70  # kg CO2e / kWh

def calculate_carbon_emissions(kwh_val: float, emission_factor: float = DEFAULT_EMISSION_FACTOR) -> float:
    """
    Calculate carbon dioxide equivalent emissions (kg CO2e) for a given kWh consumption.
    Formula: CO2e = kWh * Emission Factor
    """
    if kwh_val is None or kwh_val < 0:
        return 0.0
    return float(kwh_val * emission_factor)

def calculate_carbon_summary(monthly_kwh: float, emission_factor: float = DEFAULT_EMISSION_FACTOR) -> Dict[str, Any]:
    """
    Calculate monthly and annual carbon emissions and avoided emissions.
    """
    monthly_co2 = calculate_carbon_emissions(monthly_kwh, emission_factor)
    annual_kwh = monthly_kwh * 12.0
    annual_co2 = calculate_carbon_emissions(annual_kwh, emission_factor)
    
    return {
        "monthly_kwh": monthly_kwh,
        "annual_kwh": annual_kwh,
        "emission_factor": emission_factor,
        "monthly_co2_kg": monthly_co2,
        "annual_co2_kg": annual_co2,
        "monthly_co2_tons": monthly_co2 / 1000.0,
        "annual_co2_tons": annual_co2 / 1000.0
    }
