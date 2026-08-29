"""
Utility formatting functions for ENERGYSCAPE application.
Provides consistent display formatting for currency, energy, emissions, and percentages.
"""

def format_currency(val: float, precision: int = 2) -> str:
    """Format numeric value as Philippine Peso currency."""
    if val is None or (isinstance(val, float) and float('nan') == val):
        return "N/A"
    return f"₱{val:,.{precision}f}"

def format_kwh(val: float, precision: int = 2) -> str:
    """Format numeric value as energy in kWh."""
    if val is None or (isinstance(val, float) and float('nan') == val):
        return "N/A"
    return f"{val:,.{precision}f} kWh"

def format_co2(val: float, precision: int = 2) -> str:
    """Format numeric value as carbon emissions in kg CO2e."""
    if val is None or (isinstance(val, float) and float('nan') == val):
        return "N/A"
    return f"{val:,.{precision}f} kg CO₂e"

def format_pct(val: float, precision: int = 2) -> str:
    """Format numeric value as percentage."""
    if val is None or (isinstance(val, float) and float('nan') == val):
        return "N/A"
    return f"{val:.{precision}f}%"

def format_num(val: float, precision: int = 2) -> str:
    """Format general numbers with comma separators."""
    if val is None or (isinstance(val, float) and float('nan') == val):
        return "N/A"
    return f"{val:,.{precision}f}"
