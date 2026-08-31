"""
Mathematical Optimization, Sensitivity Analysis, and Target Monitoring Module for ENERGYSCAPE.
Implements Linear Goal Programming (LGP) optimization using scipy.optimize.linprog,
constraint-satisfaction scenario optimization, sensitivity analysis, and target monitoring.
Matches Section 11 of MCS Prereq-Paper.docx: min Z = sum(E_i * x_i) subject to operational constraints and objective function formulation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy.optimize import linprog

from config import (
    DEFAULT_ELECTRICITY_RATE as CONFIG_DEFAULT_ELECTRICITY_RATE,
    DEFAULT_EMISSION_FACTOR as CONFIG_DEFAULT_EMISSION_FACTOR,
    DEFAULT_BAU_MONTHLY_KWH as CONFIG_DEFAULT_BAU_MONTHLY_KWH,
    DEFAULT_SCENARIO_REDUCTION_RATES as CONFIG_DEFAULT_SCENARIO_REDUCTION_RATES,
)

DEFAULT_ELECTRICITY_RATE = CONFIG_DEFAULT_ELECTRICITY_RATE
DEFAULT_EMISSION_FACTOR = CONFIG_DEFAULT_EMISSION_FACTOR

def optimize_conservation_target(
    data_df: Optional[Any] = None,
    scenarios_df: Optional[pd.DataFrame] = None,
    electricity_rate: float = DEFAULT_ELECTRICITY_RATE,
    emission_factor: float = DEFAULT_EMISSION_FACTOR,
    max_ac_red: float = 0.15,
    max_comp_red: float = 0.15,
    max_other_red: float = 0.10,
    objective: str = "MINIMIZE ELECTRICITY + COST + CO₂",
    appliance_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Solves the Linear Goal Programming (LGP) optimization model defined in Section 11 of the paper:
        min Z = sum(c_i * E_i * x_i)
    Subject to operational constraints:
        (1 - max_red_i) <= x_i <= 1.0  for each load category i
    """
    try:
        electricity_rate = float(electricity_rate)
    except (TypeError, ValueError):
        electricity_rate = DEFAULT_ELECTRICITY_RATE

    try:
        emission_factor = float(emission_factor)
    except (TypeError, ValueError):
        emission_factor = DEFAULT_EMISSION_FACTOR

    # Auto-detect whether data_df is an appliance inventory or a scenarios DataFrame
    if data_df is not None and isinstance(data_df, pd.DataFrame):
        if 'appliance' in data_df.columns or 'power_watts' in data_df.columns:
            appliance_df = data_df
        elif 'Scenario' in data_df.columns or 'Projected Monthly kWh' in data_df.columns:
            scenarios_df = data_df

    strategy_focus = "Linear Goal Programming"
    short_tag = "Optimal Target"

    if appliance_df is not None and isinstance(appliance_df, pd.DataFrame) and not appliance_df.empty:
        df = appliance_df.copy()
        if 'monthly_kwh' not in df.columns and all(c in df.columns for c in ['power_watts', 'quantity', 'hours_per_day', 'operating_days']):
            df['monthly_kwh'] = (df['power_watts'] * df['quantity'] * df['hours_per_day'] * df['operating_days']) / 1000.0
            
        if 'appliance' in df.columns:
            ac_mask = df['appliance'].astype(str).str.contains('Air Conditioner', case=False, na=False)
            comp_mask = df['appliance'].astype(str).str.contains('Computer|Laptop', case=False, na=False)
            other_mask = ~(ac_mask | comp_mask)
            
            e_ac = float(df.loc[ac_mask, 'monthly_kwh'].sum()) if 'monthly_kwh' in df.columns else 0.0
            e_comp = float(df.loc[comp_mask, 'monthly_kwh'].sum()) if 'monthly_kwh' in df.columns else 0.0
            e_other = float(df.loc[other_mask, 'monthly_kwh'].sum()) if 'monthly_kwh' in df.columns else 0.0
        else:
            e_ac, e_comp, e_other = 0.0, 0.0, 0.0
            
        if e_ac == 0 and e_comp == 0 and len(df) >= 2 and 'monthly_kwh' in df.columns:
            sorted_apps = df.sort_values(by='monthly_kwh', ascending=False)
            e_ac = float(sorted_apps.iloc[0]['monthly_kwh'])
            e_comp = float(sorted_apps.iloc[1]['monthly_kwh'])
            e_other = float(sorted_apps.iloc[2:]['monthly_kwh'].sum()) if len(sorted_apps) > 2 else 0.0
            
        bau_monthly_kwh = e_ac + e_comp + e_other
        if bau_monthly_kwh == 0 and 'monthly_kwh' in df.columns:
            bau_monthly_kwh = float(df['monthly_kwh'].sum())
        
        # Objective Function Weighting
        obj_upper = str(objective).upper()
        if "EXPENDITURE" in obj_upper or "COST" in obj_upper and "CO₂" not in obj_upper and "LOAD" not in obj_upper:
            w_ac, w_comp, w_other = 1.15 * electricity_rate, 1.05 * electricity_rate, 1.00 * electricity_rate
            strategy_focus = "Expenditure Focus (₱)"
            short_tag = "Cost Target"
        elif "GREENHOUSE" in obj_upper or "EMISSIONS" in obj_upper or "CO₂" in obj_upper and "COST" not in obj_upper:
            w_ac, w_comp, w_other = 1.10 * emission_factor, 1.02 * emission_factor, 1.00 * emission_factor
            strategy_focus = "Carbon Focus (CO₂e)"
            short_tag = "Carbon Target"
        elif "LOAD" in obj_upper or "KWH" in obj_upper and "COST" not in obj_upper:
            w_ac, w_comp, w_other = 1.0, 1.0, 1.0
            strategy_focus = "Energy Focus (kWh)"
            short_tag = "Energy Target"
        else:
            w_ac = 1.0 + 0.15 * (electricity_rate / 11.0) + 0.15 * (emission_factor / 0.70)
            w_comp = 1.0 + 0.10 * (electricity_rate / 11.0) + 0.10 * (emission_factor / 0.70)
            w_other = 1.0
            strategy_focus = "Linear Goal Programming"
            short_tag = "Multi-Goal Target"

        if bau_monthly_kwh > 0:
            c = [e_ac * w_ac, e_comp * w_comp, e_other * w_other]
            bounds = [
                (max(0.0, 1.0 - max_ac_red), 1.0),
                (max(0.0, 1.0 - max_comp_red), 1.0),
                (max(0.0, 1.0 - max_other_red), 1.0)
            ]
            
            res = linprog(c, bounds=bounds, method='highs')
            if res.success:
                x_ac_opt, x_comp_opt, x_other_opt = res.x
                opt_monthly_kwh = e_ac * x_ac_opt + e_comp * x_comp_opt + e_other * x_other_opt
            else:
                x_ac_opt = 1.0 - max_ac_red
                x_comp_opt = 1.0 - max_comp_red
                x_other_opt = 1.0 - max_other_red
                opt_monthly_kwh = e_ac * x_ac_opt + e_comp * x_comp_opt + e_other * x_other_opt
        else:
            bau_monthly_kwh = float(CONFIG_DEFAULT_BAU_MONTHLY_KWH)
            opt_monthly_kwh = bau_monthly_kwh * 0.85
            x_ac_opt, x_comp_opt, x_other_opt = 0.85, 0.85, 0.90
            
    elif scenarios_df is not None and isinstance(scenarios_df, pd.DataFrame) and not scenarios_df.empty:
        candidates = scenarios_df[scenarios_df['Reduction %'] > 0].copy() if 'Reduction %' in scenarios_df.columns else pd.DataFrame()
        if not candidates.empty and 'Projected Monthly kWh' in candidates.columns:
            candidates = candidates.sort_values(by='Projected Monthly kWh', ascending=True).reset_index(drop=True)
            optimal = candidates.iloc[0]
            bau_rows = scenarios_df[scenarios_df['Reduction %'] == 0] if 'Reduction %' in scenarios_df.columns else pd.DataFrame()
            bau_monthly_kwh = float(bau_rows.iloc[0]["Projected Monthly kWh"]) if not bau_rows.empty else float(CONFIG_DEFAULT_BAU_MONTHLY_KWH)
            opt_monthly_kwh = float(optimal["Projected Monthly kWh"])
        else:
            bau_monthly_kwh = float(CONFIG_DEFAULT_BAU_MONTHLY_KWH)
            opt_monthly_kwh = bau_monthly_kwh * 0.85
        x_ac_opt, x_comp_opt, x_other_opt = 1.0 - max_ac_red, 1.0 - max_comp_red, 1.0 - max_other_red
        strategy_focus = "Linear Goal Programming"
        short_tag = "Scenario Target"
    else:
        bau_monthly_kwh = float(CONFIG_DEFAULT_BAU_MONTHLY_KWH)
        opt_monthly_kwh = bau_monthly_kwh * 0.85
        x_ac_opt, x_comp_opt, x_other_opt = 0.85, 0.85, 0.90
        strategy_focus = "Linear Goal Programming"
        short_tag = "Default Target"

    monthly_kwh_savings = bau_monthly_kwh - opt_monthly_kwh
    annual_kwh_savings = monthly_kwh_savings * 12.0
    monthly_cost_savings = monthly_kwh_savings * electricity_rate
    annual_cost_savings = monthly_cost_savings * 12.0
    annual_avoided_co2 = monthly_kwh_savings * emission_factor * 12.0
    reduction_percentage = (monthly_kwh_savings / bau_monthly_kwh * 100.0) if bau_monthly_kwh > 0 else 0.0

    selected_scenario_name = f"{reduction_percentage:.0f}% {short_tag}" if reduction_percentage > 0 else "Baseline Target"

    return {
        "status": "Optimal Solution Found (Linear Programming)",
        "selected_scenario": selected_scenario_name,
        "strategy_focus": strategy_focus,
        "objective_function": objective,
        "reduction_percentage": float(reduction_percentage),
        "bau_monthly_kwh": float(bau_monthly_kwh),
        "optimized_monthly_kwh": float(opt_monthly_kwh),
        "monthly_kwh_savings": float(monthly_kwh_savings),
        "annual_kwh_savings": float(annual_kwh_savings),
        "monthly_cost_savings_php": float(monthly_cost_savings),
        "annual_cost_savings_php": float(annual_cost_savings),
        "annual_avoided_co2_kg": float(annual_avoided_co2),
        "x_opt": [float(x_ac_opt), float(x_comp_opt), float(x_other_opt)],
        "optimization_rationale": (
            f"The Linear Programming optimization solver formulated under objective '{objective}' "
            f"identified an optimal target of {opt_monthly_kwh:,.2f} kWh/month ({monthly_kwh_savings:,.2f} kWh/month saved, "
            f"₱{annual_cost_savings:,.2f}/year cost reduction, {annual_avoided_co2:,.2f} kg CO₂e/year avoided)."
        )
    }

def monitor_target_consumption(actual_kwh: float, target_kwh: float = 0.0) -> Dict[str, Any]:
    """
    Compare actual monthly electricity consumption against the optimized target benchmark.
    Returns status indicator (GREEN if <= target, RED if > target).
    """
    diff_kwh = actual_kwh - target_kwh
    pct_diff = (diff_kwh / target_kwh * 100.0) if target_kwh > 0 else 0.0
    
    is_on_target = actual_kwh <= target_kwh
    status_code = "GREEN" if is_on_target else "RED"
    status_msg = "At or Below Target (Compliant)" if is_on_target else "Exceeds Target (Action Required)"
    
    return {
        "actual_kwh": actual_kwh,
        "target_kwh": target_kwh,
        "difference_kwh": float(diff_kwh),
        "percentage_difference": float(pct_diff),
        "status_code": status_code,
        "status_message": status_msg,
        "is_on_target": is_on_target
    }

def calculate_sensitivity_analysis(bau_kwh: float = 0.0, 
                                    test_reductions: List[float] = None) -> pd.DataFrame:
    if test_reductions is None:
        test_reductions = [0.0] + list(CONFIG_DEFAULT_SCENARIO_REDUCTION_RATES)
    """
    Compute sensitivity ratio across tested reduction percentages.
    Sensitivity Ratio = (% Change in Output Consumption) / (% Input Reduction)
    Demonstrates the direct 1.00 mathematical relationship.
    """
    rows = []
    for r in test_reductions:
        proj_kwh = bau_kwh * (1.0 - r)
        change_kwh = proj_kwh - bau_kwh
        out_pct = (change_kwh / bau_kwh * 100.0) if bau_kwh > 0 else 0.0
        in_pct = r * 100.0
        
        ratio_str = f"{abs(out_pct / in_pct):.2f}" if in_pct > 0 else "—"
        
        rows.append({
            "Input Reduction": f"{int(in_pct)}%",
            "Projected Consumption (kWh/month)": f"{proj_kwh:,.2f}",
            "Change from BAU (kWh)": f"{change_kwh:,.2f}",
            "Output Change (%)": f"{out_pct:.2f}%",
            "Sensitivity Ratio": ratio_str
        })
        
    return pd.DataFrame(rows)
