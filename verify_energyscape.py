"""
Automated Verification Script for ENERGYSCAPE Prototype.
Tests computational accuracy against paper reference targets.
"""

import sys
import pandas as pd
import numpy as np

from modules.data_processing import load_historical_bills, load_appliance_loads, load_seasonal_data, validate_dataset
from modules.historical_analysis import calculate_historical_metrics
from modules.seasonal_analysis import calculate_seasonal_metrics
from modules.load_analysis import calculate_appliance_loads, get_load_summary
from modules.forecasting import fit_ets_forecast
from modules.validation import calculate_mape, calculate_rmse, verify_computational_consistency
from modules.carbon import calculate_carbon_emissions
from modules.scenarios import calculate_bau_baseline, simulate_conservation_scenarios
from modules.optimization import optimize_conservation_target, monitor_target_consumption, calculate_sensitivity_analysis

def run_tests():
    print("==================================================")
    print("⚡ RUNNING ENERGYSCAPE COMPUTATIONAL VERIFICATION ⚡")
    print("==================================================")
    
    # 1. Data Processing
    hist = load_historical_bills()
    apps = load_appliance_loads()
    seas = load_seasonal_data()
    
    print(f"✓ Datasets loaded. Historical rows: {len(hist)}, Appliance rows: {len(apps)}, Seasonal rows: {len(seas)}")
    
    val_hist = validate_dataset(hist, "historical")
    assert val_hist["tbf_missing_count"] == 12, f"Expected 12 TBF entries across both schools, got {val_hist['tbf_missing_count']}"
    print("✓ TBF missing values correctly identified and preserved as NaN.")
    
    # 2. Appliance Load Quantification
    an_apps = calculate_appliance_loads(apps, electricity_rate=11.00, school_name="An-anaao Integrated School")
    load_sum = get_load_summary(an_apps, electricity_rate=11.00)
    
    total_kwh = load_sum["total_kwh"]
    assert abs(total_kwh - 2289.10) < 0.01, f"Expected total appliance load 2289.10 kWh, got {total_kwh:.2f}"
    print(f"✓ Total Appliance Load verified: {total_kwh:.2f} kWh/month")
    
    ac_kwh = float(an_apps[an_apps['appliance'] == 'Air Conditioner']['monthly_kwh'].iloc[0])
    comp_kwh = float(an_apps[an_apps['appliance'] == 'Computers']['monthly_kwh'].iloc[0])
    assert abs(ac_kwh - 792.00) < 0.01, f"Expected Air Conditioner load 792.00 kWh, got {ac_kwh:.2f}"
    assert abs(comp_kwh - 577.50) < 0.01, f"Expected Computers load 577.50 kWh, got {comp_kwh:.2f}"
    print(f"✓ Major loads verified: Air Conditioner = {ac_kwh:.2f} kWh, Computers = {comp_kwh:.2f} kWh")
    
    # 3. BAU Baseline
    bau = calculate_bau_baseline(total_kwh, electricity_rate=11.00, emission_factor=0.70)
    assert abs(bau["monthly_cost_php"] - 25180.10) < 0.1, f"Expected monthly cost 25180.10, got {bau['monthly_cost_php']:.2f}"
    assert abs(bau["annual_cost_php"] - 302161.20) < 0.1, f"Expected annual cost 302161.20, got {bau['annual_cost_php']:.2f}"
    assert abs(bau["monthly_co2_kg"] - 1602.37) < 0.1, f"Expected monthly CO2 1602.37 kg, got {bau['monthly_co2_kg']:.2f}"
    assert abs(bau["annual_co2_kg"] - 19228.44) < 0.1, f"Expected annual CO2 19228.44 kg, got {bau['annual_co2_kg']:.2f}"
    print("✓ Business-as-Usual (BAU) baseline metrics verified.")
    
    # 4. Conservation Scenarios
    scenarios_df = simulate_conservation_scenarios(bau)
    s5 = scenarios_df[scenarios_df['Reduction %'] == 5.0].iloc[0]
    s10 = scenarios_df[scenarios_df['Reduction %'] == 10.0].iloc[0]
    s15 = scenarios_df[scenarios_df['Reduction %'] == 15.0].iloc[0]
    
    assert abs(s5["Projected Monthly kWh"] - 2174.645) < 0.1, f"5% scenario error: {s5['Projected Monthly kWh']}"
    assert abs(s10["Projected Monthly kWh"] - 2060.19) < 0.1, f"10% scenario error: {s10['Projected Monthly kWh']}"
    assert abs(s15["Projected Monthly kWh"] - 1945.735) < 0.1, f"15% scenario error: {s15['Projected Monthly kWh']}"
    
    assert abs(s15["Annual Energy Saved (kWh)"] - 4120.32) < 1.0, f"Annual energy saved error: {s15['Annual Energy Saved (kWh)']}"
    assert abs(s15["Annual Cost Savings (₱)"] - 45323.52) < 1.0, f"Annual cost savings error: {s15['Annual Cost Savings (₱)']}"
    assert abs(s15["Annual Avoided CO₂e (kg)"] - 2884.22) < 1.0, f"Annual avoided CO2 error: {s15['Annual Avoided CO₂e (kg)']}"
    print("✓ 5%, 10%, and 15% Conservation Scenarios verified.")
    
    # 5. Optimization
    opt = optimize_conservation_target(scenarios_df)
    assert abs(opt["optimized_monthly_kwh"] - 1945.735) < 0.1, f"Optimization error: {opt['optimized_monthly_kwh']}"
    print(f"✓ Mathematical Optimization target verified: {opt['optimized_monthly_kwh']:.2f} kWh/month (15% reduction)")
    
    # 6. Forecasting & Validation
    fc_an = fit_ets_forecast(hist, "An-anaao Integrated School")
    mape = calculate_mape(fc_an["val_actuals"], fc_an["val_predictions"])
    rmse = calculate_rmse(fc_an["val_actuals"], fc_an["val_predictions"])
    print(f"✓ ETS Forecasting executed successfully. Calculated validation MAPE = {mape:.2f}%, RMSE = {rmse:.2f}")
    
    print("\n==================================================")
    print("🎉 ALL COMPUTATIONAL VERIFICATION TESTS PASSED! 🎉")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
