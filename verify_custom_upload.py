import sys
from pathlib import Path
import pandas as pd

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from modules.data_processing import load_historical_bills, load_appliance_loads, load_seasonal_data
from modules.load_analysis import calculate_appliance_loads, get_load_summary
from modules.seasonal_analysis import calculate_seasonal_metrics
from modules.historical_analysis import calculate_historical_metrics
from modules.forecasting import fit_ets_forecast
from modules.carbon import calculate_carbon_emissions
from modules.scenarios import calculate_bau_baseline, simulate_conservation_scenarios
from modules.optimization import optimize_conservation_target
from modules.recommendations import generate_executive_summary_recommendation, generate_appliance_recommendations

def run_dynamic_dataset_test():
    print("=" * 60)
    print("⚡ RUNNING DYNAMIC DATASET UPLOAD & SYSTEM-WIDE TEST ⚡")
    print("=" * 60)
    
    # 1. Load default historical & seasonal datasets
    hist_df = load_historical_bills()
    season_df = load_seasonal_data()
    print("✓ Default historical bills and seasonal data loaded successfully.")
    
    # 2. Load custom test appliance dataset (test_custom_appliances.csv)
    custom_app_path = Path("data/test_custom_appliances.csv")
    custom_app_df = load_appliance_loads(custom_app_path)
    print(f"✓ Custom Appliance dataset '{custom_app_path.name}' loaded: {len(custom_app_df)} rows")
    
    target_school = "An-anaao Integrated School"
    rate = 11.00
    ef = 0.70
    
    # 3. Test Load Analysis Module with custom dataset
    apps_proc = calculate_appliance_loads(custom_app_df, rate, target_school)
    load_summary = get_load_summary(apps_proc, rate)
    custom_total_kwh = load_summary["total_kwh"]
    custom_total_cost = load_summary["total_cost_php"]
    print(f"✓ Custom Appliance Monthly Load: {custom_total_kwh:.2f} kWh/month (₱{custom_total_cost:,.2f}/month)")
    
    print(f"  Top Priority Load: {load_summary['top_appliance']} ({load_summary['top_kwh']:.2f} kWh, {load_summary['top_share']:.1f}% share)")
    
    # 4. Test Seasonal Analysis Module
    seasonal_metrics = calculate_seasonal_metrics(hist_df, ["January", "February", "March", "April", "May", "December"], ["June", "July", "August", "September", "October", "November"])
    print(f"✓ Seasonal Index Peak: Dry Season average {seasonal_metrics['dry_avg']:.2f} kWh vs Wet Season {seasonal_metrics['wet_avg']:.2f} kWh ({seasonal_metrics['percentage_difference']:.2f}% diff)")
    
    # 5. Test ETS Forecasting Module
    ets_res = fit_ets_forecast(hist_df, target_school, forecast_horizon=12)
    avg_fc = float(ets_res['forecast_df']['forecast_bill'].mean())
    print(f"✓ ETS Forecast computed. 12-month avg: ₱{avg_fc:,.2f} (Validation MAPE: {ets_res['val_mape']:.2f}%, RMSE: {ets_res['val_rmse']:.2f})")
    
    # 6. Test Carbon Module
    from modules.carbon import calculate_carbon_summary
    carbon_res = calculate_carbon_summary(custom_total_kwh, ef)
    print(f"✓ Custom Monthly Carbon Footprint: {carbon_res['monthly_co2_kg']:.2f} kg CO2e ({carbon_res['annual_co2_tons']:.2f} metric tons/year)")
    
    # 7. Test BAU Baseline Module
    bau = calculate_bau_baseline(custom_total_kwh, rate, ef)
    print(f"✓ Custom BAU Baseline: {bau['monthly_kwh']:.2f} kWh/mo, ₱{bau['monthly_cost_php']:,.2f}/mo, {bau['annual_co2_kg']:,.2f} kg CO2e/yr")
    
    # 8. Test Scenario Simulation Module
    scenarios = simulate_conservation_scenarios(bau)
    s15_row = scenarios[scenarios['Reduction %'] == 15.0].iloc[0]
    print(f"✓ 15% Scenario Simulation: Consumption = {s15_row['Projected Monthly kWh']:.2f} kWh/mo, Monthly Savings = ₱{s15_row['Monthly Cost Savings (₱)']:,.2f}, Avoided CO2 = {s15_row['Annual Avoided CO₂e (kg)']:,.2f} kg CO2e/yr")
    
    # 9. Test Goal Programming Optimization Module
    opt = optimize_conservation_target(scenarios)
    print(f"✓ Linear Goal Programming Solved. Target = {opt['optimized_monthly_kwh']:.2f} kWh/mo (Saved {opt['annual_kwh_savings']:,.2f} kWh/yr, ₱{opt['annual_cost_savings_php']:,.2f}/yr)")
    
    # 10. Test Executive Report Generation Module
    exec_rec = generate_executive_summary_recommendation(load_summary["top_appliance"], load_summary["top_share"], opt["optimized_monthly_kwh"], seasonal_metrics["percentage_difference"])
    app_recs = generate_appliance_recommendations(apps_proc)
    print("✓ Executive Report Generated Successfully!")
    print(f"  Executive Statement: '{exec_rec[:120]}...'")
    print(f"  Appliance Priority #1: {app_recs.iloc[0]['Target Load']} -> {app_recs.iloc[0]['Recommended Management Action']}")
    
    print("=" * 60)
    print("🎉 DYNAMIC DATASET WORKFLOW INTEGRATION VERIFIED! 🎉")
    print("=" * 60)

if __name__ == "__main__":
    run_dynamic_dataset_test()
