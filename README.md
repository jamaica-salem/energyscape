# ENERGYSCAPE: Multi-Seasonal Mathematical-Computational Framework for Predictive Energy Management and Carbon Reduction

Grade 12 Mathematical and Computational Science Research Prototype  
**Participating Institutions**: An-anaao Integrated School & La Paz Integrated School, La Paz, Abra, Philippines

---

## 📌 Overview

**ENERGYSCAPE** is an interactive mathematical-computational decision-support application designed to analyze historical electricity billing data, characterize seasonal demand patterns, quantify appliance electrical loads, forecast future electricity bills, model carbon emissions, evaluate conservation scenarios, and optimize electricity consumption targets.

The project functions strictly as an **analytical decision-support system**. It does **NOT** perform physical appliance switching, direct equipment control, or automated hardware control.

---

## 🔄 Core ENERGYSCAPE Workflow

The conceptual workflow follows a 15-step analytical progression:

$$\text{Historical Bills} \rightarrow \text{Validation} \rightarrow \text{Historical Analysis} \rightarrow \text{Seasonal Analysis} \rightarrow \text{Appliance Load Quantification} \rightarrow \text{Major Load Identification} \rightarrow \text{ETS Forecasting} \rightarrow \text{MAPE/RMSE Validation} \rightarrow \text{Carbon Modeling} \rightarrow \text{BAU Baseline} \rightarrow \text{Conservation Scenarios (5\%, 10\%, 15\%)} \rightarrow \text{Optimization} \rightarrow \text{Savings Assessment} \rightarrow \text{School Comparison} \rightarrow \text{Target Monitoring}$$

---

## 🧮 Mathematical Formulas & Models

### 1. Electrical Load Quantification
$$\text{Monthly Energy (kWh)} = \frac{P \times Q \times H \times D}{1000}$$
- $P$: Rated Power in Watts (W)
- $Q$: Quantity of appliances
- $H$: Estimated operating hours per day
- $D$: Operating days per month (default 22 days)

### 2. Carbon Emission Quantification
$$\text{CO}_2\text{e (kg)} = \text{Electricity Consumption (kWh)} \times \text{Emission Factor (0.70 kg CO}_2\text{e/kWh)}$$

### 3. Forecast Validation Metrics
$$\text{MAPE} = \frac{100}{n} \sum_{i=1}^n \left| \frac{\text{Actual}_i - \text{Forecast}_i}{\text{Actual}_i} \right|$$

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (\text{Actual}_i - \text{Forecast}_i)^2}$$

### 4. Conservation Scenarios & Optimization
For reduction rate $r \in \{0.05, 0.10, 0.15\}$:
$$\text{Scenario Consumption} = \text{BAU} \times (1 - r)$$
$$\text{Energy Savings} = \text{BAU} - \text{Scenario Consumption}$$
$$\text{Cost Savings} = \text{Energy Savings} \times \text{Electricity Rate (₱11.00/kWh)}$$
$$\text{Avoided CO}_2\text{e} = \text{Energy Savings} \times \text{Emission Factor (0.70 kg/kWh)}$$

---

## 📁 Dataset Structure

The application operates on clean, normalized long-form CSV datasets located in `data/`:

- **`historical_bills.csv`**: Contains monthly electricity billing records in Philippine Pesos (₱) for SY 2021–2022 to 2025–2026. Unavailable observations (TBF) are preserved as `NaN`.
- **`appliance_loads.csv`**: Contains appliance inventory parameters (`quantity`, `power_watts`, `hours_per_day`, `operating_days`) for major loads.
- **`seasonal_data.csv`**: Contains seasonal consumption records (in kWh) grouped into Dry (Dec–May) and Wet (Jun–Nov) seasons.

---

## 🚀 Installation & Running

### Prerequisites
- Python 3.10+
- Virtual environment (included in `./venv`)

### Step 1: Install Dependencies
```bash
./venv/bin/pip install -r requirements.txt
```

### Step 2: Run the Streamlit Application
```bash
./venv/bin/streamlit run app.py
```

The application will automatically launch in your default web browser at `http://localhost:8501`.

---

## 🔬 Limitations & Study Scope

1. **Model Estimates**: All calculated appliance energy loads, financial cost savings, and avoided emissions represent mathematical model estimates, not experimentally measured physical reductions.
2. **Electricity Scope Only**: Carbon calculations exclusively measure electricity-related emissions within the study boundary. Transportation, waste, or secondary institutional emissions are not included.
3. **Decision-Support Focus**: The framework provides analytical benchmarks and recommendations for decision support. It does not perform automated hardware control.