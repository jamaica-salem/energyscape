"""Central configuration for paper-backed default assumptions.

This file stores document-backed operational defaults used when no uploaded data
or explicit overrides are provided. The values are intentionally separated from
business logic so they can be reviewed, updated, and configured without editing
calculation modules.
"""

# Paper-backed default values derived from the attached MCS Prereq-Paper.
electricity_rate_php_per_kwh = 11.00
emission_factor_kg_per_kwh = 0.70
bau_monthly_kwh = 2289.10
scenario_reductions = [0.05, 0.10, 0.15]

# Operational defaults that are not UI configuration.
# These can be overridden by uploaded data or user-provided settings.
DEFAULT_ELECTRICITY_RATE = electricity_rate_php_per_kwh
DEFAULT_EMISSION_FACTOR = emission_factor_kg_per_kwh
DEFAULT_BAU_MONTHLY_KWH = bau_monthly_kwh
DEFAULT_SCENARIO_REDUCTION_RATES = scenario_reductions
