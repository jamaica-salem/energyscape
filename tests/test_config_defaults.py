import importlib


def test_document_backed_defaults_are_defined_in_config():
    config = importlib.import_module("config")

    assert config.electricity_rate_php_per_kwh == 11.0
    assert config.emission_factor_kg_per_kwh == 0.70
    assert config.bau_monthly_kwh == 2289.10
    assert config.scenario_reductions == [0.05, 0.10, 0.15]


def test_module_defaults_match_config():
    config = importlib.import_module("config")
    load_analysis = importlib.import_module("modules.load_analysis")
    carbon = importlib.import_module("modules.carbon")
    scenarios = importlib.import_module("modules.scenarios")

    assert load_analysis.DEFAULT_ELECTRICITY_RATE == config.electricity_rate_php_per_kwh
    assert carbon.DEFAULT_EMISSION_FACTOR == config.emission_factor_kg_per_kwh
    assert scenarios.DEFAULT_SCENARIOS == config.scenario_reductions
