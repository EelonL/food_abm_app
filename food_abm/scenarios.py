SCENARIOS = {
    "baseline": {
        "neutral_hub_enabled": False,
        "public_buyer_enabled": False,
        "horeca_enabled": True,
        "fuel_cost_multiplier": 1.0,
        "retail_access_threshold": 0.65,
    },
    "neutral_hub": {
        "neutral_hub_enabled": True,
        "public_buyer_enabled": False,
        "horeca_enabled": True,
        "fuel_cost_multiplier": 1.0,
        "retail_access_threshold": 0.65,
    },
    "multichannel": {
        "neutral_hub_enabled": True,
        "public_buyer_enabled": True,
        "horeca_enabled": True,
        "fuel_cost_multiplier": 1.0,
        "retail_access_threshold": 0.60,
    },
    "shock": {
        "neutral_hub_enabled": True,
        "public_buyer_enabled": True,
        "horeca_enabled": True,
        "fuel_cost_multiplier": 1.6,
        "retail_access_threshold": 0.65,
    },
}
