DIFFICULTY = {
    "fps": 2,
    "max_frames": 400,
    "rows": 5,
    "cols": 9,
    "initial_sun": 50,
    "natural_sun_production": 25,
    "natural_sun_cooldown": 10,
    "mowers": False,
    "survival_score": 0,
    "survival_step": 20,
    "alive_plant_score": 0,
    "alive_mower_score": 0,
}


ZOMBIE_SPAWN = {
    "initial_offset_sec": 10,
    "spawn_interval_sec": 8,
    "first_wave_delay_multiplier": 10,
    "wave_interval_multiplier": 20,
    "post_wave_spawn_delay_multiplier": 10,
    "flag_wave_modulo": 10,
    "flag_wave_remainder": 9,
    "base_advanced_probability": 0.05,
    "advanced_probability_growth": 2.0,
    "max_advanced_probability": 1.0,
    "cone_probability_multiplier": 3.0,
}


REWARDS = {
    "zombie_kill": {
        "use_type_rewards": False,
        "default": 0.30,
        "zombie": 0.30,
        "flag": 0.30,
        "conehead": 0.45,
        "buckethead": 0.70,
    },
    "sun_collect": 0.01,
    "wave_complete": 4.0,
    "game_win": 18.0,
    "streak_bonus": 0.0,
    "survival_per_step": 0.0,
    "plant_sunflower": 0.10,
    "plant_attacker": 0.35,
    "plant_wall": 0.18,
    "plant_other": 0.30,
    "plant_lost": -0.25,
    "sunflower_lost": -0.80,
    "lawnmower_triggered": -50.0,
    "game_lose": -12.0,
    "invalid_action": -0.01,
    "wait_with_sun": -0.02,
    "wait_sun_threshold": 300,
    "coverage": {
        "scale": 0.0,
    },
    "proximity": {
        "scale": 0.0,
    },
    "potential": {
        "sun_scale": 0.06,
        "sun_cap": 300.0,
        "plant_scale": 0.22,
        "spread_bonus": 0.06,
        "lawnmower_scale": 0.35,
        "zombie_threat_scale": 0.35,
        "zombie_distance_bonus": 0.75,
        "wave_scale": 0.05,
        "delta_scale": 0.18,
    },
}
