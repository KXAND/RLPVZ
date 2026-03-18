from config import load_config

# Load global config once
cfg = load_config()

# Default paths
DEFAULT_GAME_PATH = cfg.game_path
MODEL_PATH = cfg.model_save_path
LOAD_PATH = cfg.model_load_path
