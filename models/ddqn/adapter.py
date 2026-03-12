import numpy as np

EXPECTED_ROWS = 6
EXPECTED_COLS = 9
EXPECTED_PLANTS = 10
SUN_NORM = 200.0


def _extract_action_mask(obs, env):
    if isinstance(obs, dict) and "action_mask" in obs:
        return np.asarray(obs["action_mask"], dtype=np.int8)
    if hasattr(env, "_get_action_mask") and hasattr(env, "pvz"):
        try:
            game_state = env.pvz.get_game_state() if env.pvz else None
            return np.asarray(env._get_action_mask(game_state), dtype=np.int8)
        except Exception:
            pass
    return np.ones(env.action_space.n, dtype=np.int8)


def _extract_sun(obs, info):
    if isinstance(info, dict) and "sun" in info:
        return float(info["sun"])
    if isinstance(obs, dict) and "global_features" in obs:
        # global_features[0] = sun / 9999.0
        return float(obs["global_features"][0]) * 9999.0
    return 0.0


def _plant_availability_from_mask(action_mask, num_cards, grid_size):
    mask = np.asarray(action_mask, dtype=bool)
    avail = np.zeros(num_cards, dtype=np.float32)
    for i in range(num_cards):
        start = i * grid_size
        end = start + grid_size
        if end <= mask.size and np.any(mask[start:end]):
            avail[i] = 1.0
    return avail


def build_state_vector(obs, action_mask, sun, rows, cols, num_cards):
    if not isinstance(obs, dict) or "grid" not in obs:
        raise ValueError("DDQN requires dict observation with 'grid'.")

    grid = obs["grid"]
    if grid.shape[0] != rows or grid.shape[1] != cols:
        raise ValueError(
            f"Grid shape mismatch: expected ({rows},{cols},C), got {grid.shape}."
        )

    # Plant grid: 1 if plant exists, 0 if empty
    plant_grid = (grid[:, :, 0] > 0).astype(np.float32)
    # Zombie HP sum per tile (already normalized in env)
    zombie_grid = grid[:, :, 4].astype(np.float32)

    grid_size = rows * cols
    plant_avail = _plant_availability_from_mask(action_mask, num_cards, grid_size)

    sun_norm = float(sun) / float(SUN_NORM)

    state = np.concatenate(
        [
            plant_grid.reshape(-1),
            zombie_grid.reshape(-1),
            plant_avail,
            np.array([sun_norm], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    return state


class DDQNEnvAdapter:
    """
    Wrap PVZEnv to provide DDQN-compatible observations and action masks.

    Observation vector:
    - plant grid (6x9)
    - zombie HP grid (6x9)
    - plant availability (10)
    - sun (1)
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.rows = env.rows
        self.cols = env.cols
        self.num_cards = env.num_cards
        self.plant_deck = list(getattr(env, "card_plant_ids", []))
        self._last_action_mask = None
        if self.rows != EXPECTED_ROWS or self.cols != EXPECTED_COLS:
            raise ValueError(
                f"DDQN expects grid {EXPECTED_ROWS}x{EXPECTED_COLS}, "
                f"but env has {self.rows}x{self.cols}."
            )
        if self.num_cards != EXPECTED_PLANTS:
            raise ValueError(
                f"DDQN expects {EXPECTED_PLANTS} plant types, "
                f"but env has {self.num_cards}."
            )

    @property
    def steps(self):
        return getattr(self.env, "steps", 0)

    def reset(self, **kwargs):
        reset_out = self.env.reset(**kwargs)
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, info = reset_out
        else:
            obs, info = reset_out, {}

        self._last_action_mask = _extract_action_mask(obs, self.env)
        sun = _extract_sun(obs, info)
        state = build_state_vector(
            obs, self._last_action_mask, sun, self.rows, self.cols, self.num_cards
        )
        return state

    def step(self, action):
        step_out = self.env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out

        self._last_action_mask = _extract_action_mask(obs, self.env)
        sun = _extract_sun(obs, info)
        state = build_state_vector(
            obs, self._last_action_mask, sun, self.rows, self.cols, self.num_cards
        )
        return state, reward, done, info

    def mask_available_actions(self):
        if self._last_action_mask is None:
            return np.ones(self.action_space.n, dtype=bool)
        return self._last_action_mask.astype(bool)

    def close(self):
        return self.env.close()
