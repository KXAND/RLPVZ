from .zombie_spawner import ZombieSpawner
from .zombie import Zombie
from .zombie_cone import Zombie_cone
from .zombie_bucket import Zombie_bucket
from .zombie_flag import Zombie_flag
from ... import config
from simenv.config import ZOMBIE_SPAWN
import random

INITIAL_OFFSET = ZOMBIE_SPAWN["initial_offset_sec"]
SPAWN_INTERVAL = ZOMBIE_SPAWN["spawn_interval_sec"]


class WaveZombieSpawner(ZombieSpawner):
    def __init__(self):
        self._timer = INITIAL_OFFSET * config.FPS - 1
        self._wave_timer = (
            ZOMBIE_SPAWN["first_wave_delay_multiplier"]
            * INITIAL_OFFSET
            * config.FPS
            - 1
        )
        self.p = ZOMBIE_SPAWN["base_advanced_probability"]
        self.wave_index = 0
        self.completed_sublevels = 0
        self.last_wave_was_flag = False

    def spawn(self, scene):
        if self._timer <= 0 and self._wave_timer > 0:
            self.last_wave_was_flag = (
                self.wave_index % ZOMBIE_SPAWN["flag_wave_modulo"]
                == ZOMBIE_SPAWN["flag_wave_remainder"]
            )
            if self.last_wave_was_flag:
                scene.add_zombie(Zombie_flag(0))
                self.completed_sublevels += 1
            lane = random.choice(range(config.N_LANES))
            s = random.random()
            if s < self.p:
                scene.add_zombie(Zombie_bucket(lane))
            elif s < ZOMBIE_SPAWN["cone_probability_multiplier"] * self.p:
                scene.add_zombie(Zombie_cone(lane))
            else:
                scene.add_zombie(Zombie(lane))
            self.wave_index += 1
            self._timer = SPAWN_INTERVAL * config.FPS - 1
        else:
            if self._wave_timer > 0:
                self._timer -= 1
                self._wave_timer -= 1
            else:
                self.last_wave_was_flag = (
                    self.wave_index % ZOMBIE_SPAWN["flag_wave_modulo"]
                    == ZOMBIE_SPAWN["flag_wave_remainder"]
                )
                if self.last_wave_was_flag:
                    scene.add_zombie(Zombie_flag(0))
                    self.completed_sublevels += 1
                for lane in range(config.N_LANES):
                    s = random.random()
                    if s < self.p:
                        scene.add_zombie(Zombie_bucket(lane))
                    elif s < ZOMBIE_SPAWN["cone_probability_multiplier"] * self.p:
                        scene.add_zombie(Zombie_cone(lane))
                    else:
                        scene.add_zombie(Zombie(lane))
                self._wave_timer = (
                    ZOMBIE_SPAWN["wave_interval_multiplier"]
                    * SPAWN_INTERVAL
                    * config.FPS
                    - 1
                )
                self._timer = (
                    ZOMBIE_SPAWN["post_wave_spawn_delay_multiplier"]
                    * INITIAL_OFFSET
                    * config.FPS
                    - 1
                )
                self.p = min(
                    self.p * ZOMBIE_SPAWN["advanced_probability_growth"],
                    ZOMBIE_SPAWN["max_advanced_probability"],
                )
                self.wave_index += 1
