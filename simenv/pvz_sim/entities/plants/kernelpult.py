from .peashooter import Peashooter
from ..projectile.kernel import Kernel
from ... import config

KERNEL_PULT_COST = 100
KERNEL_PULT_COOLDOWN = 5
KERNEL_PULT_MAX_HP = 300
KERNEL_PULT_ATTACK = 20
KERNEL_PULT_ATTACK_COOLDOWN = 1.5
KERNEL_PULT_PROJECTILE_SPEED = 5


class KernelPult(Peashooter):
    MAX_HP = KERNEL_PULT_MAX_HP
    COOLDOWN = KERNEL_PULT_COOLDOWN
    COST = KERNEL_PULT_COST
    ATTACK = KERNEL_PULT_ATTACK
    ATTACK_COOLDOWN = KERNEL_PULT_ATTACK_COOLDOWN
    PROJECTILE_SPEED = KERNEL_PULT_PROJECTILE_SPEED

    def step(self, scene):
        if self.attack_cooldown <= 0:
            if scene.grid.is_attacked(self.lane):
                scene.projectiles.append(
                    Kernel(self.PROJECTILE_SPEED, self.ATTACK, self.lane, self.pos))
                self.attack_cooldown = self.ATTACK_COOLDOWN * config.FPS - 1
        else:
            self.attack_cooldown -= 1
