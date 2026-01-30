from __future__ import annotations
from dataclasses import dataclass

from flight.world.noise import FBMNoise, NoiseConfig

@dataclass
class HeightProvider:
    seed: int

    def __post_init__(self) -> None:
        self.noise = FBMNoise(self.seed, NoiseConfig())

    def height_at(self, x: float, z: float) -> float:
        return self.noise.value(x, z)
