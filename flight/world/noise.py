from __future__ import annotations

from dataclasses import dataclass
from opensimplex import OpenSimplex

@dataclass(frozen=True)
class NoiseConfig:
    octaves: int = 5
    lacunarity: float = 2.0
    gain: float = 0.5
    base_freq: float = 0.006
    amplitude: float = 40.0

class FBMNoise:
    def __init__(self, seed: int, cfg: NoiseConfig | None = None) -> None:
        self.seed = int(seed)
        self.cfg = cfg or NoiseConfig()
        self._simp = OpenSimplex(self.seed)

    def value(self, x: float, z: float) -> float:
        # fBm over 2D simplex
        freq = self.cfg.base_freq
        amp = 1.0
        total = 0.0
        norm = 0.0
        for _ in range(self.cfg.octaves):
            n = self._simp.noise2(x * freq, z * freq)
            total += n * amp
            norm += amp
            freq *= self.cfg.lacunarity
            amp *= self.cfg.gain
        total = total / max(norm, 1e-9)

        # Mountain shaping: ridge-ish
        ridged = 1.0 - abs(total)
        shaped = (0.65 * total + 0.35 * (ridged * 2.0 - 1.0))

        return shaped * self.cfg.amplitude
