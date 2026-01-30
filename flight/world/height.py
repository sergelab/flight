from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from flight.world.noise import NoiseConfig, FBMFastNoise, FBMSimplexNoise

@dataclass
class HeightProvider:
    seed: int
    mode: str = "fast"  # "fast" | "simplex"

    def __post_init__(self) -> None:
        cfg = NoiseConfig()
        if self.mode == "simplex":
            self.noise = FBMSimplexNoise(self.seed, cfg)
        else:
            self.noise = FBMFastNoise(self.seed, cfg)

    def height_at(self, x: float, z: float) -> float:
        return float(self.noise.value(x, z))

    def height_grid(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        # Vectorized (only supported for fast mode)
        if hasattr(self.noise, "grid"):
            return self.noise.grid(x.astype(np.float32), z.astype(np.float32)).astype(np.float32)
        # fallback: slow scalar calls
        out = np.zeros_like(x, dtype=np.float32)
        it = np.nditer(out, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            it[0] = np.float32(self.height_at(float(x[idx]), float(z[idx])))
            it.iternext()
        return out
