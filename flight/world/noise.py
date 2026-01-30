from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from opensimplex import OpenSimplex  # optional dependency
except Exception:  # pragma: no cover
    OpenSimplex = None  # type: ignore


@dataclass(frozen=True)
class NoiseConfig:
    octaves: int = 5
    lacunarity: float = 2.0
    gain: float = 0.5
    base_freq: float = 0.006
    amplitude: float = 40.0


class FastValueNoise2D:
    """Fast 2D value noise (vectorized numpy). Deterministic for seed."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _hash(self, xi: np.ndarray, zi: np.ndarray) -> np.ndarray:
        x = (xi.astype(np.uint32) * np.uint32(374761393)) ^ (zi.astype(np.uint32) * np.uint32(668265263)) ^ np.uint32(self.seed)
        x ^= (x >> np.uint32(13))
        x *= np.uint32(1274126177)
        x ^= (x >> np.uint32(16))
        return (x.astype(np.float32) / np.float32(2**32))

    def noise(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        xi0 = np.floor(x).astype(np.int32)
        zi0 = np.floor(z).astype(np.int32)
        xi1 = xi0 + 1
        zi1 = zi0 + 1

        tx = (x - xi0.astype(np.float32))
        tz = (z - zi0.astype(np.float32))
        u = self._fade(tx)
        v = self._fade(tz)

        a = self._hash(xi0, zi0)
        b = self._hash(xi1, zi0)
        c = self._hash(xi0, zi1)
        d = self._hash(xi1, zi1)

        ab = a + (b - a) * u
        cd = c + (d - c) * u
        return ab + (cd - ab) * v  # [0,1)


class FBMFastNoise:
    def __init__(self, seed: int, cfg: NoiseConfig | None = None) -> None:
        self.seed = int(seed)
        self.cfg = cfg or NoiseConfig()
        self.base = FastValueNoise2D(seed)

    def grid(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        freq = self.cfg.base_freq
        amp = 1.0
        total = np.zeros_like(x, dtype=np.float32)
        norm = 0.0
        for _ in range(self.cfg.octaves):
            n = self.base.noise(x * freq, z * freq)  # [0,1)
            n = (n * 2.0 - 1.0)  # [-1,1)
            total += n * amp
            norm += amp
            freq *= self.cfg.lacunarity
            amp *= self.cfg.gain
        total = total / np.float32(max(norm, 1e-9))

        ridged = 1.0 - np.abs(total)
        shaped = (0.65 * total + 0.35 * (ridged * 2.0 - 1.0))
        return shaped * np.float32(self.cfg.amplitude)

    def value(self, x: float, z: float) -> float:
        xv = np.array([x], dtype=np.float32)
        zv = np.array([z], dtype=np.float32)
        return float(self.grid(xv, zv)[0])


class FBMSimplexNoise:
    """Simplex-based fBm. Slower, optional."""

    def __init__(self, seed: int, cfg: NoiseConfig | None = None) -> None:
        if OpenSimplex is None:
            raise RuntimeError("opensimplex is not installed")
        self.seed = int(seed)
        self.cfg = cfg or NoiseConfig()
        self._simp = OpenSimplex(self.seed)

    def value(self, x: float, z: float) -> float:
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
        ridged = 1.0 - abs(total)
        shaped = (0.65 * total + 0.35 * (ridged * 2.0 - 1.0))
        return shaped * self.cfg.amplitude
