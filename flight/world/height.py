from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from flight.world.noise import NoiseConfig, FBMFastNoise, FBMSimplexNoise


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """GLSL-like smoothstep for numpy arrays."""
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hash01(seed: int, ix: np.ndarray, iz: np.ndarray) -> np.ndarray:
    """Deterministic hash -> [0,1) for integer grids (vectorized)."""
    x = (ix.astype(np.uint32) * np.uint32(374761393)) ^ (iz.astype(np.uint32) * np.uint32(668265263)) ^ np.uint32(seed)
    x ^= (x >> np.uint32(13))
    x *= np.uint32(1274126177)
    x ^= (x >> np.uint32(16))
    return (x.astype(np.float32) / np.float32(2**32))


def _dist2_to_points(x: np.ndarray, z: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Return min squared distance from (x,z) points to a polyline sampled as points."""
    # x,z: (N,) ; pts: (M,2)
    dx = x[:, None] - pts[None, :, 0]
    dz = z[:, None] - pts[None, :, 1]
    return np.min(dx * dx + dz * dz, axis=1)


@dataclass
class HeightProvider:
    seed: int
    mode: str = "fast"  # "fast" | "simplex"

    # v0.3 defaults ("по умолчанию")
    water_level: float = -6.0
    river_cell: float = 256.0  # spacing of potential river sources
    river_spawn_p: float = 0.03
    river_step: float = 16.0
    river_max_steps: int = 320
    river_width: float = 7.0

    def __post_init__(self) -> None:
        # Base terrain
        cfg = NoiseConfig(octaves=6, base_freq=0.0045, amplitude=45.0)
        if self.mode == "simplex":
            try:
                self.noise = FBMSimplexNoise(self.seed, cfg)
            except Exception:
                # Fallback to fast if opensimplex isn't available
                self.noise = FBMFastNoise(self.seed, cfg)
                self.mode = "fast"
        else:
            self.noise = FBMFastNoise(self.seed, cfg)

        # Extra layers for diversity (still low-frequency to avoid shimmer)
        self._macro = FBMFastNoise(self.seed + 1337, NoiseConfig(octaves=4, base_freq=0.0012, amplitude=28.0))
        self._warp = FBMFastNoise(self.seed + 9001, NoiseConfig(octaves=3, base_freq=0.0022, amplitude=1.0))

    def height_at(self, x: float, z: float) -> float:
        # Domain-warp + macro variation for more diverse shapes.
        wx = float(self._warp.value(x + 12.3, z - 9.7))
        wz = float(self._warp.value(x - 31.1, z + 27.8))
        xw = x + wx * 35.0
        zw = z + wz * 35.0
        base = float(self.noise.value(xw, zw))
        macro = float(self._macro.value(x * 0.85, z * 0.85))
        # Gentle shaping: broaden valleys and sharpen ridges a bit.
        shaped = 0.85 * base + 0.55 * macro
        return shaped

    def height_grid(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        # Keep a compatibility method (height only).
        h, _w = self.terrain_grid(x, z)
        return h

    def terrain_grid(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (height, water_mask) for a vertex grid.

        water_mask:
          - 0.0 = no water
          - 0..~0.75 = river (strength)
          - ~1.0 = lake
        """
        xf = x.astype(np.float32)
        zf = z.astype(np.float32)

        # --- terrain height ---
        # Domain warp (vectorized)
        wx = self._warp.grid(xf + np.float32(12.3), zf - np.float32(9.7))
        wz = self._warp.grid(xf - np.float32(31.1), zf + np.float32(27.8))
        xw = xf + wx * np.float32(35.0)
        zw = zf + wz * np.float32(35.0)

        base = self.noise.grid(xw, zw).astype(np.float32)
        macro = self._macro.grid(xf * np.float32(0.85), zf * np.float32(0.85)).astype(np.float32)
        h = (np.float32(0.85) * base + np.float32(0.55) * macro).astype(np.float32)

        # --- lakes ---
        wl = np.float32(self.water_level)
        lake = h < wl
        lake_mask = lake.astype(np.float32)

        # Flatten lakes to a stable water surface.
        h = np.where(lake, wl, h)

        # --- rivers ---
        river_mask = self._river_mask_from_sources(xf, zf)

        # Combine: lakes dominate.
        water = np.maximum(lake_mask, river_mask)
        return h.astype(np.float32), water.astype(np.float32)

    def _river_mask_from_sources(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Generate a river mask for the given vertex grid.

        This is a lightweight, deterministic "downhill" tracer:
        - pick sparse sources on a coarse world grid
        - follow steepest descent by sampling 8 directions
        - render as distance-to-polyline (sampled points)
        """
        # Flatten for distance computations
        xf = x.reshape(-1).astype(np.float32)
        zf = z.reshape(-1).astype(np.float32)
        out = np.zeros_like(xf, dtype=np.float32)

        # Chunk bounds in world coords
        xmin, xmax = float(np.min(xf)), float(np.max(xf))
        zmin, zmax = float(np.min(zf)), float(np.max(zf))

        # Search in neighborhood of coarse cells
        cell = float(self.river_cell)
        pad = cell * 0.75
        i0 = int(np.floor((xmin - pad) / cell))
        i1 = int(np.floor((xmax + pad) / cell))
        j0 = int(np.floor((zmin - pad) / cell))
        j1 = int(np.floor((zmax + pad) / cell))

        wl = float(self.water_level)
        spawn_p = float(self.river_spawn_p)
        step = float(self.river_step)
        max_steps = int(self.river_max_steps)
        width = float(self.river_width)
        w2 = max(width * width, 1e-6)

        # Precompute 8 directions
        dirs = np.array([
            [1, 0], [0, 1], [-1, 0], [0, -1],
            [1, 1], [-1, 1], [-1, -1], [1, -1],
        ], dtype=np.float32)
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

        for ix in range(i0, i1 + 1):
            for iz in range(j0, j1 + 1):
                # Deterministic spawn decision
                r = float(_hash01(self.seed + 4242, np.array([ix], dtype=np.int32), np.array([iz], dtype=np.int32))[0])
                if r >= spawn_p:
                    continue

                sx = (ix + 0.5) * cell
                sz = (iz + 0.5) * cell
                sh = float(self.height_at(sx, sz))
                if sh < (wl + 10.0):
                    continue

                # Trace downhill
                pts: list[tuple[float, float]] = [(sx, sz)]
                cx, cz = sx, sz
                ch = sh
                for _ in range(max_steps):
                    # Sample 8 neighbors
                    best_h = ch
                    best_x, best_z = cx, cz
                    for d in dirs:
                        nx = cx + float(d[0]) * step
                        nz = cz + float(d[1]) * step
                        nh = float(self.height_at(nx, nz))
                        if nh < best_h:
                            best_h = nh
                            best_x, best_z = nx, nz
                    if (best_x == cx and best_z == cz):
                        break
                    cx, cz, ch = best_x, best_z, best_h
                    pts.append((cx, cz))
                    # Stop if we reached lake level
                    if ch <= wl:
                        break
                    # Early stop if far away from this chunk neighborhood
                    if (cx < xmin - cell * 2) or (cx > xmax + cell * 2) or (cz < zmin - cell * 2) or (cz > zmax + cell * 2):
                        break

                if len(pts) < 6:
                    continue

                pts_arr = np.array(pts, dtype=np.float32)

                # Quick bbox check
                rxmin, rzmin = float(np.min(pts_arr[:, 0])), float(np.min(pts_arr[:, 1]))
                rxmax, rzmax = float(np.max(pts_arr[:, 0])), float(np.max(pts_arr[:, 1]))
                if rxmax < xmin - width or rxmin > xmax + width or rzmax < zmin - width or rzmin > zmax + width:
                    continue

                d2 = _dist2_to_points(xf, zf, pts_arr)
                # Gaussian-ish falloff, capped below lake mask
                strength = np.exp(-d2 / np.float32(w2)).astype(np.float32) * np.float32(0.75)
                out = np.maximum(out, strength)

        return out.reshape(x.shape)
