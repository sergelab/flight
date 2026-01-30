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

    # v0.6 defaults ("по умолчанию")
    # Rivers are now carved into the terrain to create valleys/canyons.
    river_carve_width: float = 22.0
    river_carve_depth: float = 18.0
    river_carve_power: float = 1.35

    # Ruggedness & peaks
    # - crinkle: frequent small height changes (less "smooth" look)
    # - peaks: sharp mountains (sometimes in small groups)
    crinkle_amp: float = 6.0
    peak_cell: float = 420.0
    peak_spawn_p: float = 0.10
    peak_sigma: float = 150.0
    peak_amp: float = 115.0
    peak_group_p: float = 0.55  # probability to spawn extra nearby peaks

    # Rare high snowy plateaus
    plateau_gate_lo: float = 0.82
    plateau_gate_hi: float = 0.93
    plateau_step: float = 9.0

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

        # Extra layers for diversity (keep it mostly low-frequency to avoid shimmer)
        self._macro = FBMFastNoise(self.seed + 1337, NoiseConfig(octaves=4, base_freq=0.0012, amplitude=28.0))
        self._warp = FBMFastNoise(self.seed + 9001, NoiseConfig(octaves=3, base_freq=0.0022, amplitude=1.0))

        # v0.6: less smooth terrain + sharp peaks
        # Rolling hills: mid-frequency layer
        self._hills = FBMFastNoise(self.seed + 777, NoiseConfig(octaves=5, base_freq=0.012, amplitude=14.0))

        # Crinkle: frequent small height changes (higher freq, small amplitude)
        self._crinkle = FBMFastNoise(
            self.seed + 31337,
            NoiseConfig(octaves=3, base_freq=0.045, amplitude=float(self.crinkle_amp)),
        )

        # Peak detail (ridged-ish) - used to add sharpness inside peak regions
        self._peak_detail = FBMFastNoise(self.seed + 2222, NoiseConfig(octaves=3, base_freq=0.018, amplitude=1.0))

        # Rare plateau mask (very low frequency)
        self._plateau_mask = FBMFastNoise(self.seed + 9090, NoiseConfig(octaves=2, base_freq=0.00045, amplitude=1.0))

        # Forest moisture (biome helper). Low frequency so forests form coherent regions.
        self._moisture = FBMFastNoise(self.seed + 5050, NoiseConfig(octaves=3, base_freq=0.0016, amplitude=1.0))

    def forest_density_grid(self, x: np.ndarray, z: np.ndarray, h: np.ndarray, n: np.ndarray, water: np.ndarray) -> np.ndarray:
        """Return forest density 0..1 for a vertex grid.

        This is purely a biome mask used for tree placement (variant C).
        We gate forests by:
          - moisture (low-frequency noise)
          - altitude (no dense forests on high snowy peaks)
          - slope (less trees on steep cliffs)
          - water proximity (slightly more trees near rivers/lakes)
        """
        # Moisture in [0,1]
        m = self._moisture.grid(x.astype(np.float32), z.astype(np.float32)).astype(np.float32)
        m01 = np.clip((m + np.float32(1.0)) * np.float32(0.5), 0.0, 1.0)

        # Base from moisture
        base = _smoothstep(0.52, 0.82, m01).astype(np.float32)

        # Altitude gate (avoid waterline and high snow)
        lo = _smoothstep(float(self.water_level + 4.0), float(self.water_level + 14.0), h)
        hi = (np.float32(1.0) - _smoothstep(65.0, 90.0, h)).astype(np.float32)
        alt = (lo * hi).astype(np.float32)

        # Slope gate (n.y is 1 on flat ground)
        slope = np.clip(n[..., 1], 0.0, 1.0).astype(np.float32)
        slope_gate = _smoothstep(0.55, 0.90, slope).astype(np.float32)

        # Water proximity boost: slightly greener near rivers/lakes, but not on water itself.
        near_water = _smoothstep(0.05, 0.35, water).astype(np.float32)
        water_boost = (np.float32(1.0) + near_water * np.float32(0.25)).astype(np.float32)
        not_water = (np.float32(1.0) - _smoothstep(0.25, 0.85, water)).astype(np.float32)

        return np.clip(base * alt * slope_gate * water_boost * not_water, 0.0, 1.0).astype(np.float32)

    def height_at(self, x: float, z: float) -> float:
        # Domain-warp + macro variation for more diverse shapes.
        wx = float(self._warp.value(x + 12.3, z - 9.7))
        wz = float(self._warp.value(x - 31.1, z + 27.8))
        xw = x + wx * 35.0
        zw = z + wz * 35.0
        base = float(self.noise.value(xw, zw))
        macro = float(self._macro.value(x * 0.85, z * 0.85))

        hills = float(self._hills.value(x * 1.05, z * 1.05))
        crinkle = float(self._crinkle.value(x, z))

        # Peaks (sharp mountains) are generated as local "fields" so they look like distinct peaks
        peak = float(self._peaks_at_point(x, z))

        shaped = 0.80 * base + 0.55 * macro + 0.85 * hills + 1.00 * crinkle + 1.00 * peak
        shaped = float(self._apply_plateau(np.array([shaped], dtype=np.float32), np.array([x], dtype=np.float32), np.array([z], dtype=np.float32))[0])
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
        hills = self._hills.grid(xf * np.float32(1.05), zf * np.float32(1.05)).astype(np.float32)
        crinkle = self._crinkle.grid(xf, zf).astype(np.float32)
        peaks = self._peaks_grid(xf, zf).astype(np.float32)

        h = (
            np.float32(0.80) * base
            + np.float32(0.55) * macro
            + np.float32(0.85) * hills
            + np.float32(1.00) * crinkle
            + np.float32(1.00) * peaks
        ).astype(np.float32)

        # Precompute lake mask *before* carving rivers so river carving does not change lake frequency.
        wl = np.float32(self.water_level)
        lake = h < wl
        lake_mask = lake.astype(np.float32)

        # --- rivers (mask + carving) ---
        river_mask, river_carve = self._river_mask_and_carve_from_sources(xf, zf)

        # Carve valleys/canyons for rivers (do not affect precomputed lake mask)
        if river_carve is not None:
            h = h - river_carve

        # Flatten lakes to a stable water surface (based on the pre-carve mask).
        h = np.where(lake, wl, h)

        # Combine: lakes dominate.
        water = np.maximum(lake_mask, river_mask)

        # Rare snowy plateaus (visual + geometric): terrace only at high altitude and only where the mask gates.
        h = self._apply_plateau(h, xf, zf)
        return h.astype(np.float32), water.astype(np.float32)

    def _river_mask_and_carve_from_sources(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
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
        carve = np.zeros_like(xf, dtype=np.float32)

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

        cwidth = float(self.river_carve_width)
        cw2 = max(cwidth * cwidth, 1e-6)
        cdepth = float(self.river_carve_depth)
        cpower = float(self.river_carve_power)

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

                # Carve a wider valley around the river. Use a slightly different width and a power curve.
                c = np.exp(-d2 / np.float32(cw2)).astype(np.float32)
                c = np.power(np.clip(c, 0.0, 1.0), np.float32(cpower))
                carve = np.maximum(carve, c * np.float32(cdepth))

        return out.reshape(x.shape), carve.reshape(x.shape)

    def _peak_centers_for_bounds(self, xmin: float, xmax: float, zmin: float, zmax: float) -> np.ndarray:
        """Generate peak centers (x,z) in an expanded neighborhood of the given bounds.

        Peaks are generated on a coarse grid (peak_cell). Each cell can spawn a peak.
        With some probability, we also spawn 1-2 nearby peaks (small groups).
        """
        cell = float(self.peak_cell)
        pad = cell * 1.25
        i0 = int(np.floor((xmin - pad) / cell))
        i1 = int(np.floor((xmax + pad) / cell))
        j0 = int(np.floor((zmin - pad) / cell))
        j1 = int(np.floor((zmax + pad) / cell))

        centers: list[tuple[float, float]] = []
        spawn_p = float(self.peak_spawn_p)
        group_p = float(self.peak_group_p)

        for ix in range(i0, i1 + 1):
            for iz in range(j0, j1 + 1):
                r = float(_hash01(self.seed + 8086, np.array([ix], dtype=np.int32), np.array([iz], dtype=np.int32))[0])
                if r >= spawn_p:
                    continue

                # Jitter within cell
                jx = float(_hash01(self.seed + 8087, np.array([ix], dtype=np.int32), np.array([iz], dtype=np.int32))[0])
                jz = float(_hash01(self.seed + 8088, np.array([ix], dtype=np.int32), np.array([iz], dtype=np.int32))[0])
                cx = (ix + 0.20 + 0.60 * jx) * cell
                cz = (iz + 0.20 + 0.60 * jz) * cell
                centers.append((cx, cz))

                # Sometimes spawn a small group (1-2 nearby peaks)
                rg = float(_hash01(self.seed + 8089, np.array([ix], dtype=np.int32), np.array([iz], dtype=np.int32))[0])
                if rg < group_p:
                    n_extra = 1 + int(rg * 2.0)  # 1..2
                    for k in range(n_extra):
                        # Small offset within the same cell neighborhood
                        ox = (float(_hash01(self.seed + 8090 + k, np.array([ix], dtype=np.int32), np.array([iz], dtype=np.int32))[0]) - 0.5) * cell * 0.55
                        oz = (float(_hash01(self.seed + 8092 + k, np.array([ix], dtype=np.int32), np.array([iz], dtype=np.int32))[0]) - 0.5) * cell * 0.55
                        centers.append((cx + ox, cz + oz))

        if not centers:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(centers, dtype=np.float32)

    def _peaks_grid(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute a peak contribution grid (sharp mountains, sometimes grouped)."""
        xf = x.reshape(-1).astype(np.float32)
        zf = z.reshape(-1).astype(np.float32)
        xmin, xmax = float(np.min(xf)), float(np.max(xf))
        zmin, zmax = float(np.min(zf)), float(np.max(zf))

        centers = self._peak_centers_for_bounds(xmin, xmax, zmin, zmax)
        if centers.shape[0] == 0:
            return np.zeros_like(x, dtype=np.float32)

        dx = xf[:, None] - centers[None, :, 0]
        dz = zf[:, None] - centers[None, :, 1]
        d2 = dx * dx + dz * dz
        sigma = np.float32(self.peak_sigma)
        s2 = np.maximum(sigma * sigma, np.float32(1e-6))
        field = np.exp(-d2 / s2).astype(np.float32)
        # Make them look like peaks (sharper falloff)
        field = field * field
        peak_mask = np.max(field, axis=1).astype(np.float32)

        # Add sharpness inside peak regions
        detail = self._peak_detail.grid(x.astype(np.float32), z.astype(np.float32)).reshape(-1).astype(np.float32)
        ridged = np.clip(np.float32(1.0) - np.abs(detail), 0.0, 1.0)
        sharp = (np.float32(0.55) + np.float32(0.45) * ridged)

        amp = np.float32(self.peak_amp)
        peaks = peak_mask * sharp * amp
        return peaks.reshape(x.shape).astype(np.float32)

    def _peaks_at_point(self, x: float, z: float) -> float:
        """Point-sampled peaks (used by the river tracer)."""
        cell = float(self.peak_cell)
        ix = int(np.floor(x / cell))
        iz = int(np.floor(z / cell))
        # Evaluate small neighborhood of cells around point
        centers: list[tuple[float, float]] = []
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cx_i = ix + dx
                cz_i = iz + dz
                r = float(_hash01(self.seed + 8086, np.array([cx_i], dtype=np.int32), np.array([cz_i], dtype=np.int32))[0])
                if r >= float(self.peak_spawn_p):
                    continue
                jx = float(_hash01(self.seed + 8087, np.array([cx_i], dtype=np.int32), np.array([cz_i], dtype=np.int32))[0])
                jz = float(_hash01(self.seed + 8088, np.array([cx_i], dtype=np.int32), np.array([cz_i], dtype=np.int32))[0])
                px = (cx_i + 0.20 + 0.60 * jx) * cell
                pz = (cz_i + 0.20 + 0.60 * jz) * cell
                centers.append((px, pz))
                rg = float(_hash01(self.seed + 8089, np.array([cx_i], dtype=np.int32), np.array([cz_i], dtype=np.int32))[0])
                if rg < float(self.peak_group_p):
                    n_extra = 1 + int(rg * 2.0)
                    for k in range(n_extra):
                        ox = (float(_hash01(self.seed + 8090 + k, np.array([cx_i], dtype=np.int32), np.array([cz_i], dtype=np.int32))[0]) - 0.5) * cell * 0.55
                        oz = (float(_hash01(self.seed + 8092 + k, np.array([cx_i], dtype=np.int32), np.array([cz_i], dtype=np.int32))[0]) - 0.5) * cell * 0.55
                        centers.append((px + ox, pz + oz))

        if not centers:
            return 0.0

        sigma = float(self.peak_sigma)
        s2 = max(sigma * sigma, 1e-6)
        best = 0.0
        for (cx, cz) in centers:
            d2 = (x - cx) * (x - cx) + (z - cz) * (z - cz)
            f = np.exp(-d2 / s2)
            f = f * f
            best = max(best, float(f))
        detail = float(self._peak_detail.value(x, z))
        ridged = max(0.0, min(1.0, 1.0 - abs(detail)))
        sharp = 0.55 + 0.45 * ridged
        return best * sharp * float(self.peak_amp)

    def _apply_plateau(self, h: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Apply rare terraced high plateaus (intended to be snowy)."""
        mask = self._plateau_mask.grid(x.astype(np.float32), z.astype(np.float32)).astype(np.float32)
        m01 = np.clip((mask + np.float32(1.0)) * np.float32(0.5), 0.0, 1.0)
        gate = _smoothstep(float(self.plateau_gate_lo), float(self.plateau_gate_hi), m01).astype(np.float32)
        # Only at high elevations
        hi = (h > np.float32(70.0)).astype(np.float32)
        gate = gate * hi
        if np.max(gate) <= 1e-6:
            return h
        step = np.float32(self.plateau_step)
        terraced = np.round(h / step) * step
        return (h * (np.float32(1.0) - gate) + terraced * gate).astype(np.float32)
