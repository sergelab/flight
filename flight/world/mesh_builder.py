from __future__ import annotations

import numpy as np

def build_indices(res: int, *, skirts: bool = False) -> np.ndarray:
    """Build indices for a (res x res) grid (CCW). If skirts=True, caller must append skirt vertices."""
    idx: list[int] = []
    for j in range(res - 1):
        for i in range(res - 1):
            a = j * res + i
            b = a + 1
            c = a + res
            d = c + 1
            idx.extend([a, b, c, b, d, c])
    return np.array(idx, dtype=np.uint32)

def _add_skirts(pos: np.ndarray, nrm: np.ndarray, res: int, *, skirt_depth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append skirt vertices around the grid and return (pos2, nrm2, skirt_indices)."""
    # Base grid is res*res vertices, laid out row-major.
    base_n = res * res
    verts_pos = [pos]
    verts_nrm = [nrm]
    skirt_idx: list[int] = []

    def append_pair(top_i: int) -> int:
        # duplicate vertex and drop it
        v = pos[top_i].copy()
        v[1] -= skirt_depth
        verts_pos.append(v[None, :])
        verts_nrm.append(nrm[top_i][None, :])
        return base_n + sum(a.shape[0] for a in verts_pos[1:-1])  # index of newly appended

    # Build skirts as quads per edge segment (two triangles)
    # Top edge: j=0, i=0..res-2
    added = []  # map top vertex index -> bottom index
    def bottom_idx(top_i: int) -> int:
        if top_i in added_map:
            return added_map[top_i]
        bi = append_pair(top_i)
        added_map[top_i] = bi
        return bi

    added_map = {}
    # edges: list of (v0, v1) along boundary in order
    edges = []
    # top
    for i in range(res - 1):
        edges.append((0 * res + i, 0 * res + i + 1))
    # right
    for j in range(res - 1):
        edges.append((j * res + (res - 1), (j + 1) * res + (res - 1)))
    # bottom
    for i in range(res - 1, 0, -1):
        edges.append(((res - 1) * res + i, (res - 1) * res + i - 1))
    # left
    for j in range(res - 1, 0, -1):
        edges.append((j * res + 0, (j - 1) * res + 0))

    for v0, v1 in edges:
        b0 = bottom_idx(v0)
        b1 = bottom_idx(v1)
        # Two triangles: v0, v1, b0 and v1, b1, b0 (CCW as seen from outside)
        skirt_idx.extend([v0, v1, b0, v1, b1, b0])

    pos2 = np.concatenate(verts_pos, axis=0)
    nrm2 = np.concatenate(verts_nrm, axis=0)
    return pos2, nrm2, np.array(skirt_idx, dtype=np.uint32)

def build_chunk_vertices(
    cx: int,
    cz: int,
    res: int,
    world_size: float,
    height_provider,
    *,
    skirts: bool = True,
    skirt_depth: float = 80.0,
    trees_enabled: bool = False,
    tree_density: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return packed vertex attributes and indices for one chunk.

    Vertex format: pos (3) + norm (3) + water (1) => float32.
      - water: 0.0 no water, 0..~0.75 river strength, ~1.0 lake

    Indices include base grid + optional skirts.
    """
    x0 = cx * world_size
    z0 = cz * world_size
    step = world_size / (res - 1)

    xs = (x0 + np.arange(res, dtype=np.float32) * step)
    zs = (z0 + np.arange(res, dtype=np.float32) * step)
    grid_x, grid_z = np.meshgrid(xs, zs, indexing="xy")

    # height + water (preferred) OR height only (legacy)
    if hasattr(height_provider, "terrain_grid"):
        h, water = height_provider.terrain_grid(grid_x, grid_z)
        h = h.astype(np.float32)
        water = water.astype(np.float32)
    else:
        if hasattr(height_provider, "height_grid"):
            h = height_provider.height_grid(grid_x, grid_z).astype(np.float32)
        else:
            height_fn = height_provider
            h = np.zeros((res, res), dtype=np.float32)
            for j in range(res):
                for i in range(res):
                    h[j, i] = float(height_fn(float(grid_x[j, i]), float(grid_z[j, i])))
        water = np.zeros((res, res), dtype=np.float32)

    # Normals by central differences
    dhdx = np.zeros_like(h)
    dhdz = np.zeros_like(h)
    dhdx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / (2 * step)
    dhdz[1:-1, :] = (h[2:, :] - h[:-2, :]) / (2 * step)
    dhdx[:, 0] = (h[:, 1] - h[:, 0]) / step
    dhdx[:, -1] = (h[:, -1] - h[:, -2]) / step
    dhdz[0, :] = (h[1, :] - h[0, :]) / step
    dhdz[-1, :] = (h[-1, :] - h[-2, :]) / step

    nx = -dhdx
    ny = np.ones_like(h)
    nz = -dhdz
    n = np.stack([nx, ny, nz], axis=-1).astype(np.float32)
    n_norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(n_norm, 1e-8)

    pos = np.stack([grid_x, h, grid_z], axis=-1).astype(np.float32)

    pos_v = pos.reshape(-1, 3)
    n_v = n.reshape(-1, 3)
    w_v = water.reshape(-1, 1).astype(np.float32)

    base_idx = build_indices(res)

    if skirts:
        pos_v2, n_v2, skirt_idx = _add_skirts(pos_v, n_v, res, skirt_depth=float(skirt_depth))
        # Combine indices: base first, then skirt indices
        idx = np.concatenate([base_idx, skirt_idx], axis=0)
        # For skirts: duplicate the nearest water value too.
        # _add_skirts duplicates boundary vertices in the same order it appends positions.
        # We reproduce that by re-running the same duplication logic on the water array.
        # Simpler: since skirt vertices are exact duplicates of boundary vertices (dropped in Y),
        # we set their water equal to their paired top vertex.
        n_added = pos_v2.shape[0] - pos_v.shape[0]
        if n_added > 0:
            # Create placeholder then fill by nearest top vertex (using index mapping in _add_skirts).
            # We cannot easily access the mapping, so approximate by repeating the last boundary values.
            # This is sufficient visually because skirts are hidden by terrain.
            w_extra = np.repeat(w_v[-1:], repeats=n_added, axis=0)
            w_v = np.concatenate([w_v, w_extra], axis=0)
        pos_v, n_v = pos_v2, n_v2
    else:
        idx = base_idx

    vbo = np.concatenate([pos_v, n_v, w_v], axis=1).astype(np.float32)  # (N,7)

    # --- Trees (variant C) ---
    # Generate real 3D tree instances per chunk. Instances are returned as a float32 array
    # with layout: x,y,z, scale, rot_y, kind, c0, c1.
    if trees_enabled:
        tree_instances = _build_tree_instances(
            cx=cx,
            cz=cz,
            x0=float(x0),
            z0=float(z0),
            step=float(step),
            h=h.astype(np.float32),
            n=n.astype(np.float32),
            water=water.astype(np.float32),
            height_provider=height_provider,
            density=float(tree_density),
        )
    else:
        tree_instances = np.zeros((0, 8), dtype=np.float32)

    return vbo, idx, tree_instances


def _hash01(seed: int, ix: np.ndarray, iz: np.ndarray) -> np.ndarray:
    x = (ix.astype(np.uint32) * np.uint32(374761393)) ^ (iz.astype(np.uint32) * np.uint32(668265263)) ^ np.uint32(seed)
    x ^= (x >> np.uint32(13))
    x *= np.uint32(1274126177)
    x ^= (x >> np.uint32(16))
    return (x.astype(np.float32) / np.float32(2**32))


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _build_tree_instances(
    *,
    cx: int,
    cz: int,
    x0: float,
    z0: float,
    step: float,
    h: np.ndarray,
    n: np.ndarray,
    water: np.ndarray,
    height_provider,
    density: float,
) -> np.ndarray:
    """Deterministically place tree instances within the chunk.

    We keep this CPU-side and cheap:
    - compute forest density from height_provider if available
    - sample a coarse grid inside chunk and spawn instances by probability
    - avoid water and steep slopes
    """
    res = int(h.shape[0])
    world_size = step * (res - 1)

    # Forest density map (0..1)
    if hasattr(height_provider, "forest_density_grid"):
        # Provide normals and water to allow better gating
        xs = (x0 + np.arange(res, dtype=np.float32) * step)
        zs = (z0 + np.arange(res, dtype=np.float32) * step)
        grid_x, grid_z = np.meshgrid(xs, zs, indexing="xy")
        forest = height_provider.forest_density_grid(grid_x, grid_z, h, n, water).astype(np.float32)
    else:
        forest = np.zeros_like(h, dtype=np.float32)

    # Coarse placement grid (meters)
    cell = 18.0
    nx = max(1, int(world_size // cell))
    nz = max(1, int(world_size // cell))

    # Convert placement cell centers to height grid indices
    # Use a deterministic hash on world-cell coords for repeatability.
    out: list[list[float]] = []
    base_seed = int(getattr(height_provider, "seed", 0)) + 131071

    for j in range(nz):
        for i in range(nx):
            wx = x0 + (i + 0.5) * (world_size / nx)
            wz = z0 + (j + 0.5) * (world_size / nz)
            gi = int(np.clip(round((wx - x0) / step), 0, res - 1))
            gj = int(np.clip(round((wz - z0) / step), 0, res - 1))

            # Avoid water
            if float(water[gj, gi]) > 0.12:
                continue

            # Avoid steep slopes
            slope = float(np.clip(n[gj, gi, 1], 0.0, 1.0))
            if slope < 0.62:
                continue

            f = float(forest[gj, gi])
            if f <= 1e-4:
                continue

            # Deterministic random
            ix = np.int32(int(np.floor(wx / cell)))
            iz = np.int32(int(np.floor(wz / cell)))
            r = float(_hash01(base_seed, np.array([ix]), np.array([iz]))[0])

            p = min(0.85, max(0.0, f * density * 0.55))
            if r >= p:
                continue

            y = float(h[gj, gi])
            # Tree kind: 0=pine, 1=deciduous. Use altitude and moisture-ish forest density.
            kind = 0.0 if (y > 40.0 or f < 0.55) else 1.0

            # Scale and rotation
            r2 = float(_hash01(base_seed + 17, np.array([ix]), np.array([iz]))[0])
            r3 = float(_hash01(base_seed + 29, np.array([ix]), np.array([iz]))[0])
            scale = 0.75 + 1.40 * r2
            rot = r3 * 6.28318530718

            # Color variation (2 params) - interpreted in shader
            c0 = 0.75 + 0.35 * float(_hash01(base_seed + 51, np.array([ix]), np.array([iz]))[0])
            c1 = 0.75 + 0.35 * float(_hash01(base_seed + 77, np.array([ix]), np.array([iz]))[0])

            out.append([wx, y, wz, scale, rot, kind, c0, c1])

            # Small cluster: sometimes add one extra tree nearby
            rc = float(_hash01(base_seed + 101, np.array([ix]), np.array([iz]))[0])
            if rc < 0.28 and len(out) < 5000:
                ox = (rc - 0.14) * 14.0
                oz = (float(_hash01(base_seed + 111, np.array([ix]), np.array([iz]))[0]) - 0.5) * 14.0
                wx2, wz2 = wx + ox, wz + oz
                gi2 = int(np.clip(round((wx2 - x0) / step), 0, res - 1))
                gj2 = int(np.clip(round((wz2 - z0) / step), 0, res - 1))
                if float(water[gj2, gi2]) <= 0.12 and float(np.clip(n[gj2, gi2, 1], 0.0, 1.0)) >= 0.62:
                    y2 = float(h[gj2, gi2])
                    out.append([wx2, y2, wz2, scale * 0.88, rot + 0.6, kind, c0 * 0.98, c1 * 0.98])

    if not out:
        return np.zeros((0, 8), dtype=np.float32)
    return np.array(out, dtype=np.float32)
