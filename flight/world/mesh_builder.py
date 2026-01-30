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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return packed vertex attributes and indices for one chunk.

    Vertex format: pos (3) + norm (3) => float32.
    Indices include base grid + optional skirts.
    """
    x0 = cx * world_size
    z0 = cz * world_size
    step = world_size / (res - 1)

    xs = (x0 + np.arange(res, dtype=np.float32) * step)
    zs = (z0 + np.arange(res, dtype=np.float32) * step)
    grid_x, grid_z = np.meshgrid(xs, zs, indexing="xy")

    if hasattr(height_provider, "height_grid"):
        h = height_provider.height_grid(grid_x, grid_z).astype(np.float32)
    else:
        height_fn = height_provider
        h = np.zeros((res, res), dtype=np.float32)
        for j in range(res):
            for i in range(res):
                h[j, i] = float(height_fn(float(grid_x[j, i]), float(grid_z[j, i])))

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

    base_idx = build_indices(res)

    if skirts:
        pos_v2, n_v2, skirt_idx = _add_skirts(pos_v, n_v, res, skirt_depth=float(skirt_depth))
        # Combine indices: base first, then skirt indices
        idx = np.concatenate([base_idx, skirt_idx], axis=0)
        pos_v, n_v = pos_v2, n_v2
    else:
        idx = base_idx

    vbo = np.concatenate([pos_v, n_v], axis=1).astype(np.float32)  # (N,6)
    return vbo, idx, pos_v  # pos_v returned for potential metrics/debug
