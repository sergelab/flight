from __future__ import annotations

import numpy as np

def build_indices(res: int) -> np.ndarray:
    """Build triangle indices for a (res x res) grid.

    Winding is CCW when viewed from +Y (so the surface front-face points up).
    """
    idx: list[int] = []
    for j in range(res - 1):
        for i in range(res - 1):
            a = j * res + i
            b = a + 1
            c = a + res
            d = c + 1
            # CCW (a, b, c) and (b, d, c)
            idx.extend([a, b, c, b, d, c])
    return np.array(idx, dtype=np.uint32)

def build_chunk_vertices(cx: int, cz: int, res: int, world_size: float, height_fn) -> tuple[np.ndarray, np.ndarray]:
    # returns positions (N,3) and normals (N,3)
    x0 = cx * world_size
    z0 = cz * world_size
    step = world_size / (res - 1)

    xs = (x0 + np.arange(res, dtype=np.float32) * step)
    zs = (z0 + np.arange(res, dtype=np.float32) * step)
    grid_x, grid_z = np.meshgrid(xs, zs, indexing="xy")  # (res,res)

    # Height sampling
    h = np.zeros((res, res), dtype=np.float32)
    for j in range(res):
        for i in range(res):
            h[j, i] = float(height_fn(float(grid_x[j, i]), float(grid_z[j, i])))

    # Compute normals from central differences
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
    return pos.reshape(-1, 3), n.reshape(-1, 3)
