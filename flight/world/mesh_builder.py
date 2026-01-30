from __future__ import annotations

import numpy as np

def build_indices(res: int) -> np.ndarray:
    idx: list[int] = []
    for j in range(res - 1):
        for i in range(res - 1):
            a = j * res + i
            b = a + 1
            c = a + res
            d = c + 1
            idx.extend([a, b, c, b, d, c])
    return np.array(idx, dtype=np.uint32)

def build_chunk_vertices(cx: int, cz: int, res: int, world_size: float, height_provider) -> tuple[np.ndarray, np.ndarray]:
    x0 = cx * world_size
    z0 = cz * world_size
    step = world_size / (res - 1)

    xs = (x0 + np.arange(res, dtype=np.float32) * step)
    zs = (z0 + np.arange(res, dtype=np.float32) * step)
    grid_x, grid_z = np.meshgrid(xs, zs, indexing="xy")  # (res,res)

    # Vectorized height sampling (fast path)
    if hasattr(height_provider, "height_grid"):
        h = height_provider.height_grid(grid_x, grid_z).astype(np.float32)
    else:
        # fallback: callable height_fn(x,z)
        height_fn = height_provider
        h = np.zeros((res, res), dtype=np.float32)
        for j in range(res):
            for i in range(res):
                h[j, i] = float(height_fn(float(grid_x[j, i]), float(grid_z[j, i])))

    # Central differences for normals (vectorized)
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
