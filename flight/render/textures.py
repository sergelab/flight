from __future__ import annotations

import numpy as np
import moderngl


def _soft_noise(h: int, w: int, *, seed: int, octaves: int = 4) -> np.ndarray:
    """Low-frequency noise suitable for tiling textures.

    We intentionally keep frequency low and rely on mipmaps for stability.
    """
    rng = np.random.default_rng(int(seed))
    img = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    total = 0.0
    for o in range(int(octaves)):
        step = max(1, 2 ** (o + 2))  # low frequency
        gh = max(2, h // step)
        gw = max(2, w // step)
        grid = rng.random((gh + 1, gw + 1), dtype=np.float32)

        # Bilinear upsample
        yy = np.linspace(0.0, gh, h, endpoint=False)
        xx = np.linspace(0.0, gw, w, endpoint=False)
        y0 = np.floor(yy).astype(np.int32)
        x0 = np.floor(xx).astype(np.int32)
        y1 = np.minimum(y0 + 1, gh)
        x1 = np.minimum(x0 + 1, gw)
        fy = (yy - y0).astype(np.float32)
        fx = (xx - x0).astype(np.float32)

        g00 = grid[y0[:, None], x0[None, :]]
        g10 = grid[y1[:, None], x0[None, :]]
        g01 = grid[y0[:, None], x1[None, :]]
        g11 = grid[y1[:, None], x1[None, :]]

        a = g00 * (1.0 - fy)[:, None] + g10 * fy[:, None]
        b = g01 * (1.0 - fy)[:, None] + g11 * fy[:, None]
        up = a * (1.0 - fx)[None, :] + b * fx[None, :]

        img += up * amp
        total += amp
        amp *= 0.55

    img /= max(1e-6, total)
    return img


def _to_rgba(col: np.ndarray) -> bytes:
    col8 = np.clip(col * 255.0, 0, 255).astype(np.uint8)
    return col8.tobytes(order="C")


def build_terrain_textures(ctx: moderngl.Context, *, seed: int, size: int = 256) -> tuple[moderngl.Texture, moderngl.Texture]:
    """Create two small, stable tile textures (grass/rock) with mipmaps."""
    s = int(size)

    n1 = _soft_noise(s, s, seed=seed + 101, octaves=4)
    n2 = _soft_noise(s, s, seed=seed + 202, octaves=4)

    # Grass: mostly green with subtle variation
    grass = np.zeros((s, s, 4), dtype=np.float32)
    g = 0.55 + 0.25 * (n1 - 0.5)
    grass[..., 0] = 0.12 + 0.08 * (n2 - 0.5)
    grass[..., 1] = g
    grass[..., 2] = 0.14 + 0.10 * (n1 - 0.5)
    grass[..., 3] = 1.0

    # Rock: gray-ish with slightly higher contrast
    rock = np.zeros((s, s, 4), dtype=np.float32)
    r = 0.55 + 0.35 * (n2 - 0.5)
    rock[..., 0] = r
    rock[..., 1] = r
    rock[..., 2] = r + 0.03
    rock[..., 3] = 1.0

    t_grass = ctx.texture((s, s), 4, data=_to_rgba(grass))
    t_rock = ctx.texture((s, s), 4, data=_to_rgba(rock))
    for t in (t_grass, t_rock):
        t.repeat_x = True
        t.repeat_y = True
        # Trilinear-ish filtering, stability via mipmaps
        t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        try:
            t.build_mipmaps()
        except Exception:
            # Some drivers may fail; still usable with base level
            pass
        try:
            # Optional anisotropy; safe to ignore if unsupported
            t.anisotropy = 4.0
        except Exception:
            pass

    return t_grass, t_rock
