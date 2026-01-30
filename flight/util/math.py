from __future__ import annotations
import numpy as np

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return a column-major view matrix suitable for OpenGL."""
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0] = s[0]; m[1, 0] = s[1]; m[2, 0] = s[2]
    m[0, 1] = u[0]; m[1, 1] = u[1]; m[2, 1] = u[2]
    m[0, 2] = -f[0]; m[1, 2] = -f[1]; m[2, 2] = -f[2]
    m[3, 0] = -np.dot(s, eye)
    m[3, 1] = -np.dot(u, eye)
    m[3, 2] = np.dot(f, eye)
    return m

def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Return a column-major projection matrix suitable for OpenGL."""
    f = 1.0 / np.tan(np.deg2rad(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = -1.0
    m[3, 2] = (2.0 * far * near) / (near - far)
    return m

def exp_smooth(current: float, target: float, k: float, dt: float) -> float:
    alpha = 1.0 - float(np.exp(-k * dt))
    return current + (target - current) * alpha
