from __future__ import annotations

import numpy as np

from flight.util.math import exp_smooth, look_at

class CameraRail:
    """Camera moves forward along +Z with constant speed.
    No rotation controls. Height follows terrain with smoothing.
    """

    def __init__(self, speed: float, height_offset: float, smooth_k: float) -> None:
        self.speed = float(speed)
        self.height_offset = float(height_offset)
        self.smooth_k = float(smooth_k)

        self.x = 0.0
        self.z = 0.0
        self.y = height_offset

        # View tuning
        self.look_ahead = 60.0
        self.look_down = 8.0

    def update(self, dt: float, height_fn) -> None:
        self.z += self.speed * dt

        y_target = float(height_fn(self.x, self.z)) + self.height_offset
        self.y = exp_smooth(self.y, y_target, self.smooth_k, dt)

    def eye(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def view_matrix(self) -> np.ndarray:
        eye = self.eye()
        target = np.array([self.x, self.y - self.look_down, self.z + self.look_ahead], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return look_at(eye, target, up)


class CameraYaw:
    """Camera that can rotate around Y (yaw) and move forward/back.

    Coordinate system:
    - yaw=0 looks towards +Z (same as the old rail camera)
    - yaw increases when turning right
    """

    def __init__(self, speed: float, height_offset: float, smooth_k: float, turn_rate: float) -> None:
        self.speed = float(speed)
        self.height_offset = float(height_offset)
        self.smooth_k = float(smooth_k)
        self.turn_rate = float(turn_rate)

        self.x = 0.0
        self.z = 0.0
        self.y = height_offset

        self.yaw = 0.0

        # View tuning
        self.look_ahead = 60.0
        self.look_down = 8.0

    def update(self, dt: float, height_fn, *, move_axis: float, turn_axis: float) -> None:
        # Turn
        self.yaw += float(turn_axis) * self.turn_rate * dt

        # Move
        if move_axis != 0.0:
            fx = float(np.sin(self.yaw))
            fz = float(np.cos(self.yaw))
            self.x += fx * self.speed * dt * float(move_axis)
            self.z += fz * self.speed * dt * float(move_axis)

        # Follow terrain height
        y_target = float(height_fn(self.x, self.z)) + self.height_offset
        self.y = exp_smooth(self.y, y_target, self.smooth_k, dt)

    def eye(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def view_matrix(self) -> np.ndarray:
        eye = self.eye()
        fx = float(np.sin(self.yaw))
        fz = float(np.cos(self.yaw))
        target = np.array(
            [
                self.x + fx * self.look_ahead,
                self.y - self.look_down,
                self.z + fz * self.look_ahead,
            ],
            dtype=np.float32,
        )
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return look_at(eye, target, up)
