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


class CameraFlight:
    """Manual flight camera.

    - Moves in the XZ plane based on yaw and forward/back input.
    - Height follows terrain with smoothing.
    - View looks ahead along the forward direction.

    Coordinate conventions:
    - +Z is "forward" when yaw == 0.
    - User expectation: RIGHT arrow yaws the view to the right.

    Note:
    In our current world/view setup, the effective X-axis handedness in camera
    space results in "turn" needing to be applied with a negative sign to match
    intuitive left/right input.
    """

    def __init__(self, speed: float, turn_rate: float, height_offset: float, smooth_k: float) -> None:
        self.speed = float(speed)
        self.turn_rate = float(turn_rate)
        self.height_offset = float(height_offset)
        self.smooth_k = float(smooth_k)

        self.x = 0.0
        self.z = 0.0
        self.y = height_offset
        self.yaw = 0.0

        # View tuning
        self.look_ahead = 60.0
        self.look_down = 8.0

    def _forward(self) -> np.ndarray:
        # yaw==0 -> +Z
        s = float(np.sin(self.yaw))
        c = float(np.cos(self.yaw))
        return np.array([s, 0.0, c], dtype=np.float32)

    def update(self, dt: float, height_fn, *, forward: float, turn: float) -> None:
        """Update camera.

        Args:
            forward: -1..1 (back..forward)
            turn: -1..1 (left..right)
        """

        # Apply rotation first.
        # turn: -1..1 (left..right). We negate to match intuitive screen-space.
        self.yaw -= float(turn) * self.turn_rate * dt

        # Move in XZ.
        v = float(forward) * self.speed
        if v != 0.0:
            fwd = self._forward()
            self.x += float(fwd[0]) * v * dt
            self.z += float(fwd[2]) * v * dt

        # Height follow.
        y_target = float(height_fn(self.x, self.z)) + self.height_offset
        self.y = exp_smooth(self.y, y_target, self.smooth_k, dt)

    def eye(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def view_matrix(self) -> np.ndarray:
        eye = self.eye()
        fwd = self._forward()
        target = eye + fwd * np.float32(self.look_ahead)
        target = np.array([float(target[0]), float(target[1]) - self.look_down, float(target[2])], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return look_at(eye, target, up)
