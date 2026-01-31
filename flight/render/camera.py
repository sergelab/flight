from __future__ import annotations

import numpy as np

from flight.util.math import exp_smooth, look_at


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _rotate_around_axis(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector v around unit axis by angle (radians)."""
    # Rodrigues' rotation formula.
    a = axis / (np.linalg.norm(axis) + 1e-8)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return v * c + np.cross(a, v) * s + a * float(np.dot(a, v)) * (1.0 - c)


def _wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    a = float(angle)
    a = (a + float(np.pi)) % (2.0 * float(np.pi)) - float(np.pi)
    return a


def exp_smooth_angle(current: float, target: float, k: float, dt: float) -> float:
    """Exponentially smooth an angle, taking wrapping into account."""
    cur = float(current)
    tgt = float(target)
    d = _wrap_pi(tgt - cur)
    return _wrap_pi(cur + (d * (1.0 - float(np.exp(-float(k) * float(dt))))))


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

    def __init__(
        self,
        *,
        max_speed: float,
        accel: float,
        brake: float,
        drag: float,
        max_yaw_rate_slow: float,
        max_yaw_rate_fast: float,
        yaw_accel: float,
        yaw_decel: float,
        yaw_drag: float,
        bank_gain: float,
        bank_max: float,
        bank_smooth_k: float,
        input_smooth_k: float,
        cam_yaw_smooth_k: float,
        pitch_gain: float,
        pitch_max: float,
        pitch_smooth_k: float,
        height_offset: float,
        smooth_k: float,
    ) -> None:
        # Linear
        self.max_speed = float(max_speed)
        self.accel = float(accel)
        self.brake = float(brake)
        self.drag = float(drag)

        # Angular
        self.max_yaw_rate_slow = float(max_yaw_rate_slow)
        self.max_yaw_rate_fast = float(max_yaw_rate_fast)
        self.yaw_accel = float(yaw_accel)
        self.yaw_decel = float(yaw_decel)
        self.yaw_drag = float(yaw_drag)

        # Camera reactions
        self.bank_gain = float(bank_gain)
        self.bank_max = float(bank_max)
        self.bank_smooth_k = float(bank_smooth_k)

        # Input + view smoothing
        self.input_smooth_k = float(input_smooth_k)
        self.cam_yaw_smooth_k = float(cam_yaw_smooth_k)
        self.pitch_gain = float(pitch_gain)
        self.pitch_max = float(pitch_max)
        self.pitch_smooth_k = float(pitch_smooth_k)

        # Terrain follow
        self.height_offset = float(height_offset)
        self.smooth_k = float(smooth_k)

        # State
        self.x = 0.0
        self.z = 0.0
        self.y = float(height_offset)
        self.yaw = 0.0
        self.speed = 0.0
        self.yaw_rate = 0.0
        self.bank = 0.0

        # Smoothed input and view state
        self._throttle = 0.0
        self._turn = 0.0
        self._cam_yaw = 0.0
        self._pitch = 0.0
        self._prev_speed = 0.0

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

        dt = float(dt)
        forward = _clamp(float(forward), -1.0, 1.0)
        turn = _clamp(float(turn), -1.0, 1.0)

        # Smooth digital inputs so arrow keys don't feel like on/off switches.
        # This is critical for perceiving inertia even with modest accel limits.
        self._throttle = exp_smooth(self._throttle, forward, self.input_smooth_k, dt)
        self._turn = exp_smooth(self._turn, turn, self.input_smooth_k, dt)

        # --- Linear: speed follows throttle with accel/brake and drag.
        speed_target = self._throttle * self.max_speed
        delta = speed_target - self.speed
        limit = self.accel if abs(speed_target) > abs(self.speed) else self.brake
        step = _clamp(delta, -limit * dt, limit * dt)
        self.speed += step
        if self.drag > 0.0:
            self.speed *= float(np.exp(-self.drag * dt))

        # --- Angular: yaw_rate follows turn with accel/decel and drag.
        speed_norm = 0.0 if self.max_speed <= 1e-6 else _clamp(abs(self.speed) / self.max_speed, 0.0, 1.0)
        yaw_rate_cap = self.max_yaw_rate_slow + (self.max_yaw_rate_fast - self.max_yaw_rate_slow) * speed_norm

        # turn: -1..1 (left..right). We negate to match intuitive screen-space.
        yaw_rate_target = -self._turn * yaw_rate_cap
        a = self.yaw_accel if abs(yaw_rate_target) > abs(self.yaw_rate) else self.yaw_decel
        delta_r = yaw_rate_target - self.yaw_rate
        self.yaw_rate += _clamp(delta_r, -a * dt, a * dt)
        if self.yaw_drag > 0.0:
            self.yaw_rate *= float(np.exp(-self.yaw_drag * dt))

        self.yaw += self.yaw_rate * dt

        # View yaw lags behind physics yaw a bit ("weight" in turns).
        self._cam_yaw = exp_smooth_angle(self._cam_yaw, self.yaw, self.cam_yaw_smooth_k, dt)

        # --- Move in XZ.
        if self.speed != 0.0:
            fwd = self._forward()
            self.x += float(fwd[0]) * self.speed * dt
            self.z += float(fwd[2]) * self.speed * dt

        # --- Camera reaction: bank into turn.
        # NOTE: yaw_rate already includes a sign flip to match intuitive left/right input.
        # Bank should visually lean *into* the perceived turn direction, so we flip the sign here.
        bank_target = _clamp(-self.yaw_rate * self.bank_gain * speed_norm, -self.bank_max, self.bank_max)
        self.bank = exp_smooth(self.bank, bank_target, self.bank_smooth_k, dt)

        # Visual cue: pitch on longitudinal acceleration.
        if dt > 1e-6:
            long_accel = (self.speed - self._prev_speed) / dt
        else:
            long_accel = 0.0
        self._prev_speed = self.speed

        # Positive accel -> pitch up (reduce look_down), braking -> pitch down.
        pitch_target = _clamp(-long_accel * self.pitch_gain, -self.pitch_max, self.pitch_max)
        self._pitch = exp_smooth(self._pitch, pitch_target, self.pitch_smooth_k, dt)

        # Height follow.
        y_target = float(height_fn(self.x, self.z)) + self.height_offset
        self.y = exp_smooth(self.y, y_target, self.smooth_k, dt)

    def eye(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def view_matrix(self) -> np.ndarray:
        eye = self.eye()

        # Use a slightly lagged yaw for view to enhance "weight".
        s = float(np.sin(self._cam_yaw))
        c = float(np.cos(self._cam_yaw))
        fwd = np.array([s, 0.0, c], dtype=np.float32)

        target = eye + fwd * np.float32(self.look_ahead)
        look_down = float(self.look_down) + float(self._pitch)
        target = np.array([float(target[0]), float(target[1]) - look_down, float(target[2])], dtype=np.float32)
        up0 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        up = _rotate_around_axis(up0, fwd, float(self.bank)).astype(np.float32)
        return look_at(eye, target, up)
