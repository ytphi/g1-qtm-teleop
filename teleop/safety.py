"""
Safety filter between the retargeter and the robot.

Every command goes through SafetyFilter before rt/lowcmd:
    1. non-finite (NaN / inf) targets are rejected → hold last command
    2. clamp to G1 joint limits (minus a small margin)
    3. rate-limit: each joint moves at most max_vel * dt per tick
       (this also gives a smooth start after calibration, instead of a jump)

Usage:
    safety = SafetyFilter(dt=0.01, max_vel=1.0)
    safety.reset(q_start)
    cmd = safety(target)         # target may be None → hold
    robot.set_joint_positions(cmd)
"""

import numpy as np

from teleop.retarget import G1_JOINT_LIMITS, G1_JOINT_NAMES

_LIMITS = np.array([G1_JOINT_LIMITS[n] for n in G1_JOINT_NAMES])  # (29, 2) motor order


class SafetyFilter:
    def __init__(self, dt: float, max_vel: float = 1.0, limit_margin: float = 0.05):
        """
        dt           control period (s)
        max_vel      max joint speed (rad/s) for every joint
        limit_margin stay this far (rad) inside the URDF joint limits
        """
        self.max_step = max_vel * dt
        self.lo = _LIMITS[:, 0] + limit_margin
        self.hi = _LIMITS[:, 1] - limit_margin
        self.q = None
        self.rejected = 0       # count of non-finite targets
        self.rate_limited = False

    def reset(self, q: np.ndarray) -> None:
        """Set the last commanded position (e.g. after move_to_zero)."""
        self.q = np.clip(np.asarray(q, dtype=float), self.lo, self.hi)

    def __call__(self, target: np.ndarray | None) -> np.ndarray:
        assert self.q is not None, "call reset() first"
        if target is None:
            return self.q.copy()
        target = np.asarray(target, dtype=float)
        if target.shape != self.q.shape or not np.all(np.isfinite(target)):
            self.rejected += 1
            return self.q.copy()

        target = np.clip(target, self.lo, self.hi)
        step = target - self.q
        self.rate_limited = bool(np.any(np.abs(step) > self.max_step))
        self.q = self.q + np.clip(step, -self.max_step, self.max_step)
        return self.q.copy()

    def ramp_to(self, target: np.ndarray, tol: float = 1e-3) -> bool:
        """One rate-limited step toward target; True once reached."""
        self(target)
        return bool(np.all(np.abs(self.q - np.clip(target, self.lo, self.hi)) < tol))
