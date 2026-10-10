"""
Live MuJoCo viewer for the teleop loop.
Runs the viewer in a background thread so it doesn't block the control loop.

Usage in teleop_session.py:
    from teleop.live_viewer import LiveViewer
    viewer = LiveViewer()
    viewer.start()
    ...
    viewer.update(joint_angles)   # call each control tick
    ...
    viewer.stop()
"""

import sys
import threading
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "GMR"))   # cloned + patched by setup.sh

# g1_mocap_29dof.xml joint order == G1 motor order (verified 2026-09-28), so no reorder.
# (The old _G1_TO_MJORDER table swapped right leg / waist and hid the same bug in gmr_retarget.py.)

_ROOT_POS = np.array([0.0, 0.0, 0.793])   # G1 pelvis standing height
_ROOT_ROT = np.array([1.0, 0.0, 0.0, 0.0])  # identity (w,x,y,z)


class LiveViewer:
    """Thread-safe live MuJoCo viewer. update() is non-blocking."""

    def __init__(self, fps: float = 50.0):
        self._fps = fps
        self._lock = threading.Lock()
        self._dof_mj = np.zeros(29)
        self._running = False
        self._thread = None
        self._viewer = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, joint_angles: np.ndarray) -> None:
        """Call each control tick with G1_JOINT_NAMES-ordered joint angles (29,)."""
        dof_mj = np.asarray(joint_angles, dtype=float).copy()
        with self._lock:
            self._dof_mj = dof_mj

    def stop(self) -> None:
        self._running = False
        if self._viewer is not None:
            self._viewer.close()

    def _run(self) -> None:
        from general_motion_retargeting import RobotMotionViewer
        self._viewer = RobotMotionViewer(
            robot_type="unitree_g1",
            motion_fps=self._fps,
            camera_follow=False,
            record_video=False,
        )
        while self._running:
            with self._lock:
                dof = self._dof_mj.copy()
            self._viewer.step(_ROOT_POS, _ROOT_ROT, dof, rate_limit=True)
        self._viewer.close()
