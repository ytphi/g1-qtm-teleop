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

    def __init__(self, fps: float = 50.0, camera_follow: bool = False, ground_feet: bool = False):
        self._fps = fps
        self._camera_follow = camera_follow
        self._ground_feet = ground_feet   # shift the pelvis up/down so the lower sole touches z=0
        self._lock = threading.Lock()
        self._dof_mj = np.zeros(29)
        self._root_pos, self._root_rot = _ROOT_POS, _ROOT_ROT
        self._running = False
        self._thread = None
        self._viewer = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, joint_angles: np.ndarray, root_pos=None, root_rot=None) -> None:
        """Call each control tick with G1_JOINT_NAMES-ordered joint angles (29,).
        root_pos/root_rot (w,x,y,z) move the pelvis; omitted → pelvis fixed, like the gantry."""
        dof_mj = np.asarray(joint_angles, dtype=float).copy()
        with self._lock:
            self._dof_mj = dof_mj
            if root_pos is not None and root_rot is not None:
                self._root_pos, self._root_rot = np.array(root_pos), np.array(root_rot)

    def stop(self) -> None:
        self._running = False
        if self._viewer is not None:
            self._viewer.close()

    def _run(self) -> None:
        from general_motion_retargeting import RobotMotionViewer
        self._viewer = RobotMotionViewer(
            robot_type="unitree_g1",
            motion_fps=self._fps,
            camera_follow=self._camera_follow,
            record_video=False,
        )
        if self._ground_feet:
            import mujoco as mj
            model = self._viewer.model
            scratch = mj.MjData(model)
            # the 4 sole-corner contact spheres of each foot
            soles = [g for g in range(model.ngeom)
                     if model.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE
                     and model.body(model.geom_bodyid[g]).name in ("left_ankle_roll_link", "right_ankle_roll_link")]
            sole_radius = model.geom_size[soles, 0]
        while self._running:
            with self._lock:
                dof = self._dof_mj.copy()
                root_pos, root_rot = self._root_pos.copy(), self._root_rot
            if self._ground_feet:
                scratch.qpos[:3], scratch.qpos[3:7], scratch.qpos[7:] = root_pos, root_rot, dof
                mj.mj_kinematics(model, scratch)
                root_pos[2] -= np.min(scratch.geom_xpos[soles, 2] - sole_radius)
            self._viewer.step(root_pos, root_rot, dof, rate_limit=True)
        self._viewer.close()
