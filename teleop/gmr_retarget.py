"""
QTM → G1 retargeter using YanjieZe/GMR (IK-based, mink + MuJoCo).

Uses GMR (cloned into GMR/ and patched by setup.sh) with the `qtm_rt` IK config
(QTM segment names: Hips, Spine2, LeftArm, LeftForeArm, ...).

Usage:
    retargeter = GMRRetargeter()
    retargeter.setup()
    q = retargeter.retarget(qtm_stream.latest)  # → (29,) joint angles, motor order
"""

import sys
import numpy as np
from pathlib import Path

from teleop.retarget import G1_JOINT_NAMES

GMR_ROOT = Path(__file__).resolve().parent.parent / "GMR"   # cloned + patched by setup.sh
SRC_HUMAN = "qtm_rt"       # ik_configs/qtm_rt_to_g1.json (live QTM RT; qtm_bvh offsets are for BVH files)
TGT_ROBOT = "unitree_g1"   # assets/unitree_g1/g1_mocap_29dof.xml


class GMRRetargeter:
    """
    Wraps GeneralMotionRetargeting for live QTM streaming.
    Call setup() once, then retarget() every frame.
    """

    def __init__(self, human_height: float | None = None):
        self._retargeter = None
        self._human_height = human_height
        self._dof_index = None   # qpos index for each motor, in G1_JOINT_NAMES order
        self.root_pos = None     # last pelvis position (3,) m; the real robot can't follow it
        self.root_rot = None     # last pelvis orientation (w, x, y, z)

    def setup(self) -> None:
        if not GMR_ROOT.exists():
            raise RuntimeError(f"GMR repo not found at {GMR_ROOT}")

        sys.path.insert(0, str(GMR_ROOT))
        from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting

        self._retargeter = GeneralMotionRetargeting(
            src_human=SRC_HUMAN,
            tgt_robot=TGT_ROBOT,
            actual_human_height=self._human_height,
            verbose=False,
        )
        # Map by joint name — never assume the XML joint order matches the motor order
        model = self._retargeter.model
        self._dof_index = np.array([
            model.jnt_qposadr[model.joint(f"{name}_joint").id] for name in G1_JOINT_NAMES
        ])
        print(f"GMRRetargeter ready  src={SRC_HUMAN}  robot={TGT_ROBOT}")

    def retarget(self, segments: dict) -> np.ndarray:
        """
        Convert QTM segment dict → G1 joint angles (29,), in G1_JOINT_NAMES (motor) order.

        segments: dict of {name: SkeletonSegment} from QTMStream.latest (global frame)
        """
        assert self._retargeter is not None, "Call setup() first"

        human_data = {}
        for name, seg in segments.items():
            pos_m = seg.position / 1000.0                  # mm → m
            x, y, z, w = seg.rotation                      # QTM (x, y, z, w)
            human_data[name] = [pos_m, np.array([w, x, y, z])]
        # GMR's foot target: foot position + toe orientation (same as its QTM BVH loader)
        for side in ("Left", "Right"):
            if f"{side}Foot" in human_data and f"{side}ToeBase" in human_data:
                human_data[f"{side}FootMod"] = [human_data[f"{side}Foot"][0],
                                                human_data[f"{side}ToeBase"][1]]

        qpos = np.asarray(self._retargeter.retarget(human_data, offset_to_ground=False))
        self.root_pos, self.root_rot = qpos[:3].copy(), qpos[3:7].copy()
        return qpos[self._dof_index]
