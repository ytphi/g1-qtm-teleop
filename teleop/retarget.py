import numpy as np
from scipy.spatial.transform import Rotation


# G1 U2 joint order as exposed by unitree_sdk2 low-level API (29 joints total)
# Source: unitree_sdk2_python/example/g1/low_level/g1_loco_client.py
G1_JOINT_NAMES = [
    # Left leg
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    # Right leg
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    # Waist
    "waist_yaw", "waist_roll", "waist_pitch",
    # Left arm
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    # Right arm
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

# G1 joint limits (radians) — from Unitree G1 URDF
# Format: (lower, upper)
G1_JOINT_LIMITS = {
    "left_hip_pitch":     (-2.87, 2.87),
    "left_hip_roll":      (-0.52, 2.53),
    "left_hip_yaw":       (-2.44, 2.44),
    "left_knee":          (-0.26, 2.53),
    "left_ankle_pitch":   (-0.87, 0.52),
    "left_ankle_roll":    (-0.26, 0.26),
    "right_hip_pitch":    (-2.87, 2.87),
    "right_hip_roll":     (-2.53, 0.52),
    "right_hip_yaw":      (-2.44, 2.44),
    "right_knee":         (-0.26, 2.53),
    "right_ankle_pitch":  (-0.87, 0.52),
    "right_ankle_roll":   (-0.26, 0.26),
    "waist_yaw":          (-2.62, 2.62),
    "waist_roll":         (-0.52, 0.52),
    "waist_pitch":        (-0.52, 0.52),
    "left_shoulder_pitch": (-3.14, 3.14),
    "left_shoulder_roll":  (-1.57, 2.87),
    "left_shoulder_yaw":   (-2.87, 2.87),
    "left_elbow":          (-1.57, 2.87),
    "left_wrist_roll":     (-1.57, 1.57),
    "left_wrist_pitch":    (-1.57, 1.57),
    "left_wrist_yaw":      (-1.57, 1.57),
    "right_shoulder_pitch": (-3.14, 3.14),
    "right_shoulder_roll":  (-2.87, 1.57),
    "right_shoulder_yaw":   (-2.87, 2.87),
    "right_elbow":          (-1.57, 2.87),
    "right_wrist_roll":     (-1.57, 1.57),
    "right_wrist_pitch":    (-1.57, 1.57),
    "right_wrist_yaw":      (-1.57, 1.57),
}


def clamp(value: float, joint: str) -> float:
    lo, hi = G1_JOINT_LIMITS[joint]
    return float(np.clip(value, lo, hi))


class G1Retargeter:
    """
    Maps QTM skeleton segments to G1 joint angles.

    This is a geometric retargeter — it extracts joint angles from
    relative rotations between adjacent segments. It runs frame-by-frame
    with no learned model, so it's suitable for real-time use.

    For higher quality (reduced self-collision, smoother output), pipe the
    recorded episodes through GMR or OmniRetarget offline.
    """

    def __init__(self):
        self._t_pose: dict | None = None  # calibration frame

    def calibrate(self, segments: dict) -> None:
        """Call once with the operator copying the G1 zero pose (arms down) — this frame maps to q = 0."""
        self._t_pose = {k: v.rotation.copy() for k, v in segments.items()}

    def retarget(self, segments: dict) -> np.ndarray:
        """
        Convert QTM segment dict to G1 joint angle array (29,).
        Returns zeros for joints that cannot be computed from available segments.
        """
        q = np.zeros(len(G1_JOINT_NAMES))

        def get_rot(name: str) -> Rotation:
            r = Rotation.from_quat(segments[name].rotation)
            if self._t_pose and name in self._t_pose:
                r = Rotation.from_quat(self._t_pose[name]).inv() * r
            return r

        def rel(parent_name: str, child_name: str) -> np.ndarray:
            if parent_name not in segments or child_name not in segments:
                return np.zeros(3)
            r_parent = get_rot(parent_name)
            r_child  = get_rot(child_name)
            return (r_parent.inv() * r_child).as_euler("xyz")

        # --- Waist ---
        waist = rel("Hips", "Spine")
        q[12] = clamp(waist[1], "waist_yaw")
        q[13] = clamp(waist[2], "waist_roll")
        q[14] = clamp(waist[0], "waist_pitch")

        # --- Left arm ---
        l_sh = rel("LeftShoulder", "LeftArm")
        q[15] = clamp(l_sh[0], "left_shoulder_pitch")
        q[16] = clamp(l_sh[2], "left_shoulder_roll")
        q[17] = clamp(l_sh[1], "left_shoulder_yaw")
        l_el = rel("LeftArm", "LeftForeArm")
        q[18] = clamp(l_el[0], "left_elbow")
        l_wr = rel("LeftForeArm", "LeftHand")
        q[19] = clamp(l_wr[0], "left_wrist_roll")
        q[20] = clamp(l_wr[1], "left_wrist_pitch")
        q[21] = clamp(l_wr[2], "left_wrist_yaw")

        # --- Right arm ---
        r_sh = rel("RightShoulder", "RightArm")
        q[22] = clamp(r_sh[0], "right_shoulder_pitch")
        q[23] = clamp(r_sh[2], "right_shoulder_roll")
        q[24] = clamp(r_sh[1], "right_shoulder_yaw")
        r_el = rel("RightArm", "RightForeArm")
        q[25] = clamp(r_el[0], "right_elbow")
        r_wr = rel("RightForeArm", "RightHand")
        q[26] = clamp(r_wr[0], "right_wrist_roll")
        q[27] = clamp(r_wr[1], "right_wrist_pitch")
        q[28] = clamp(r_wr[2], "right_wrist_yaw")

        # --- Left leg ---
        l_hp = rel("Hips", "LeftUpLeg")
        q[0] = clamp(l_hp[0], "left_hip_pitch")
        q[1] = clamp(l_hp[2], "left_hip_roll")
        q[2] = clamp(l_hp[1], "left_hip_yaw")
        l_kn = rel("LeftUpLeg", "LeftLeg")
        q[3] = clamp(l_kn[0], "left_knee")
        l_an = rel("LeftLeg", "LeftFoot")
        q[4] = clamp(l_an[0], "left_ankle_pitch")
        q[5] = clamp(l_an[2], "left_ankle_roll")

        # --- Right leg ---
        r_hp = rel("Hips", "RightUpLeg")
        q[6]  = clamp(r_hp[0], "right_hip_pitch")
        q[7]  = clamp(r_hp[2], "right_hip_roll")
        q[8]  = clamp(r_hp[1], "right_hip_yaw")
        r_kn = rel("RightUpLeg", "RightLeg")
        q[9]  = clamp(r_kn[0], "right_knee")
        r_an = rel("RightLeg", "RightFoot")
        q[10] = clamp(r_an[0], "right_ankle_pitch")
        q[11] = clamp(r_an[2], "right_ankle_roll")

        return q
