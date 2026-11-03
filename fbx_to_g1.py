"""
Qualisys FBX → Unitree G1 joint angles pipeline.

Steps performed automatically:
  1. Run Blender headless to extract per-frame bone world transforms from FBX
  2. Strip actor namespace prefix (e.g. "Joseph:Hips" → "Hips")
  3. Auto-correct initial facing direction so the robot always starts forward
  4. Retarget each frame → G1 joint angles using GMR IK (scaled by --height)
  5. Ramp out single-frame IK jumps (--no-clean to skip)
  6. Shift the root so the G1's soles touch the floor (--no-ground to skip)
  7. Save as .pkl, with the FBX's real frame rate, and optionally visualize in MuJoCo

Earlier versions always wrote fps=50 to the .pkl. QTM exports at the capture rate
(usually 100 Hz), so those files play at half speed downstream: re-convert them.

Usage:
    # Retarget a single FBX file and visualize
    python fbx_to_g1.py recording.fbx

    # Retarget a whole folder and save output
    python fbx_to_g1.py /path/to/fbx_folder --out-dir /path/to/output

    # Retarget only, no visualization (good for batch jobs)
    python fbx_to_g1.py /path/to/fbx_folder --out-dir /path/to/output --no-viz

    # Loop visualization (plays the motion on repeat)
    python fbx_to_g1.py recording.fbx --loop
"""

import argparse
import os
import subprocess
import sys
import tempfile
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

try:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting import RobotMotionViewer
except ImportError:
    print("\n[ERROR] GMR library not found.")
    print("Run `bash setup.sh` first. It clones GMR and applies the QTM patch")
    print("(gmr_patch/) that the fbx_qtm IK config needs.")
    sys.exit(1)

# --- Blender script: extracts per-frame world-space bone transforms from FBX ---
_BLENDER_EXTRACT_SCRIPT = '''
import bpy, numpy as np, sys

fbx_path = "{fbx_path}"
out_path  = "{out_path}"

bpy.ops.import_scene.fbx(filepath=fbx_path)

arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
if not arms:
    print("ERROR: no armature found in FBX"); sys.exit(1)

# QTM exports a rigid body (e.g. the chair, 6DOF) as its own one-bone armature next to the
# actor's skeleton. The actor is the armature with a Hips bone (else: the most bones).
def has_hips(o): return any(b.name.split(":")[-1] == "Hips" for b in o.data.bones)
arm_obj = max(arms, key=lambda o: (has_hips(o), len(o.data.bones)))
props   = [o for o in arms if o is not arm_obj]
print(f"Actor armature: {{arm_obj.name}}; rigid bodies: {{[o.name for o in props]}}")

bpy.context.view_layer.objects.active = arm_obj
scene  = bpy.context.scene
bone_names = [b.name for b in arm_obj.data.bones]

# Use the action's keyframe range: the scene's start/end defaults (1-250) are unrelated
# to the clip and are too short or too long for most QTM exports.
action = arm_obj.animation_data.action if arm_obj.animation_data else None
if action:
    frame_start = int(action.frame_range[0])
    frame_end   = int(action.frame_range[1])
    print(f"Using action frame range: {{frame_start}}–{{frame_end}} ({{frame_end - frame_start + 1}} frames)")
else:
    frame_start = scene.frame_start
    frame_end   = scene.frame_end
    print(f"Using scene frame range: {{frame_start}}–{{frame_end}}")

positions    = []
orientations = []
prop_pos = {{o.name.split(":")[0]: [] for o in props}}
prop_rot = {{o.name.split(":")[0]: [] for o in props}}

for f in range(frame_start, frame_end + 1):
    scene.frame_set(f)
    pos_row, rot_row = [], []
    for bname in bone_names:
        pbone = arm_obj.pose.bones[bname]
        mat   = arm_obj.matrix_world @ pbone.matrix  # world-space transform
        loc   = mat.translation                        # metres
        quat  = mat.to_quaternion()                   # w, x, y, z
        pos_row.append([loc.x, loc.y, loc.z])
        rot_row.append([quat.w, quat.x, quat.y, quat.z])
    positions.append(pos_row)
    orientations.append(rot_row)
    for o in props:                                  # root bone of each rigid body
        mat = o.matrix_world @ o.pose.bones[0].matrix
        q = mat.to_quaternion()
        prop_pos[o.name.split(":")[0]].append(list(mat.translation))
        prop_rot[o.name.split(":")[0]].append([q.w, q.x, q.y, q.z])

# The FBX importer sets the scene frame rate from the file (QTM exports at the capture rate,
# typically 100 Hz). Without it, frame counts can't be converted to seconds.
fps = scene.render.fps / scene.render.fps_base

np.savez(out_path,
         positions=np.array(positions,    dtype=np.float32),  # (T, B, 3)
         orientations=np.array(orientations, dtype=np.float32),  # (T, B, 4) wxyz
         bone_names=np.array(bone_names),
         fps=np.float64(fps),
         prop_names=np.array(list(prop_pos)),
         **{{"prop_pos_" + k: np.array(v, dtype=np.float32) for k, v in prop_pos.items()}},
         **{{"prop_rot_" + k: np.array(v, dtype=np.float32) for k, v in prop_rot.items()}})
print(f"Saved {{len(positions)}} frames at {{fps:g}} fps, {{len(bone_names)}} bones → {{out_path}}")
'''


def fbx_to_npz(fbx_path: str, npz_path: str) -> bool:
    """Run Blender headless to extract bone transforms from an FBX file."""
    script = _BLENDER_EXTRACT_SCRIPT.format(
        fbx_path=fbx_path.replace("\\", "/"),
        out_path=npz_path.replace("\\", "/"),
    )
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script)
        script_path = f.name
    result = subprocess.run(
        ["blender", "--background", "--python", script_path],
        capture_output=True, text=True,
    )
    os.unlink(script_path)
    if result.returncode != 0:
        print(f"  [Blender stderr] {result.stderr[-500:]}")
    actual = npz_path + ".npz" if not npz_path.endswith(".npz") else npz_path
    return os.path.exists(actual)


def load_npz_as_human_data(npz_path: str, start: float = 0.0, end: float | None = None,
                           fps_override: float | None = None):
    """
    Load Blender-extracted transforms and convert to GMR human_data format.

    - Trims to [start, end) seconds first, so the facing correction uses the first kept frame
      (not a T-pose at the start of the take)
    - Strips actor namespace prefix (e.g. "Joseph:Hips" → "Hips")
    - Auto-corrects initial facing direction (yaw normalisation)
    - Converts Blender Z-up to MuJoCo Z-up (coordinate flip -Y→+X)
    - Adds LeftFootMod / RightFootMod synthetic bones for GMR IK config
    - Rigid bodies (e.g. the chair) get the same yaw + axis change: {name: (pos, quat_wxyz)}
    """
    data = np.load(npz_path, allow_pickle=True)
    fps          = fps_override or (float(data["fps"]) if "fps" in data else None)
    positions    = data["positions"]                              # (T, B, 3)
    orientations = data["orientations"]                          # (T, B, 4) wxyz
    bone_names   = [str(n).split(":")[-1] for n in data["bone_names"]]
    props = {str(k): (data["prop_pos_" + str(k)], data["prop_rot_" + str(k)])
             for k in data["prop_names"]} if "prop_names" in data else {}

    if start or end is not None:
        if fps is None:
            raise ValueError("--start/--end need the frame rate; pass --fps")
        i0 = int(round(start * fps))
        i1 = len(positions) if end is None else int(round(end * fps))
        positions, orientations = positions[i0:i1], orientations[i0:i1]
        props = {k: (p[i0:i1], q[i0:i1]) for k, (p, q) in props.items()}
        print(f"  Trimmed to {start:g}–{(i1 / fps):g} s (frames {i0}–{i1 - 1})")

    # Auto-correct yaw so every recording starts facing canonical direction
    yaw_corr = R.identity()
    if "Hips" in bone_names:
        idx = bone_names.index("Hips")
        w, x, y, z = orientations[0, idx]
        hips_yaw = R.from_quat([x, y, z, w]).as_euler("zyx")[0]
        yaw_corr = R.from_euler("z", -hips_yaw)
        print(f"  Facing correction: {np.degrees(hips_yaw):.1f}° → normalised to 0°")

    frames = []
    for t in range(positions.shape[0]):
        frame = {}
        for i, bone in enumerate(bone_names):
            bx, by, bz = yaw_corr.apply(positions[t, i])
            w, x, y, z = orientations[t, i]
            q = (yaw_corr * R.from_quat([x, y, z, w])).as_quat()  # xyzw
            quat_wxyz = [q[3], q[0], q[1], q[2]]

            # Blender -Y facing → MuJoCo +X facing
            pos_mj = [-by, bx, bz]
            frame[bone] = [pos_mj, quat_wxyz]

        # Synthetic bones needed by the GMR IK config
        l_toe = frame.get("LeftToeBase",  frame["LeftFoot"])[1]
        r_toe = frame.get("RightToeBase", frame["RightFoot"])[1]
        frame["LeftFootMod"]  = [frame["LeftFoot"][0],  l_toe]
        frame["RightFootMod"] = [frame["RightFoot"][0], r_toe]
        frames.append(frame)

    # Rigid bodies: same yaw correction and the same -Y→+X axis change, applied to the
    # orientation too (a proper rotation, unlike the bone quats GMR's config expects).
    to_mj = R.from_euler("z", 90, degrees=True)
    objects = {}
    for k, (p, q) in props.items():
        pos = to_mj.apply(yaw_corr.apply(p))
        rot = (to_mj * yaw_corr * R.from_quat(q[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
        objects[k] = (pos, rot)

    return frames, bone_names, fps, objects


def retarget_frames(frames, human_height=1.68):
    """Run GMR IK on all frames and return (root_pos, root_rot, dof_pos, root_scale)."""
    retargeter = GMR(
        src_human="fbx_qtm",
        tgt_robot="unitree_g1",
        verbose=False,
        damping=0.05,
        actual_human_height=human_height,
    )
    retargeter.max_iter = 50
    # GMR scales the root translation about the world origin by this factor, so the robot
    # covers less ground than the actor; objects must be scaled the same way to line up.
    root_scale = float(retargeter.human_scale_table[retargeter.human_root_name])
    root_pos_list, root_rot_list, dof_list = [], [], []
    for human_data in tqdm(frames, desc="  retargeting", leave=False):
        qpos = retargeter.retarget(human_data, offset_to_ground=False)
        root_pos_list.append(qpos[:3].copy())
        root_rot_list.append(qpos[3:7].copy())
        dof_list.append(qpos[7:].copy())
    return (
        np.array(root_pos_list),
        np.array(root_rot_list),
        np.array(dof_list),
        root_scale,
    )


def repair_jumps(dof, fps, max_speed=15.0, max_dev=0.3):
    """Remove IK glitches from the joint trajectories.

    GMR solves each frame from the previous one, and near joint limits (often the wrists)
    it can switch to a different solution in a single frame: a joint "teleports" by up to
    ~1.5 rad and stays there. The mocap itself is smooth at those frames, so these jumps
    are artefacts, and they ask the policy for impossible joint speeds.

    1. Short flips (out and back within ~0.05 s) are found against a ~0.1 s running
       median and interpolated over.
    2. Remaining steps faster than `max_speed` rad/s are turned into linear ramps just wide
       enough to stay under that speed, centred on the jump.
    Normal motion stays well below both thresholds and is left untouched.
    """
    from scipy.signal import medfilt
    win = max(3, int(round(0.1 * fps)) | 1)           # odd window, ~0.1 s
    step = max_speed / fps                            # rad per frame allowed
    dof = dof.copy()
    t = np.arange(len(dof))
    flips = ramps = 0
    for j in range(dof.shape[1]):
        q = dof[:, j]
        bad = np.abs(q - medfilt(q, win)) > max_dev
        if bad.any() and (~bad).sum() >= 2:
            q[bad] = np.interp(t[bad], t[~bad], q[~bad])
            flips += int(bad.sum())
        for _ in range(10):                           # widen until every ramp is slow enough
            jump = np.abs(np.diff(q)) > step * 1.001
            if not jump.any():
                break
            bad = np.zeros(len(q), bool)
            for i in np.flatnonzero(jump):
                k = int(np.ceil(abs(q[i + 1] - q[i]) / step / 2)) + 1
                bad[max(1, i - k + 1):min(len(q) - 1, i + k + 1)] = True
            ramps += int(jump.sum())
            q[bad] = np.interp(t[bad], t[~bad], q[~bad])
    return dof, flips, ramps


def ground_offset(root_pos, root_rot, dof):
    """Height (m) to subtract from root z so the G1's soles touch z=0.

    Runs forward kinematics on the GMR G1 model and takes, per frame, the lowest point of
    the sole contact spheres; the median over the clip is the offset (robust to jumps and
    to a few bad frames). GMR's own offset_to_ground shifts every frame separately,
    which would erase real vertical motion, so it isn't used.
    """
    import mujoco
    from general_motion_retargeting.params import ROBOT_XML_DICT
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
    d = mujoco.MjData(m)
    feet = {m.body(b).id for b in ("left_ankle_roll_link", "right_ankle_roll_link")}
    soles = [g for g in range(m.ngeom)
             if m.geom_bodyid[g] in feet and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    lows = np.empty(len(root_pos))
    for i in range(len(root_pos)):
        d.qpos[:3], d.qpos[3:7], d.qpos[7:] = root_pos[i], root_rot[i], dof[i]
        mujoco.mj_kinematics(m, d)
        lows[i] = min(d.geom_xpos[g, 2] - m.geom_size[g, 0] for g in soles)
    return float(np.median(lows)), lows


def visualize(name, root_pos, root_rot, dof, fps=100.0, loop=False):
    T = len(root_pos)
    print(f"  Playing {T} frames  (Ctrl+C to skip)")
    viewer = RobotMotionViewer(
        robot_type="unitree_g1",
        motion_fps=fps,
        camera_follow=True,
        record_video=False,
    )
    frame = 0
    try:
        while True:
            viewer.step(root_pos[frame], root_rot[frame], dof[frame], rate_limit=True)
            frame += 1
            if frame >= T:
                if loop:
                    frame = 0
                else:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


def process_file(fbx_path: str, out_dir: str | None, do_viz: bool, loop: bool = False,
                 human_height: float = 1.68, fps_override: float | None = None,
                 clean: bool = True, ground: bool = True, max_joint_speed: float = 15.0,
                 start: float = 0.0, end: float | None = None):
    name = Path(fbx_path).stem
    print(f"\n[{name}]")

    with tempfile.NamedTemporaryFile(suffix="", delete=False) as f:
        npz_tmp = f.name

    print("  Step 1/3  Extracting bone transforms via Blender (takes ~10-30s) ...")
    ok = fbx_to_npz(fbx_path, npz_tmp)
    npz_path = npz_tmp + ".npz" if not npz_tmp.endswith(".npz") else npz_tmp
    if not ok:
        print(f"  ERROR: Blender failed on {fbx_path}")
        return

    print("  Step 2/3  Loading and converting coordinate frames ...")
    try:
        frames, clean_names, fps, objects = load_npz_as_human_data(
            npz_path, start=start, end=end, fps_override=fps_override)
    except ValueError as e:
        print(f"  ERROR: {e}")
        return
    finally:
        os.unlink(npz_path)
    if fps is None:
        print("  ERROR: FBX frame rate unknown; pass --fps (QTM capture rate, usually 100)")
        return
    print(f"  Bones ({len(clean_names)}): {clean_names[:5]} ...")
    src = "--fps override" if fps_override else "from FBX"
    print(f"  Frames: {len(frames)} at {fps:g} fps ({src}) = {len(frames) / fps:.1f} s")
    hips = frames[0].get("Hips", list(frames[0].values())[0])[0]
    print(f"  Hips height[0]: {hips[2]:.3f} m  (expected ~0.85 m)")
    if "Head" in frames[0]:
        print(f"  Head joint[0]: {frames[0]['Head'][0][2]:.3f} m  "
              f"(actor height used for scaling: {human_height:.2f} m; set with --height)")

    if objects:
        print(f"  Rigid bodies: {', '.join(objects)} (saved in the .pkl under 'objects')")

    print("  Step 3/3  Running GMR retargeting ...")
    root_pos, root_rot, dof, root_scale = retarget_frames(frames, human_height)

    if clean:
        before = np.abs(np.diff(dof, axis=0)).max() * fps
        dof, flips, ramps = repair_jumps(dof, fps, max_speed=max_joint_speed)
        speed = np.abs(np.diff(dof, axis=0)) * fps
        print(f"\n  IK glitch repair: {flips} flip frames interpolated, {ramps} jumps ramped; "
              f"max joint speed {before:.0f} → {speed.max():.1f} rad/s "
              f"(p99 {np.percentile(speed, 99):.1f})")

    off = 0.0
    if ground:
        off, lows = ground_offset(root_pos, root_rot, dof)
        root_pos = root_pos.copy()
        root_pos[:, 2] -= off
        lows = lows - off
        print(f"  Ground: shifted root by {-off * 100:+.1f} cm; lowest sole per frame now "
              f"{lows.min() * 100:+.1f} to {lows.max() * 100:+.1f} cm "
              f"(5-95%: {np.percentile(lows, 5) * 100:+.1f} to {np.percentile(lows, 95) * 100:+.1f})")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        pkl_path = os.path.join(out_dir, f"{name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos":  dof,
                "fps":      fps,
                "source":   fbx_path,
                "human_height": human_height,
                "cleaned":  clean,
                "grounded": ground,
                "trim_s":   (start, end),
                # Rigid bodies tracked in QTM (e.g. the chair), per frame, MuJoCo frame:
                #   pos / quat_wxyz      same scale and ground shift as root_pos, so they line
                #                        up with the robot (GMR shrinks root travel by root_scale)
                #   pos_human            actor-scale metres, QTM floor at z=0
                "root_scale": root_scale,
                "objects": {k: {"pos": np.c_[p[:, :2] * root_scale, p[:, 2:] * root_scale - off],
                                "quat_wxyz": q, "pos_human": p}
                            for k, (p, q) in objects.items()},
            }, f)
        print(f"  Saved → {pkl_path}")

    if do_viz:
        visualize(name, root_pos, root_rot, dof, fps=fps, loop=loop)


def main():
    parser = argparse.ArgumentParser(
        description="Qualisys FBX → Unitree G1 joint angles"
    )
    parser.add_argument("input", help="FBX file or folder containing FBX files")
    parser.add_argument("--out-dir", default=None, help="Save retargeted .pkl files here")
    parser.add_argument("--no-viz",  action="store_true", help="Skip MuJoCo visualization")
    parser.add_argument("--loop",    action="store_true", help="Loop the visualization")
    parser.add_argument("--height",  type=float, default=1.68,
                        help="Actor height in metres, used to scale the human to the G1 (default 1.68)")
    parser.add_argument("--fps",     type=float, default=None,
                        help="Override the frame rate read from the FBX (QTM capture rate)")
    parser.add_argument("--no-clean",  action="store_true",
                        help="Keep IK flips / jumps instead of smoothing them")
    parser.add_argument("--max-joint-speed", type=float, default=15.0,
                        help="rad/s; faster single-frame IK jumps are ramped (default 15)")
    parser.add_argument("--no-ground", action="store_true",
                        help="Don't shift the root height so the soles touch the floor")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Seconds to cut from the start (e.g. a T-pose); facing is taken from here")
    parser.add_argument("--end",   type=float, default=None, help="Stop at this time, in seconds")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        fbx_files = [str(input_path)]
    else:
        fbx_files = sorted(str(f) for f in input_path.glob("*.fbx"))

    if not fbx_files:
        print(f"No FBX files found in: {args.input}")
        sys.exit(1)

    print(f"Found {len(fbx_files)} FBX file(s)")
    for fbx_path in fbx_files:
        process_file(fbx_path, args.out_dir, not args.no_viz, loop=args.loop,
                     human_height=args.height, fps_override=args.fps,
                     clean=not args.no_clean, ground=not args.no_ground,
                     max_joint_speed=args.max_joint_speed,
                     start=args.start, end=args.end)

    print("\nDone.")


if __name__ == "__main__":
    main()
