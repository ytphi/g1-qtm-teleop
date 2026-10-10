"""
Qualisys FBX → Unitree G1 joint angles pipeline.

Steps performed automatically:
  1. Run Blender headless to extract per-frame bone world transforms from FBX
  2. Strip actor namespace prefix (e.g. "Joseph:Hips" → "Hips")
  3. Auto-correct initial facing direction so the robot always starts forward
  4. Retarget each frame → G1 joint angles using GMR IK
  5. Save as .pkl and optionally visualize in MuJoCo

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

arm_obj = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
if arm_obj is None:
    print("ERROR: no armature found in FBX"); sys.exit(1)

bpy.context.view_layer.objects.active = arm_obj
scene  = bpy.context.scene
bone_names = [b.name for b in arm_obj.data.bones]

# Use the action keyframe range — QTM FBX files often have 3000+ frames,
# so the scene start/end defaults may be wrong.
action = arm_obj.animation_data.action if arm_obj.animation_data else None
if action and action.frame_range[1] > scene.frame_end:
    frame_start = int(action.frame_range[0])
    frame_end   = int(action.frame_range[1])
    print(f"Using action frame range: {{frame_start}}–{{frame_end}} ({{frame_end - frame_start + 1}} frames)")
else:
    frame_start = scene.frame_start
    frame_end   = scene.frame_end
    print(f"Using scene frame range: {{frame_start}}–{{frame_end}}")

positions    = []
orientations = []

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

np.savez(out_path,
         positions=np.array(positions,    dtype=np.float32),  # (T, B, 3)
         orientations=np.array(orientations, dtype=np.float32),  # (T, B, 4) wxyz
         bone_names=np.array(bone_names))
print(f"Saved {{len(positions)}} frames, {{len(bone_names)}} bones → {{out_path}}")
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


def load_npz_as_human_data(npz_path: str):
    """
    Load Blender-extracted transforms and convert to GMR human_data format.

    - Strips actor namespace prefix (e.g. "Joseph:Hips" → "Hips")
    - Auto-corrects initial facing direction (yaw normalisation)
    - Converts Blender Z-up to MuJoCo Z-up (coordinate flip -Y→+X)
    - Adds LeftFootMod / RightFootMod synthetic bones for GMR IK config
    """
    data = np.load(npz_path, allow_pickle=True)
    positions    = data["positions"]                              # (T, B, 3)
    orientations = data["orientations"]                          # (T, B, 4) wxyz
    bone_names   = [str(n).split(":")[-1] for n in data["bone_names"]]

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

    return frames, bone_names


def retarget_frames(frames):
    """Run GMR IK on all frames and return (root_pos, root_rot, dof_pos)."""
    retargeter = GMR(
        src_human="fbx_qtm",
        tgt_robot="unitree_g1",
        verbose=False,
        damping=0.05,
        actual_human_height=1.68,
    )
    retargeter.max_iter = 50
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
    )


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


def process_file(fbx_path: str, out_dir: str | None, do_viz: bool, loop: bool = False):
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
    frames, clean_names = load_npz_as_human_data(npz_path)
    os.unlink(npz_path)
    print(f"  Bones ({len(clean_names)}): {clean_names[:5]} ...")
    print(f"  Frames: {len(frames)}")
    hips = frames[0].get("Hips", list(frames[0].values())[0])[0]
    print(f"  Hips height[0]: {hips[2]:.3f} m  (expected ~0.85 m)")

    print("  Step 3/3  Running GMR retargeting ...")
    root_pos, root_rot, dof = retarget_frames(frames)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        pkl_path = os.path.join(out_dir, f"{name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos":  dof,
                "fps":      50.0,
                "source":   fbx_path,
            }, f)
        print(f"  Saved → {pkl_path}")

    if do_viz:
        visualize(name, root_pos, root_rot, dof, loop=loop)


def main():
    parser = argparse.ArgumentParser(
        description="Qualisys FBX → Unitree G1 joint angles"
    )
    parser.add_argument("input", help="FBX file or folder containing FBX files")
    parser.add_argument("--out-dir", default=None, help="Save retargeted .pkl files here")
    parser.add_argument("--no-viz",  action="store_true", help="Skip MuJoCo visualization")
    parser.add_argument("--loop",    action="store_true", help="Loop the visualization")
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
        process_file(fbx_path, args.out_dir, not args.no_viz, loop=args.loop)

    print("\nDone.")


if __name__ == "__main__":
    main()
