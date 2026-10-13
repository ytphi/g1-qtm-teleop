# QTM Teleop G1

Drive a **Unitree G1 humanoid** from a person moving in a **Qualisys** motion capture studio — live, through the **Qualisys Track Manager (QTM)** real-time stream — or retarget recorded QTM sessions offline.

Developed at **California State University, San Bernardino (CSUSB)** — Robotics Lab.

| Mode | Input | Output | Entry point |
|---|---|---|---|
| **Live teleop** | QTM real-time skeleton stream | G1 joint commands at 100 Hz (or a MuJoCo preview) + recorded episodes | `scripts/teleop_session.py` |
| **Offline retargeting** | FBX export from QTM | `.pkl` joint-angle sequence | `fbx_to_g1.py` |

> **Status:** live teleop has been tested end-to-end against a live QTM stream in **dry-run mode** (MuJoCo viewer, no robot): 24/24 skeleton segments, ~5 ms median stream latency, arm and torso directions matching the actor within a few degrees. **It has not yet been run on the physical robot.**

---

## Install

```bash
git clone https://github.com/ytphi/g1-qtm-teleop.git
cd g1-qtm-teleop
bash setup.sh
```

`setup.sh` installs the Python packages, clones [GMR](https://github.com/YanjieZe/GMR) into `GMR/` at a pinned commit, and applies `gmr_patch/` — QTM IK configs plus small fixes that GMR needs for QTM skeletons. Python 3.10+ is recommended.

To use the modules from another project (e.g. `g1-sb01` imports `teleop.safety`), install the repo in that environment: `pip install -e .`

To drive the real robot you also need [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python). Dry runs work without it.

---

# Live teleop

```
 Qualisys studio            this repo                                          G1
┌──────────────┐  QTM RT  ┌─────────────┐  ┌────────────┐  ┌────────┐  DDS   ┌───────────┐
│ actor + QTM  │ ───────► │ qtm_stream  │─►│ retarget   │─►│ safety │ ─────► │ 29 DOF    │
│ AIM skeleton │  22223   │ 24 segments │  │ GMR IK or  │  │ filter │ 100 Hz │ low-level │
└──────────────┘          └─────────────┘  │ geometric  │  └────────┘        └───────────┘
                                           └────────────┘      │
                                                               └─► episode recorder (HDF5)
```

## 1. Set up QTM

- Fit the AIM skeleton to the actor and keep QTM in **preview or capture** so it streams.
- Turn on **real-time skeleton output**.
- Use port **22223**. On QTM 2024.2, port 22222 accepts the connection but only speaks the legacy RT protocol 1.0, so `qtm_rt` hangs silently.

## 2. Configure

Edit `config/teleop.yaml`:

```yaml
robot:
  network_interface: "eth0"   # NIC wired to the G1 (check with `ip link`)
qtm:
  ip: "192.168.0.1"           # QTM PC
  port: 22223
  skeleton_name: "Actor1"     # skeleton name in QTM
safety:
  max_joint_vel: 1.0          # rad/s per joint; lower = gentler
  qtm_timeout: 0.2            # s; older mocap data → hold pose
  limit_margin: 0.05          # rad inside URDF joint limits
```

## 3. Dry run first (no robot)

```bash
python scripts/teleop_session.py --dry-run --retargeter gmr
```

QTM → retargeting → safety filter → a MuJoCo viewer showing the G1. Check that the robot mirrors the actor before going near hardware.

By default the viewer's pelvis is fixed, as it is on the gantry. To see whole-body motion, add `--float-base` (dry run + `gmr` only): the pelvis follows GMR's root pose and the feet are kept on the ground. This is a kinematic preview with no physics or balance.

## 4. Run on the robot

```bash
python scripts/teleop_session.py --retargeter gmr
```

> ⚠️ This **releases the G1's own balance controller** and commands every joint directly. The robot must hang on a **gantry** with a **safety operator holding the remote**. The script asks you to type `yes` before it starts.

The laptop needs wired connections to both the QTM PC and the G1. Wi-Fi to QTM showed ~1 s stalls, and the robot holds its pose during each one. One cable can carry both subnets if the QTM network and the G1 share a switch. Add a second address to the port, e.g. `nmcli connection modify <profile> +ipv4.addresses 192.168.0.250/24` next to the G1's `192.168.123.x`. Otherwise use a USB Ethernet adapter.

### Keyboard controls

| Key | Action |
|---|---|
| `s` | start recording an episode |
| `e` | end and save the episode |
| `d` | discard the current episode |
| `p` | pause / resume (robot holds its pose) |
| `q` | quit: ramp to zero → damping → motors off |

`Ctrl-C` does the same safe shutdown; press it twice to skip the ramp.

### Safety filter (`teleop/safety.py`)

- Every command is clamped to the URDF joint limits (minus a margin) and rate-limited to `max_joint_vel`.
- Mocap data older than `qtm_timeout`, or with NaN segments → hold the last command.
- Retargeter errors or NaN output → hold the last command.

### Retargeters

| `--retargeter` | How it works | Notes |
|---|---|---|
| `gmr` | IK with [GMR](https://github.com/YanjieZe/GMR) (mink + MuJoCo), `qtm_rt` config | recommended; tested on live QTM data |
| `geometric` (default) | joint angles from relative rotations between adjacent QTM segments | no IK, lighter; not yet compared against GMR live |

Recorded episodes are saved as HDF5 under `episodes/` (joint commands, robot state, IMU, mocap segment poses). `recording.obs_image` is off by default: it reads `cv2` device 0, which is the laptop's own camera, not the G1 head camera.

---

# Offline retargeting: FBX recordings

Convert a recorded QTM session, exported as **FBX**, into G1 joint angles you can preview in MuJoCo or use as training data.

**Under the hood**, the script:
1. Runs **Blender** in the background (no window opens) to read the FBX skeleton data
2. Auto-corrects the actor's starting direction so the robot always faces forward
3. Uses **GMR** (General Motion Retargeting, from Stanford/CMU) to solve the inverse kinematics — figuring out which joint angles reproduce the human motion on the G1's different body proportions
4. Saves the result and optionally opens a **MuJoCo** viewer

---

### Requirements

| Software | Version | Why needed |
|---|---|---|
| Python | 3.10 or newer | runs the scripts |
| Blender | 3.6 LTS or newer | reads FBX files |
| MuJoCo | installed with GMR | 3D robot viewer |

You also need a **Qualisys FBX export** — see the section below for exact export steps.

---

### Step A — Export FBX from Qualisys Track Manager

> **Do this on the Qualisys lab computer before you leave the lab.**
> You need QTM open with your recording loaded.

#### What format does this tool expect?

Your FBX must be a **skeleton (bone) animation** — not raw marker data. QTM's AIM skeleton model must be fitted to your recording before export.

#### Export steps (QTM 2023 or newer)

**1. Open your recording in QTM**

Launch QTM and open your `.qtm` project file. Make sure the measurement you want to export is loaded and the timeline shows the correct frames.

**2. Confirm the skeleton is fitted**

In the 3D view, you should see a stick-figure skeleton overlaid on the markers. If you only see dots (raw markers) and no skeleton, you need to fit the AIM model first:
- Go to **Workflow** → **AIM Body** → **Solve**
- Accept the default settings and let it run
- You should now see the skeleton appear in the 3D view

**3. Open the export dialog**

Go to: **File → Export → FBX...**

A dialog box will appear.

**4. Configure the export settings**

Set the following options (leave everything else at default):

| Setting | Value | Notes |
|---|---|---|
| **Format** | Binary FBX | *not* ASCII — file will be smaller |
| **Up Axis** | Z | matches the MuJoCo world frame |
| **Units** | Meters | the script expects metres |
| **Include** | ✓ Skeleton | must be checked |
| **Include** | ✓ Global Positions | must be checked |
| **Frame range** | All frames | or set start/end if you only want a clip |
| **Subject** | Select your subject | if multiple people were recorded, pick one |

> **Important:** Make sure "Skeleton" and "Global Positions" are both checked. If "Skeleton" is unchecked, the FBX will contain only raw marker trajectories, which this tool cannot process.

**5. Choose a save location and export**

Click **Export**. QTM will create a `.fbx` file. Copy this file to your laptop (USB drive or network share).

#### What if my FBX has an actor prefix on bone names?

QTM sometimes names bones like `Joseph:Hips` instead of just `Hips`. This is called an **actor namespace prefix**. The script strips it automatically — you don't need to do anything.

#### Checking your FBX before running

You can quickly verify the FBX is a skeleton export (not raw markers) by opening it in Blender manually:
1. Open Blender → **File → Import → FBX**
2. Import your file
3. In the Outliner (top-right), look for an **Armature** object
4. If you see bones in the viewport, you're good

---

### Step B — Install and Run

#### 1. Clone this repository

```bash
git clone https://github.com/ytphi/g1-qtm-teleop.git
cd g1-qtm-teleop
```

#### 2. Run the setup script

```bash
bash setup.sh
```

This will automatically install the Python packages and the GMR retargeting library. It takes about 2-5 minutes the first time.

**If Blender is not installed**, follow the warning printed by setup.sh:
```bash
# Ubuntu / Debian
sudo snap install blender --classic

# Or download from blender.org and add to PATH
```

#### 3. Retarget your FBX

```bash
# Retarget a single file and open the MuJoCo viewer
python fbx_to_g1.py your_recording.fbx

# Retarget and save the output (no viewer)
python fbx_to_g1.py your_recording.fbx --out-dir output/ --no-viz

# Retarget an entire folder of FBX files
python fbx_to_g1.py /path/to/fbx_folder/ --out-dir output/
```

**What you will see in the terminal:**

```
Found 1 FBX file(s)

[your_recording]
  Step 1/3  Extracting bone transforms via Blender (takes ~10-30s) ...
  Facing correction: 34.2° → normalised to 0°
  Step 2/3  Loading and converting coordinate frames ...
  Bones (21): ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck'] ...
  Frames: 1500
  Hips height[0]: 0.847 m  (expected ~0.85 m)
  Step 3/3  Running GMR retargeting ...
  retargeting: 100%|████████████████| 1500/1500 [01:20<00:00, 18.6it/s]
  Saved → output/your_recording.pkl

Done.
```

Step 1 (Blender) takes 10–30 seconds for a typical 30-second recording. Step 3 (retargeting) takes about 1 minute per 1500 frames on a standard laptop.

---

### Step C — Visualize

If you used `--no-viz` earlier, or want to re-watch the result:

```bash
python visualize.py output/your_recording.pkl
```

The MuJoCo window will open showing the G1 robot performing the motion. Press **Ctrl+C** in the terminal to close it.

To record a video of the playback:

```bash
python visualize.py output/your_recording.pkl --record --video my_motion.mp4
```

---

### Output File Format

The `.pkl` file saved by `fbx_to_g1.py` is a Python pickle containing:

```python
{
    "root_pos": np.ndarray,  # shape (T, 3)  — robot base XYZ position, metres
    "root_rot": np.ndarray,  # shape (T, 4)  — base orientation quaternion (w, x, y, z)
    "dof_pos":  np.ndarray,  # shape (T, 29) — joint angles in radians
    "fps":      float,       # playback frame rate (50.0)
    "source":   str,         # original FBX file path
}
```

`T` is the number of frames. The 29 joints follow the **Unitree G1 arm7 layout** — 6 joints per leg (×2), 3 waist joints, and 7 joints per arm (×2).

To load and inspect the output in Python:

```python
import pickle
import numpy as np

with open("output/your_recording.pkl", "rb") as f:
    data = pickle.load(f)

print(data["dof_pos"].shape)   # (1500, 29)
print(data["fps"])             # 50.0
```

---

### Troubleshooting

#### "No armature found in FBX"

The FBX does not contain a skeleton. Go back to Step A and make sure:
- The AIM skeleton is fitted in QTM before export
- "Skeleton" is checked in the FBX export dialog

#### "blender: command not found"

Blender is not on your PATH. After installing, you may need:
```bash
# If installed via snap:
export PATH="/snap/bin:$PATH"

# Or specify the full path directly:
which blender   # find where it is
```

#### Hips height is much less than 0.85 m (e.g. 0.009 m)

The FBX was exported in **centimetres** instead of metres. In the QTM export dialog, set **Units → Meters**. If you cannot re-export, you can patch the script: find `pos_mj = [-by, bx, bz]` in `fbx_to_g1.py` and change it to `pos_mj = [-by/100, bx/100, bz/100]`.

#### The robot motion looks mirrored left-right

Check that the person was facing toward the **+X axis** (or any consistent direction) in the QTM lab. The script auto-corrects yaw but assumes the skeleton is upright. If the actor was walking sideways at capture start, the correction may flip.

#### Retargeting is very slow

The default setting runs 50 IK iterations per frame for accuracy. For a quick preview, you can reduce this by editing `fbx_to_g1.py`:
```python
retargeter.max_iter = 50   # change to 10 for faster (less accurate) results
```

#### ImportError: No module named 'general_motion_retargeting'

Run `bash setup.sh` again. If the problem persists:
```bash
cd GMR
pip install -e .
cd ..
python -c "import general_motion_retargeting; print('OK')"
```

---

## Project Structure

```
g1-qtm-teleop/
├── scripts/teleop_session.py   ← live teleop entry point
├── teleop/
│   ├── qtm_stream.py           ← QTM real-time skeleton client
│   ├── gmr_retarget.py         ← GMR IK retargeter (qtm_rt config)
│   ├── retarget.py             ← geometric retargeter + G1 joint names/limits
│   ├── safety.py               ← joint-limit clamp, rate limit, stale-data hold
│   ├── robot_control.py        ← G1 low-level DDS commands (unitree_sdk2py)
│   └── live_viewer.py          ← MuJoCo preview for --dry-run
├── record/episode_writer.py    ← HDF5 episode recorder
├── config/teleop.yaml          ← network, QTM and safety settings
├── fbx_to_g1.py                ← offline: FBX → .pkl
├── visualize.py                ← offline: MuJoCo playback of .pkl files
├── gmr_patch/                  ← QTM IK configs + GMR fixes, applied by setup.sh
├── setup.sh                    ← one-command install
├── pyproject.toml              ← makes teleop/ and record/ pip-installable
└── requirements.txt
```

---

## How the Retargeting Works (Conceptual)

A Qualisys skeleton and the Unitree G1 have different body proportions — the human subject might be 1.75 m tall with long arms, while the G1 is 1.27 m with different limb ratios. Simply copying joint angles would make the robot flail its arms in the wrong places.

**GMR (General Motion Retargeting)** solves this using **inverse kinematics (IK)**:

1. For each frame, it takes the *world-space position and orientation* of key human body landmarks (shoulders, elbows, wrists, hips, knees, ankles)
2. It scales these positions to the G1's body proportions
3. It runs an IK solver that finds the joint angles making the G1's landmarks match the scaled human landmarks as closely as possible
4. The result is a sequence of 29 joint angles per frame that faithfully recreates the human motion on the G1

This is the same approach used in professional animation pipelines for retargeting motion between characters of different sizes.

---

## Credits

- **GMR** retargeting engine: [YanjieZe/GMR](https://github.com/YanjieZe/GMR) — Yanjie Ze et al., arXiv:2505.02833
- **QTM real-time SDK**: [qualisys/qualisys_python_sdk](https://github.com/qualisys/qualisys_python_sdk)
- **Unitree G1** robot: [Unitree Robotics](https://www.unitree.com/g1/)
- Developed by the CSUSB Robotics Lab under **Prof. Yutong Liu**
