# Qualisys FBX → Unitree G1

Convert motion capture recordings from **Qualisys Track Manager (QTM)** into **Unitree G1 humanoid robot joint angles**, ready for simulation or robot playback.

Developed at **California State University, San Bernardino (CSUSB)** — Robotics Lab.

---

## What This Does

When you record a person moving in the Qualisys motion capture lab, QTM tracks reflective markers placed on their body and fits a skeleton model to them. You can export that skeleton animation as an **FBX file**.

This tool takes that FBX file and converts it into a sequence of **robot joint angles** — the exact motor positions the Unitree G1 humanoid would need to replicate the same movement. You can then preview the result in a 3D simulator (MuJoCo) and save it for robot deployment or AI training.

```
┌──────────────────────────────────────────────────────────────────┐
│                        WORKFLOW OVERVIEW                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Qualisys Lab          This Tool               Output           │
│                                                                  │
│  Person moves    →   FBX file (.fbx)   →  Robot joint angles    │
│  in capture          exported from          saved as .pkl        │
│  volume              QTM software           viewable in          │
│                                             MuJoCo sim           │
│                                                                  │
│           Step A              Step B              Step C         │
│        Export FBX          Run script          Visualize         │
│        from QTM            (automatic)         or deploy         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Under the hood**, the script:
1. Runs **Blender** in the background (no window opens) to read the FBX skeleton data
2. Auto-corrects the actor's starting direction so the robot always faces forward
3. Uses **GMR** (General Motion Retargeting, from Stanford/CMU) to solve the inverse kinematics — figuring out which joint angles reproduce the human motion on the G1's different body proportions
4. Saves the result and optionally opens a **MuJoCo** viewer

---

## Requirements

| Software | Version | Why needed |
|---|---|---|
| Python | 3.10 or newer | runs the scripts |
| Blender | 3.6 LTS or newer | reads FBX files |
| MuJoCo | installed with GMR | 3D robot viewer |

You also need a **Qualisys FBX export** — see the section below for exact export steps.

---

## Step A — Export FBX from Qualisys Track Manager

> **Do this on the Qualisys lab computer before you leave the lab.**
> You need QTM open with your recording loaded.

### What format does this tool expect?

Your FBX must be a **skeleton (bone) animation** — not raw marker data. QTM's AIM skeleton model must be fitted to your recording before export.

### Export steps (QTM 2023 or newer)

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

### What if my FBX has an actor prefix on bone names?

QTM sometimes names bones like `Joseph:Hips` instead of just `Hips`. This is called an **actor namespace prefix**. The script strips it automatically — you don't need to do anything.

### Checking your FBX before running

You can quickly verify the FBX is a skeleton export (not raw markers) by opening it in Blender manually:
1. Open Blender → **File → Import → FBX**
2. Import your file
3. In the Outliner (top-right), look for an **Armature** object
4. If you see bones in the viewport, you're good

---

## Step B — Install and Run

### 1. Clone this repository

```bash
git clone https://github.com/csusb-robotics/qualisys-fbx-to-g1.git
cd qualisys-fbx-to-g1
```

### 2. Run the setup script

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

### 3. Retarget your FBX

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

## Step C — Visualize

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

## Output File Format

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

## Troubleshooting

### "No armature found in FBX"

The FBX does not contain a skeleton. Go back to Step A and make sure:
- The AIM skeleton is fitted in QTM before export
- "Skeleton" is checked in the FBX export dialog

### "blender: command not found"

Blender is not on your PATH. After installing, you may need:
```bash
# If installed via snap:
export PATH="/snap/bin:$PATH"

# Or specify the full path directly:
which blender   # find where it is
```

### Hips height is much less than 0.85 m (e.g. 0.009 m)

The FBX was exported in **centimetres** instead of metres. In the QTM export dialog, set **Units → Meters**. If you cannot re-export, you can patch the script: find `pos_mj = [-by, bx, bz]` in `fbx_to_g1.py` and change it to `pos_mj = [-by/100, bx/100, bz/100]`.

### The robot motion looks mirrored left-right

Check that the person was facing toward the **+X axis** (or any consistent direction) in the QTM lab. The script auto-corrects yaw but assumes the skeleton is upright. If the actor was walking sideways at capture start, the correction may flip.

### Retargeting is very slow

The default setting runs 50 IK iterations per frame for accuracy. For a quick preview, you can reduce this by editing `fbx_to_g1.py`:
```python
retargeter.max_iter = 50   # change to 10 for faster (less accurate) results
```

### ImportError: No module named 'general_motion_retargeting'

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
qualisys-fbx-to-g1/
├── fbx_to_g1.py      ← main pipeline: FBX → .pkl
├── visualize.py      ← MuJoCo playback of .pkl files
├── setup.sh          ← one-command install
├── requirements.txt  ← Python dependencies
└── README.md         ← this file
```

The GMR library is cloned into a `GMR/` subfolder by `setup.sh`.

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
- **Unitree G1** robot: [Unitree Robotics](https://www.unitree.com/g1/)
- Pipeline developed by the CSUSB Robotics Lab under **Prof. Yutong Liu**
