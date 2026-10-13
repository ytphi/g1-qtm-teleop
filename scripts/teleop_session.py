"""
Main teleoperation entry point.

Usage:
    python scripts/teleop_session.py --config config/teleop.yaml
    python scripts/teleop_session.py --dry-run          # no robot: QTM → retarget → MuJoCo only

Controls (keyboard, in terminal):
    s   — start recording an episode
    e   — end and save episode
    d   — discard current episode
    p   — pause / resume (robot holds its pose while paused)
    q   — quit (arms/legs ramp to zero, then damping, then motors off)

Safety (see teleop/safety.py):
    - every command is clamped to joint limits and rate-limited (safety.max_joint_vel)
    - QTM data older than safety.qtm_timeout, or with NaN segments → hold last command
    - retargeter errors / NaN output → hold last command
    - on quit or Ctrl-C: ramp to zero → damping → motors off (Ctrl-C again skips the ramp)
"""

import argparse
import os
import select
import sys
import time

import cv2
import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from teleop.qtm_stream import QTMStream
from teleop.retarget import G1Retargeter
from teleop.safety import SafetyFilter
from record.episode_writer import EpisodeWriter

NUM_JOINTS = 29


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def read_head_camera(cap: cv2.VideoCapture) -> np.ndarray | None:
    ret, frame = cap.read()
    return frame if ret else None


def key_pressed() -> str | None:
    """Non-blocking single keypress read from stdin."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


class DryRunRobot:
    """Stands in for G1RobotControl with --dry-run: 'measured' state is simply the last command."""

    def __init__(self):
        self._q = np.zeros(NUM_JOINTS)

    def init(self):
        print("[DRY RUN] No robot — commands go to the MuJoCo viewer only.")

    joint_positions = property(lambda self: self._q.copy())
    joint_velocities = property(lambda self: np.zeros(NUM_JOINTS))
    imu_rpy = property(lambda self: np.zeros(3))

    def move_to_zero(self, duration: float = 3.0):
        self._q[:] = 0.0

    def set_joint_positions(self, q):
        self._q = np.asarray(q, dtype=float).copy()

    def damp(self, duration: float = 2.0, kd: float = 3.0):
        pass

    def stop(self):
        pass


class Throttled:
    """Print a message at most once per `period` seconds."""

    def __init__(self, period: float = 1.0):
        self.period, self._last = period, {}

    def __call__(self, key: str, msg: str):
        now = time.time()
        if now - self._last.get(key, 0.0) > self.period:
            print(msg, flush=True)
            self._last[key] = now


def safe_shutdown(robot, safety: SafetyFilter, dt: float, max_time: float = 6.0):
    """Ramp to zero posture at the safety speed, then damping, then motors off."""
    print("\n[SAFE] Shutting down: ramping to zero (Ctrl-C to skip)...", flush=True)
    try:
        t_end = time.time() + max_time
        while time.time() < t_end:
            done = safety.ramp_to(np.zeros(NUM_JOINTS))
            robot.set_joint_positions(safety.q)
            if done:
                break
            time.sleep(dt)
        print("[SAFE] Damping (2 s)...", flush=True)
        robot.damp(duration=2.0)
    except KeyboardInterrupt:
        print("[SAFE] Ramp skipped.", flush=True)
    finally:
        robot.stop()
        print("[SAFE] Motors off.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/teleop.yaml")
    parser.add_argument("--retargeter", choices=["geometric", "gmr"], default="geometric",
                        help="geometric = rule-based (default); gmr = IK-based (YanjieZe/GMR)")
    parser.add_argument("--dry-run", action="store_true",
                        help="no robot: run QTM → retarget → safety filter into a MuJoCo viewer")
    parser.add_argument("--float-base", action="store_true",
                        help="dry run + gmr only: move the viewer's pelvis with the actor, feet kept on the "
                             "ground (the real robot can't — its pelvis hangs on the gantry)")
    args = parser.parse_args()
    if args.float_base and not (args.dry_run and args.retargeter == "gmr"):
        parser.error("--float-base needs --dry-run --retargeter gmr")
    os.chdir(REPO_ROOT)   # config + episodes paths are repo-relative, wherever this is run from

    cfg = load_config(args.config)
    robot_cfg   = cfg["robot"]
    qtm_cfg     = cfg["qtm"]
    record_cfg  = cfg["recording"]
    safety_cfg  = cfg.get("safety", {})

    control_freq = robot_cfg["control_freq"]
    dt = 1.0 / control_freq
    qtm_timeout = safety_cfg.get("qtm_timeout", 0.2)
    safety = SafetyFilter(dt=dt,
                          max_vel=safety_cfg.get("max_joint_vel", 1.0),
                          limit_margin=safety_cfg.get("limit_margin", 0.05))
    warn = Throttled(1.0)

    # ── QTM stream ──────────────────────────────────────────────────────────
    qtm = QTMStream(
        host=qtm_cfg["ip"],
        port=qtm_cfg["port"],
        skeleton_name=qtm_cfg["skeleton_name"],
    )
    print(f"Connecting to QTM at {qtm_cfg['ip']}...")
    qtm.start()

    # wait up to 5s for skeleton data
    for _ in range(50):
        if qtm.ready or qtm.error:
            break
        time.sleep(0.1)
    if not qtm.ready:
        print(f"ERROR: QTM skeleton not ready ({qtm.error or 'no data'}). "
              "Check QTM IP, that QTM is in preview/capture, and RT skeleton output is on.")
        qtm.stop()
        return

    # ── Retargeter ───────────────────────────────────────────────────────────
    if args.retargeter == "gmr":
        from teleop.gmr_retarget import GMRRetargeter
        retargeter = GMRRetargeter()
        retargeter.setup()
    else:
        retargeter = G1Retargeter()

    # ── Robot ────────────────────────────────────────────────────────────────
    viewer = None
    if args.dry_run:
        robot = DryRunRobot()
        from teleop.live_viewer import LiveViewer
        viewer = LiveViewer(fps=control_freq, camera_follow=args.float_base, ground_feet=args.float_base)
        viewer.start()
    else:
        from teleop.robot_control import G1RobotControl
        print("\n!! The robot's own balance controller will be RELEASED. "
              "It must hang on the gantry, safety operator holding the remote.")
        if input("Type 'yes' to continue: ").strip().lower() != "yes":
            qtm.stop()
            return
        robot = G1RobotControl(network_interface=robot_cfg["network_interface"])
    robot.init()

    # ── Camera ───────────────────────────────────────────────────────────────
    cap = None
    if record_cfg.get("obs_image") and not args.dry_run:
        cap = cv2.VideoCapture(0)
        w, h = record_cfg["image_size"]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    # ── Recorder ─────────────────────────────────────────────────────────────
    writer = EpisodeWriter(
        save_dir=record_cfg["save_dir"],
        control_freq=control_freq,
    )

    # ── Safe start ───────────────────────────────────────────────────────────
    print("\nMoving robot to zero posture (3 s)...")
    robot.move_to_zero(duration=3.0)
    safety.reset(np.zeros(NUM_JOINTS))

    print("\n--- TELEOP READY ---")
    if args.retargeter == "geometric":
        print("Calibration: copy the robot's ZERO pose shown in the viewer (stand straight, arms\n"
              "hanging at the sides, palms like the robot's) — NOT a T-pose — then press Enter.")
        input()
        if qtm.valid():
            retargeter.calibrate(dict(qtm.latest))
            print("Calibrated.")
        else:
            print("WARNING: QTM data invalid during calibration — not calibrated.")
    print(f"Robot will follow at most {safety_cfg.get('max_joint_vel', 1.0)} rad/s per joint.")
    print("\nControls: [s] start episode  [e] end+save  [d] discard  [p] pause  [q] quit\n")

    recording = False
    paused = False
    import tty, termios
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            t0 = time.time()

            # ── keyboard ──
            k = key_pressed()
            if k == "s" and not recording:
                print("[REC] Starting episode...")
                writer.start()
                recording = True
            elif k == "e" and recording:
                path = writer.save()
                print(f"[REC] Saved → {path}")
                recording = False
            elif k == "d" and recording:
                writer.discard()
                print("[REC] Discarded.")
                recording = False
            elif k == "p":
                paused = not paused
                print("[SAFE] PAUSED — holding pose" if paused else "[SAFE] Resumed")
            elif k == "q":
                break

            # ── retarget (only on fresh, valid mocap) ──
            segments = dict(qtm.latest)
            target = None
            if paused:
                pass
            elif qtm.age > qtm_timeout:
                warn("stale", f"[SAFE] QTM data {qtm.age:.2f}s old — holding pose")
            elif not qtm.valid():
                warn("nan", "[SAFE] QTM segments missing / NaN (tracking lost) — holding pose")
            else:
                try:
                    target = retargeter.retarget(segments)
                except Exception as e:
                    warn("retarget", f"[SAFE] retarget failed ({e}) — holding pose")

            # ── safety filter → command ──
            n_rejected = safety.rejected
            action = safety(target)
            if safety.rejected > n_rejected:
                warn("nonfinite", "[SAFE] retarget produced NaN/inf — holding pose")
            robot.set_joint_positions(action)
            if viewer is not None:
                if args.float_base:
                    viewer.update(action, retargeter.root_pos, retargeter.root_rot)
                else:
                    viewer.update(action)

            # ── observe ──
            qpos    = robot.joint_positions
            qvel    = robot.joint_velocities
            imu_rpy = robot.imu_rpy
            image   = read_head_camera(cap) if cap else None

            # ── record ──
            if recording:
                writer.record_step(
                    qpos=qpos,
                    qvel=qvel,
                    imu_rpy=imu_rpy,
                    action=action,          # the command actually sent (after safety filter)
                    image=image,
                    mocap_segments=segments if segments else None,
                )

            # ── display ──
            if image is not None:
                cv2.imshow("G1 Head Camera", image)
                cv2.waitKey(1)

            # ── timing ──
            elapsed = time.time() - t0
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                warn("overrun", f"[WARN] control loop overrun: {elapsed * 1000:.1f} ms > {dt * 1000:.0f} ms")

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        if recording:
            writer.discard()
            print("[REC] Unsaved episode discarded.")
        safe_shutdown(robot, safety, dt)
        qtm.stop()
        if viewer is not None:
            viewer.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("Session ended.")


if __name__ == "__main__":
    main()
