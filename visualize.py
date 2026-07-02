"""
Visualize a retargeted .pkl file in MuJoCo.

Usage:
    python visualize.py output/my_recording.pkl
    python visualize.py output/my_recording.pkl --fps 50
    python visualize.py output/my_recording.pkl --record --video out.mp4
"""

import argparse
import pickle
import sys

try:
    from general_motion_retargeting import RobotMotionViewer
except ImportError:
    print("[ERROR] GMR not installed. Run setup.sh first.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Visualize a retargeted G1 .pkl file")
    parser.add_argument("pkl", help="Path to .pkl file produced by fbx_to_g1.py")
    parser.add_argument("--fps",     type=float, default=50.0, help="Playback FPS (default: 50)")
    parser.add_argument("--record",  action="store_true",      help="Record to video file")
    parser.add_argument("--video",   default="visualization.mp4")
    parser.add_argument("--no-loop", action="store_true",      help="Play once then exit")
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    root_pos = data["root_pos"]   # (T, 3)
    root_rot = data["root_rot"]   # (T, 4) wxyz
    dof_pos  = data["dof_pos"]    # (T, 29)
    fps      = data.get("fps", args.fps)
    T        = len(root_pos)

    print(f"Loaded: {args.pkl}")
    print(f"  Frames: {T}  |  FPS: {fps}  |  Duration: {T/fps:.1f}s")
    print("  Press Ctrl+C to stop")

    viewer = RobotMotionViewer(
        robot_type="unitree_g1",
        motion_fps=fps,
        camera_follow=True,
        record_video=args.record,
        video_path=args.video,
    )

    frame = 0
    try:
        while True:
            viewer.step(root_pos[frame], root_rot[frame], dof_pos[frame], rate_limit=True)
            frame += 1
            if frame >= T:
                if args.no_loop:
                    break
                frame = 0
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()

    if args.record:
        print(f"Video saved to: {args.video}")


if __name__ == "__main__":
    main()
