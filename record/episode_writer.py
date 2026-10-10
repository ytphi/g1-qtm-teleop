import os
import time
import numpy as np
import h5py
import cv2


class EpisodeWriter:
    """
    Records a single teleop episode to an HDF5 file.

    File layout:
        episode_NNNN.hdf5
        ├── observations/
        │   ├── qpos          (T, 29)  joint positions
        │   ├── qvel          (T, 29)  joint velocities
        │   ├── imu_rpy       (T, 3)   roll/pitch/yaw
        │   └── images/
        │       └── head      (T, H, W, 3)  BGR uint8
        ├── actions           (T, 29)  commanded joint angles
        ├── mocap/
        │   ├── segment_names (S,)     string array
        │   └── segment_poses (T, S, 7) position(3) + quaternion(4)
        └── metadata/
            ├── timestamp_start  scalar float
            ├── control_freq     scalar float
            └── episode_id       scalar int
    """

    def __init__(self, save_dir: str, control_freq: float = 100.0):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.control_freq = control_freq

        self._qpos: list[np.ndarray] = []
        self._qvel: list[np.ndarray] = []
        self._imu: list[np.ndarray] = []
        self._images: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._mocap_poses: list[np.ndarray] = []
        self._mocap_names: list[str] | None = None
        self._t_start: float | None = None

    def start(self):
        self._t_start = time.time()

    def record_step(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        imu_rpy: np.ndarray,
        action: np.ndarray,
        image: np.ndarray | None,
        mocap_segments: dict | None,
    ):
        if self._t_start is None:
            self.start()

        self._qpos.append(qpos.copy())
        self._qvel.append(qvel.copy())
        self._imu.append(imu_rpy.copy())
        self._actions.append(action.copy())

        if image is not None:
            self._images.append(image.copy())

        if mocap_segments is not None and self._mocap_names is None:
            self._mocap_names = list(mocap_segments.keys())
        if self._mocap_names is not None:
            # one row per step (NaN when mocap is missing) so mocap stays aligned with qpos
            row = []
            for name in self._mocap_names:
                seg = (mocap_segments or {}).get(name)
                if seg is None:
                    row.append(np.full(7, np.nan))
                else:
                    row.append(np.concatenate([seg.position / 1000.0, seg.rotation]))  # mm→m
            self._mocap_poses.append(np.stack(row))

    def save(self) -> str:
        episode_id = self._next_episode_id()
        path = os.path.join(self.save_dir, f"episode_{episode_id:04d}.hdf5")

        with h5py.File(path, "w") as f:
            obs = f.create_group("observations")
            obs.create_dataset("qpos",    data=np.stack(self._qpos))
            obs.create_dataset("qvel",    data=np.stack(self._qvel))
            obs.create_dataset("imu_rpy", data=np.stack(self._imu))

            if self._images:
                imgs = obs.create_group("images")
                imgs.create_dataset(
                    "head", data=np.stack(self._images),
                    compression="lzf",
                )

            f.create_dataset("actions", data=np.stack(self._actions))

            if self._mocap_poses:
                mc = f.create_group("mocap")
                mc.create_dataset(
                    "segment_names",
                    data=np.array(self._mocap_names, dtype=h5py.string_dtype()),
                )
                mc.create_dataset("segment_poses", data=np.stack(self._mocap_poses))

            meta = f.create_group("metadata")
            meta.create_dataset("timestamp_start", data=self._t_start)
            meta.create_dataset("control_freq",    data=self.control_freq)
            meta.create_dataset("episode_id",      data=episode_id)

        print(f"Saved {len(self._qpos)} steps → {path}")
        self._reset()
        return path

    def discard(self):
        self._reset()

    def _reset(self):
        self._qpos.clear()
        self._qvel.clear()
        self._imu.clear()
        self._images.clear()
        self._actions.clear()
        self._mocap_poses.clear()
        self._mocap_names = None
        self._t_start = None

    def _next_episode_id(self) -> int:
        existing = [
            f for f in os.listdir(self.save_dir) if f.startswith("episode_") and f.endswith(".hdf5")
        ]
        if not existing:
            return 0
        ids = [int(f.split("_")[1].split(".")[0]) for f in existing]
        return max(ids) + 1
