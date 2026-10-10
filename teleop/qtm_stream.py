import asyncio
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import qtm_rt


class SkeletonSegment:
    def __init__(self, name, position, rotation):
        self.name = name
        self.position = np.array(position, dtype=float)   # (x, y, z) in mm, global
        self.rotation = np.array(rotation, dtype=float)   # quaternion (x, y, z, w), global


def parse_skeleton_names(xml_string: str) -> list[tuple[str, dict[int, str]]]:
    """
    QTM skeleton parameter XML → [(skeleton_name, {segment_id: segment_name}), ...]
    in the same order as skeletons appear in streamed packets.
    """
    root = ET.fromstring(xml_string)
    skeletons = []
    for sk in root.iter("Skeleton"):
        ids = {int(seg.attrib["ID"]): seg.attrib["Name"] for seg in sk.iter("Segment")}
        skeletons.append((sk.attrib.get("Name", ""), ids))
    return skeletons


class QTMStream:
    """Connects to QTM and streams skeleton segment poses (global frame)."""

    # Segments the G1 U2 retargeter needs (no hand fingers)
    REQUIRED_SEGMENTS = [
        "Hips",
        "Spine", "Spine1", "Spine2",
        "Neck", "Head",
        "RightShoulder", "LeftShoulder",
        "RightArm", "LeftArm",
        "RightForeArm", "LeftForeArm",
        "RightHand", "LeftHand",
        "RightUpLeg", "LeftUpLeg",
        "RightLeg", "LeftLeg",
        "RightFoot", "LeftFoot",
        "RightToeBase", "LeftToeBase",
    ]

    def __init__(self, host: str, port: int = 22222, skeleton_name: str = "Actor1"):
        self.host = host
        self.port = port
        self.skeleton_name = skeleton_name
        self._conn = None
        self._loop = None
        self._skeleton_index = None           # which skeleton in the packet is ours
        self._segment_names: dict[int, str] = {}
        self._latest: dict[str, SkeletonSegment] = {}
        self._last_update: float | None = None
        self.error: str | None = None         # set if connecting / setup failed

    @property
    def latest(self) -> dict[str, SkeletonSegment]:
        return self._latest

    @property
    def ready(self) -> bool:
        return len(self._latest) >= len(self.REQUIRED_SEGMENTS)

    @property
    def age(self) -> float:
        """Seconds since the last skeleton packet (inf if none yet)."""
        return float("inf") if self._last_update is None else time.time() - self._last_update

    def valid(self) -> bool:
        """All required segments present with finite pose (QTM sends NaN when tracking is lost)."""
        latest = self._latest
        for name in self.REQUIRED_SEGMENTS:
            seg = latest.get(name)
            if seg is None or not (np.all(np.isfinite(seg.position)) and np.all(np.isfinite(seg.rotation))):
                return False
        return True

    def _set_skeleton_params(self, xml_string: str) -> None:
        skeletons = parse_skeleton_names(xml_string)
        if not skeletons:
            raise RuntimeError("QTM reports no skeletons — is the skeleton defined and calibrated?")
        names = [n for n, _ in skeletons]
        if self.skeleton_name in names:
            self._skeleton_index = names.index(self.skeleton_name)
        elif len(skeletons) == 1:
            print(f"WARNING: skeleton '{self.skeleton_name}' not found, using the only one: '{names[0]}'")
            self._skeleton_index = 0
        else:
            # never guess between several actors — could follow the wrong person
            raise RuntimeError(f"Skeleton '{self.skeleton_name}' not found in QTM; available: {names}")
        self._segment_names = skeletons[self._skeleton_index][1]

    def _on_packet(self, packet):
        result = packet.get_skeletons()            # (component_info, skeletons) or None
        if result is None:
            return
        _, skeletons = result
        if not skeletons or self._skeleton_index is None or self._skeleton_index >= len(skeletons):
            return
        # build a new dict and swap it in, so readers never see a half-updated frame
        latest = {}
        for seg_id, p, r in skeletons[self._skeleton_index]:   # p: (x,y,z) mm, r: (x,y,z,w)
            name = self._segment_names.get(seg_id)
            if name is not None:
                latest[name] = SkeletonSegment(name, (p.x, p.y, p.z), (r.x, r.y, r.z, r.w))
        self._latest = latest
        self._last_update = time.time()

    async def _connect(self):
        self._conn = await qtm_rt.connect(self.host, port=self.port)
        if self._conn is None:
            raise ConnectionError(f"Cannot connect to QTM at {self.host}:{self.port}")
        self._set_skeleton_params(await self._conn.get_parameters(parameters=["skeleton"]))
        await self._conn.stream_frames(components=["skeleton:global"], on_packet=self._on_packet)

    def _run(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
        except Exception as e:  # surfaced via .error; teleop sees ready == False
            self.error = str(e)
            print(f"QTM ERROR: {e}")
            return
        self._loop.run_forever()   # keep processing incoming packets

    def start(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._conn and self._loop and self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._conn.stream_frames_stop(), self._loop)
            try:
                fut.result(timeout=1.0)
            except Exception:
                pass
            try:
                self._conn.disconnect()
            except Exception:   # transport already gone (e.g. QTM dropped the connection)
                pass
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
