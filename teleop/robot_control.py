import time
import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

NUM_JOINTS = 29

# PD gains from Unitree example — tuned for position tracking
KP = [
    60, 60, 60, 100, 40, 40,      # left leg
    60, 60, 60, 100, 40, 40,      # right leg
    60, 40, 40,                   # waist
    40, 40, 40, 40, 40, 40, 40,   # left arm
    40, 40, 40, 40, 40, 40, 40,   # right arm
]
KD = [
    1, 1, 1, 2, 1, 1,
    1, 1, 1, 2, 1, 1,
    1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
]

class G1RobotControl:
    """
    Low-level joint position control for Unitree G1 via DDS.

    Usage:
        robot = G1RobotControl(network_interface="eth0")
        robot.init()
        robot.move_to_zero(duration=3.0)   # safe start
        robot.set_joint_positions(q)        # teleop loop
        robot.stop()
    """

    def __init__(self, network_interface: str = "eth0"):
        self.network_interface = network_interface
        self._low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._low_state: LowState_ | None = None
        self._mode_machine = 0
        self._mode_machine_ready = False
        self._crc = CRC()

    def init(self):
        ChannelFactoryInitialize(0, self.network_interface)

        self._msc = MotionSwitcherClient()
        self._msc.SetTimeout(5.0)
        self._msc.Init()

        # release any high-level mode (sport/loco) before taking low-level control
        status, result = self._msc.CheckMode()
        while result["name"]:
            print(f"Releasing mode: {result['name']}")
            self._msc.ReleaseMode()
            status, result = self._msc.CheckMode()
            time.sleep(1.0)

        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()

        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_low_state, 10)

        # wait for first state message
        print("Waiting for robot state...")
        while not self._mode_machine_ready:
            time.sleep(0.1)
        print("Robot ready.")

    def _on_low_state(self, msg: LowState_):
        self._low_state = msg
        if not self._mode_machine_ready:
            self._mode_machine = msg.mode_machine
            self._mode_machine_ready = True

    @property
    def joint_positions(self) -> np.ndarray:
        """Current motor positions in radians (29,)."""
        if self._low_state is None:
            return np.zeros(NUM_JOINTS)
        return np.array([self._low_state.motor_state[i].q for i in range(NUM_JOINTS)])

    @property
    def joint_velocities(self) -> np.ndarray:
        if self._low_state is None:
            return np.zeros(NUM_JOINTS)
        return np.array([self._low_state.motor_state[i].dq for i in range(NUM_JOINTS)])

    @property
    def imu_rpy(self) -> np.ndarray:
        if self._low_state is None:
            return np.zeros(3)
        return np.array(self._low_state.imu_state.rpy)

    def move_to_zero(self, duration: float = 3.0):
        """Smoothly interpolate all joints to zero over `duration` seconds."""
        start = self.joint_positions.copy()
        t0 = time.time()
        dt = 0.01
        while True:
            elapsed = time.time() - t0
            if elapsed >= duration:
                break
            ratio = np.clip(elapsed / duration, 0.0, 1.0)
            target = start * (1.0 - ratio)
            self._publish(target)
            time.sleep(dt)
        self._publish(np.zeros(NUM_JOINTS))

    def set_joint_positions(self, q: np.ndarray):
        """Send a joint position command immediately. Call at your control rate."""
        assert len(q) == NUM_JOINTS
        self._publish(q)

    def _publish(self, q: np.ndarray):
        self._low_cmd.mode_pr = 0  # PR mode
        self._low_cmd.mode_machine = self._mode_machine
        for i in range(NUM_JOINTS):
            self._low_cmd.motor_cmd[i].mode = 1
            self._low_cmd.motor_cmd[i].q = float(q[i])
            self._low_cmd.motor_cmd[i].dq = 0.0
            self._low_cmd.motor_cmd[i].tau = 0.0
            self._low_cmd.motor_cmd[i].kp = KP[i]
            self._low_cmd.motor_cmd[i].kd = KD[i]
        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._pub.Write(self._low_cmd)

    def damp(self, duration: float = 2.0, kd: float = 3.0):
        """Damping mode: kp=0, kd>0 — joints resist fast motion but sink slowly under gravity."""
        t_end = time.time() + duration
        while time.time() < t_end:
            self._low_cmd.mode_pr = 0
            self._low_cmd.mode_machine = self._mode_machine
            for i in range(NUM_JOINTS):
                cmd = self._low_cmd.motor_cmd[i]
                cmd.mode = 1
                cmd.q, cmd.dq, cmd.tau = 0.0, 0.0, 0.0
                cmd.kp, cmd.kd = 0.0, kd
            self._low_cmd.crc = self._crc.Crc(self._low_cmd)
            self._pub.Write(self._low_cmd)
            time.sleep(0.01)

    def stop(self):
        """Disable all motors."""
        self._low_cmd.mode_machine = self._mode_machine
        for i in range(NUM_JOINTS):
            self._low_cmd.motor_cmd[i].mode = 0
        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._pub.Write(self._low_cmd)
