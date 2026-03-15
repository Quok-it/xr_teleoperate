from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
import zmq
from enum import IntEnum
import threading
import time

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

brainco_Num_Motors = 6
kTopicbraincoLeftCommand = "rt/brainco/left/cmd"
kTopicbraincoLeftState = "rt/brainco/left/state"
kTopicbraincoRightCommand = "rt/brainco/right/cmd"
kTopicbraincoRightState = "rt/brainco/right/state"

# inference_server.py publishes 6 float32 joint angles per hand via ZMQ PUB.
# revo2 joint order: [thumb_mc, thumb_px, index, middle, ring, pinky]
# Maps 1:1 to brainco motor order: [thumb, thumb-aux, index, middle, ring, pinky]
WALDO_RIGHT_HAND_PORT = 5555
WALDO_LEFT_HAND_PORT = 5556

# GeoRT revo2 joint upper limits in radians (lower limits are all 0.0)
# Order: [thumb_mc, thumb_px, index, middle, ring, pinky]
_REVO2_UPPER_LIMITS = np.array([1.57, 1.03, 1.41, 1.41, 1.41, 1.41])

def _normalize_to_brainco(q_radians):
    """Convert GeoRT revo2 joint angles (radians) to brainco motor range [0, 1].
    0.0 = fully open, 1.0 = fully closed."""
    return np.clip(q_radians / _REVO2_UPPER_LIMITS, 0.0, 1.0)


class Waldo_Brainco_Controller:
    """Brainco hand controller that receives joint angles directly from
    inference_server.py via ZMQ, bypassing XR hand skeleton retargeting.

    Subscribes to ZMQ PUB sockets for joint angle data (6 float32 per hand),
    publishes motor commands via DDS, and exposes state/action arrays for recording.
    """

    def __init__(self, dual_hand_data_lock=None, dual_hand_state_array=None,
                 dual_hand_action_array=None, fps=120.0, simulation_mode=False,
                 right_hand_port=WALDO_RIGHT_HAND_PORT, left_hand_port=WALDO_LEFT_HAND_PORT):
        logger_mp.info("Initialize Waldo_Brainco_Controller...")
        self.fps = fps
        self.simulation_mode = simulation_mode
        self.hand_sub_ready = False
        self.dual_hand_data_lock = dual_hand_data_lock
        self.dual_hand_state_array = dual_hand_state_array
        self.dual_hand_action_array = dual_hand_action_array
        self.running = True

        # ZMQ subscriber setup for inference_server joint angles
        self.right_hand_port = right_hand_port
        self.left_hand_port = left_hand_port

        # latest joint angles from ZMQ (protected by lock)
        self._zmq_lock = threading.Lock()
        self._right_q_target = np.zeros(brainco_Num_Motors, dtype=np.float64)
        self._left_q_target = np.zeros(brainco_Num_Motors, dtype=np.float64)
        self._zmq_connected = False

        # DDS publishers for motor commands
        self.LeftHandCmd_publisher = ChannelPublisher(kTopicbraincoLeftCommand, MotorCmds_)
        self.LeftHandCmd_publisher.Init()
        self.RightHandCmd_publisher = ChannelPublisher(kTopicbraincoRightCommand, MotorCmds_)
        self.RightHandCmd_publisher.Init()

        # DDS subscribers for motor state feedback
        self.LeftHandState_subscriber = ChannelSubscriber(kTopicbraincoLeftState, MotorStates_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicbraincoRightState, MotorStates_)
        self.RightHandState_subscriber.Init()

        # current motor state from DDS feedback
        self._state_lock = threading.Lock()
        self._left_hand_state = np.zeros(brainco_Num_Motors, dtype=np.float64)
        self._right_hand_state = np.zeros(brainco_Num_Motors, dtype=np.float64)

        # initialize DDS cmd messages
        self.left_hand_msg = MotorCmds_()
        self.left_hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(brainco_Num_Motors)]
        self.right_hand_msg = MotorCmds_()
        self.right_hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(brainco_Num_Motors)]

        for i in range(brainco_Num_Motors):
            self.left_hand_msg.cmds[i].q = 0.0
            self.left_hand_msg.cmds[i].dq = 1.0
            self.right_hand_msg.cmds[i].q = 0.0
            self.right_hand_msg.cmds[i].dq = 1.0

        # start DDS state subscriber thread
        self._dds_state_thread = threading.Thread(target=self._subscribe_hand_state, daemon=True)
        self._dds_state_thread.start()

        while not self.hand_sub_ready:
            time.sleep(0.1)
            logger_mp.warning("[Waldo_Brainco] Waiting for DDS hand state...")
        logger_mp.info("[Waldo_Brainco] DDS hand state connected.")

        # start ZMQ subscriber thread
        self._zmq_thread = threading.Thread(target=self._subscribe_zmq, daemon=True)
        self._zmq_thread.start()

        while not self._zmq_connected:
            time.sleep(0.1)
            logger_mp.warning("[Waldo_Brainco] Waiting for ZMQ inference data...")
        logger_mp.info("[Waldo_Brainco] ZMQ inference connected.")

        # start control loop thread
        self._ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._ctrl_thread.start()

        logger_mp.info("Initialize Waldo_Brainco_Controller OK!")

    def _subscribe_hand_state(self):
        """Read motor state feedback from DDS (runs in background thread)."""
        while self.running:
            left_msg = self.LeftHandState_subscriber.Read()
            right_msg = self.RightHandState_subscriber.Read()
            self.hand_sub_ready = True
            if left_msg is not None and right_msg is not None:
                with self._state_lock:
                    for idx, jid in enumerate(Brainco_Left_Hand_JointIndex):
                        self._left_hand_state[idx] = left_msg.states[jid].q
                    for idx, jid in enumerate(Brainco_Right_Hand_JointIndex):
                        self._right_hand_state[idx] = right_msg.states[jid].q
            time.sleep(0.002)

    def _subscribe_zmq(self):
        """Subscribe to inference_server.py ZMQ PUB sockets for joint angles.

        inference_server publishes 6 float32 values as raw bytes per message.
        revo2 joint order: [thumb_mc, thumb_px, index, middle, ring, pinky]
        """
        ctx = zmq.Context()

        right_sub = ctx.socket(zmq.SUB)
        right_sub.connect(f"tcp://localhost:{self.right_hand_port}")
        right_sub.setsockopt(zmq.SUBSCRIBE, b"")
        right_sub.setsockopt(zmq.CONFLATE, 1)  # only keep latest message

        left_sub = ctx.socket(zmq.SUB)
        left_sub.connect(f"tcp://localhost:{self.left_hand_port}")
        left_sub.setsockopt(zmq.SUBSCRIBE, b"")
        left_sub.setsockopt(zmq.CONFLATE, 1)

        poller = zmq.Poller()
        poller.register(right_sub, zmq.POLLIN)
        poller.register(left_sub, zmq.POLLIN)

        try:
            while self.running:
                socks = dict(poller.poll(timeout=100))

                if right_sub in socks:
                    data = right_sub.recv()
                    joints = np.frombuffer(data, dtype=np.float32).astype(np.float64)
                    if len(joints) == brainco_Num_Motors:
                        with self._zmq_lock:
                            self._right_q_target[:] = joints
                            self._zmq_connected = True

                if left_sub in socks:
                    data = left_sub.recv()
                    joints = np.frombuffer(data, dtype=np.float32).astype(np.float64)
                    if len(joints) == brainco_Num_Motors:
                        with self._zmq_lock:
                            self._left_q_target[:] = joints
                            self._zmq_connected = True
        finally:
            right_sub.close()
            left_sub.close()
            ctx.term()

    def _control_loop(self):
        """Main control loop: read ZMQ targets, publish DDS commands, update recording arrays."""
        try:
            while self.running:
                start_time = time.time()

                # read latest joint angle targets from ZMQ (in radians from GeoRT)
                with self._zmq_lock:
                    left_q_raw = self._left_q_target.copy()
                    right_q_raw = self._right_q_target.copy()

                # normalize radians to [0, 1] range for brainco motors
                # GeoRT revo2 outputs radians with lower=0 for all joints.
                # Brainco expects 0.0 = fully open, 1.0 = fully closed.
                # Joint upper limits from revo2 config (radians):
                #   idx 0 (thumb_mc): 1.57,  idx 1 (thumb_px): 1.03
                #   idx 2 (index):    1.41,  idx 3 (middle):   1.41
                #   idx 4 (ring):     1.41,  idx 5 (pinky):    1.41
                left_q_target = _normalize_to_brainco(left_q_raw)
                right_q_target = _normalize_to_brainco(right_q_raw)

                # read current motor state from DDS feedback
                with self._state_lock:
                    state_data = np.concatenate((self._left_hand_state.copy(), self._right_hand_state.copy()))

                # build action data for recording (normalized [0,1] values)
                action_data = np.concatenate((left_q_target, right_q_target))

                # update recording arrays
                if self.dual_hand_state_array is not None and self.dual_hand_action_array is not None:
                    with self.dual_hand_data_lock:
                        self.dual_hand_state_array[:] = state_data
                        self.dual_hand_action_array[:] = action_data

                # don't publish until we've received real data from inference_server
                if not self._zmq_connected:
                    continue

                # publish motor commands via DDS
                self._ctrl_dual_hand(left_q_target, right_q_target)

                elapsed = time.time() - start_time
                sleep_time = max(0, (1 / self.fps) - elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Waldo_Brainco_Controller control loop closed.")

    def _ctrl_dual_hand(self, left_q_target, right_q_target):
        """Publish joint angle targets to brainco hands via DDS."""
        for idx, jid in enumerate(Brainco_Left_Hand_JointIndex):
            self.left_hand_msg.cmds[jid].q = left_q_target[idx]
        for idx, jid in enumerate(Brainco_Right_Hand_JointIndex):
            self.right_hand_msg.cmds[jid].q = right_q_target[idx]

        self.LeftHandCmd_publisher.Write(self.left_hand_msg)
        self.RightHandCmd_publisher.Write(self.right_hand_msg)
        print(f"Published DDS commands - Left: {left_q_target}, Right: {right_q_target}")

    def get_hand_state(self):
        """Return current hand motor state as (left_state, right_state), each np.array of shape (6,).
        For use in recording."""
        with self._state_lock:
            return self._left_hand_state.copy(), self._right_hand_state.copy()

    def get_hand_action(self):
        """Return latest hand joint angle targets as (left_action, right_action), each np.array of shape (6,).
        For use in recording."""
        with self._zmq_lock:
            return self._left_q_target.copy(), self._right_q_target.copy()

    def stop(self):
        """Stop all threads and clean up."""
        self.running = False


# Motor joint order (same as original brainco controller)
# ┌──────┬───────┬────────────┬────────┬────────┬────────┬────────┐
# │ Id   │   0   │     1      │   2    │   3    │   4    │   5    │
# ├──────┼───────┼────────────┼────────┼────────┼────────┼────────┤
# │Joint │ thumb │ thumb-aux  |  index │ middle │  ring  │  pinky │
# └──────┴───────┴────────────┴────────┴────────┴────────┴────────┘
class Brainco_Right_Hand_JointIndex(IntEnum):
    kRightHandThumb = 0
    kRightHandThumbAux = 1
    kRightHandIndex = 2
    kRightHandMiddle = 3
    kRightHandRing = 4
    kRightHandPinky = 5

class Brainco_Left_Hand_JointIndex(IntEnum):
    kLeftHandThumb = 0
    kLeftHandThumbAux = 1
    kLeftHandIndex = 2
    kLeftHandMiddle = 3
    kLeftHandRing = 4
    kLeftHandPinky = 5
