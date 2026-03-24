from teleop.robot_control.robot_arm import G1_29_ArmController, G1_29_JointArmIndex, G1_29_JointIndex

import numpy as np
import pinocchio as pin
import zmq
import threading
import time

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

ARM_NUM_JOINTS = 14

# stream_arm_zmq.py publishes 14x float32 joint angles (radians) via ZMQ PUB
# in sequential order: left[0:7] + right[7:14]
# Maps 1:1 to G1_29_ArmController.ctrl_dual_arm() expected order.
WALDO_ARM_PORT = 5557


class Waldo_Arm_Controller:
    """Arm controller that receives joint angles from stream_arm_zmq.py via ZMQ,
    and forwards them to the robot via G1_29_ArmController (DDS).

    Subscribes to ZMQ PUB socket for joint angle data (14x float32, sequential L/R),
    computes feedforward torques via RNEA, and publishes motor commands via DDS.
    Exposes state/action for recording.
    """

    def __init__(self, motion_mode=False, simulation_mode=False, fps=60.0,
                 arm_port=WALDO_ARM_PORT):
        logger_mp.info("Initialize Waldo_Arm_Controller...")
        self.fps = fps
        self.simulation_mode = simulation_mode
        self.running = True
        self._idle = True
        self.arm_port = arm_port
        self._zmq_connected = False

        # latest joint angles from ZMQ (protected by lock)
        self._zmq_lock = threading.Lock()
        self._q_target = np.zeros(ARM_NUM_JOINTS, dtype=np.float64)

        # latest sol_q and tauff sent to arm_ctrl (for recording)
        self._action_lock = threading.Lock()
        self._sol_q = np.zeros(ARM_NUM_JOINTS, dtype=np.float64)
        self._sol_tauff = np.zeros(ARM_NUM_JOINTS, dtype=np.float64)

        # DDS arm controller (handles motor commands, state feedback, velocity clipping)
        self.arm_ctrl = G1_29_ArmController(motion_mode=motion_mode, simulation_mode=simulation_mode)

        # reduced pinocchio model for RNEA torque computation
        from teleop.robot_control.arm_pink_real import load_model
        self._rnea_model, self._rnea_data, _, _, _ = load_model()

        # start control loop thread (publishes zeros while idle / waiting for ZMQ)
        self._ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._ctrl_thread.start()

        # start ZMQ subscriber thread
        self._zmq_thread = threading.Thread(target=self._subscribe_zmq, daemon=True)
        self._zmq_thread.start()

        while not self._zmq_connected:
            time.sleep(0.1)
            logger_mp.warning("[Waldo_Arm] Waiting for ZMQ arm data on port %d...", self.arm_port)
        logger_mp.info("[Waldo_Arm] ZMQ arm data connected.")

        logger_mp.info("Initialize Waldo_Arm_Controller OK!")

    def start(self):
        """Activate: forward ZMQ data to robot."""
        # restore gains for active control
        arm_indices = set(member.value for member in G1_29_JointArmIndex)
        for id in G1_29_JointIndex:
            if id.value in arm_indices:
                if self.arm_ctrl._Is_wrist_motor(id):
                    self.arm_ctrl.msg.motor_cmd[id].kp = self.arm_ctrl.kp_wrist
                else:
                    self.arm_ctrl.msg.motor_cmd[id].kp = self.arm_ctrl.kp_low
        self._idle = False
        self.speed_gradual_max(t=1.0)

    def _subscribe_zmq(self):
        """Subscribe to stream_arm_zmq.py ZMQ PUB socket for joint angles.

        stream_arm_zmq publishes 14 float32 values as raw bytes per message,
        in sequential order: left[0:7] + right[7:14].
        """
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.connect(f"tcp://192.168.4.46:{self.arm_port}")
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        sub.setsockopt(zmq.CONFLATE, 1)  # only keep latest message

        try:
            while self.running:
                if sub.poll(timeout=100):
                    data = sub.recv()
                    joints = np.frombuffer(data, dtype=np.float32).astype(np.float64)
                    if len(joints) == ARM_NUM_JOINTS:
                        with self._zmq_lock:
                            self._q_target[:] = joints
                            self._zmq_connected = True
        finally:
            sub.close()
            ctx.term()

    def _control_loop(self):
        """Main control loop: read ZMQ targets, compute torques, send to arm_ctrl."""
        try:
            while self.running:
                start_time = time.time()

                if not self._zmq_connected:
                    time.sleep(1.0 / self.fps)
                    continue

                if self._idle:
                    time.sleep(1.0 / self.fps)
                    continue
                else:
                    # active: forward ZMQ data
                    with self._zmq_lock:
                        q_target = self._q_target.copy()

                # compute feedforward torques via RNEA
                v = np.zeros(self._rnea_model.nv)
                tauff = pin.rnea(self._rnea_model, self._rnea_data, q_target, v, v)

                # send to DDS arm controller
                self.arm_ctrl.ctrl_dual_arm(q_target, tauff)

                # store for recording
                with self._action_lock:
                    self._sol_q[:] = q_target
                    self._sol_tauff[:] = tauff

                elapsed = time.time() - start_time
                sleep_time = max(0, (1 / self.fps) - elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Waldo_Arm_Controller control loop closed.")

    # --- Interface methods (match G1_29_ArmController) ---

    def get_current_dual_arm_q(self):
        """Return current arm joint positions from DDS feedback."""
        return self.arm_ctrl.get_current_dual_arm_q()

    def get_current_dual_arm_dq(self):
        """Return current arm joint velocities from DDS feedback."""
        return self.arm_ctrl.get_current_dual_arm_dq()

    def get_current_dual_arm_ddq(self):
        """Return current arm joint accelerations from DDS feedback."""
        return self.arm_ctrl.get_current_dual_arm_ddq()

    def get_current_dual_arm_tau_est(self):
        """Return current arm estimated torques from DDS feedback."""
        return self.arm_ctrl.get_current_dual_arm_tau_est()

    def get_current_motor_q(self):
        """Return current all-body motor positions from DDS feedback."""
        return self.arm_ctrl.get_current_motor_q()

    def get_arm_action(self):
        """Return latest arm joint angle targets sent this frame.
        For use in recording."""
        with self._action_lock:
            return self._sol_q.copy()

    def get_arm_tauff(self):
        """Return latest feedforward torques sent this frame.
        For use in recording."""
        with self._action_lock:
            return self._sol_tauff.copy()

    def ctrl_dual_arm_go_home(self):
        """Return arms to home position."""
        self.arm_ctrl.ctrl_dual_arm_go_home()

    def speed_gradual_max(self, t=5.0):
        """Ramp arm velocity limit 20->30 rad/s over t seconds."""
        self.arm_ctrl.speed_gradual_max(t)

    def stop(self):
        """Idle: block ZMQ and return arms home."""
        self._idle = True
        self.arm_ctrl.arm_velocity_limit = 0.5
        self.arm_ctrl.ctrl_dual_arm_go_home()
