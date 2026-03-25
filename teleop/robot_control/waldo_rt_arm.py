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
        self._sol_q_prev = np.zeros(ARM_NUM_JOINTS, dtype=np.float64)
        self._sol_dq = np.zeros(ARM_NUM_JOINTS, dtype=np.float64)
        self._sol_dq_prev = np.zeros(ARM_NUM_JOINTS, dtype=np.float64)
        self._sol_ddq = np.zeros(ARM_NUM_JOINTS, dtype=np.float64)
        self._sol_tauff = np.zeros(ARM_NUM_JOINTS, dtype=np.float64)
        self._frame_count = 0
        self._sol_timestamp = 0.0

        # DDS arm controller (handles motor commands, state feedback, velocity clipping)
        self.arm_ctrl = G1_29_ArmController(motion_mode=motion_mode, simulation_mode=simulation_mode)

        # reduced pinocchio model for RNEA / FK computation
        from teleop.robot_control.arm_pink_real import load_model
        self._rnea_model, self._rnea_data, self._left_frame_id, self._right_frame_id, _ = load_model()

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

                # store for recording (+ finite-difference velocity & acceleration)
                with self._action_lock:
                    new_dq = (q_target - self._sol_q_prev) * self.fps
                    self._sol_ddq[:] = (new_dq - self._sol_dq_prev) * self.fps
                    self._sol_dq_prev[:] = self._sol_dq
                    self._sol_dq[:] = new_dq
                    self._sol_q_prev[:] = self._sol_q
                    self._sol_q[:] = q_target
                    self._sol_tauff[:] = tauff
                    self._frame_count += 1
                    self._sol_timestamp = time.time()

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

    def get_current_dual_arm_temperature(self):
        """Return current arm motor temperatures from DDS feedback. 14 motors x 2 sensors each."""
        return self.arm_ctrl.get_current_dual_arm_temperature()

    def get_current_tick(self):
        """Return current motor cycle tick from DDS feedback."""
        return self.arm_ctrl.get_current_tick()

    def get_action_frame_count(self):
        """Return leader-side frame counter (incremented each control loop iteration)."""
        with self._action_lock:
            return self._frame_count

    def get_arm_action_timestamp(self):
        """Return UNIX timestamp (seconds) of when the latest action was computed."""
        with self._action_lock:
            return self._sol_timestamp

    def _fk_wrist_poses(self, q):
        """Run FK and return (left_pos, left_axisangle, right_pos, right_axisangle)."""
        pin.forwardKinematics(self._rnea_model, self._rnea_data, q)
        pin.updateFramePlacements(self._rnea_model, self._rnea_data)
        left_se3 = self._rnea_data.oMf[self._left_frame_id]
        right_se3 = self._rnea_data.oMf[self._right_frame_id]
        left_aa = pin.log3(left_se3.rotation)
        right_aa = pin.log3(right_se3.rotation)
        return (left_se3.translation.copy(), left_aa,
                right_se3.translation.copy(), right_aa)

    def get_current_dual_arm_cartesian_pos(self):
        """Return FK cartesian poses at current sensed joint positions."""
        q = self.arm_ctrl.get_current_dual_arm_q()
        return self._fk_wrist_poses(q)

    def get_arm_action_cartesian_pos(self):
        """Return FK cartesian poses at current action joint targets."""
        with self._action_lock:
            q = self._sol_q.copy()
        return self._fk_wrist_poses(q)

    def _fk_wrist_velocities(self, q, dq):
        """Run FK + Jacobian and return (left_twist_6d, right_twist_6d)."""
        pin.forwardKinematics(self._rnea_model, self._rnea_data, q)
        pin.updateFramePlacements(self._rnea_model, self._rnea_data)
        J_l = pin.computeFrameJacobian(self._rnea_model, self._rnea_data, q,
                                        self._left_frame_id, pin.LOCAL_WORLD_ALIGNED)
        J_r = pin.computeFrameJacobian(self._rnea_model, self._rnea_data, q,
                                        self._right_frame_id, pin.LOCAL_WORLD_ALIGNED)
        return J_l @ dq, J_r @ dq

    def get_current_dual_arm_cartesian_vel(self):
        """Return 6D cartesian twist at current sensed state."""
        q = self.arm_ctrl.get_current_dual_arm_q()
        dq = self.arm_ctrl.get_current_dual_arm_dq()
        return self._fk_wrist_velocities(q, dq)

    def get_arm_action_cartesian_vel(self):
        """Return 6D cartesian twist at action targets (using finite-diff dq)."""
        with self._action_lock:
            q = self._sol_q.copy()
            dq = self._sol_dq.copy()
        return self._fk_wrist_velocities(q, dq)

    def _fk_wrist_accelerations(self, q, dq, ddq):
        """Run FK with accelerations and return (left_accel_6d, right_accel_6d)."""
        pin.forwardKinematics(self._rnea_model, self._rnea_data, q, dq, ddq)
        pin.updateFramePlacements(self._rnea_model, self._rnea_data)
        left_a = pin.getFrameClassicalAcceleration(
            self._rnea_model, self._rnea_data,
            self._left_frame_id, pin.LOCAL_WORLD_ALIGNED)
        right_a = pin.getFrameClassicalAcceleration(
            self._rnea_model, self._rnea_data,
            self._right_frame_id, pin.LOCAL_WORLD_ALIGNED)
        return left_a.vector.copy(), right_a.vector.copy()

    def get_current_dual_arm_cartesian_accel(self):
        """Return 6D cartesian acceleration at current sensed state."""
        q = self.arm_ctrl.get_current_dual_arm_q()
        dq = self.arm_ctrl.get_current_dual_arm_dq()
        ddq = self.arm_ctrl.get_current_dual_arm_ddq()
        return self._fk_wrist_accelerations(q, dq, ddq)

    def get_arm_action_cartesian_accel(self):
        """Return 6D cartesian acceleration at action targets (using finite-diff dq/ddq)."""
        with self._action_lock:
            q = self._sol_q.copy()
            dq = self._sol_dq.copy()
            ddq = self._sol_ddq.copy()
        return self._fk_wrist_accelerations(q, dq, ddq)

    def get_current_dual_arm_cartesian_external_wrench(self):
        """Return estimated Cartesian external wrench (6D) at each wrist: J^{-T} @ tau_ext."""
        q = self.arm_ctrl.get_current_dual_arm_q()
        tau_ext = self.get_current_dual_arm_external_tau()
        pin.forwardKinematics(self._rnea_model, self._rnea_data, q)
        pin.updateFramePlacements(self._rnea_model, self._rnea_data)
        J_l_full = pin.computeFrameJacobian(self._rnea_model, self._rnea_data, q,
                                        self._left_frame_id, pin.LOCAL_WORLD_ALIGNED)
        J_r_full = pin.computeFrameJacobian(self._rnea_model, self._rnea_data, q,
                                        self._right_frame_id, pin.LOCAL_WORLD_ALIGNED)
        # Slice to each arm's own joints (left=0:7, right=7:14)
        J_l = J_l_full[:, :7]
        J_r = J_r_full[:, 7:]
        # J^{-T} @ tau = (J @ J^T)^{-1} @ J @ tau  (pseudoinverse transpose)
        wrench_l = np.linalg.lstsq(J_l.T, tau_ext[:7], rcond=None)[0]
        wrench_r = np.linalg.lstsq(J_r.T, tau_ext[7:], rcond=None)[0]
        return wrench_l, wrench_r

    def get_current_dual_arm_compensation_tau(self):
        """Return dynamics compensation torque: rnea(q_sensed, dq_sensed, 0)."""
        q = self.arm_ctrl.get_current_dual_arm_q()
        dq = self.arm_ctrl.get_current_dual_arm_dq()
        a = np.zeros(self._rnea_model.nv)
        return pin.rnea(self._rnea_model, self._rnea_data, q, dq, a)

    def get_current_dual_arm_external_tau(self):
        """Return estimated external torques: tau_est - rnea(q, dq, 0)."""
        tau_est = self.arm_ctrl.get_current_dual_arm_tau_est()
        tau_comp = self.get_current_dual_arm_compensation_tau()
        return tau_est - tau_comp

    def get_current_motor_q(self):
        """Return current all-body motor positions from DDS feedback."""
        return self.arm_ctrl.get_current_motor_q()

    def get_arm_action(self):
        """Return latest arm joint angle targets sent this frame.
        For use in recording."""
        with self._action_lock:
            return self._sol_q.copy()

    def get_arm_action_velocity(self):
        """Return finite-difference velocity of arm action targets (rad/s)."""
        with self._action_lock:
            return self._sol_dq.copy()

    def get_arm_action_acceleration(self):
        """Return finite-difference acceleration of arm action targets (rad/s^2)."""
        with self._action_lock:
            return self._sol_ddq.copy()

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
