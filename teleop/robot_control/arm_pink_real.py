#!/usr/bin/env python3
"""
VIVE tracker teleoperation for Unitree G1 --> 6DOF IK using pink library.

Uses pink (QP-based differential IK on Pinocchio) with a reduced model
that locks all non-arm joints. Tracks both position and orientation
of wrist trackers.

Usage:
  python arm_pink_real.py                    # press-to-capture mode
  python arm_pink_real.py --continuous       # continuous streaming mode
  python arm_pink_real.py --continuous --rate 60  # streaming at 60 Hz
  python arm_pink_real.py --test             # test mode (no trackers)
  python arm_pink_real.py --pos-only         # position-only (ignore orientation)
"""

import sys
import os
import time
import argparse
import numpy as np
import pinocchio as pin
import pink
import zmq
from pink import Configuration
from pink.tasks import FrameTask, PostureTask

# --- CONFIG ---
# Use g1_body29_hand14.urdf (same arm joints as g1_29dof, plus 14 hand joints)
# Relative to this file: teleop/robot_control/ -> ../../assets/g1/
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "../../assets/g1/g1_body29_hand14.urdf")

LEFT_EE_FRAME = "left_wrist_yaw_link"
RIGHT_EE_FRAME = "right_wrist_yaw_link"

# joint IDs (1-indexed) in full model for each arm (g1_body29_hand14 layout)
# Left arm: joints 16-22, Right arm: joints 30-36 (hand joints 23-29, 37-43 in between)
_LEFT_ARM_JOINT_IDS = [16, 17, 18, 19, 20, 21, 22]
_RIGHT_ARM_JOINT_IDS = [30, 31, 32, 33, 34, 35, 36]
_ARM_JOINT_IDS = set(_LEFT_ARM_JOINT_IDS + _RIGHT_ARM_JOINT_IDS)

# q-indices in FULL model (for output reordering)
_FULL_LEFT_ARM_IDX = [15, 16, 17, 18, 19, 20, 21]
_FULL_RIGHT_ARM_IDX = [29, 30, 31, 32, 33, 34, 35]

# interleaved L/R output order (full model q-indices)
REORDER_IDX = [15, 29, 16, 30, 17, 31, 18, 32, 19, 33, 20, 34, 21, 35]

# sequential output order: left[0:7] then right[7:14] (matches G1_29_ArmController)
SEQUENTIAL_IDX = _FULL_LEFT_ARM_IDX + _FULL_RIGHT_ARM_IDX

JOINT_NAMES_INTERLEAVED = [
    "L_shoulder_pitch", "R_shoulder_pitch",
    "L_shoulder_roll",  "R_shoulder_roll",
    "L_shoulder_yaw",   "R_shoulder_yaw",
    "L_elbow",          "R_elbow",
    "L_wrist_roll",     "R_wrist_roll",
    "L_wrist_pitch",    "R_wrist_pitch",
    "L_wrist_yaw",      "R_wrist_yaw",
]

# calibration poses in reduced model (q-indices -> degrees)
# reduced model: q[0:7] = left (sp,sr,sy,e,wr,wp,wy), q[7:14] = right
#
# pose 1: t-pose palms down
T_POSE_REDUCED = {
    1: 90.0, 3: 90.0,          # left shoulder_roll, elbow
    8: -90.0, 10: 90.0,        # right shoulder_roll, elbow
}
# pose 2: arms forward palms down
ARMS_FWD_REDUCED = {
    0: -90.0, 1: 15.0, 3: 90.0, 4: 90.0,       # left: sp, sr, e, wr
    7: -90.0, 8: -15.0, 10: 90.0, 11: -90.0,    # right: sp, sr, e, wr
}

# Pink IK settings (from Pinocchio)
IK_DT = 0.01
IK_POS_COST = 1.0
IK_ORI_COST = 0.3  # orientation cost --> lower than position so arm prioritizes actually getting there
IK_POSTURE_COST = 7e-2
IK_ITERS = 200
IK_ITERS_CONTINUOUS = 10  # fewer iters when warm-starting frame-to-frame
IK_SOLVER = "quadprog"
SMOOTH_ALPHA = 0.1  # EMA smoothing for position (0=no change, 1=no smoothing)
SMOOTH_ALPHA_ROT = 0.2  # SLERP smoothing for rotation (lower = more filtering)
ROT_JUMP_THRESHOLD = np.radians(60)  # reject single-frame jumps larger than this (otherwise jitter)
MAX_JOINT_VEL = np.radians(5)  # max joint change per frame (rad) — velocity clamp

SMOOTH_FILTER_WEIGHTS = np.array([0.25, 0.2, 0.18, 0.15, 0.12, 0.1])

MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "../../assets/g1/")


class WeightedMovingFilter:
    """weighted moving average filter for joint-space smoothing."""
    def __init__(self, weights, dim):
        self.weights = weights / weights.sum()
        self.dim = dim
        self.buffer = []
        self.filtered_data = np.zeros(dim)

    def add_data(self, data):
        self.buffer.append(data.copy())
        if len(self.buffer) > len(self.weights):
            self.buffer.pop(0)
        n = len(self.buffer)
        w = self.weights[-n:]
        w = w / w.sum()
        self.filtered_data = sum(w[i] * self.buffer[i] for i in range(n))


def load_model():
    """load URDF and build reduced model with only arm joints."""
    model_full = pin.buildModelFromUrdf(URDF_PATH)

    # lock all joints except arms
    joints_to_lock = [i for i in range(1, model_full.njoints)
                      if i not in _ARM_JOINT_IDS]
    q_lock = pin.neutral(model_full)
    model = pin.buildReducedModel(model_full, joints_to_lock, q_lock)

    # tighten shoulder yaw upper limits: URDF allows up to 2.618,
    # clamp to 2.0 so IK stays in a natural human range
#    model.upperPositionLimit[2] = 2.0      # left shoulder yaw
#    model.upperPositionLimit[9] = 2.0      # right shoulder yaw

    # tighten elbow upper limits: URDF allows up to 2.0944 (120 deg),
    # clamp to 1.5708 (90 deg) so IK stays in a natural range
    model.upperPositionLimit[3] = 1.5708   # left elbow
    model.upperPositionLimit[10] = 1.5708  # right elbow

    data = model.createData()

    left_id = model.getFrameId(LEFT_EE_FRAME)
    right_id = model.getFrameId(RIGHT_EE_FRAME)

    return model, data, left_id, right_id, model_full


def make_tpose_q(model):
    """build t-pose config for reduced model."""
    q = pin.neutral(model)
    for idx, deg in T_POSE_REDUCED.items():
        q[idx] = np.radians(deg)
    return q


def make_armsfwd_q(model):
    """build arms-forward config for reduced model."""
    q = pin.neutral(model)
    for idx, deg in ARMS_FWD_REDUCED.items():
        q[idx] = np.radians(deg)
    return q


def reduced_to_full(q_reduced, model_full):
    """map reduced model q (14 DOF) back to full model q (43 DOF) for output."""
    q_full = pin.neutral(model_full)
    # reduced q[0:7] -> full q[15:22] (left arm)
    # reduced q[7:14] -> full q[29:36] (right arm, after 7 left hand joints)
    q_full[15:22] = q_reduced[0:7]
    q_full[29:36] = q_reduced[7:14]
    return q_full


def extract_pose(mat):
    """extract position (3,) and rotation (3x3) from OpenVR 3x4 pose matrix."""
    pos = np.array([mat[r][3] for r in range(3)])
    rot = np.array([[mat[r][c] for c in range(3)] for r in range(3)])
    return pos, rot


def smooth_rotation(R_prev, R_new, alpha=SMOOTH_ALPHA_ROT):
    """SLERP-based EMA for rotation matrices via log/exp map."""
    dR = R_prev.T @ R_new
    log_dR = pin.log3(dR)
    return R_prev @ pin.exp3(alpha * log_dR)


def filter_rotation(R_prev, R_new, threshold=ROT_JUMP_THRESHOLD):
    """reject large single-frame orientation jumps (quaternion sign flips / glitches).

    returns R_new if the angular change is below threshold, else R_prev.
    """
    if R_prev is None:
        return R_new
    dR = R_prev.T @ R_new
    angle = np.linalg.norm(pin.log3(dR))
    if angle > threshold:
        return R_prev
    return R_new


def solve_ik_pink(config, tasks, n_iters=IK_ITERS):
    """run n_iters of pink differential IK."""
    for _ in range(n_iters):
        vel = pink.solve_ik(config, tasks, IK_DT, solver=IK_SOLVER)
        config.integrate_inplace(vel, IK_DT)


def _setup_viz(model_full):
    """set up meshcat 3D visualization. 

    returns visualizer or None.
    """
    try:
        from pinocchio.visualize import MeshcatVisualizer
        visual_model = pin.buildGeomFromUrdf(
            model_full, URDF_PATH, pin.GeometryType.VISUAL, MESH_DIR)
        viz = MeshcatVisualizer(model_full, pin.GeometryModel(), visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()
        viz.display(pin.neutral(model_full))
        print(f"Visualizer: {viz.viewer.url()}")
        return viz
    except Exception as e:
        print(f"Warning: visualization failed: {e}")
        return None


def run_test(model, data, left_id, right_id, model_full, pos_only=False):
    """test IK without trackers --> set known targets and solve."""
    print("=== IK Test Mode (pink, reduced model) ===")
    print(f"Model: {model.nq} DOF, {model.nframes} frames")

    q_tpose = make_tpose_q(model)
    pin.forwardKinematics(model, data, q_tpose)
    pin.updateFramePlacements(model, data)
    target_l = data.oMf[left_id].copy()
    target_r = data.oMf[right_id].copy()

    print(f"T-pose left wrist:  {target_l.translation}")
    print(f"T-pose right wrist: {target_r.translation}")

    # offset targets slightly to test non-trivial IK
    target_l_offset = target_l.copy()
    target_r_offset = target_r.copy()
    target_l_offset.translation[2] += 0.05  # raise left wrist 5cm
    target_r_offset.translation[2] -= 0.05  # lower right wrist 5cm

    ori_cost = 0.0 if pos_only else IK_ORI_COST
    left_task = FrameTask(LEFT_EE_FRAME, position_cost=IK_POS_COST,
                          orientation_cost=ori_cost)
    right_task = FrameTask(RIGHT_EE_FRAME, position_cost=IK_POS_COST,
                           orientation_cost=ori_cost)
    posture_task = PostureTask(cost=IK_POSTURE_COST)

    posture_task.set_target(q_tpose)
    tasks = [left_task, right_task, posture_task]

    # test 1: t-pose target from t-pose seed (trivial)
    print("\n--- Test 1: t-pose target from t-pose seed ---")
    config = Configuration(model, data, q_tpose.copy())
    left_task.set_target(target_l)
    right_task.set_target(target_r)
    solve_ik_pink(config, tasks, 50)
    _print_ik_result(model, data, config.q, left_id, right_id,
                     target_l, target_r, q_tpose, model_full)

    # test 2: t-pose target from neutral seed
    print("\n--- Test 2: t-pose target from neutral seed ---")
    config2 = Configuration(model, data, pin.neutral(model))
    left_task.set_target(target_l)
    right_task.set_target(target_r)
    solve_ik_pink(config2, tasks, 300)
    _print_ik_result(model, data, config2.q, left_id, right_id,
                     target_l, target_r, q_tpose, model_full)

    # test 3: offset target from t-pose seed
    print("\n--- Test 3: offset target from t-pose seed ---")
    config3 = Configuration(model, data, q_tpose.copy())
    left_task.set_target(target_l_offset)
    right_task.set_target(target_r_offset)
    solve_ik_pink(config3, tasks, 300)
    _print_ik_result(model, data, config3.q, left_id, right_id,
                     target_l_offset, target_r_offset, q_tpose, model_full)


def _print_ik_result(model, data, q, left_id, right_id,
                     target_l, target_r, q_tpose, model_full):
    """print IK result summary."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    err_l = np.linalg.norm(data.oMf[left_id].translation - target_l.translation)
    err_r = np.linalg.norm(data.oMf[right_id].translation - target_r.translation)

    # orientation error (angle of rotation difference)
    dR_l = data.oMf[left_id].rotation.T @ target_l.rotation
    dR_r = data.oMf[right_id].rotation.T @ target_r.rotation
    ori_err_l = np.linalg.norm(pin.log3(dR_l))
    ori_err_r = np.linalg.norm(pin.log3(dR_r))

    print(f"  pos err:  L={err_l:.6f}m  R={err_r:.6f}m")
    print(f"  ori err:  L={np.degrees(ori_err_l):.2f}deg  R={np.degrees(ori_err_r):.2f}deg")

    q_full = reduced_to_full(q, model_full)
    print(f"  Joint angles:")
    for i, name in enumerate(JOINT_NAMES_INTERLEAVED):
        idx = REORDER_IDX[i]
        print(f"    {name:20s}: {np.degrees(q_full[idx]):+7.2f}deg")


def run_live(model, data, left_id, right_id, model_full,
             rate=60, pos_only=False, debug=False, continuous=False, viz=None,
             use_waist=False, frame_callback=None):
    """Live teleoperation from VIVE trackers."""
    # triad_openvr lives at xr_teleoperate/triad_openvr/
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../../triad_openvr"))
    import triad_openvr as tvr

    print("init VIVE tracking...")
    v = tvr.triad_openvr()
    v.print_discovered_objects()

    trackers = sorted(k for k in v.devices if "tracker" in k)
    if len(trackers) < 2:
        print(f"Error: Need 2 trackers, found {len(trackers)}: {trackers}")
        sys.exit(1)

    # --- detect left/right/waist by waving ---
    def _measure_movement(device_keys, duration_s=3):
        """track movement of devices over duration, return {key: total_dist}."""
        pose = v.get_pose()
        mv = {t: 0.0 for t in device_keys}
        pv = {}
        for t in device_keys:
            mat = v.devices[t].get_pose_matrix(pose)
            if mat is None:
                print(f"Error: {t} not tracked")
                sys.exit(1)
            pv[t] = np.array([mat[r][3] for r in range(3)])
        n_samples = int(duration_s / 0.05)
        for _ in range(n_samples):
            time.sleep(0.05)
            p = v.get_pose()
            for t in device_keys:
                m = v.devices[t].get_pose_matrix(p)
                if m is not None:
                    pos = np.array([m[r][3] for r in range(3)])
                    mv[t] += np.linalg.norm(pos - pv[t])
                    pv[t] = pos
        return mv

    print("\nWave your LEFT hand for 3 seconds...")
    movement = _measure_movement(trackers)
    left_tracker = max(trackers, key=lambda t: movement[t])
    remaining = [t for t in trackers if t != left_tracker]
    print(f"Left wrist:  {left_tracker} (moved {movement[left_tracker]:.3f}m)")

    print("\nWave your RIGHT hand for 3 seconds...")
    movement2 = _measure_movement(remaining)
    right_tracker = max(remaining, key=lambda t: movement2[t])
    print(f"Right wrist: {right_tracker} (moved {movement2[right_tracker]:.3f}m)")

    waist_tracker = None
    if use_waist and len(remaining) >= 2:
        waist_tracker = [t for t in remaining if t != right_tracker][0]
        print(f"Waist:       {waist_tracker} (moved {movement2[waist_tracker]:.3f}m)")

    # --- calibration pose 1: T-pose palms down ---
    print("\nHold T-POSE (palms down) for calibration...")
    for s in range(5, 0, -1):
        print(f"  {s}...")
        time.sleep(1)
    print("Calibrating t-pose!")

    pose = v.get_pose()
    l_mat = v.devices[left_tracker].get_pose_matrix(pose)
    r_mat = v.devices[right_tracker].get_pose_matrix(pose)
    if l_mat is None or r_mat is None:
        print("Error: trackers lost during calibration")
        sys.exit(1)

    human_l_pos_1, human_l_rot_1 = extract_pose(l_mat)
    human_r_pos_1, human_r_rot_1 = extract_pose(r_mat)

    # --- pre-compute robot FK for both calibration poses ---
    q_tpose = make_tpose_q(model)
    pin.forwardKinematics(model, data, q_tpose)
    pin.updateFramePlacements(model, data)
    robot_l_se3 = data.oMf[left_id].copy()
    robot_r_se3 = data.oMf[right_id].copy()
    robot_l_1 = robot_l_se3.translation.copy()
    robot_r_1 = robot_r_se3.translation.copy()

    q_armsfwd = make_armsfwd_q(model)
    pin.forwardKinematics(model, data, q_armsfwd)
    pin.updateFramePlacements(model, data)
    robot_l_2 = data.oMf[left_id].translation.copy()
    robot_r_2 = data.oMf[right_id].translation.copy()
    robot_rot_l_2 = data.oMf[left_id].rotation.copy()
    robot_rot_r_2 = data.oMf[right_id].rotation.copy()

    robot_center = (robot_l_1 + robot_r_1) / 2

    # position calibration from t-pose (always use wrist midpoint for R_calib/T_map)
    human_center = (human_l_pos_1 + human_r_pos_1) / 2
    human_dist = np.linalg.norm(human_l_pos_1 - human_r_pos_1)
    robot_dist = np.linalg.norm(robot_l_1 - robot_r_1)
    arm_scale = robot_dist / human_dist
    print(f"Scale: {arm_scale:.3f} (human {human_dist:.3f}m, robot {robot_dist:.3f}m)")

    # --- calibration pose 2: arms forward palms down ---
    print("\nHold ARMS FORWARD (palms down) for calibration...")
    for s in range(5, 0, -1):
        print(f"  {s}...")
        time.sleep(1)
    print("Calibrating arms-forward!")

    pose = v.get_pose()
    l_mat = v.devices[left_tracker].get_pose_matrix(pose)
    r_mat = v.devices[right_tracker].get_pose_matrix(pose)
    if l_mat is None or r_mat is None:
        print("Error: trackers lost during calibration")
        sys.exit(1)

    human_l_pos_2, human_l_rot_2 = extract_pose(l_mat)
    human_r_pos_2, human_r_rot_2 = extract_pose(r_mat)

    # --- compute R_calib using SVD with both calibration poses ---
    human_pts = np.vstack([
        human_l_pos_1 - human_center,
        human_r_pos_1 - human_center,
        human_l_pos_2 - human_center,
        human_r_pos_2 - human_center,
    ])
    robot_pts = np.vstack([
        robot_l_1 - robot_center,
        robot_r_1 - robot_center,
        robot_l_2 - robot_center,
        robot_r_2 - robot_center,
    ])
    R_calib = _compute_R_calib(human_pts, robot_pts)

    # per-axis scaling in robot frame (handles different arm proportions)
    aligned = R_calib @ human_pts.T      # 3x4: human points rotated to robot frame
    robot_centered = robot_pts.T         # 3x4: robot points centered
    axis_scale = np.ones(3)
    for i in range(3):
        denom = np.dot(aligned[i], aligned[i])
        if denom > 1e-6:
            axis_scale[i] = np.dot(robot_centered[i], aligned[i]) / denom
        else:
            axis_scale[i] = arm_scale  # fallback for underdetermined axis (e.g. Z)
    T_map = np.diag(axis_scale) @ R_calib
    print(f"Per-axis scale: X={axis_scale[0]:.3f} Y={axis_scale[1]:.3f} Z={axis_scale[2]:.3f}")

    # verify position calibration at both poses
    test_l_1 = robot_center + T_map @ (human_l_pos_1 - human_center)
    test_r_1 = robot_center + T_map @ (human_r_pos_1 - human_center)
    test_l_2 = robot_center + T_map @ (human_l_pos_2 - human_center)
    test_r_2 = robot_center + T_map @ (human_r_pos_2 - human_center)
    print(f"Calib check t-pose:  L err={np.linalg.norm(test_l_1 - robot_l_1):.4f}m  "
          f"R err={np.linalg.norm(test_r_1 - robot_r_1):.4f}m")
    print(f"Calib check armsfwd: L err={np.linalg.norm(test_l_2 - robot_l_2):.4f}m  "
          f"R err={np.linalg.norm(test_r_2 - robot_r_2):.4f}m")

    # --- calibration pose 3: arms forward, palms facing each other ---
    # used to compute per-arm R_corr empirically from tracker vs robot rotation
    print("\nNow ROTATE BOTH PALMS to face each other (keep arms in same position)...")
    for s in range(3, 0, -1):
        print(f"  {s}...")
        time.sleep(1)
    print("Calibrating palms-facing!")

    pose = v.get_pose()
    l_mat = v.devices[left_tracker].get_pose_matrix(pose)
    r_mat = v.devices[right_tracker].get_pose_matrix(pose)
    if l_mat is None or r_mat is None:
        print("Error: trackers lost during calibration")
        sys.exit(1)
    _, human_l_rot_roll = extract_pose(l_mat)
    _, human_r_rot_roll = extract_pose(r_mat)

    # robot FK for palms-facing (arms-forward with wrist_roll=0)
    q_palms_facing = make_armsfwd_q(model)
    q_palms_facing[4] = 0.0   # left wrist_roll = 0
    q_palms_facing[11] = 0.0  # right wrist_roll = 0
    pin.forwardKinematics(model, data, q_palms_facing)
    pin.updateFramePlacements(model, data)
    robot_rot_l_pf = data.oMf[left_id].rotation.copy()
    robot_rot_r_pf = data.oMf[right_id].rotation.copy()

    # --- compute per-arm R_corr from poses 1→2 and 2→3 via Wahba SVD ---
    # Two independent rotation correspondences fully constrain R_corr:
    #   axis 1: t-pose → arms-forward (shoulder/elbow movement)
    #   axis 2: palms-down → palms-facing (wrist roll)
    # R_corr^T maps tracker rotation axes to robot rotation axes.
    def _compute_R_corr(t_rot_1, t_rot_2, t_rot_3, r_rot_1, r_rot_2, r_rot_3):
        """Find R_corr via Wahba SVD from two rotation correspondences.

        Uses pose 1→2 and pose 2→3 to get two independent axis mappings.
        R_corr^T maps tracker rotation axes to robot rotation axes.
        """
        # axis pair 1: pose 1 → 2 (t-pose → arms-forward)
        log_t_12 = pin.log3(t_rot_1.T @ t_rot_2)
        log_r_12 = pin.log3(r_rot_1.T @ r_rot_2)
        # axis pair 2: pose 2 → 3 (palms-down → palms-facing)
        log_t_23 = pin.log3(t_rot_2.T @ t_rot_3)
        log_r_23 = pin.log3(r_rot_2.T @ r_rot_3)

        norms = [np.linalg.norm(v) for v in [log_t_12, log_r_12, log_t_23, log_r_23]]
        if any(n < 1e-6 for n in norms):
            print("    Warning: degenerate calibration rotation, using identity R_corr")
            return np.eye(3)

        n_t_12 = log_t_12 / norms[0]
        n_r_12 = log_r_12 / norms[1]
        n_t_23 = log_t_23 / norms[2]
        n_r_23 = log_r_23 / norms[3]

        print(f"    axis1 (tpose→armsfwd): tracker={np.degrees(log_t_12).astype(int)} "
              f"robot={np.degrees(log_r_12).astype(int)}  dot={np.dot(n_t_12, n_r_12):.3f}")
        print(f"    axis2 (pd→pf):         tracker={np.degrees(log_t_23).astype(int)} "
              f"robot={np.degrees(log_r_23).astype(int)}  dot={np.dot(n_t_23, n_r_23):.3f}")

        # Wahba SVD: find R_corr^T that best maps n_t_i → n_r_i
        H = np.outer(n_r_12, n_t_12) + np.outer(n_r_23, n_t_23)
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(U @ Vt)
        R_corr_T = U @ np.diag([1.0, 1.0, np.sign(d)]) @ Vt

        # verify axis mapping
        dot1 = np.dot(R_corr_T @ n_t_12, n_r_12)
        dot2 = np.dot(R_corr_T @ n_t_23, n_r_23)
        print(f"    R_corr^T mapped: axis1 dot={dot1:.3f}  axis2 dot={dot2:.3f}")

        return R_corr_T.T  # R_corr

    R_corr_l = _compute_R_corr(human_l_rot_1, human_l_rot_2, human_l_rot_roll,
                                robot_l_se3.rotation, robot_rot_l_2, robot_rot_l_pf)
    R_corr_r = _compute_R_corr(human_r_rot_1, human_r_rot_2, human_r_rot_roll,
                                robot_r_se3.rotation, robot_rot_r_2, robot_rot_r_pf)

    # verify: does R_corr make dR_tracker match dR_robot for both axis pairs?
    for side, R_corr, rots_t, rots_r in [
        ("Left",  R_corr_l,
         (human_l_rot_1, human_l_rot_2, human_l_rot_roll),
         (robot_l_se3.rotation, robot_rot_l_2, robot_rot_l_pf)),
        ("Right", R_corr_r,
         (human_r_rot_1, human_r_rot_2, human_r_rot_roll),
         (robot_r_se3.rotation, robot_rot_r_2, robot_rot_r_pf))]:
        for i, j, label in [(0, 1, "tpose→armsfwd"), (1, 2, "pd→pf")]:
            dR_t = rots_t[i].T @ rots_t[j]
            dR_r = rots_r[i].T @ rots_r[j]
            corrected = R_corr.T @ dR_t @ R_corr
            err = np.degrees(np.linalg.norm(pin.log3(corrected.T @ dR_r)))
            log_corr = np.degrees(pin.log3(corrected))
            log_want = np.degrees(pin.log3(dR_r))
            print(f"  {side} {label}: err={err:.1f}deg  "
                  f"got=[{log_corr[0]:+.0f},{log_corr[1]:+.0f},{log_corr[2]:+.0f}] "
                  f"want=[{log_want[0]:+.0f},{log_want[1]:+.0f},{log_want[2]:+.0f}]")

    # apply R_corr to all tracker rotations before computing offsets
    human_l_rot_1 = human_l_rot_1 @ R_corr_l
    human_r_rot_1 = human_r_rot_1 @ R_corr_r
    human_l_rot_2 = human_l_rot_2 @ R_corr_l
    human_r_rot_2 = human_r_rot_2 @ R_corr_r

    # --- compute orientation offsets at poses 1 & 2 (with corrected tracker rotations) ---
    R_offset_l_1 = robot_l_se3.rotation @ (R_calib @ human_l_rot_1).T
    R_offset_r_1 = robot_r_se3.rotation @ (R_calib @ human_r_rot_1).T
    R_offset_l_2 = robot_rot_l_2 @ (R_calib @ human_l_rot_2).T
    R_offset_r_2 = robot_rot_r_2 @ (R_calib @ human_r_rot_2).T

    # compute log of relative rotation for slerp interpolation
    dR_offset_l = R_offset_l_1.T @ R_offset_l_2
    dR_offset_r = R_offset_r_1.T @ R_offset_r_2
    log_dR_l = pin.log3(dR_offset_l)
    log_dR_r = pin.log3(dR_offset_r)

    # verify orientation at calibration poses
    print("\n=== Orientation calibration check ===")
    for side, offsets in [("Left", (R_offset_l_1, R_offset_l_2)),
                          ("Right", (R_offset_r_1, R_offset_r_2))]:
        robot_rots = (robot_l_se3.rotation, robot_rot_l_2) if side == "Left" \
            else (robot_r_se3.rotation, robot_rot_r_2)
        tracker_rots = (human_l_rot_1, human_l_rot_2) if side == "Left" \
            else (human_r_rot_1, human_r_rot_2)
        for i, label in enumerate(["t-pose", "arms-fwd"]):
            computed = offsets[i] @ R_calib @ tracker_rots[i]
            dR = computed.T @ robot_rots[i]
            err = np.degrees(np.linalg.norm(pin.log3(dR)))
            print(f"  {side} {label}: err={err:.1f}deg")

    # bundle calibration data
    cal = {
        'human_center': human_center,
        'robot_center': robot_center,
        'T_map': T_map,
        'R_calib': R_calib,
        'R_corr_l': R_corr_l,
        'R_corr_r': R_corr_r,
        'R_offset_l_1': R_offset_l_1,
        'R_offset_r_1': R_offset_r_1,
        'log_dR_l': log_dR_l,
        'log_dR_r': log_dR_r,
        # raw tracker positions for IK seed blending
        'tracker_pos_l_1': human_l_pos_1.copy(),
        'tracker_pos_r_1': human_r_pos_1.copy(),
        'tracker_pos_l_2': human_l_pos_2.copy(),
        'tracker_pos_r_2': human_r_pos_2.copy(),
        # waist reference for runtime delta tracking
        'waist_pos_calib': None,
        'waist_rot_calib': None,
    }
    # snapshot waist position NOW (after all calibration) as the delta baseline
    if waist_tracker is not None:
        for _ in range(10):  # retry briefly if tracker not visible
            pose = v.get_pose()
            w_mat = v.devices[waist_tracker].get_pose_matrix(pose)
            if w_mat is not None:
                waist_pos, waist_rot = extract_pose(w_mat)
                cal['waist_pos_calib'] = waist_pos
                cal['waist_rot_calib'] = waist_rot
                print(f"Waist baseline set: [{cal['waist_pos_calib'][0]:.3f}, "
                      f"{cal['waist_pos_calib'][1]:.3f}, {cal['waist_pos_calib'][2]:.3f}]")
                break
            time.sleep(0.1)
        if cal['waist_pos_calib'] is None:
            print("Warning: waist tracker lost, disabling body tracking")
            waist_tracker = None
    print("Calibrated.\n")

    # --- set up pink IK (coupled position + orientation) ---
    ori_cost = 0.0 if pos_only else IK_ORI_COST
    left_task = FrameTask(LEFT_EE_FRAME, position_cost=IK_POS_COST,
                          orientation_cost=ori_cost)
    right_task = FrameTask(RIGHT_EE_FRAME, position_cost=IK_POS_COST,
                           orientation_cost=ori_cost)
    posture_task = PostureTask(cost=IK_POSTURE_COST)
    # mid-range for shoulders + elbows (prevents getting stuck at limits),
    # t-pose for wrists (lets them track orientation freely)
    q_posture_ref = q_tpose.copy()
    q_mid = (model.upperPositionLimit + model.lowerPositionLimit) / 2
    for i in [0, 1, 2, 3, 7, 8, 9, 10]:  # shoulder + elbow joints only
        q_posture_ref[i] = q_mid[i]
    posture_task.set_target(q_posture_ref)

    left_task.set_target(robot_l_se3)
    right_task.set_target(robot_r_se3)
    tasks = [left_task, right_task, posture_task]

    config = Configuration(model, data, q_tpose.copy())

    if continuous:
        _run_continuous(v, left_tracker, right_tracker, waist_tracker,
                        model, data, left_id, right_id, model_full,
                        config, tasks, left_task, right_task, q_tpose,
                        cal, rate, debug, pos_only, viz, frame_callback)
    else:
        _run_capture(v, left_tracker, right_tracker, waist_tracker,
                     model, data, left_id, right_id, model_full,
                     config, tasks, left_task, right_task, q_tpose,
                     cal, debug, pos_only, viz)


def _compute_R_calib(human_pts, robot_pts):
    """compute rotation from human frame to robot frame using SVD (Kabsch).

    human_pts and robot_pts are Nx3 arrays of corresponding centered points.
    
    returns R such that R @ human_vec ≈ robot_vec.
    """
    H = human_pts.T @ robot_pts
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    d = np.linalg.det(V @ U.T)
    return V @ np.diag([1.0, 1.0, np.sign(d)]) @ U.T


def _transform_tracker(l_pos, l_rot, r_pos, r_rot, cal,
                       human_center=None, blend_w_l=0.0, blend_w_r=0.0,
                       R_body=None):
    """transform raw tracker poses to robot-frame SE3 targets.

    blend_w_l/r: per-arm blend (0=t-pose offset, 1=arms-fwd offset).
    R_body: optional 3x3 rotation delta from waist tracker; rotates arm
            positions around robot_center and arm orientations.
    """
    T = cal['T_map']
    R = cal['R_calib']
    center = human_center if human_center is not None else cal['human_center']
    left_pos = cal['robot_center'] + T @ (l_pos - center)
    right_pos = cal['robot_center'] + T @ (r_pos - center)

    # slerp between t-pose and arms-fwd orientation offsets (per arm)
    wl = np.clip(blend_w_l, 0.0, 1.0)
    wr = np.clip(blend_w_r, 0.0, 1.0)
    R_off_l = cal['R_offset_l_1'] @ pin.exp3(wl * cal['log_dR_l'])
    R_off_r = cal['R_offset_r_1'] @ pin.exp3(wr * cal['log_dR_r'])
    left_rot = R_off_l @ R @ (l_rot @ cal['R_corr_l'])
    right_rot = R_off_r @ R @ (r_rot @ cal['R_corr_r'])

    # apply waist rotation delta: rotate arms around robot center
    if R_body is not None:
        rc = cal['robot_center']
        left_pos = rc + R_body @ (left_pos - rc)
        right_pos = rc + R_body @ (right_pos - rc)
        left_rot = R_body @ left_rot
        right_rot = R_body @ right_rot

    return left_pos, left_rot, right_pos, right_rot



def _run_continuous(v, left_tracker, right_tracker, waist_tracker,
                    model, data, left_id, right_id, model_full, config,
                    tasks, left_task, right_task, q_tpose,
                    cal, rate, debug, pos_only=False, viz=None,
                    frame_callback=None):
    """continuous streaming mode --> solves IK every frame at target rate."""
    period = 1.0 / rate
    alpha = SMOOTH_ALPHA
    smooth_l_pos = smooth_r_pos = None
    smooth_l_rot = smooth_r_rot = None
    smooth_blend_l = smooth_blend_r = 0.5
    joint_filter = WeightedMovingFilter(SMOOTH_FILTER_WEIGHTS, model.nq)
    q_prev = None  # for velocity clamping
    frame = 0

    # t-pose reset detection: if both hands stay near t-pose for 3s, reset IK
    pin.forwardKinematics(model, data, q_tpose)
    pin.updateFramePlacements(model, data)
    tpose_l_pos = data.oMf[left_id].translation.copy()
    tpose_r_pos = data.oMf[right_id].translation.copy()
    TPOSE_RESET_THRESHOLD = 0.05  # 5cm
    TPOSE_RESET_FRAMES = int(3 * rate)  # 3 seconds
    tpose_hold_count = 0
    sum_err_l = sum_err_r = sum_ori_l = sum_ori_r = 0.0

    print(f"Streaming at {rate} Hz. Ctrl-C to stop.\n")

    try:
        while True:
            t0 = time.time()

            pose = v.get_pose()
            l_m = v.devices[left_tracker].get_pose_matrix(pose)
            r_m = v.devices[right_tracker].get_pose_matrix(pose)
            if l_m is None or r_m is None:
                print("\r  trackers lost...", end="", flush=True)
                time.sleep(period)
                continue

            l_pos, l_rot = extract_pose(l_m)
            r_pos, r_rot = extract_pose(r_m)

            # de-rotate hand poses into calibration body frame so torso
            # rotation doesn't leak into robot-frame arm targets
            ik_l_pos, ik_l_rot = l_pos, l_rot
            ik_r_pos, ik_r_rot = r_pos, r_rot
            live_center = None
            if waist_tracker is not None:
                w_m = v.devices[waist_tracker].get_pose_matrix(pose)
                if w_m is not None:
                    waist_now, waist_rot_now = extract_pose(w_m)
                    if cal['waist_rot_calib'] is not None:
                        R_derot = cal['waist_rot_calib'] @ waist_rot_now.T
                        ik_l_pos = cal['waist_pos_calib'] + R_derot @ (l_pos - waist_now)
                        ik_r_pos = cal['waist_pos_calib'] + R_derot @ (r_pos - waist_now)
                        ik_l_rot = R_derot @ l_rot
                        ik_r_rot = R_derot @ r_rot
                    else:
                        waist_delta = waist_now - cal['waist_pos_calib']
                        live_center = cal['human_center'] + waist_delta

            # per-arm blend weights (derotated positions are body-relative)
            d1_l = np.linalg.norm(ik_l_pos - cal['tracker_pos_l_1'])
            d2_l = np.linalg.norm(ik_l_pos - cal['tracker_pos_l_2'])
            blend_w_l = d1_l / (d1_l + d2_l + 1e-8)
            d1_r = np.linalg.norm(ik_r_pos - cal['tracker_pos_r_1'])
            d2_r = np.linalg.norm(ik_r_pos - cal['tracker_pos_r_2'])
            blend_w_r = d1_r / (d1_r + d2_r + 1e-8)

            left_pos, left_rot, right_pos, right_rot = _transform_tracker(
                ik_l_pos, ik_l_rot, ik_r_pos, ik_r_rot, cal,
                human_center=live_center,
                blend_w_l=blend_w_l, blend_w_r=blend_w_r)

            # EMA smoothing on positions
            if smooth_l_pos is None:
                smooth_l_pos = left_pos.copy()
                smooth_r_pos = right_pos.copy()
            else:
                smooth_l_pos += alpha * (left_pos - smooth_l_pos)
                smooth_r_pos += alpha * (right_pos - smooth_r_pos)

            # SLERP smoothing on rotations + discontinuity rejection
            if not pos_only:
                left_rot = filter_rotation(smooth_l_rot, left_rot)
                right_rot = filter_rotation(smooth_r_rot, right_rot)
                if smooth_l_rot is None:
                    smooth_l_rot = left_rot.copy()
                    smooth_r_rot = right_rot.copy()
                else:
                    smooth_l_rot = smooth_rotation(smooth_l_rot, left_rot)
                    smooth_r_rot = smooth_rotation(smooth_r_rot, right_rot)
                left_rot = smooth_l_rot
                right_rot = smooth_r_rot

            left_task.set_target(pin.SE3(left_rot, smooth_l_pos))
            right_task.set_target(pin.SE3(right_rot, smooth_r_pos))

            # warm-start coupled IK from previous frame
            solve_ik_pink(config, tasks, IK_ITERS_CONTINUOUS)
            joint_filter.add_data(config.q.copy())
            q_out = joint_filter.filtered_data.copy()

            # velocity clamp: cap max joint change per frame
            if q_prev is not None:
                delta = q_out - q_prev
                q_out = q_prev + np.clip(delta, -MAX_JOINT_VEL, MAX_JOINT_VEL)
            q_prev = q_out.copy()

            # compute errors
            pin.forwardKinematics(model, data, q_out)
            pin.updateFramePlacements(model, data)
            err_l = np.linalg.norm(data.oMf[left_id].translation - smooth_l_pos)
            err_r = np.linalg.norm(data.oMf[right_id].translation - smooth_r_pos)

            # orientation errors (always compute for stats)
            dR_l = data.oMf[left_id].rotation.T @ left_rot
            dR_r = data.oMf[right_id].rotation.T @ right_rot
            ol = np.degrees(np.linalg.norm(pin.log3(dR_l)))
            ori = np.degrees(np.linalg.norm(pin.log3(dR_r)))

            sum_err_l += err_l
            sum_err_r += err_r
            sum_ori_l += ol
            sum_ori_r += ori

            # t-pose reset detection
            d_l = np.linalg.norm(smooth_l_pos - tpose_l_pos)
            d_r = np.linalg.norm(smooth_r_pos - tpose_r_pos)
            if d_l < TPOSE_RESET_THRESHOLD and d_r < TPOSE_RESET_THRESHOLD:
                tpose_hold_count += 1
                if tpose_hold_count == TPOSE_RESET_FRAMES:
                    print("\n*** T-POSE RESET: resetting IK to t-pose ***\n")
                    config.update(q_tpose.copy())
                    joint_filter = WeightedMovingFilter(SMOOTH_FILTER_WEIGHTS, model.nq)
                    q_prev = None
                    q_out = q_tpose.copy()
            else:
                tpose_hold_count = 0

            q_full = reduced_to_full(q_out, model_full)
            if viz is not None:
                viz.display(q_full)

            if frame_callback is not None:
                joints_rad = np.array([q_full[SEQUENTIAL_IDX[i]] for i in range(14)],
                                      dtype=np.float32)
                frame_callback(joints_rad)

            joints = " ".join(f"{np.degrees(q_full[REORDER_IDX[i]]):+6.1f}"
                              for i in range(14))

            elapsed = time.time() - t0
            hz = 1.0 / elapsed if elapsed > 0 else 0

            if debug:
                print(f"\r[{frame:5d}] err_l={err_l:.4f} err_r={err_r:.4f} "
                      f"ori_l={ol:.1f} ori_r={ori:.1f} {hz:4.0f}Hz "
                      f"j=[{joints}]", end="", flush=True)
            else:
                print(f"\r[{frame:5d}] err_l={err_l:.4f} err_r={err_r:.4f} "
                      f"{hz:4.0f}Hz j=[{joints}]", end="", flush=True)

            frame += 1

            # sleep to maintain target rate
            remaining = period - (time.time() - t0)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print(f"\nStopped after {frame} frames.")
        if frame > 0:
            print(f"\n=== Session averages ({frame} frames) ===")
            print(f"  pos err:  L={sum_err_l/frame:.4f}m  R={sum_err_r/frame:.4f}m")
            print(f"  ori err:  L={sum_ori_l/frame:.1f}deg  R={sum_ori_r/frame:.1f}deg")


def _run_capture(v, left_tracker, right_tracker, waist_tracker,
                 model, data, left_id, right_id, model_full, config,
                 tasks, left_task, right_task, q_tpose,
                 cal, debug, pos_only=False, viz=None):
    """press-to-capture mode —-> records samples, average, solve IK."""
    capture_num = 0
    try:
        while True:
            capture_num += 1
            input(f"Press Enter to capture pose #{capture_num}...")
            print("Move into position...")
            time.sleep(1)
            print("Recording for 2 seconds...")

            samples_l_pos, samples_r_pos = [], []
            samples_l_rot, samples_r_rot = [], []
            samples_w_pos = []
            for _ in range(40):
                pose = v.get_pose()
                l_m = v.devices[left_tracker].get_pose_matrix(pose)
                r_m = v.devices[right_tracker].get_pose_matrix(pose)
                if l_m is not None and r_m is not None:
                    lp, lr = extract_pose(l_m)
                    rp, rr = extract_pose(r_m)
                    samples_l_pos.append(lp)
                    samples_r_pos.append(rp)
                    samples_l_rot.append(lr)
                    samples_r_rot.append(rr)
                    if waist_tracker is not None:
                        w_m = v.devices[waist_tracker].get_pose_matrix(pose)
                        if w_m is not None:
                            wp, _ = extract_pose(w_m)
                            samples_w_pos.append(wp)
                time.sleep(0.05)

            if len(samples_l_pos) < 10:
                print("Error: trackers lost, try again.")
                continue

            avg_l_pos = np.mean(samples_l_pos, axis=0)
            avg_r_pos = np.mean(samples_r_pos, axis=0)
            avg_l_rot = samples_l_rot[-1]
            avg_r_rot = samples_r_rot[-1]

            # de-rotate hand poses into calibration body frame
            ik_l_pos, ik_l_rot = avg_l_pos, avg_l_rot
            ik_r_pos, ik_r_rot = avg_r_pos, avg_r_rot
            live_center = None
            if waist_tracker is not None and len(samples_w_pos) > 0:
                waist_now = np.mean(samples_w_pos, axis=0)
                if cal['waist_rot_calib'] is not None:
                    w_m = v.devices[waist_tracker].get_pose_matrix(v.get_pose())
                    if w_m is not None:
                        _, waist_rot_now = extract_pose(w_m)
                        R_derot = cal['waist_rot_calib'] @ waist_rot_now.T
                        ik_l_pos = cal['waist_pos_calib'] + R_derot @ (avg_l_pos - waist_now)
                        ik_r_pos = cal['waist_pos_calib'] + R_derot @ (avg_r_pos - waist_now)
                        ik_l_rot = R_derot @ avg_l_rot
                        ik_r_rot = R_derot @ avg_r_rot
                    else:
                        waist_delta = waist_now - cal['waist_pos_calib']
                        live_center = cal['human_center'] + waist_delta
                else:
                    waist_delta = waist_now - cal['waist_pos_calib']
                    live_center = cal['human_center'] + waist_delta

            # per-arm blend weights (derotated positions are body-relative)
            d1_l = np.linalg.norm(ik_l_pos - cal['tracker_pos_l_1'])
            d2_l = np.linalg.norm(ik_l_pos - cal['tracker_pos_l_2'])
            w_l = d1_l / (d1_l + d2_l + 1e-8)
            d1_r = np.linalg.norm(ik_r_pos - cal['tracker_pos_r_1'])
            d2_r = np.linalg.norm(ik_r_pos - cal['tracker_pos_r_2'])
            w_r = d1_r / (d1_r + d2_r + 1e-8)

            left_target_pos, left_target_rot, right_target_pos, right_target_rot = \
                _transform_tracker(ik_l_pos, ik_l_rot, ik_r_pos, ik_r_rot, cal,
                                   human_center=live_center,
                                   blend_w_l=w_l, blend_w_r=w_r)

            target_l_se3 = pin.SE3(left_target_rot, left_target_pos)
            target_r_se3 = pin.SE3(right_target_rot, right_target_pos)

            left_task.set_target(target_l_se3)
            right_task.set_target(target_r_se3)

            # blend IK seed and posture reference
            q_armsfwd = make_armsfwd_q(model)
            w = (w_l + w_r) / 2
            q_blend = (1 - w) * q_tpose + w * q_armsfwd
            print(f"  IK seed blend: w={w:.3f} (w_l={w_l:.3f} w_r={w_r:.3f})")

            posture_task = tasks[2]
            posture_task.set_target(q_blend)
            config = Configuration(model, data, q_blend)
            solve_ik_pink(config, tasks, IK_ITERS)

            # coupled IK handles both position + orientation
            q_out = config.q.copy()

            pin.forwardKinematics(model, data, q_out)
            pin.updateFramePlacements(model, data)
            err_l = np.linalg.norm(data.oMf[left_id].translation - left_target_pos)
            err_r = np.linalg.norm(data.oMf[right_id].translation - right_target_pos)

            q_full = reduced_to_full(q_out, model_full)
            if viz is not None:
                viz.display(q_full)

            print(f"\nCapture #{capture_num}")
            print(f"  target_l=[{left_target_pos[0]:+.4f},{left_target_pos[1]:+.4f},{left_target_pos[2]:+.4f}]")
            print(f"  target_r=[{right_target_pos[0]:+.4f},{right_target_pos[1]:+.4f},{right_target_pos[2]:+.4f}]")
            print(f"  err_l={err_l:.4f}m  err_r={err_r:.4f}m")
            print(f"  Joint angles (deg):")
            for i, name in enumerate(JOINT_NAMES_INTERLEAVED):
                idx = REORDER_IDX[i]
                print(f"    {name:20s}: {np.degrees(q_full[idx]):+7.2f}")

            if debug:
                dR_l = data.oMf[left_id].rotation.T @ left_target_rot
                dR_r = data.oMf[right_id].rotation.T @ right_target_rot
                ori_err_l = np.degrees(np.linalg.norm(pin.log3(dR_l)))
                ori_err_r = np.degrees(np.linalg.norm(pin.log3(dR_r)))
                print(f"  ori_err: L={ori_err_l:.2f}deg  R={ori_err_r:.2f}deg")

    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(
        description="G1 upper body 6DOF IK from VIVE trackers (pink)")
    parser.add_argument("--test", action="store_true",
                        help="Run IK test (no trackers)")
    parser.add_argument("--rate", type=int, default=60,
                        help="Update rate in Hz")
    parser.add_argument("--continuous", action="store_true",
                        help="Continuous streaming mode (vs press-to-capture)")
    parser.add_argument("--pos-only", action="store_true",
                        help="Position-only IK (ignore tracker orientation)")
    parser.add_argument("--debug", action="store_true",
                        help="Show detailed IK diagnostics")
    parser.add_argument("--viz", action="store_true",
                        help="Launch meshcat 3D visualizer in browser")
    parser.add_argument("--waist", action="store_true",
                        help="Use 3rd tracker on waist for body tracking")
    args = parser.parse_args()

    model, data, left_id, right_id, model_full = load_model()
    print(f"Loaded G1 reduced model: {model.nq} DOF (arms only), "
          f"{model.nframes} frames")

    viz = _setup_viz(model_full) if args.viz else None

    if args.test:
        run_test(model, data, left_id, right_id, model_full,
                 pos_only=args.pos_only)
    else:
        run_live(model, data, left_id, right_id, model_full,
                 rate=args.rate, pos_only=args.pos_only,
                 debug=args.debug, continuous=args.continuous, viz=viz,
                 use_waist=args.waist)


if __name__ == "__main__":
    main()
