#!/usr/bin/env python3
"""Recompute fingertip cartesian positions in an episode with the
Rz(π/2) wrist→hand correction rotation applied.

Usage:
    python utils/recompute_fingertips.py <episode_dir>

Overwrites fingertip_positions in states and actions of data.json.
A backup is saved as data.json.bak.
"""

import json
import os
import shutil
import sys
import numpy as np
import pinocchio as pin

# --- Paths ---------------------------------------------------------------
_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
URDF_PATH = os.path.join(_REPO_ROOT, 'assets', 'g1', 'g1_body29_hand14.urdf')
HAND_DIR = os.path.join(_REPO_ROOT, 'assets', 'brainco_hand')

# Same joint IDs as arm_pink_real.py
_LEFT_ARM_JOINT_IDS = [16, 17, 18, 19, 20, 21, 22]
_RIGHT_ARM_JOINT_IDS = [30, 31, 32, 33, 34, 35, 36]
_ARM_JOINT_IDS = set(_LEFT_ARM_JOINT_IDS + _RIGHT_ARM_JOINT_IDS)


# --- Build arm reduced model (same as waldo_rt_arm.py) -------------------
def build_arm_model():
    model_full = pin.buildModelFromUrdf(URDF_PATH)
    joints_to_lock = [i for i in range(1, model_full.njoints)
                      if i not in _ARM_JOINT_IDS]
    q_lock = pin.neutral(model_full)
    model = pin.buildReducedModel(model_full, joints_to_lock, q_lock)
    data = model.createData()
    left_id = model.getFrameId('left_wrist_yaw_link')
    right_id = model.getFrameId('right_wrist_yaw_link')

    # T_cam_base (camera optical frame ← robot base)
    data_full = model_full.createData()
    q_n = pin.neutral(model_full)
    pin.forwardKinematics(model_full, data_full, q_n)
    pin.updateFramePlacements(model_full, data_full)
    d435_id = model_full.getFrameId('d435_link')
    T_base_link = data_full.oMf[d435_id]
    R_optical = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    T_optical = pin.SE3(R_optical, np.zeros(3))
    T_base_optical = T_base_link * T_optical.inverse()
    T_cam_base = T_base_optical.inverse()

    return model, data, left_id, right_id, T_cam_base


# --- Build hand FK info (mirrors waldo_rt_arm._build_hand_fk_info) -------
def build_hand_fk_info(model, side):
    actuated_names = [
        f'{side}_thumb_metacarpal_joint', f'{side}_thumb_proximal_joint',
        f'{side}_index_proximal_joint', f'{side}_middle_proximal_joint',
        f'{side}_ring_proximal_joint', f'{side}_pinky_proximal_joint',
    ]
    actuated_q_idx = [model.joints[model.getJointId(n)].idx_q
                      for n in actuated_names]
    upper_limits = np.array([model.upperPositionLimit[i]
                             for i in actuated_q_idx])

    mimic_defs = [
        (f'{side}_thumb_distal_joint', f'{side}_thumb_proximal_joint', 1.0),
        (f'{side}_index_distal_joint', f'{side}_index_proximal_joint', 1.155),
        (f'{side}_middle_distal_joint', f'{side}_middle_proximal_joint', 1.155),
        (f'{side}_ring_distal_joint', f'{side}_ring_proximal_joint', 1.155),
        (f'{side}_pinky_distal_joint', f'{side}_pinky_proximal_joint', 1.155),
    ]
    mimic_map = []
    for child_name, parent_name, mult in mimic_defs:
        child_qi = model.joints[model.getJointId(child_name)].idx_q
        parent_qi = model.joints[model.getJointId(parent_name)].idx_q
        mimic_map.append((child_qi, parent_qi, mult))

    tip_names = [f'{side}_thumb_tip', f'{side}_index_tip', f'{side}_middle_tip',
                 f'{side}_ring_tip', f'{side}_pinky_tip']
    tip_frame_ids = [model.getFrameId(n, pin.FrameType.BODY) for n in tip_names]

    return {
        'actuated_q_idx': actuated_q_idx,
        'upper_limits': upper_limits,
        'mimic_map': mimic_map,
        'tip_frame_ids': tip_frame_ids,
    }


def fingertip_positions(hand_model, hand_data, info, hand_normalized,
                        T_cam_wrist, T_wrist_hand):
    """Compute 5 fingertip XYZ in camera optical frame."""
    q = pin.neutral(hand_model)
    q_rad = np.asarray(hand_normalized, dtype=np.float64) * info['upper_limits']
    for i, qi in enumerate(info['actuated_q_idx']):
        q[qi] = q_rad[i]
    for child_qi, parent_qi, mult in info['mimic_map']:
        q[child_qi] = q[parent_qi] * mult
    pin.forwardKinematics(hand_model, hand_data, q)
    pin.updateFramePlacements(hand_model, hand_data)
    tips = np.empty((5, 3))
    for i, fid in enumerate(info['tip_frame_ids']):
        tip_cam = T_cam_wrist * T_wrist_hand * hand_data.oMf[fid]
        tips[i] = tip_cam.translation
    return tips


def process_episode(episode_dir, arm_model, arm_data, left_fid, right_fid,
                     T_cam_base, hand_l_model, hand_l_data, hand_r_model,
                     hand_r_data, l_info, r_info, T_wrist_hand_l, T_wrist_hand_r):
    """Recompute fingertip positions for a single episode. Returns True on success."""
    data_path = os.path.join(episode_dir, 'data.json')
    if not os.path.isfile(data_path):
        return False

    with open(data_path) as f:
        episode = json.load(f)

    for frame in episode['data']:
        for section_key in ('states', 'actions'):
            section = frame.get(section_key, {})
            left_arm = section.get('left_arm', {})
            right_arm = section.get('right_arm', {})
            left_ee = section.get('left_ee', {})
            right_ee = section.get('right_ee', {})

            arm_qpos_l = left_arm.get('qpos')
            arm_qpos_r = right_arm.get('qpos')
            hand_qpos_l = left_ee.get('qpos')
            hand_qpos_r = right_ee.get('qpos')

            if arm_qpos_l is None or arm_qpos_r is None:
                continue
            if hand_qpos_l is None or hand_qpos_r is None:
                continue

            arm_q = np.array(arm_qpos_l + arm_qpos_r, dtype=np.float64)
            pin.forwardKinematics(arm_model, arm_data, arm_q)
            pin.updateFramePlacements(arm_model, arm_data)
            T_cam_lw = T_cam_base * arm_data.oMf[left_fid]
            T_cam_rw = T_cam_base * arm_data.oMf[right_fid]

            l_tips = fingertip_positions(
                hand_l_model, hand_l_data, l_info,
                hand_qpos_l, T_cam_lw, T_wrist_hand_l)
            r_tips = fingertip_positions(
                hand_r_model, hand_r_data, r_info,
                hand_qpos_r, T_cam_rw, T_wrist_hand_r)

            left_ee['fingertip_positions'] = l_tips.tolist()
            right_ee['fingertip_positions'] = r_tips.tolist()

    with open(data_path, 'w') as f:
        json.dump(episode, f, indent=4)
    return True


def main():
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} <episode_dir> [episode_dir2 ...]")

    # --- Load models (once) -----------------------------------------------
    print("Loading models...")
    arm_model, arm_data, left_fid, right_fid, T_cam_base = build_arm_model()

    hand_l_model = pin.buildModelFromUrdf(
        os.path.join(HAND_DIR, 'brainco_left.urdf'))
    hand_l_data = hand_l_model.createData()
    hand_r_model = pin.buildModelFromUrdf(
        os.path.join(HAND_DIR, 'brainco_right.urdf'))
    hand_r_data = hand_r_model.createData()
    l_info = build_hand_fk_info(hand_l_model, 'left')
    r_info = build_hand_fk_info(hand_r_model, 'right')

    _R_left = np.array([[ 0, -1,  0],
                         [ 1,  0,  0],
                         [ 0,  0,  1]], dtype=float)
    _R_right = np.array([[ 0, -1,  0],
                          [-1,  0,  0],
                          [ 0,  0, -1]], dtype=float)
    T_wrist_hand_l = pin.SE3(_R_left, np.zeros(3))
    T_wrist_hand_r = pin.SE3(_R_right, np.zeros(3))

    dirs = sys.argv[1:]
    total = len(dirs)
    for i, episode_dir in enumerate(dirs):
        ok = process_episode(
            episode_dir, arm_model, arm_data, left_fid, right_fid,
            T_cam_base, hand_l_model, hand_l_data, hand_r_model,
            hand_r_data, l_info, r_info, T_wrist_hand_l, T_wrist_hand_r)
        status = "OK" if ok else "SKIP"
        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] {status} {os.path.basename(episode_dir)}")

    print(f"Done! Processed {total} episodes.")


if __name__ == '__main__':
    main()
