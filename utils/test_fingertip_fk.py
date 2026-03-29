"""Test script to isolate errors in BrainCo hand fingertip cartesian FK."""
import os
import sys
import numpy as np

# ── Step 1: Load URDFs ──────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Load pinocchio and hand URDFs")
print("=" * 60)

try:
    import pinocchio as pin
    print(f"[OK] pinocchio {pin.__version__}")
except ImportError as e:
    sys.exit(f"[FAIL] Cannot import pinocchio: {e}")

asset_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'brainco_hand')
left_urdf = os.path.join(asset_dir, 'brainco_left.urdf')
right_urdf = os.path.join(asset_dir, 'brainco_right.urdf')

for label, path in [("Left", left_urdf), ("Right", right_urdf)]:
    if os.path.isfile(path):
        print(f"[OK] {label} URDF exists: {path}")
    else:
        print(f"[FAIL] {label} URDF missing: {path}")

try:
    hand_l_model = pin.buildModelFromUrdf(left_urdf)
    hand_l_data = hand_l_model.createData()
    print(f"[OK] Left model loaded — nq={hand_l_model.nq}, nv={hand_l_model.nv}, njoints={hand_l_model.njoints}, nframes={hand_l_model.nframes}")
except Exception as e:
    sys.exit(f"[FAIL] Left URDF load: {e}")

try:
    hand_r_model = pin.buildModelFromUrdf(right_urdf)
    hand_r_data = hand_r_model.createData()
    print(f"[OK] Right model loaded — nq={hand_r_model.nq}, nv={hand_r_model.nv}, njoints={hand_r_model.njoints}, nframes={hand_r_model.nframes}")
except Exception as e:
    sys.exit(f"[FAIL] Right URDF load: {e}")

# ── Step 2: Build FK info (mirrors _build_hand_fk_info) ────────────
print()
print("=" * 60)
print("STEP 2: Build hand FK info (actuated joints, mimic map, tip frames)")
print("=" * 60)

def build_hand_fk_info(model, side):
    errors = []

    # -- actuated joints --
    actuated_names = [
        f'{side}_thumb_metacarpal_joint', f'{side}_thumb_proximal_joint',
        f'{side}_index_proximal_joint', f'{side}_middle_proximal_joint',
        f'{side}_ring_proximal_joint', f'{side}_pinky_proximal_joint',
    ]
    actuated_q_idx = []
    for name in actuated_names:
        jid = model.getJointId(name)
        if jid >= model.njoints:
            errors.append(f"  [FAIL] Joint '{name}' not found")
        else:
            qi = model.joints[jid].idx_q
            actuated_q_idx.append(qi)
            print(f"  [OK] Joint '{name}' -> id={jid}, idx_q={qi}")

    upper_limits = np.array([model.upperPositionLimit[i] for i in actuated_q_idx])
    print(f"  Upper limits: {upper_limits}")

    # -- mimic joints --
    mimic_defs = [
        (f'{side}_thumb_distal_joint', f'{side}_thumb_proximal_joint', 1.0),
        (f'{side}_index_distal_joint', f'{side}_index_proximal_joint', 1.155),
        (f'{side}_middle_distal_joint', f'{side}_middle_proximal_joint', 1.155),
        (f'{side}_ring_distal_joint', f'{side}_ring_proximal_joint', 1.155),
        (f'{side}_pinky_distal_joint', f'{side}_pinky_proximal_joint', 1.155),
    ]
    mimic_map = []
    for child_name, parent_name, mult in mimic_defs:
        cid = model.getJointId(child_name)
        pid = model.getJointId(parent_name)
        if cid >= model.njoints:
            errors.append(f"  [FAIL] Mimic child joint '{child_name}' not found")
        elif pid >= model.njoints:
            errors.append(f"  [FAIL] Mimic parent joint '{parent_name}' not found")
        else:
            child_qi = model.joints[cid].idx_q
            parent_qi = model.joints[pid].idx_q
            mimic_map.append((child_qi, parent_qi, mult))
            print(f"  [OK] Mimic: '{child_name}'(q{child_qi}) = '{parent_name}'(q{parent_qi}) * {mult}")

    # -- tip frames --
    tip_names = [f'{side}_thumb_tip', f'{side}_index_tip', f'{side}_middle_tip',
                 f'{side}_ring_tip', f'{side}_pinky_tip']
    tip_frame_ids = []
    for name in tip_names:
        try:
            fid = model.getFrameId(name)
            tip_frame_ids.append(fid)
            print(f"  [OK] Frame '{name}' -> id={fid}")
        except ValueError as e:
            print(f"  [WARN] getFrameId('{name}') ambiguous: {e}")
            print(f"         Retrying with FrameType.BODY...")
            try:
                fid = model.getFrameId(name, pin.FrameType.BODY)
                tip_frame_ids.append(fid)
                print(f"  [OK] Frame '{name}' (BODY) -> id={fid}")
            except Exception as e2:
                errors.append(f"  [FAIL] Frame '{name}' not found even with BODY type: {e2}")

    # dump all frame names for debugging
    print(f"\n  All frames in {side} model:")
    for i, f in enumerate(model.frames):
        print(f"    [{i}] {f.name}  (type={f.type})")

    for e in errors:
        print(e)

    return {
        'actuated_q_idx': actuated_q_idx,
        'upper_limits': upper_limits,
        'mimic_map': mimic_map,
        'tip_frame_ids': tip_frame_ids,
    }, errors

print("\n--- LEFT HAND ---")
l_info, l_errors = build_hand_fk_info(hand_l_model, 'left')
print("\n--- RIGHT HAND ---")
r_info, r_errors = build_hand_fk_info(hand_r_model, 'right')

if l_errors or r_errors:
    print(f"\n[BLOCKED] FK info has errors, cannot proceed to step 3.")
    sys.exit(1)

# ── Step 3: Run FK with dummy inputs ───────────────────────────────
print()
print("=" * 60)
print("STEP 3: Compute fingertip FK with dummy hand inputs")
print("=" * 60)

def fingertip_positions(hand_model, hand_data, info, hand_normalized):
    """Mirrors _fingertip_positions but without the wrist transform."""
    q = pin.neutral(hand_model)
    q_rad = np.asarray(hand_normalized, dtype=np.float64) * info['upper_limits']
    for i, qi in enumerate(info['actuated_q_idx']):
        q[qi] = q_rad[i]
    for child_qi, parent_qi, mult in info['mimic_map']:
        q[child_qi] = q[parent_qi] * mult

    print(f"  q vector ({len(q)}): {q.round(4)}")

    pin.forwardKinematics(hand_model, hand_data, q)
    pin.updateFramePlacements(hand_model, hand_data)

    tips = np.empty((5, 3))
    for i, fid in enumerate(info['tip_frame_ids']):
        tips[i] = hand_data.oMf[fid].translation
    return tips

test_inputs = [
    ("zeros (open)", np.zeros(6)),
    ("ones (closed)", np.ones(6)),
    ("half", np.full(6, 0.5)),
]

for label, hand_norm in test_inputs:
    print(f"\n--- Input: {label} = {hand_norm} ---")
    try:
        l_tips = fingertip_positions(hand_l_model, hand_l_data, l_info, hand_norm)
        print(f"  [OK] Left tips (m):\n{l_tips.round(5)}")
    except Exception as e:
        print(f"  [FAIL] Left FK: {e}")

    try:
        r_tips = fingertip_positions(hand_r_model, hand_r_data, r_info, hand_norm)
        print(f"  [OK] Right tips (m):\n{r_tips.round(5)}")
    except Exception as e:
        print(f"  [FAIL] Right FK: {e}")

print()
print("=" * 60)
print("DONE")
print("=" * 60)
