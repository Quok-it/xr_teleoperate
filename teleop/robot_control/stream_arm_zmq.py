#!/usr/bin/env python3
"""
stream arm joint angles via ZMQ.

runs arm_pink_real in continuous mode and publishes
14x float32 values (radians) over ZMQ PUB on tcp://*:5557.

joint order (sequential, matches G1_29_ArmController):
  0: left_shoulder_pitch     7: right_shoulder_pitch
  1: left_shoulder_roll      8: right_shoulder_roll
  2: left_shoulder_yaw       9: right_shoulder_yaw
  3: left_elbow             10: right_elbow
  4: left_wrist_roll        11: right_wrist_roll
  5: left_wrist_pitch       12: right_wrist_pitch
  6: left_wrist_yaw         13: right_wrist_yaw

usage:
  python stream_arm_zmq.py
  python stream_arm_zmq.py --port 5557 --rate 60
  python stream_arm_zmq.py --waist --debug
"""

import argparse
import time

import numpy as np
import zmq

ZMQ_PORT = 5557

STATIC_POSE = {
    "left_shoulder_pitch_joint": np.radians(-90.0),
    "right_shoulder_pitch_joint": np.radians(-90.0),
    "left_shoulder_roll_joint": np.radians(15),
    "right_shoulder_roll_joint": np.radians(-15.0),
    "left_shoulder_yaw_joint": np.radians(0.0),
    "right_shoulder_yaw_joint": np.radians(0.0),
    "left_elbow_joint": np.radians(90.0),
    "right_elbow_joint": np.radians(90.0),
    "left_wrist_roll_joint": np.radians(90.0),
    "right_wrist_roll_joint": np.radians(-90.0),
    "left_wrist_pitch_joint": np.radians(0.0),
    "right_wrist_pitch_joint": np.radians(0.0),
    "left_wrist_yaw_joint": np.radians(0.0),
    "right_wrist_yaw_joint": np.radians(0.0),
}

# sequential L then R order (matches G1_29_ArmController)
STATIC_JOINT_ORDER = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def main():
    parser = argparse.ArgumentParser(description="Stream arm joints via ZMQ")
    parser.add_argument("--port", type=int, default=ZMQ_PORT,
                        help="ZMQ PUB port (default: 5557)")
    parser.add_argument("--rate", type=int, default=60,
                        help="Streaming rate in Hz (default: 60)")
    parser.add_argument("--pos-only", action="store_true",
                        help="Position-only IK (ignore orientation)")
    parser.add_argument("--static", action="store_true",
                        help="Stream a fixed static pose (no trackers needed)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--viz", action="store_true",
                        help="Launch meshcat visualizer")
    parser.add_argument("--waist", action="store_true",
                        help="Use 3rd tracker for waist")
    parser.add_argument("--smooth", type=float, default=None,
                        help="Smoothing strength 0-1 (0=max smoothing, 1=no smoothing). "
                             "Overrides SMOOTH_ALPHA and SMOOTH_ALPHA_ROT in arm_pink_real.")
    args = parser.parse_args()

    # set up ZMQ publisher
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"ZMQ PUB on tcp://*:{args.port} --> 14x float32 arm joints in radians (sequential L/R)")

    if args.static:
        joints = np.array([STATIC_POSE[j] for j in STATIC_JOINT_ORDER], dtype=np.float32)
        print(f"Streaming static pose at {args.rate} Hz")
        dt = 1.0 / args.rate
        try:
            while True:
                sock.send(joints.tobytes())
                time.sleep(dt)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    from arm_pink_real import load_model, run_live, _setup_viz

    if args.smooth is not None:
        import arm_pink_real as apr
        apr.SMOOTH_ALPHA = args.smooth
        apr.SMOOTH_ALPHA_ROT = args.smooth * 0.7
        print(f"Smoothing override: alpha={apr.SMOOTH_ALPHA:.2f}, alpha_rot={apr.SMOOTH_ALPHA_ROT:.2f}")

    def on_frame(joints_rad):
        sock.send(joints_rad.tobytes())

    # load model and run (frame_callback receives 14x float32 in sequential order)
    model, data, left_id, right_id, model_full = load_model()
    viz = _setup_viz(model_full) if args.viz else None

    run_live(model, data, left_id, right_id, model_full,
             rate=args.rate, pos_only=args.pos_only,
             debug=args.debug, continuous=True, viz=viz,
             use_waist=args.waist, frame_callback=on_frame)


if __name__ == "__main__":
    main()
