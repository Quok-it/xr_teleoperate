"""Quick test: stream head camera and print AprilTag head pose in real time."""
import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "teleop"))

from teleimager.image_client import ImageClient
from teleop.utils.apriltag_tracker import AprilTagTracker

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.164"

img_client = ImageClient(host=ROBOT_IP, request_bgr=True)
cam_config = img_client.get_cam_config()

intr = cam_config['head_camera'].get('intrinsics')
if not intr:
    print("ERROR: No intrinsics in cam_config. Update image_server on robot.")
    sys.exit(1)

print(f"Intrinsics: fx={intr['fx']:.1f} fy={intr['fy']:.1f} cx={intr['cx']:.1f} cy={intr['cy']:.1f}")

tracker = AprilTagTracker(
    camera_params=(intr['fx'], intr['fy'], intr['cx'], intr['cy']),
)

try:
    while True:
        head_img = img_client.get_head_frame()
        if head_img is not None and head_img.bgr is not None:
            import cv2
            cv2.imwrite("/tmp/head_frame.png", head_img.bgr)
            break
            tracker.update_frame(head_img.bgr)

        pose, detected = tracker.get_pose()
        if pose is not None:
            pos = pose[:3, 3]
            status = "DETECTED" if detected else "LOST    "
            print(f"{status}  x={pos[0]:+.3f}  y={pos[1]:+.3f}  z={pos[2]:+.3f}", end="\r")
        else:
            print("NO TAG DETECTED                                      ", end="\r")

        time.sleep(0.033)
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    tracker.stop()
    img_client.close()
