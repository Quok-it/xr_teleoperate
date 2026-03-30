#!/usr/bin/env python3
"""Grab frames from local RealSense (D405), detect AprilTags with multiple families,
and display annotated live view. Press 'q' to quit."""

import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

# Families to try
FAMILIES = [
    "tagStandard41h12",
    "tag36h11",
    "tag25h9",
    "tag16h5",
    "tagStandard52h13",
    "tagCircle21h7",
    "tagCircle49h12",
    "tagCustom48h12",
]

# Set up RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

dev = profile.get_device()
serial = dev.get_info(rs.camera_info.serial_number)
name = dev.get_info(rs.camera_info.name)
print(f"Camera: {name} (serial: {serial})")

stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = stream.get_intrinsics()
camera_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
print(f"Intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}")

# Build one detector per family
detectors = {}
for fam in FAMILIES:
    try:
        detectors[fam] = Detector(families=fam, nthreads=2, quad_decimate=1.0)
    except Exception as e:
        print(f"  Skipping {fam}: {e}")

print(f"\nLoaded {len(detectors)} detectors. Hold your AprilTag in view. Press 'q' to quit.\n")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        bgr = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        display = bgr.copy()

        found_any = False
        for fam, det in detectors.items():
            results = det.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=0.105,
            )
            for r in results:
                found_any = True
                # Draw corners
                corners = r.corners.astype(int)
                for i in range(4):
                    cv2.line(display, tuple(corners[i-1]), tuple(corners[i]), (0, 255, 0), 2)
                # Label
                cx, cy = int(r.center[0]), int(r.center[1])
                label = f"{fam} ID={r.tag_id}"
                cv2.putText(display, label, (cx - 80, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                margin = f"margin={r.decision_margin:.1f}"
                cv2.putText(display, margin, (cx - 80, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)

                # Pose info
                if r.pose_t is not None:
                    t = r.pose_t.flatten()
                    dist = np.linalg.norm(t)
                    pose_str = f"d={dist:.3f}m  t=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}]"
                    cv2.putText(display, pose_str, (cx - 120, cy + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)

                print(f"  {fam} ID={r.tag_id}  margin={r.decision_margin:.1f}  "
                      f"center=({cx},{cy})  dist={np.linalg.norm(r.pose_t):.3f}m")

        if not found_any:
            cv2.putText(display, "No tags detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("AprilTag Detection", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Done.")
