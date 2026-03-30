"""Grab frames from local RealSense surround camera, try ALL pupil-apriltags families
+ OpenCV ArUco dicts, and report what's detected. Press 'q' to quit."""

import cv2
import numpy as np
from pupil_apriltags import Detector

# All families supported by pupil-apriltags
FAMILIES = [
    "tag36h11",
    "tag25h9",
    "tag16h5",
    "tagCircle21h7",
    "tagCircle49h12",
    "tagCustom48h12",
    "tagStandard41h12",
    "tagStandard52h13",
]

# Build detectors
detectors = {}
for fam in FAMILIES:
    detectors[fam] = Detector(families=fam, nthreads=2, quad_decimate=1.0)

# Open surround camera via RealSense
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, 30)
profile = pipeline.start(config)

print("Streaming from surround camera. Press 'q' to quit.\n")

try:
    while True:
        frames = pipeline.wait_for_frames(timeout_ms=2000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        bgr = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        display = bgr.copy()

        found_any = False
        for fam, det in detectors.items():
            results = det.detect(gray)
            for r in results:
                found_any = True
                cx, cy = int(r.center[0]), int(r.center[1])
                label = f"{fam} ID={r.tag_id} margin={r.decision_margin:.1f}"
                print(f"  FOUND: {label} at ({cx},{cy})")
                # draw on display
                corners = r.corners.astype(int)
                for i in range(4):
                    cv2.line(display, tuple(corners[i]), tuple(corners[(i+1)%4]), (0,255,0), 2)
                cv2.putText(display, label, (cx-80, cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

        # Also try OpenCV ArUco dicts
        aruco_dicts = {
            "APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
            "APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "APRILTAG_25h9":  cv2.aruco.DICT_APRILTAG_25h9,
            "APRILTAG_16h5":  cv2.aruco.DICT_APRILTAG_16h5,
            "ARUCO_4X4_50":   cv2.aruco.DICT_4X4_50,
            "ARUCO_5X5_50":   cv2.aruco.DICT_5X5_50,
            "ARUCO_6X6_50":   cv2.aruco.DICT_6X6_50,
        }
        for name, dict_id in aruco_dicts.items():
            d = cv2.aruco.getPredefinedDictionary(dict_id)
            p = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(d, p)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                for i, mid in enumerate(ids.flatten()):
                    found_any = True
                    c = corners[i][0].mean(axis=0).astype(int)
                    label = f"cv2:{name} ID={mid}"
                    print(f"  FOUND: {label} at ({c[0]},{c[1]})")

        if found_any:
            print("---")

        cv2.imshow("AprilTag Detection", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("\nDone.")
