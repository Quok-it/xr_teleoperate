"""Debug: grab one frame, try tag36h11 with various settings, save annotated output."""
import sys
import zmq
import numpy as np
import cv2
from pupil_apriltags import Detector

ROBOT_IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.164"
ZMQ_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 55555

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect(f"tcp://{ROBOT_IP}:{ZMQ_PORT}")
sock.setsockopt(zmq.SUBSCRIBE, b"")
sock.setsockopt(zmq.RCVTIMEO, 5000)
sock.setsockopt(zmq.CONFLATE, 1)

print(f"Grabbing frame from {ROBOT_IP}:{ZMQ_PORT} ...")
jpg_bytes = sock.recv()
np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
h, w = bgr.shape[:2]
print(f"Frame: {h}x{w}")

cv2.imwrite("/tmp/at_debug.jpg", bgr)

# tagStandard41h12, ID 1
det = Detector(families="tagStandard41h12", nthreads=2, quad_decimate=1.0)
results = det.detect(gray)
print(f"\ntagStandard41h12: Found {len(results)} detections")
for r in results:
    print(f"  ID={r.tag_id} center=({int(r.center[0])},{int(r.center[1])}) margin={r.decision_margin:.1f}")

# --- Try OpenCV's AprilTag dictionaries ---
print("\n--- OpenCV AprilTag dictionaries ---")
cv_apriltag_dicts = {
    "APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "APRILTAG_25h9":  cv2.aruco.DICT_APRILTAG_25h9,
    "APRILTAG_16h5":  cv2.aruco.DICT_APRILTAG_16h5,
}
for name, dict_id in cv_apriltag_dicts.items():
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    p = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(d, p)
    corners, ids, rejected = detector.detectMarkers(gray)
    n_det = 0 if ids is None else len(ids)
    n_rej = len(rejected) if rejected is not None else 0
    print(f"  {name}: detected={n_det}, rejected_candidates={n_rej}")
    if ids is not None:
        for i, mid in enumerate(ids.flatten()):
            c = corners[i][0].mean(axis=0).astype(int)
            print(f"    ID={mid} center=({c[0]},{c[1]})")

# --- Try ArUco dictionaries ---
print("\n--- ArUco dictionaries ---")
aruco_dicts = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "6X6_50": cv2.aruco.DICT_6X6_50,
    "7X7_50": cv2.aruco.DICT_7X7_50,
    "ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}
for name, dict_id in aruco_dicts.items():
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    p = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(d, p)
    corners, ids, rejected = detector.detectMarkers(gray)
    n_det = 0 if ids is None else len(ids)
    n_rej = len(rejected) if rejected is not None else 0
    print(f"  {name}: detected={n_det}, rejected_candidates={n_rej}")
    if ids is not None:
        for i, mid in enumerate(ids.flatten()):
            c = corners[i][0].mean(axis=0).astype(int)
            print(f"    ID={mid} center=({c[0]},{c[1]})")

cv2.imwrite("/tmp/at_debug_region.jpg", gray[50:350, 800:1200])
print("\nSaved /tmp/at_debug_region.jpg")

sock.close()
ctx.term()
