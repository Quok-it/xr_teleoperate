import threading
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def _fast_mat_inv(mat):
    """Invert a rigid-body (SE3) 4x4 matrix using R^T shortcut."""
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


class AprilTagHeadTracker:
    """Detects a fixed AprilTag in the environment and estimates the camera (head) pose
    in the tag's coordinate frame (world frame).

    All public methods are non-blocking. Detection runs in a background daemon thread.

    Args:
        camera_params: (fx, fy, cx, cy) camera intrinsics.
        tag_family: AprilTag family string, e.g. "tag36h11".
        tag_size: Physical side length of the tag in meters.
        target_tag_id: Which tag ID to track.
    """

    def __init__(self, camera_params, tag_family="tagStandard41h12", tag_size=0.2, target_tag_id=2):
        try:
            from pupil_apriltags import Detector
        except ImportError:
            raise ImportError("pupil-apriltags is required: pip install pupil-apriltags")

        self._camera_params = camera_params  # (fx, fy, cx, cy)
        self._tag_size = tag_size
        self._target_tag_id = target_tag_id

        self._detector = Detector(families=tag_family, nthreads=2, quad_decimate=1.0)

        # Latest frame (lock-protected)
        self._frame_lock = threading.Lock()
        self._latest_frame = None

        # Latest pose result (lock-protected)
        self._pose_lock = threading.Lock()
        self._latest_pose = None  # (4,4) SE(3) T_world_camera or None

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        logger.info(f"AprilTagHeadTracker started: family={tag_family}, size={tag_size}m, id={target_tag_id}")

    def update_frame(self, bgr_frame):
        """Non-blocking: provide the latest BGR frame for detection."""
        with self._frame_lock:
            self._latest_frame = bgr_frame

    def get_pose(self):
        """Non-blocking: return the latest head pose as (4,4) SE(3) matrix, or None if no tag detected."""
        with self._pose_lock:
            return self._latest_pose

    def stop(self):
        """Stop the detection thread."""
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        logger.info("AprilTagHeadTracker stopped.")

    def _detection_loop(self):
        frames_since_detection = 0

        while not self._stop_event.is_set():
            # Grab latest frame
            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None  # consume it

            if frame is None:
                self._stop_event.wait(timeout=0.005)  # ~200Hz poll when idle
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect tags with pose estimation
            detections = self._detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=self._camera_params,
                tag_size=self._tag_size,
            )

            # Find the target tag
            target = None
            for det in detections:
                if det.tag_id == self._target_tag_id:
                    target = det
                    break

            if target is not None and target.pose_R is not None and target.pose_t is not None:
                # Build T_camera_tag (tag pose in camera frame)
                T_cam_tag = np.eye(4)
                T_cam_tag[:3, :3] = target.pose_R
                T_cam_tag[:3, 3] = target.pose_t.flatten()

                # Invert to get T_tag_camera (camera/head pose in tag/world frame)
                T_world_camera = _fast_mat_inv(T_cam_tag)

                with self._pose_lock:
                    self._latest_pose = T_world_camera

                frames_since_detection = 0
                pos = T_world_camera[:3, 3]
                logger.debug(f"AprilTag {self._target_tag_id} detected: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}")
            else:
                frames_since_detection += 1
                if frames_since_detection == 60:  # ~2 seconds at 30fps
                    logger.warning(f"AprilTag {self._target_tag_id} not detected for ~2 seconds")
                # Keep last valid pose (don't set to None)
