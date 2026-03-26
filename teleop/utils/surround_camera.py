"""Surrounding camera: local RealSense with background capture.

Mirrors ImageClient's API so the main loop can use it the same way as the head camera.
Tracker is kept separate — initialize and feed it in the main loop, same as head_tracker.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import logging

logger = logging.getLogger(__name__)


class SurroundImage:
    """Mirrors TeleImage: .bgr (np.ndarray) and .jpg (bytes)."""
    __slots__ = ['bgr', 'jpg', 'fps']

    def __init__(self, bgr, jpg, fps):
        self.bgr = bgr
        self.jpg = jpg
        self.fps = fps


class SurroundCamera:
    """Local RealSense camera client. Background thread captures frames
    so get_frame() / get_depth_frame() are non-blocking, same as ImageClient."""

    def __init__(self, serial=None, width=480, height=270, fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 4)  # High Density

        # Intrinsics (full format matching image_server cam_config)
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = stream.get_intrinsics()
        self.intrinsics = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy}
        self.color_intrinsics = {
            "width": intr.width, "height": intr.height,
            "fx": intr.fx, "fy": intr.fy,
            "ppx": intr.ppx, "ppy": intr.ppy,
            "distortion_model": str(intr.model),
            "distortion_coeffs": list(intr.coeffs),
        }
        dintr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = {
            "width": dintr.width, "height": dintr.height,
            "fx": dintr.fx, "fy": dintr.fy,
            "ppx": dintr.ppx, "ppy": dintr.ppy,
            "distortion_model": str(dintr.model),
            "distortion_coeffs": list(dintr.coeffs),
        }
        self.image_shape = [height, width]
        self.depth_scale = depth_sensor.get_depth_scale()
        self.fps = fps

        # Device info
        dev = profile.get_device()
        self.serial = dev.get_info(rs.camera_info.serial_number)
        self.name = dev.get_info(rs.camera_info.name)
        logger.info(f"SurroundCamera: {self.name} (serial: {self.serial})")
        logger.info(f"  Intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}")

        # Shared frame buffers
        self._lock = threading.Lock()
        self._color = None      # BGR ndarray
        self._color_jpg = None  # JPEG bytes
        self._depth_raw = None  # uint16 ndarray

        # Background capture
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while not self._stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
            except RuntimeError:
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            _, jpg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 95])

            with self._lock:
                self._color = color
                self._color_jpg = jpg.tobytes()
                self._depth_raw = depth

    def get_frame(self):
        """Return SurroundImage (like TeleImage) or None."""
        with self._lock:
            if self._color is None:
                return None
            return SurroundImage(bgr=self._color, jpg=self._color_jpg, fps=self.fps)

    def get_depth_frame(self):
        """Return raw uint16 depth ndarray or None."""
        with self._lock:
            return self._depth_raw

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self.pipeline.stop()
        logger.info("SurroundCamera closed.")
