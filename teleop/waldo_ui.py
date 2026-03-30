"""
RealSense camera streaming API.

Endpoints:
  GET  /cameras                 — list connected cameras
  GET  /stream/{index}/color    — MJPEG color stream
  GET  /stream/{index}/depth    — MJPEG depth colormap stream

React usage:
  <img src="http://<host>:8585/stream/0/color" />
  <img src="http://<host>:8585/stream/0/depth" />
"""f

import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- camera state ---

cameras = []  # list of {index, serial, name, pipeline, lock, color, depth}


def _init_cameras():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        return

    print(f"Found {len(devices)} RealSense device(s):")
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"  [{i}] {name} (serial: {serial})")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 30)

        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 4)  # High Density

        cameras.append({
            "index": i,
            "serial": serial,
            "name": name,
            "pipeline": pipeline,
            "lock": threading.Lock(),
            "color": None,
            "depth": None,
        })


def _capture_loop():
    """Background thread: grab frames from all cameras into shared buffers."""
    while True:
        for cam in cameras:
            try:
                frames = cam["pipeline"].wait_for_frames(timeout_ms=100)
            except RuntimeError:
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET
            )

            with cam["lock"]:
                cam["color"] = color
                cam["depth"] = depth_colormap


# --- endpoints ---

@app.get("/cameras")
def list_cameras():
    return [{"index": c["index"], "serial": c["serial"], "name": c["name"]} for c in cameras]


def _mjpeg_generator(cam_index: int, stream_type: str):
    if cam_index < 0 or cam_index >= len(cameras):
        return
    cam = cameras[cam_index]
    while True:
        with cam["lock"]:
            frame = cam[stream_type]
        if frame is None:
            time.sleep(0.03)
            continue
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )
        time.sleep(0.03)


@app.get("/stream/{index}/color")
def stream_color(index: int):
    if index < 0 or index >= len(cameras):
        raise HTTPException(status_code=404, detail=f"Camera {index} not found")
    return StreamingResponse(
        _mjpeg_generator(index, "color"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/{index}/depth")
def stream_depth(index: int):
    if index < 0 or index >= len(cameras):
        raise HTTPException(status_code=404, detail=f"Camera {index} not found")
    return StreamingResponse(
        _mjpeg_generator(index, "depth"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# --- startup / shutdown ---

@app.on_event("startup")
def on_startup():
    _init_cameras()
    t = threading.Thread(target=_capture_loop, daemon=True)
    t.start()


@app.on_event("shutdown")
def on_shutdown():
    for cam in cameras:
        cam["pipeline"].stop()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8585)
