#!/usr/bin/env python3
"""Simple browser viewer for depth camera via ZMQ → MJPEG over HTTP."""

import argparse
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
import zmq


class DepthReceiver:
    def __init__(self, host, port, height, width, raw=False):
        self.host = host
        self.port = port
        self.h = height
        self.w = width
        self.raw = raw
        self._frame = None
        self._lock = threading.Lock()
        self._ctx = zmq.Context()
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        sock = self._ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 1)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{self.host}:{self.port}")
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        while self._running:
            events = dict(poller.poll(timeout=200))
            if sock in events:
                data = sock.recv()
                expected = self.h * self.w * 2  # uint16 = 2 bytes
                if len(data) != expected:
                    print(f"Bad frame: got {len(data)} bytes, expected {expected}")
                    continue
                depth = np.frombuffer(data, dtype=np.uint16).reshape(self.h, self.w)
                if self.raw:
                    # Raw: scale 0–3m range to 0–255 grayscale
                    gray = (np.clip(depth, 0, 3000) * (255.0 / 3000)).astype(np.uint8)
                    _, jpg = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    # D435i operating range: 0.3m–3m, fixed range avoids per-frame flicker
                    depth_clipped = np.clip(depth, 300, 3000)
                    depth_norm = ((depth_clipped - 300) * (255.0 / 2700)).astype(np.uint8)
                    colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
                    colored[depth == 0] = 0
                    _, jpg = cv2.imencode('.jpg', colored, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self._lock:
                    self._frame = jpg.tobytes()

        sock.close()

    def get_frame(self):
        with self._lock:
            return self._frame

    def stop(self):
        self._running = False


def make_handler(receiver):
    class MJPEGHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(b'<html><body style="margin:0;background:#000">'
                                 b'<img src="/stream" style="width:100%;height:auto">'
                                 b'</body></html>')
                return

            if self.path == '/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                try:
                    while True:
                        frame = receiver.get_frame()
                        if frame is not None:
                            self.wfile.write(b'--frame\r\n'
                                             b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        time.sleep(0.033)  # ~30fps
                except BrokenPipeError:
                    pass
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            pass  # silence logs

    return MJPEGHandler


def main():
    parser = argparse.ArgumentParser(description='View depth camera in browser')
    parser.add_argument('--host', default='192.168.123.164', help='Robot IP')
    parser.add_argument('--depth-port', type=int, default=55556, help='ZMQ depth port')
    parser.add_argument('--http-port', type=int, default=8080, help='HTTP server port')
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--raw', action='store_true', help='Show raw grayscale depth (no colormap/clipping)')
    args = parser.parse_args()

    receiver = DepthReceiver(args.host, args.depth_port, args.height, args.width, raw=args.raw)
    server = HTTPServer(('0.0.0.0', args.http_port), make_handler(receiver))
    print(f"Depth viewer running at http://localhost:{args.http_port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        server.server_close()


if __name__ == '__main__':
    main()
