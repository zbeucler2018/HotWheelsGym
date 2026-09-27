"""Dependency-free local HTTP server for the RAM observation visualizer."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import struct
from threading import Condition, RLock, Thread
from time import monotonic
from typing import Any
from urllib.parse import urlparse
import zlib

import numpy as np

from .adapter import ObservationVisualizerAdapter

STATIC_ROOT = Path(__file__).with_name("static")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


class PlaybackController:
    """Advance the emulator independently of browser/network round trips."""

    def __init__(self, adapter: ObservationVisualizerAdapter, lock: RLock) -> None:
        self.adapter = adapter
        self.lock = lock
        self._condition = Condition()
        self._playing = False
        self._closed = False
        self._speed = 1.0
        self._frames_per_tick = 4
        self._thread = Thread(target=self._run, name="ram-visualizer-playback", daemon=True)
        self._thread.start()

    def play(self, speed: float) -> None:
        if not 0.25 <= speed <= 8.0:
            raise ValueError("playback speed must be in 0.25..8")
        with self._condition:
            self._speed = speed
            self._playing = True
            self._condition.notify_all()

    def pause(self) -> None:
        with self._condition:
            self._playing = False

    def status(self) -> dict[str, float | bool]:
        with self._condition:
            return {"playing": self._playing, "speed": self._speed}

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._playing = False
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._playing and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                speed = self._speed
            started = monotonic()
            with self.lock:
                self.adapter.step(self._frames_per_tick)
            target_seconds = self._frames_per_tick / (60.0 * speed)
            remaining = target_seconds - (monotonic() - started)
            if remaining > 0:
                with self._condition:
                    self._condition.wait(timeout=remaining)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def encode_png(framebuffer: np.ndarray) -> bytes:
    """Encode an RGB/RGBA uint8 framebuffer without a web or image dependency."""

    frame = np.asarray(framebuffer, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(f"unexpected framebuffer shape {frame.shape}")
    height, width, channels = frame.shape
    color_type = 2 if channels == 3 else 6
    scanlines = b"".join(b"\x00" + row.tobytes() for row in frame)
    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return signature + _png_chunk(b"IHDR", header) + _png_chunk(
        b"IDAT", zlib.compress(scanlines, level=3)
    ) + _png_chunk(b"IEND", b"")


def create_server(
    adapter: ObservationVisualizerAdapter,
    *,
    host: str = "0.0.0.0",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create the local server without starting its blocking serve loop."""

    lock = RLock()
    playback = PlaybackController(adapter, lock)

    class Handler(BaseHTTPRequestHandler):
        server_version = "HotWheelsRAMVisualizer/1"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

        def _send(
            self,
            status: HTTPStatus,
            body: bytes,
            content_type: str,
            *,
            cache: str = "no-store",
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(
                status,
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _snapshot_payload(self) -> dict[str, Any]:
            payload = adapter.snapshot().payload()
            payload["playback"] = playback.status()
            return payload

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 16_384:
                raise ValueError("request body is too large")
            if not length:
                return {}
            raw = self.rfile.read(length)
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            try:
                if path in STATIC_FILES:
                    filename, content_type = STATIC_FILES[path]
                    body = (STATIC_ROOT / filename).read_bytes()
                    self._send(
                        HTTPStatus.OK,
                        body,
                        content_type,
                        cache="no-cache",
                    )
                    return
                if path == "/api/snapshot":
                    with lock:
                        self._json(self._snapshot_payload())
                    return
                if path == "/api/frame":
                    with lock:
                        body = encode_png(adapter.snapshot().framebuffer)
                    self._send(HTTPStatus.OK, body, "image/png")
                    return
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as error:  # pragma: no cover - defensive HTTP boundary
                self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                body = self._body()
                with lock:
                    if path == "/api/reset":
                        playback.pause()
                        if "controlled_slot" in body:
                            slot = int(body["controlled_slot"])
                            if slot not in range(4):
                                raise ValueError(
                                    "controlled_slot must be 0, 1, 2, or 3"
                                )
                            adapter.controlled_slot = slot
                        snapshot = adapter.reset()
                    elif path == "/api/step":
                        playback.pause()
                        snapshot = adapter.step(int(body.get("frames", 1)))
                    elif path == "/api/play":
                        playback.play(float(body.get("speed", 1.0)))
                        snapshot = adapter.snapshot()
                    elif path == "/api/pause":
                        playback.pause()
                        snapshot = adapter.snapshot()
                    elif path == "/api/controlled-slot":
                        snapshot = adapter.set_controlled_slot(int(body["slot"]))
                    else:
                        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                        return
                payload = snapshot.payload()
                payload["playback"] = playback.status()
                self._json(payload)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:  # pragma: no cover - defensive HTTP boundary
                self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    server = ThreadingHTTPServer((host, port), Handler)
    server.playback_controller = playback  # type: ignore[attr-defined]
    return server


def serve(
    adapter: ObservationVisualizerAdapter,
    *,
    host: str = "0.0.0.0",
    port: int = 8765,
) -> None:
    server = create_server(adapter, host=host, port=port)
    print(f"HotWheels RAM visualizer listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.playback_controller.close()  # type: ignore[attr-defined]
        server.server_close()
        adapter.close()
