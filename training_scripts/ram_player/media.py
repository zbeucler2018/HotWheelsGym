"""Small MP4 and TensorBoard replay helpers for RAM-policy evaluation."""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
from tensorboard.compat.proto.summary_pb2 import Summary


class RGBVideoWriter:
    """Stream RGB arrays to H.264 without retaining a race in memory."""

    def __init__(self, path: Path, first_frame: np.ndarray, fps: float) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        first_frame = self._frame(first_frame)
        height, width = first_frame.shape[:2]
        self._encoder = imageio_ffmpeg.write_frames(
            str(self.path),
            (width, height),
            fps=fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            ffmpeg_log_level="warning",
        )
        self._encoder.send(None)
        self.write(first_frame)

    @staticmethod
    def _frame(frame: Any) -> np.ndarray:
        result = np.asarray(frame, dtype=np.uint8)
        if result.ndim != 3 or result.shape[2] != 3:
            raise RuntimeError(f"unexpected RGB frame shape: {result.shape}")
        return np.ascontiguousarray(result)

    def write(self, frame: Any) -> None:
        self._encoder.send(self._frame(frame))

    def close(self) -> None:
        self._encoder.close()


def log_video_replay(
    writer: Any,
    tag: str,
    video_path: Path,
    *,
    global_step: int = 0,
    sample_fps: int = 1,
    playback_fps: int = 8,
    width: int = 240,
) -> Path:
    """Embed a compact, accelerated GIF replay in TensorBoard's Images tab."""

    if sample_fps < 1 or playback_fps < 1 or width < 1:
        raise ValueError("replay FPS and width must be positive")
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    gif_path = video_path.with_suffix(".tensorboard.gif")
    speedup = playback_fps / sample_fps
    video_filter = (
        f"fps={sample_fps},scale={width}:-2:flags=lanczos," f"setpts=PTS/{speedup:g}"
    )
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            video_filter,
            "-loop",
            "0",
            str(gif_path),
        ],
        check=True,
    )
    encoded = gif_path.read_bytes()
    if encoded[:6] not in (b"GIF87a", b"GIF89a"):
        raise RuntimeError(f"ffmpeg did not create a GIF: {gif_path}")
    image_width, image_height = struct.unpack("<HH", encoded[6:10])
    summary = Summary(
        value=[
            Summary.Value(
                tag=tag,
                image=Summary.Image(
                    height=image_height,
                    width=image_width,
                    colorspace=3,
                    encoded_image_string=encoded,
                ),
            )
        ]
    )
    writer._get_file_writer().add_summary(summary, global_step)
    writer.flush()
    return gif_path
