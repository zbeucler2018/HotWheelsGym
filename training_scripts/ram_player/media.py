"""Small MP4 helper for RAM-policy evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np


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
