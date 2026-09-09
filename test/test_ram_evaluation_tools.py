import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from training_scripts.ram_player.common import require_all_opponent_slots
from training_scripts.ram_player.media import RGBVideoWriter, log_video_replay
from training_scripts.ram_player.sweep import checkpoint_sort_key


class RAMEvaluationToolTests(unittest.TestCase):
    def test_native_button_runs_require_every_opponent_slot(self):
        require_all_opponent_slots({})
        require_all_opponent_slots({1: "model", 2: "model", 3: "model"})
        with self.assertRaisesRegex(ValueError, "slots 1, 2, and 3"):
            require_all_opponent_slots({1: "model"})

    def test_checkpoint_selection_prefers_finish_then_speed(self):
        partial = {
            "finish_rate": 0.0,
            "mean_finish_frames": 0.0,
            "mean_completion": 0.99,
            "mean_rank": 1.0,
            "mean_reward": 999.0,
        }
        slow_finish = {
            "finish_rate": 1.0,
            "mean_finish_frames": 20_000.0,
            "mean_completion": 1.0,
            "mean_rank": 1.0,
            "mean_reward": 900.0,
        }
        fast_finish = dict(slow_finish, mean_finish_frames=18_000.0)
        self.assertIs(
            max((partial, slow_finish, fast_finish), key=checkpoint_sort_key),
            fast_finish,
        )

    def test_tensorboard_replay_is_an_animated_image_summary(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "race.mp4"
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            recorder = RGBVideoWriter(video, frame, fps=4)
            for red in (32, 64, 128, 255):
                next_frame = frame.copy()
                next_frame[:, :, 0] = red
                recorder.write(next_frame)
            recorder.close()

            log_dir = root / "events"
            writer = SummaryWriter(log_dir=str(log_dir))
            gif = log_video_replay(
                writer,
                "evaluation_replays/test",
                video,
                sample_fps=1,
                playback_fps=2,
                width=16,
            )
            writer.close()

            self.assertTrue(gif.is_file())
            accumulator = EventAccumulator(str(log_dir))
            accumulator.Reload()
            self.assertIn("evaluation_replays/test", accumulator.Tags()["images"])


if __name__ == "__main__":
    unittest.main()
