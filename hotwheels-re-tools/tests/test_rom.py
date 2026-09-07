import unittest

from hotwheels_re_tools.rom import (
    MIRROR_CPU_PATCHES,
    PLAYER_RACER_CONSTRUCTOR,
    apply_checked_patches,
    decode_thumb_bl,
    encode_thumb_bl,
)


class ThumbBranchTests(unittest.TestCase):
    def test_decode_original_cpu_constructor_calls(self) -> None:
        self.assertEqual(
            decode_thumb_bl(bytes.fromhex("f9f7e6fe"), 0x080FBAB8), 0x080F5888
        )
        self.assertEqual(
            decode_thumb_bl(bytes.fromhex("f9f71afc"), 0x080FC050), 0x080F5888
        )

    def test_new_calls_target_player_constructor(self) -> None:
        for patch in MIRROR_CPU_PATCHES:
            if "constructor" in patch.name:
                self.assertEqual(
                    decode_thumb_bl(patch.replacement, patch.address),
                    PLAYER_RACER_CONSTRUCTOR,
                )

    def test_encode_round_trip_for_backward_and_forward_calls(self) -> None:
        for source, target in [
            (0x080FBAB8, 0x080F5888),
            (0x080FBAB8, 0x08101618),
        ]:
            self.assertEqual(decode_thumb_bl(encode_thumb_bl(source, target), source), target)

    def test_checked_patch_rejects_unexpected_input(self) -> None:
        size = max(patch.offset + len(patch.expected) for patch in MIRROR_CPU_PATCHES)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            apply_checked_patches(bytes(size), MIRROR_CPU_PATCHES)


if __name__ == "__main__":
    unittest.main()

