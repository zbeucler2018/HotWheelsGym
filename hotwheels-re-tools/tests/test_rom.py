import unittest

from hotwheels_re_tools.rom import (
    NPC_BUTTON_CONTROL_SITE,
    NPC_BUTTON_CPU_COUNT_SITES,
    NPC_BUTTON_PLAYER_COUNT_SITES,
    NPC_BUTTON_CONTROL_TRAMPOLINE,
    apply_checked_patches,
    button_controlled_vehicle_indices,
    decode_thumb_bl,
    encode_thumb_bl,
    npc_button_control_patches,
)


class ThumbBranchTests(unittest.TestCase):
    def test_encode_round_trip_for_backward_and_forward_calls(self) -> None:
        for source, target in [
            (NPC_BUTTON_CONTROL_SITE, NPC_BUTTON_CONTROL_TRAMPOLINE),
            (NPC_BUTTON_CONTROL_SITE, 0x081025D4),
        ]:
            self.assertEqual(
                decode_thumb_bl(encode_thumb_bl(source, target), source), target
            )

    def test_checked_patch_rejects_unexpected_input(self) -> None:
        patches = npc_button_control_patches()
        size = max(patch.offset + len(patch.expected) for patch in patches)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            apply_checked_patches(bytes(size), patches)

    def test_native_button_patch_converts_every_cpu_and_embeds_marker(self) -> None:
        patches = npc_button_control_patches()
        self.assertEqual(
            tuple(patch.address for patch in patches[:4]),
            (
                NPC_BUTTON_CPU_COUNT_SITES[0],
                NPC_BUTTON_PLAYER_COUNT_SITES[0],
                NPC_BUTTON_CPU_COUNT_SITES[1],
                NPC_BUTTON_PLAYER_COUNT_SITES[1],
            ),
        )
        self.assertEqual(patches[4].address, NPC_BUTTON_CONTROL_SITE)
        self.assertEqual(
            decode_thumb_bl(patches[4].replacement, patches[4].address),
            NPC_BUTTON_CONTROL_TRAMPOLINE,
        )
        size = max(patch.offset + len(patch.expected) for patch in patches)
        source = bytearray(size)
        for patch in patches:
            source[patch.offset : patch.offset + len(patch.expected)] = patch.expected
        output = apply_checked_patches(bytes(source), patches)
        self.assertEqual(button_controlled_vehicle_indices(output), (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
