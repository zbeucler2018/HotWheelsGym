import unittest

from hotwheels_re_tools.rom import (
    EXTERNAL_CPU_HEADING_PATCHES,
    MIRROR_CPU_PATCHES,
    NPC_CONTROL_HOOK,
    NPC_CONTROL_ROM_SHA1S,
    NPC_CONTROL_SITE,
    NPC_CONTROL_TRAMPOLINE,
    PLAYER_RACER_CONSTRUCTOR,
    apply_checked_patches,
    controlled_vehicle_indices,
    decode_thumb_bl,
    encode_thumb_bl,
    npc_control_patches,
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

    def test_external_heading_patch_nops_checked_store(self) -> None:
        patch = EXTERNAL_CPU_HEADING_PATCHES[0]
        data = bytearray(patch.offset + len(patch.expected))
        data[patch.offset : patch.offset + len(patch.expected)] = patch.expected
        output = apply_checked_patches(bytes(data), EXTERNAL_CPU_HEADING_PATCHES)
        self.assertEqual(
            output[patch.offset : patch.offset + len(patch.replacement)],
            bytes.fromhex("c046"),
        )

    def test_selected_npc_patch_uses_checked_trampoline_and_marker(self) -> None:
        patches = npc_control_patches((3, 1, 1))
        self.assertEqual(patches[0].address, NPC_CONTROL_SITE)
        self.assertEqual(
            decode_thumb_bl(patches[0].replacement, NPC_CONTROL_SITE),
            NPC_CONTROL_TRAMPOLINE,
        )
        self.assertEqual(patches[1].address, NPC_CONTROL_TRAMPOLINE)
        self.assertEqual(patches[2].address, NPC_CONTROL_HOOK)

        size = max(patch.offset + len(patch.expected) for patch in patches)
        source = bytearray(size)
        for patch in patches:
            source[patch.offset : patch.offset + len(patch.expected)] = patch.expected
        output = apply_checked_patches(bytes(source), patches)
        self.assertEqual(controlled_vehicle_indices(output), (1, 3))

    def test_selected_npc_patch_rejects_non_cpu_vehicle_indices(self) -> None:
        for indices in ((), (0,), (4,)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                npc_control_patches(indices)

    def test_known_hashes_cover_every_nonempty_cpu_selection(self) -> None:
        self.assertEqual(
            set(NPC_CONTROL_ROM_SHA1S),
            {(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)},
        )


if __name__ == "__main__":
    unittest.main()
