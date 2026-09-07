import gzip
from pathlib import Path
import struct
import tempfile
import unittest

from hotwheels_re_tools.savestate import (
    CPU_RACER,
    EWRAM_BASE,
    EWRAM_OFFSET,
    EWRAM_SIZE,
    PLAYER_RACER,
    SAVESTATE_SIZE,
    inspect_state,
)


class SavestateTests(unittest.TestCase):
    def test_discovers_manager_pointer_list_and_racers(self) -> None:
        payload = bytearray(SAVESTATE_SIZE)
        ewram = memoryview(payload)[EWRAM_OFFSET : EWRAM_OFFSET + EWRAM_SIZE]
        manager = EWRAM_BASE + 0x1000
        pointer_list = EWRAM_BASE + 0x1800
        player = EWRAM_BASE + 0x2000
        cpu = EWRAM_BASE + 0x2400

        def put_u8(address: int, value: int) -> None:
            ewram[address - EWRAM_BASE] = value

        def put_u16(address: int, value: int) -> None:
            struct.pack_into("<H", ewram, address - EWRAM_BASE, value)

        def put_u32(address: int, value: int) -> None:
            struct.pack_into("<I", ewram, address - EWRAM_BASE, value)

        put_u8(manager + 0x448, 1)
        put_u8(manager + 0x449, 1)
        put_u8(manager + 0x44A, 2)
        put_u32(manager + 0x450, pointer_list)
        put_u32(pointer_list, player)
        put_u32(pointer_list + 4, cpu)
        put_u32(player, PLAYER_RACER.vtable)
        put_u32(cpu, CPU_RACER.vtable)
        for address, progress in [(player, 71), (cpu, 73)]:
            put_u32(address + 0x50, manager)
            put_u16(address + 0x148, progress)
        put_u16(player + 0x306, 0x21)
        put_u16(cpu + 0xDE, 0x123)
        put_u16(cpu + 0xE0, 0x120)
        put_u32(cpu + 0x2F0, 0xEA00)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.state"
            with gzip.open(path, "wb") as handle:
                handle.write(payload)
            result = inspect_state(path)

        self.assertEqual(result.manager, manager)
        self.assertEqual(result.pointer_list, pointer_list)
        self.assertEqual((result.player_count, result.cpu_count, result.total_count), (1, 1, 2))
        self.assertEqual([racer.kind for racer in result.racers], ["player", "cpu"])
        self.assertEqual([racer.progress for racer in result.racers], [71, 73])
        self.assertEqual(result.racers[0].held, 0x21)
        self.assertEqual(result.racers[1].current_heading, 0x123)
        self.assertEqual(result.racers[1].desired_heading, 0x120)
        self.assertEqual(result.racers[1].target_speed, 0xEA00)


if __name__ == "__main__":
    unittest.main()
