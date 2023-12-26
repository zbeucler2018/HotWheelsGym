import enum


class Tracks(enum.Enum):
    """
    Tracks to race on
    """

    TRex_Valley = "trex_valley"
    Dino_Boneyard = "dino_boneyard"


class RaceMode(enum.Enum):
    """
    Racing modes
    """

    SINGLE = "single"  # Play as the only car on the track. Trex Valley only!
    MULTI = "multi"  # Play against other NPCs
