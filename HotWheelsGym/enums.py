import enum


class Tracks(enum.Enum):
    """
    Tracks to race on
    """

    TRex_Valley = "trex_valley"
    Dino_Boneyard = "dino_boneyard"
    Monsters_of_the_Deep = "monsters_of_the_deep"
    Whiteskull_Cliffs = "whiteskull_cliffs"
    Jungle_Snakepit = "jungle_snakepit"
    Gator_Forest = "gator_forest"
    Solar_Strip = "solar_strip"
    Satellite_Mission = "satellite_mission"
    


class RaceMode(enum.Enum):
    """
    Racing modes
    """

    SINGLE = "single"  # Play as the only car on the track. Trex Valley only!
    MULTI = "multi"  # Play against other NPCs
