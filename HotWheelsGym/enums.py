import enum


class Tracks(enum.Enum):
    """
    Tracks to race on
    """

    TRex_Valley = "trex_valley"
    Dino_Boneyard = "dino_boneyard"
    Black_Widows_Nest = "black_widows_nest"
    Insect_Hive = "insect_hive"
    Monsters_of_the_Deep = "monsters_of_the_deep"
    Whiteskull_Cliffs = "whiteskull_cliffs"
    Jungle_Snakepit = "jungle_snakepit"
    Gator_Forest = "gator_forest"
    Satellite_Mission = "satellite_mission"
    Solar_Strip = "solar_strip"
    Fire_Mountain = "fire_mountain"
    Volcano_Battle = "volcano_battle"
    


class RaceMode(enum.Enum):
    """
    Racing modes
    """

    SINGLE = "single"  # Play as the only car on the track
    MULTI = "multi"  # Play against 3 other NPCs
