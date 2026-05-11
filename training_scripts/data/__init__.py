from pathlib import Path
import os


__all__ = ["DATA_DIR", "LOGS_DIR", "MODELS_DIR", "BEST_MODELS_DIR", "GEN_T_STATES_DIR", "MAN_T_STATES_DIR", "ALL_T_STATES_SIR"]

DATA_DIR: str = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/data/logs"
MODELS_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/data/models"
BEST_MODELS_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/data/best_models"
GEN_T_STATES_DIR = [str(f.absolute()) for f in Path("./data/states/generated/").iterdir() if f.suffix == ".state"]
MAN_T_STATES_DIR = [str(f.absolute()) for f in Path("./data/states/specific/").iterdir() if f.suffix == ".state"]
ALL_T_STATES_SIR = [*GEN_T_STATES_DIR, *MAN_T_STATES_DIR]