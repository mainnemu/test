from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class Config:
    id: str
    recip_id: str

# 12 Configurations (Fixed Order E1..D2r)
CONFIGS: List[Config] = [
    Config("E1", "E1r"), Config("E1r", "E1"),
    Config("E2", "E2r"), Config("E2r", "E2"),
    Config("E3", "E3r"), Config("E3r", "E3"),
    Config("E4", "E4r"), Config("E4r", "E4"),
    Config("D1", "D1r"), Config("D1r", "D1"),
    Config("D2", "D2r"), Config("D2r", "D2"),
]

ID_TO_INDEX = {c.id: i for i, c in enumerate(CONFIGS)}

# Gate 1 Pairs (Fixed)
PAIR_IDS = [
    ("E1", "E1r"), ("E2", "E2r"), ("E3", "E3r"),
    ("E4", "E4r"), ("D1", "D1r"), ("D2", "D2r")
]
PAIR_INDICES = [(ID_TO_INDEX[a], ID_TO_INDEX[b]) for a, b in PAIR_IDS]

# Gate 0 Targets (E1..E4)
GATE0_IDS = ["E1", "E2", "E3", "E4"]
GATE0_INDICES = [ID_TO_INDEX[cid] for cid in GATE0_IDS]
