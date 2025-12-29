from dataclasses import dataclass
from typing import Any, Dict
import yaml, os

@dataclass
class Cfg:
    raw: Dict[str, Any]

def load_config(path: str) -> Cfg:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return Cfg(raw=cfg)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
