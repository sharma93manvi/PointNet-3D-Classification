from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

