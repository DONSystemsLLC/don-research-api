from __future__ import annotations

import copy
from typing import Any, Dict

_system_health: Dict[str, Any] = {
    "don_stack": {
        "mode": "unknown",
        "don_gpu": False,
        "tace": False,
        "qac": False,
        "adapter_loaded": False,
    }
}


def set_system_health(snapshot: Dict[str, Any]) -> None:
    global _system_health
    _system_health = copy.deepcopy(snapshot)


def get_system_health() -> Dict[str, Any]:
    return copy.deepcopy(_system_health)
