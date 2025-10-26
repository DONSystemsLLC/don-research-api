from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

_DEFAULT_INSTITUTIONS: Dict[str, Dict[str, Any]] = {
    "demo_token": {
        "name": "Demo Access",
        "contact": "demo@donsystems.com",
        "rate_limit": 100,
    }
}


def _parse_mapping(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    parsed: Dict[str, Dict[str, Any]] = {}
    for token, payload in raw.items():
        if not isinstance(token, str):
            raise ValueError("Institution token keys must be strings")
        if not isinstance(payload, dict):
            raise ValueError("Institution metadata must be objects")
        name = payload.get("name")
        contact = payload.get("contact")
        rate_limit = payload.get("rate_limit")
        if not name or not isinstance(name, str):
            raise ValueError(f"Institution '{token}' missing valid 'name'")
        if not contact or not isinstance(contact, str):
            raise ValueError(f"Institution '{token}' missing valid 'contact'")
        if not isinstance(rate_limit, int) or rate_limit <= 0:
            raise ValueError(f"Institution '{token}' missing positive 'rate_limit'")
        parsed[token] = {
            "name": name,
            "contact": contact,
            "rate_limit": rate_limit,
        }
    return parsed


def _load_from_json_string(value: str) -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Invalid JSON in DON_AUTHORIZED_INSTITUTIONS_JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("DON_AUTHORIZED_INSTITUTIONS_JSON must encode an object")
    return _parse_mapping(data)


def _load_from_file(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():  # pragma: no cover - defensive guard
        raise FileNotFoundError(f"Institution file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Invalid JSON in {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return _parse_mapping(data)


def load_authorized_institutions() -> Dict[str, Dict[str, Any]]:
    """Return the institution map, applying environment overrides."""

    institutions = dict(_DEFAULT_INSTITUTIONS)

    env_json = os.getenv("DON_AUTHORIZED_INSTITUTIONS_JSON")
    if env_json:
        institutions.update(_load_from_json_string(env_json))

    env_file = os.getenv("DON_AUTHORIZED_INSTITUTIONS_FILE")
    if env_file:
        institutions.update(_load_from_file(Path(env_file)))

    return institutions
