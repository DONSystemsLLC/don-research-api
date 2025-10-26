import json
from pathlib import Path

import pytest

from src.auth.authorized_institutions import load_authorized_institutions


def test_load_authorized_institutions_default(monkeypatch):
    monkeypatch.delenv("DON_AUTHORIZED_INSTITUTIONS_JSON", raising=False)
    monkeypatch.delenv("DON_AUTHORIZED_INSTITUTIONS_FILE", raising=False)

    institutions = load_authorized_institutions()
    assert "demo_token" in institutions
    assert institutions["demo_token"]["rate_limit"] == 100


def test_load_authorized_institutions_with_json_override(monkeypatch):
    override = {
        "token_a": {
            "name": "Override A",
            "contact": "a@example.org",
            "rate_limit": 500,
        }
    }
    monkeypatch.setenv("DON_AUTHORIZED_INSTITUTIONS_JSON", json.dumps(override))
    monkeypatch.delenv("DON_AUTHORIZED_INSTITUTIONS_FILE", raising=False)

    institutions = load_authorized_institutions()
    assert "token_a" in institutions
    assert institutions["token_a"]["name"] == "Override A"


def test_load_authorized_institutions_with_file_override(tmp_path: Path, monkeypatch):
    payload = {
        "token_b": {
            "name": "Override B",
            "contact": "b@example.org",
            "rate_limit": 750,
        }
    }
    file_path = tmp_path / "institutions.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setenv("DON_AUTHORIZED_INSTITUTIONS_FILE", str(file_path))
    monkeypatch.delenv("DON_AUTHORIZED_INSTITUTIONS_JSON", raising=False)

    institutions = load_authorized_institutions()
    assert "token_b" in institutions
    assert institutions["token_b"]["rate_limit"] == 750
