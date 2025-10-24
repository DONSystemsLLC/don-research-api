from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from don_research.genomics import geoadapter


def test_resolve_local_h5ad_path(h5ad_file: str) -> None:
    resolved = geoadapter.resolve_accession_to_h5ad(h5ad_file)
    assert resolved == os.path.abspath(h5ad_file)


def test_resolve_url_downloads(tmp_path, monkeypatch) -> None:
    data = b"test"

    class DummyResponse:
        def __init__(self) -> None:
            self.raw = io.BytesIO(data)

        def __enter__(self) -> "DummyResponse":
            self.raw.seek(0)
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - no cleanup needed
            return None

        def raise_for_status(self) -> None:
            return None

    def fake_get(url, stream=False, timeout=0):  # pragma: no cover - used in test only
        assert stream is True
        assert "http://example.com/mock.h5ad" == url
        return DummyResponse()

    monkeypatch.setattr(geoadapter, "requests", type("Req", (), {"get": staticmethod(fake_get)}))
    target = geoadapter.resolve_accession_to_h5ad("http://example.com/mock.h5ad", cache_dir=str(tmp_path))
    assert Path(target).exists()
    assert Path(target).read_bytes() == data


@pytest.mark.parametrize("cell_key,tissue_key", [("cell_type", "tissue"), ("missing", "missing")])
def test_extract_vocab(h5ad_file: str, cell_key: str, tissue_key: str) -> None:
    vocab = geoadapter.extract_adata_vocab(h5ad_file, cell_type_key=cell_key, tissue_key=tissue_key)
    assert "genes" in vocab
    assert all(token.isupper() for token in vocab["genes"])