from __future__ import annotations

from typing import Dict

import numpy as np

from don_research.genomics.vector_encoder import encode_query_vector


def test_encode_query_vector_shape() -> None:
    psi = encode_query_vector(text="Interferon gamma high T cells", gene_list=["IFNG", "GZMB"])
    assert isinstance(psi, list)
    assert len(psi) == 128
    assert np.isclose(np.linalg.norm(np.asarray(psi)), 1.0, atol=1e-6)
    assert psi[0:28] == [0.0] * 28


def test_encode_query_vector_respects_vocab(monkeypatch) -> None:
    vocab = {
        "genes": {"IFNG", "GZMB"},
        "cell_types": {"T-cell"},
        "tissues": {"PBMC"},
    }
    captured: Dict[str, list[str]] = {}

    def fake_hashed_bigrams(tokens, out_dim):
        captured["tokens"] = list(tokens)
        return np.ones(out_dim, dtype=np.float32)

    from don_research.genomics import vector_encoder

    monkeypatch.setattr(vector_encoder, "_hashed_bigrams", fake_hashed_bigrams)
    psi = encode_query_vector(
        text="IFNG driven PBMC T-cell response",
        gene_list=["IFNG", "GZMB", "FOO1"],
        cell_type="T-cell",
        tissue="PBMC",
        vocab=vocab,
    )
    assert len(psi) == 128
    tokens = captured.get("tokens", [])
    assert "GENE:FOO1" not in tokens
    assert "GENE:IFNG" in tokens
    assert "CELLTYPE:T-cell" in tokens