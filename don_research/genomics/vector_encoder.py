from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Optional

import numpy as np

GENE_RE = re.compile(r"^[A-Za-z0-9\-.]{2,}$")


def _hash_bucket(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % dim


def _hashed_bigrams(tokens: Iterable[str], out_dim: int) -> np.ndarray:
    vector = np.zeros(out_dim, dtype=np.float32)
    ordered = [token for token in tokens if token]
    for index in range(len(ordered) - 1):
        bucket = _hash_bucket(f"{ordered[index]}||{ordered[index + 1]}", out_dim)
        vector[bucket] += 1.0
    total = vector.sum()
    return vector / total if total > 0 else vector


def _normalize_gene(symbol: str) -> str:
    return symbol.strip().upper().replace(" ", "").replace("_", "-")


def encode_query_vector(
    text: Optional[str] = None,
    gene_list: Optional[List[str]] = None,
    cell_type: Optional[str] = None,
    tissue: Optional[str] = None,
    vocab: Optional[Dict[str, set]] = None,
    out_dim: int = 128,
) -> List[float]:
    """Encode query metadata into the 128-D ψ_query vector space."""
    tokens: List[str] = []

    genes_in_query = set()
    if gene_list:
        for gene in gene_list:
            normalized = _normalize_gene(gene)
            if vocab and "genes" in vocab and normalized not in vocab["genes"]:
                continue
            if GENE_RE.match(normalized):
                genes_in_query.add(normalized)
    if text:
        for raw in re.split(r"[^A-Za-z0-9\-._]+", text):
            normalized = _normalize_gene(raw)
            if len(normalized) >= 2 and GENE_RE.match(normalized):
                if not vocab or ("genes" in vocab and normalized in vocab["genes"]):
                    genes_in_query.add(normalized)
    for symbol in sorted(genes_in_query):
        tokens.append(f"GENE:{symbol}")

    ct = cell_type
    tt = tissue
    if text and vocab:
        lowered = text.lower()
        for candidate in vocab.get("cell_types", set()):
            if str(candidate).lower() in lowered:
                ct = candidate
                break
        for candidate in vocab.get("tissues", set()):
            if str(candidate).lower() in lowered:
                tt = candidate
                break
    if ct:
        tokens.append(f"CELLTYPE:{ct}")
    if tt:
        tokens.append(f"TISSUE:{tt}")

    if text:
        for word in re.findall(r"[A-Za-z]{3,}", text.lower()):
            if word in {"the", "and", "for", "with", "from", "this", "that", "cell", "cells", "gene", "genes"}:
                continue
            tokens.append(f"KEY:{word}")

    vector = np.zeros(out_dim, dtype=np.float32)
    vector[28:] = _hashed_bigrams(tokens, out_dim - 28)
    norm = np.linalg.norm(vector) + 1e-12
    return (vector / norm).astype(np.float32).tolist()
