from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None


class VectorIndex:
    """Vector index with FAISS (if available) or a NumPy fallback."""

    def __init__(self, dim: int, metric: str = "cosine") -> None:
        self.dim = dim
        self.metric = metric
        self.ids: List[str] = []
        self.meta: List[Dict] = []
        self.X: Optional[np.ndarray] = None
        self.faiss_index = None

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            norms = np.linalg.norm(values, axis=1, keepdims=True) + 1e-12
            return values / norms
        return values

    def build(self, items: List[Dict]) -> None:
        self.ids = [item["vector_id"] for item in items]
        self.meta = [item["meta"] for item in items]
        matrix = np.array([item["psi"] for item in items], dtype=np.float32)
        matrix = self._normalize(matrix)
        self.X = matrix

        if faiss is not None:
            if self.metric == "cosine":
                index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efSearch = 128
            else:
                index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_L2)
            index.add(matrix)
            self.faiss_index = index

    def search(
        self,
        psi: List[float],
        k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        query = np.array(psi, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(f"Query dim {query.shape[1]} != index dim {self.dim}")
        query = self._normalize(query)

        if self.faiss_index is not None:
            if self.metric == "cosine":
                similarities, indices = self.faiss_index.search(query, k)
                distances = (1.0 - similarities).ravel()
            else:
                distances, indices = self.faiss_index.search(query, k)
                distances = distances.ravel()
            indices = indices.ravel()
        else:
            matrix = self.X
            if matrix is None:
                return []
            if self.metric == "cosine":
                similarities = (matrix @ query.T).ravel()
                indices = np.argsort(-similarities)[:k]
                distances = 1.0 - similarities[indices]
            else:
                distances = np.linalg.norm(matrix - query, axis=1)
                indices = np.argsort(distances)[:k]
                distances = distances[indices]

        results: List[Dict] = []
        for rank, (idx, distance) in enumerate(zip(indices, distances), start=1):
            record = {
                "rank": rank,
                "vector_id": self.ids[idx],
                "distance": float(distance),
                "meta": self.meta[idx],
            }
            results.append(record)

        if filters:
            def _matches(meta: Dict) -> bool:
                for key, expected in filters.items():
                    if str(meta.get(key)) != str(expected):
                        return False
                return True

            results = [record for record in results if _matches(record["meta"])]

        return results

    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".jsonl", "w", encoding="utf-8") as handle:
            for vector_id, meta, vector in zip(self.ids, self.meta, self.X or []):
                handle.write(
                    json.dumps({"vector_id": vector_id, "meta": meta, "psi": vector.tolist()}) + "\n"
                )

    @classmethod
    def load_jsonl(cls, path: str, metric: str = "cosine") -> "VectorIndex":
        items: List[Dict] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                items.append(json.loads(line))
        dimension = len(items[0]["psi"])
        index = cls(dim=dimension, metric=metric)
        index.build(items)
        return index
