from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from don_research.genomics.entropy_map import generate_entropy_map
from don_research.genomics.geoadapter import extract_adata_vocab, resolve_accession_to_h5ad
from don_research.genomics.index import VectorIndex
from don_research.genomics.vector_builder import build_vectors_from_h5ad
from don_research.genomics.vector_encoder import encode_query_vector

router = APIRouter(prefix="/api/v1/genomics", tags=["genomics"])

DATA_DIR = os.environ.get("DMP_DATA_DIR", "./data")
VEC_DIR = os.path.join(DATA_DIR, "vectors")
MAP_DIR = os.path.join(DATA_DIR, "maps")
os.makedirs(VEC_DIR, exist_ok=True)
os.makedirs(MAP_DIR, exist_ok=True)


@router.post("/vectors/build")
async def build_vectors(
    file: UploadFile = File(..., description=".h5ad single-cell dataset"),
    mode: str = Form("cluster"),
):
    if not file.filename.endswith(".h5ad"):
        raise HTTPException(status_code=400, detail="Expected .h5ad file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad") as tmp:
        tmp.write(await file.read())
        h5ad_path = tmp.name

    try:
        items = build_vectors_from_h5ad(h5ad_path, mode=mode)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Vector build failed: {exc}") from exc

    base = os.path.splitext(os.path.basename(file.filename))[0]
    out_path = os.path.join(VEC_DIR, f"{base}.{mode}.jsonl")
    with open(out_path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")

    preview = items[: min(5, len(items))]
    return JSONResponse({"ok": True, "mode": mode, "jsonl": out_path, "count": len(items), "preview": preview})


@router.post("/vectors/search")
async def search_vectors(
    jsonl_path: str = Form(..., description="Path to vectors JSONL (from /vectors/build)"),
    k: int = Form(10),
    psi: str = Form(..., description="JSON array of floats (query vector)"),
    filters_json: Optional[str] = Form(None, description="Optional JSON filters"),
):
    if not os.path.exists(jsonl_path):
        raise HTTPException(status_code=400, detail=f"jsonl not found: {jsonl_path}")

    try:
        index = VectorIndex.load_jsonl(jsonl_path, metric="cosine")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Index load failed: {exc}") from exc

    try:
        query = json.loads(psi)
        if not isinstance(query, list):
            raise ValueError("psi must be JSON list of floats")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Invalid psi: {exc}") from exc

    filters = None
    if filters_json:
        try:
            filters = json.loads(filters_json)
        except Exception:  # pragma: no cover
            filters = None

    try:
        hits = index.search(query, k=k, filters=filters)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    return JSONResponse({"ok": True, "hits": hits})


@router.post("/entropy-map")
async def entropy_map(
    file: UploadFile = File(..., description=".h5ad single-cell dataset"),
    label_key: Optional[str] = Form(None),
    emb_key: Optional[str] = Form(None),
):
    if not file.filename.endswith(".h5ad"):
        raise HTTPException(status_code=400, detail="Expected .h5ad file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad") as tmp:
        tmp.write(await file.read())
        h5ad_path = tmp.name

    base = os.path.splitext(os.path.basename(file.filename))[0]
    out_png = os.path.join(MAP_DIR, f"{base}.entropy_map.png")
    try:
        png_path, stats = generate_entropy_map(h5ad_path, label_key=label_key, emb_key=emb_key, out_png=out_png)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Entropy map failed: {exc}") from exc

    return JSONResponse({"ok": True, "png": png_path, "stats": stats})


@router.post("/load")
async def resolve_h5ad(
    accession_or_path: str = Form(..., description="GEO accession (GSE*), direct .h5ad URL, or local .h5ad path")
):
    """Resolve a GEO accession or URL to a cached .h5ad path."""
    try:
        path = resolve_accession_to_h5ad(accession_or_path, cache_dir=os.path.join(DATA_DIR, "cache"))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Resolution failed: {exc}") from exc
    return JSONResponse({"ok": True, "h5ad_path": path})


@router.post("/query/encode")
async def encode_query(
    text: Optional[str] = Form(None),
    gene_list_json: Optional[str] = Form(None, description='JSON list of genes, e.g. ["IFNG","GZMB"]'),
    h5ad_path: Optional[str] = Form(None, description="Optional .h5ad to derive vocab (genes/cell types/tissues)"),
    cell_type: Optional[str] = Form(None),
    tissue: Optional[str] = Form(None),
):
    """Encode a text/gene query into the ψ_query vector space."""
    genes = None
    if gene_list_json:
        try:
            genes = json.loads(gene_list_json)
            if not isinstance(genes, list):
                raise ValueError("gene_list_json must be a JSON list")
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=400, detail=f"Invalid gene_list_json: {exc}") from exc

    vocab = None
    if h5ad_path:
        if not os.path.exists(h5ad_path):
            raise HTTPException(status_code=400, detail=f"h5ad_path not found: {h5ad_path}")
        try:
            vocab = extract_adata_vocab(h5ad_path)
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Failed to load vocab from h5ad: {exc}") from exc

    try:
        psi = encode_query_vector(text=text, gene_list=genes, cell_type=cell_type, tissue=tissue, vocab=vocab)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Encoding failed: {exc}") from exc

    return JSONResponse({"ok": True, "psi": psi})
