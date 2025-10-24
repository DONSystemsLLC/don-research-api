from __future__ import annotations

import os
import re
import shutil
from typing import Dict

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    import GEOparse
except Exception:  # pragma: no cover
    GEOparse = None

try:
    import anndata as ad
except Exception:  # pragma: no cover
    ad = None

_GSE_RE = re.compile(r"^GSE\d+$", re.IGNORECASE)


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def resolve_accession_to_h5ad(accession_or_path: str, cache_dir: str = "./data/cache") -> str:
    """Resolve a GEO accession, direct URL, or path to a local .h5ad file."""
    os.makedirs(cache_dir, exist_ok=True)
    query = accession_or_path.strip()

    if os.path.exists(query) and query.lower().endswith(".h5ad"):
        return os.path.abspath(query)

    if _is_url(query):
        if requests is None:
            raise RuntimeError("requests not installed; cannot download URL.")
        filename = os.path.join(cache_dir, os.path.basename(query.split("?")[0]))
        with requests.get(query, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(filename, "wb") as handle:
                shutil.copyfileobj(response.raw, handle)
        if not filename.lower().endswith(".h5ad"):
            raise RuntimeError(f"Downloaded file is not .h5ad: {filename}")
        return os.path.abspath(filename)

    if _GSE_RE.match(query):
        accession = query.upper()
        for entry in os.listdir(cache_dir):
            if entry.startswith(accession) and entry.lower().endswith(".h5ad"):
                return os.path.abspath(os.path.join(cache_dir, entry))

        if GEOparse is None:
            raise RuntimeError("GEOparse not installed; provide a direct .h5ad URL/path.")

        gse = GEOparse.get_GEO(geo=accession, destdir=cache_dir, how="full")
        candidates = []
        for gsm in gse.gsms.values():
            supp = gsm.metadata.get("supplementary_file", []) or gsm.metadata.get("supplementary_file_1", [])
            urls = supp if isinstance(supp, list) else [supp]
            for url in urls:
                if str(url).lower().endswith(".h5ad"):
                    candidates.append(url)
        if not candidates:
            raise RuntimeError(
                f"No .h5ad supplementary files for {accession}. Provide a direct .h5ad URL/path."
            )
        if requests is None:
            raise RuntimeError("requests not installed; cannot download GEO supplementary file.")
        target_url = str(candidates[0])
        destination = os.path.join(cache_dir, f"{accession}__{os.path.basename(target_url)}")
        with requests.get(target_url, stream=True, timeout=90) as response:
            response.raise_for_status()
            with open(destination, "wb") as handle:
                shutil.copyfileobj(response.raw, handle)
        return os.path.abspath(destination)

    raise RuntimeError(
        "Provide a local .h5ad path, a direct .h5ad URL, or a GEO accession such as 'GSE12345'."
    )


def extract_adata_vocab(
    h5ad_path: str,
    cell_type_key: str = "cell_type",
    tissue_key: str = "tissue",
) -> Dict[str, set]:
    """Extract gene, cell-type, and tissue vocabularies from an AnnData file."""
    if ad is None:
        raise RuntimeError("anndata not installed; cannot extract vocab.")
    adata = ad.read_h5ad(h5ad_path)
    genes = {str(gene).upper() for gene in adata.var_names.tolist()}
    cell_types: set = set()
    tissues: set = set()
    if cell_type_key in adata.obs.columns:
        cell_types = {str(value) for value in adata.obs[cell_type_key].astype(str).values}
    if tissue_key in adata.obs.columns:
        tissues = {str(value) for value in adata.obs[tissue_key].astype(str).values}
    return {"genes": genes, "cell_types": cell_types, "tissues": tissues}
