"""
AnnData Adapter - Export H5AD to DON Bio Artifacts
Converts single-cell data to collapse_map.json + collapse_vectors.jsonl
"""

import json
import numpy as np
import scanpy as sc
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def export_artifacts(
    h5ad_path: str,
    cluster_key: str = "leiden",
    latent_key: str = "X_pca",
    paga_key: Optional[str] = "paga",
    sample_cells: Optional[int] = None,
    output_dir: str = "artifacts",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Export AnnData to DON bio artifacts format
    
    Args:
        h5ad_path: Path to .h5ad file
        cluster_key: Column in adata.obs for clusters
        latent_key: Key in adata.obsm for latent space (X_pca, X_scVI, etc.)
        paga_key: Key in adata.uns for PAGA connectivity (None to skip)
        sample_cells: Subsample cells (0 = all)
        output_dir: Output directory for artifacts
        seed: Random seed for subsampling
        
    Returns:
        Dict with node/edge counts and artifact paths
    """
    logger.info(f"Loading AnnData from {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    
    # Subsample if requested (handle None case from Form)
    if sample_cells and sample_cells > 0 and adata.n_obs > sample_cells:
        sc.pp.subsample(adata, n_obs=sample_cells, random_state=seed)
        logger.info(f"Subsampled to {sample_cells} cells")
    
    # Ensure cluster key exists
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")
    
    # Ensure latent space exists
    if latent_key not in adata.obsm.keys():
        raise ValueError(f"Latent key '{latent_key}' not found in adata.obsm")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract clusters
    clusters = adata.obs[cluster_key].astype(str).values
    cluster_ids = sorted(set(clusters))
    n_clusters = len(cluster_ids)
    
    logger.info(f"Found {n_clusters} clusters, {adata.n_obs} cells")
    
    # Build collapse map (nodes + edges)
    collapse_map = {
        "nodes": [],
        "edges": [],
        "metadata": {
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_clusters": n_clusters,
            "cluster_key": cluster_key,
            "latent_key": latent_key
        }
    }
    
    # Add nodes (one per cluster)
    for cluster_id in cluster_ids:
        mask = clusters == cluster_id
        n_cells = int(mask.sum())
        
        # Compute cluster centroid in latent space
        latent = adata.obsm[latent_key][mask]
        centroid = latent.mean(axis=0).tolist()
        
        # Get marker genes if available
        markers = []
        if 'rank_genes_groups' in adata.uns:
            try:
                cluster_markers = adata.uns['rank_genes_groups']['names'][cluster_id][:5]
                markers = [str(g) for g in cluster_markers]
            except:
                pass
        
        node = {
            "id": f"cluster_{cluster_id}",
            "cluster": cluster_id,
            "n_cells": n_cells,
            "centroid": centroid,
            "markers": markers
        }
        collapse_map["nodes"].append(node)
    
    # Add edges from PAGA if available
    if paga_key and paga_key in adata.uns:
        try:
            connectivities = adata.uns[paga_key]['connectivities'].toarray()
            threshold = 0.1  # Minimum connectivity to include edge
            
            for i, source_id in enumerate(cluster_ids):
                for j, target_id in enumerate(cluster_ids):
                    if i < j:  # Undirected graph, only add once
                        weight = float(connectivities[i, j])
                        if weight > threshold:
                            edge = {
                                "source": f"cluster_{source_id}",
                                "target": f"cluster_{target_id}",
                                "weight": weight
                            }
                            collapse_map["edges"].append(edge)
            
            logger.info(f"Added {len(collapse_map['edges'])} PAGA edges")
        except Exception as e:
            logger.warning(f"Could not extract PAGA edges: {e}")
    
    # Save collapse map
    map_path = output_path / "collapse_map.json"
    with open(map_path, 'w') as f:
        json.dump(collapse_map, f, indent=2)
    logger.info(f"Saved collapse map to {map_path}")
    
    # Build per-cell vectors
    vectors = []
    latent = adata.obsm[latent_key]
    
    for i in range(adata.n_obs):
        # Get QC metrics
        n_genes = int(adata.obs['n_genes'][i]) if 'n_genes' in adata.obs else 0
        n_counts = int(adata.obs['n_counts'][i]) if 'n_counts' in adata.obs else 0
        pct_mt = float(adata.obs['pct_counts_mt'][i]) if 'pct_counts_mt' in adata.obs else 0.0
        
        # Get doublet score if available
        doublet_score = 0.0
        if 'doublet_score' in adata.obs:
            doublet_score = float(adata.obs['doublet_score'][i])
        elif 'scrublet_score' in adata.obs:
            doublet_score = float(adata.obs['scrublet_score'][i])
        
        # Get batch if available
        batch = "unknown"
        if 'batch' in adata.obs:
            batch = str(adata.obs['batch'][i])
        elif 'sample' in adata.obs:
            batch = str(adata.obs['sample'][i])
        
        vector = {
            "cell_id": adata.obs_names[i],
            "cluster": str(clusters[i]),
            "latent": latent[i].tolist(),
            "qc": {
                "n_genes": n_genes,
                "n_counts": n_counts,
                "pct_mt": pct_mt,
                "doublet_score": doublet_score
            },
            "batch": batch
        }
        vectors.append(vector)
    
    # Save vectors as JSONL
    vectors_path = output_path / "collapse_vectors.jsonl"
    with open(vectors_path, 'w') as f:
        for vec in vectors:
            f.write(json.dumps(vec) + '\n')
    logger.info(f"Saved {len(vectors)} cell vectors to {vectors_path}")
    
    return {
        "nodes": n_clusters,
        "edges": len(collapse_map["edges"]),
        "vectors": len(vectors),
        "artifacts": [str(map_path), str(vectors_path)],
        "metadata": collapse_map["metadata"]
    }


def stream_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Stream JSONL file and return list of records"""
    records = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
