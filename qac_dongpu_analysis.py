#!/usr/bin/env python3
"""
QAC Genomics Analysis

Applies Quantum Adjacency Code error correction to gene expression patterns.
Uses cell-cell adjacency from expression similarity to stabilize gene vectors.
"""

import sys
import json
import numpy as np
import scanpy as sc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Add stack directory to path for QAC
current_dir = Path(__file__).parent
stack_dir = current_dir / "stack"
if str(stack_dir) not in sys.path:
    sys.path.insert(0, str(stack_dir))

try:
    from tace.core import QACEngine
    QAC_AVAILABLE = True
    print("✓ QAC Engine available")
except Exception as e:
    QAC_AVAILABLE = False
    print(f"⚠️  Warning: QAC Engine not available: {e}")

def load_data_with_alpha():
    """Load PBMC3K dataset with precomputed alpha values"""
    print("\nLoading PBMC3K dataset...")
    adata = sc.read_h5ad("data/pbmc3k.h5ad")
    
    # Standard preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    
    # PCA and clustering
    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_pcs=40)
    sc.tl.leiden(adata, resolution=0.5)
    
    print(f"✓ Loaded {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Load alpha values
    alpha_file = "gene_analysis_alpha_results.json"
    if Path(alpha_file).exists():
        with open(alpha_file, 'r') as f:
            alpha_data = json.load(f)
            alpha_values = np.array(alpha_data['alpha_values'])
            print(f"✓ Loaded alpha values for {len(alpha_values)} cells")
    else:
        print("⚠ No alpha values found")
        alpha_values = None
    
    return adata, alpha_values

def build_cell_adjacency(expr_matrix: np.ndarray, n_neighbors: int = 10) -> np.ndarray:
    """
    Build cell-cell adjacency matrix based on expression similarity.
    
    Uses k-nearest neighbors in expression space to build sparse adjacency.
    This is what QAC needs - similarity relationships between cells.
    """
    n_cells = expr_matrix.shape[0]
    print(f"Building cell adjacency matrix for {n_cells} cells...")
    
    # Normalize expression matrix (standard normalization, not DON-GPU compression)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    expr_normalized = scaler.fit_transform(expr_matrix)
    
    print(f"Computing k-nearest neighbors (k={n_neighbors})...")
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='correlation', n_jobs=-1)
    nbrs.fit(expr_normalized)
    distances, indices = nbrs.kneighbors(expr_normalized)
    
    print(f"✓ Found {n_neighbors} nearest neighbors for each cell")
    
    # Build sparse adjacency matrix
    adjacency = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        for j_idx, j in enumerate(indices[i]):
            # Similarity = 1 - normalized_distance
            # Use correlation distance (1 - correlation)
            similarity = 1.0 - distances[i, j_idx]
            adjacency[i, j] = similarity
            adjacency[j, i] = similarity  # Make symmetric
    
    # Threshold to maintain sparsity
    threshold = 0.5
    adjacency[adjacency < threshold] = 0.0
    
    print(f"✓ Built adjacency matrix: {adjacency.shape}, sparsity: {100 * (1 - np.count_nonzero(adjacency) / adjacency.size):.1f}%")
    return adjacency

def apply_qac_genomics(expr_matrix, gene_names, alpha_values):
    """Apply QAC stabilization to gene expression patterns"""
    n_cells, n_features = expr_matrix.shape
    
    print("\n" + "="*80)
    print("QAC GENOMICS ANALYSIS")
    print("="*80)
    
    # Build cell-cell adjacency matrix
    adjacency = build_cell_adjacency(expr_matrix, n_neighbors=15)
    
    # Initialize QAC engine
    qac = QACEngine(
        num_qubits=n_cells,
        reinforce_rate=0.05,
        layers=50
    )
    
    print(f"\nQAC Engine initialized: {n_cells} qubits, 50 layers")
    
    # Inject cell adjacency into QAC
    try:
        import jax.numpy as jnp
        qac.base_adj = jnp.array(adjacency, dtype=jnp.float32)
        print("✓ Using JAX for QAC computation")
    except Exception:
        qac.base_adj = adjacency
        print("✓ Using NumPy for QAC computation")
    
    # Stabilize each gene
    stabilized_matrix = np.zeros_like(expr_matrix)
    
    print("\nApplying QAC stabilization to gene expression patterns...")
    for i, gene in enumerate(gene_names):
        if (i + 1) % 20 == 0:
            print(f"  Stabilized {i + 1}/{len(gene_names)} genes...")
        
        gene_expr = expr_matrix[:, i]
        stabilized = qac.stabilize(gene_expr.tolist())
        stabilized_matrix[:, i] = np.array(stabilized)
    
    print(f"\n✓ QAC stabilization complete for {len(gene_names)} genes")
    
    # Analyze results
    return analyze_results(expr_matrix, stabilized_matrix, gene_names, alpha_values)

def analyze_results(original, stabilized, gene_names, alpha_values):
    """Analyze QAC stabilization results"""
    print("\n" + "="*80)
    print("QAC STABILIZATION RESULTS")
    print("="*80)
    
    results = []
    
    for i, gene in enumerate(gene_names):
        orig_expr = original[:, i]
        stab_expr = stabilized[:, i]
        
        change = np.abs(stab_expr - orig_expr)
        
        result = {
            'gene': gene,
            'original_mean': float(orig_expr.mean()),
            'stabilized_mean': float(stab_expr.mean()),
            'mean_change': float(change.mean()),
            'max_change': float(change.max()),
            'stability_score': float(1.0 - change.mean() / (np.abs(orig_expr).mean() + 1e-9)),
            'expressing_cells': int((orig_expr > 0).sum()),
            'expressing_pct': float((orig_expr > 0).sum() / len(orig_expr) * 100)
        }
        
        if alpha_values is not None:
            result['original_alpha_corr'] = float(np.corrcoef(orig_expr, alpha_values)[0, 1])
            result['stabilized_alpha_corr'] = float(np.corrcoef(stab_expr, alpha_values)[0, 1])
            
            low_mask = alpha_values < 0.3
            high_mask = alpha_values >= 0.7
            
            result['orig_low_alpha'] = float(orig_expr[low_mask].mean() if low_mask.sum() > 0 else 0)
            result['orig_high_alpha'] = float(orig_expr[high_mask].mean() if high_mask.sum() > 0 else 0)
            result['stab_low_alpha'] = float(stab_expr[low_mask].mean() if low_mask.sum() > 0 else 0)
            result['stab_high_alpha'] = float(stab_expr[high_mask].mean() if high_mask.sum() > 0 else 0)
        
        results.append(result)
    
    results.sort(key=lambda x: x['stability_score'], reverse=True)
    return results

def print_summary(results, alpha_values):
    """Print summary"""
    print("\n" + "="*80)
    print("TOP 20 MOST STABLE GENES (Quantum-Robust Programs)")
    print("="*80)
    print(f"\n{'Rank':<5} {'Gene':<15} {'Stability':<10} {'Alpha Corr':<12} {'Pattern'}")
    print("-"*70)
    
    for i, r in enumerate(results[:20], 1):
        pattern = ""
        if alpha_values is not None:
            if r['original_alpha_corr'] > 0.3:
                pattern = "High-α enriched"
            elif r['original_alpha_corr'] < -0.3:
                pattern = "Low-α enriched"
            else:
                pattern = "α-independent"
        
        print(f"{i:<5} {r['gene']:<15} {r['stability_score']:.4f}   {r.get('original_alpha_corr', 0):>6.4f}       {pattern}")
    
    print("\n" + "="*80)
    print("TOP 20 MOST DYNAMIC GENES (Quantum-Labile States)")
    print("="*80)
    print(f"\n{'Rank':<5} {'Gene':<15} {'Stability':<10} {'Alpha Corr':<12} {'Pattern'}")
    print("-"*70)
    
    for i, r in enumerate(results[-20:], 1):
        pattern = ""
        if alpha_values is not None:
            if r['original_alpha_corr'] > 0.3:
                pattern = "High-α enriched"
            elif r['original_alpha_corr'] < -0.3:
                pattern = "Low-α enriched"
            else:
                pattern = "α-independent"
        
        print(f"{i:<5} {r['gene']:<15} {r['stability_score']:.4f}   {r.get('original_alpha_corr', 0):>6.4f}       {pattern}")

def save_results(results):
    """Save results"""
    output_file = "qac_dongpu_results.json"
    print(f"\nSaving results to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_genes': len(results),
            'method': 'QAC + DON-GPU',
            'results': results
        }, f, indent=2)
    
    print(f"✓ Saved analysis for {len(results)} genes")

def main():
    """Main entry point"""
    print("="*80)
    print("QAC GENE EXPRESSION ANALYSIS")
    print("Quantum Error Correction via Cell Adjacency")
    print("="*80)
    
    # Load data
    adata, alpha_values = load_data_with_alpha()
    
    # Get highly variable genes
    highly_var_genes = adata.var_names[adata.var['highly_variable']][:100]
    expr_matrix = adata[:, highly_var_genes].X.toarray() if hasattr(adata.X, 'toarray') else adata[:, highly_var_genes].X
    
    print(f"Analyzing {len(highly_var_genes)} highly variable genes")
    
    if QAC_AVAILABLE:
        results = apply_qac_genomics(expr_matrix, highly_var_genes, alpha_values)
        print_summary(results, alpha_values)
        save_results(results)
    else:
        print("\n⚠ QAC Engine not available - cannot proceed")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
