#!/usr/bin/env python3
"""
QAC Genomics Analysis - Cluster-Level Approach

CORRECT usage of DON Stack for genomics:
1. DON-GPU compresses cells into cluster vectors (2700 cells → 8 clusters)
2. QAC stabilizes cluster-level gene expression patterns
3. Analyze gene stability across clusters (not individual cells)

This follows the working examples from qac_exploration.py
"""

import sys
import json
import numpy as np
import scanpy as sc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add stack directory to path
current_dir = Path(__file__).parent
stack_dir = current_dir / "stack"
src_dir = current_dir / "src"
if str(stack_dir) not in sys.path:
    sys.path.insert(0, str(stack_dir))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import DON Stack components
try:
    from don_memory.adapters.don_stack_adapter import DONStackAdapter
    ADAPTER_AVAILABLE = True
except Exception as e:
    ADAPTER_AVAILABLE = False
    print(f"⚠️  DON Stack adapter not available: {e}")

try:
    from tace.core import QACEngine
    QAC_AVAILABLE = True
    print("✓ QAC Engine available")
except Exception as e:
    QAC_AVAILABLE = False
    print(f"⚠️  QAC Engine not available: {e}")

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
        print("⚠️  No alpha values found")
        alpha_values = None
    
    return adata, alpha_values

def build_adjacency_tensor(expr_matrix: np.ndarray, k_neighbors: int = 15) -> jnp.ndarray:
    """
    Build adjacency tensor using DON-GPU style matrix operations.
    
    This is the CORRECT usage of DON-GPU principles:
    - Compute all pairwise relationships in parallel (matrix ops)
    - Use adjacency tensor for collapse dynamics
    - Vectorized, not sequential loops
    """
    n_cells = expr_matrix.shape[0]
    print(f"\nBuilding adjacency tensor for {n_cells} cells...")
    
    # Convert to JAX array for GPU acceleration
    expr_jax = jnp.array(expr_matrix)
    
    # Compute pairwise correlation matrix (vectorized)
    # This is O(n²) but done in parallel on GPU
    print("Computing correlation matrix (vectorized)...")
    
    # Standardize each cell's expression
    expr_mean = jnp.mean(expr_jax, axis=1, keepdims=True)
    expr_std = jnp.std(expr_jax, axis=1, keepdims=True) + 1e-9
    expr_norm = (expr_jax - expr_mean) / expr_std
    
    # Correlation matrix via matrix multiplication
    # corr[i,j] = dot(expr_norm[i], expr_norm[j]) / n_features
    corr_matrix = (expr_norm @ expr_norm.T) / expr_matrix.shape[1]
    
    print(f"✓ Correlation matrix computed: {corr_matrix.shape}")
    
    # Build k-nearest neighbor adjacency
    # For each cell, keep only top-k correlations
    print(f"Building k-NN adjacency (k={k_neighbors})...")
    
    # Get top-k indices per row (vectorized)
    top_k_indices = jnp.argsort(corr_matrix, axis=1)[:, -(k_neighbors+1):]  # +1 to exclude self
    
    # Build sparse adjacency matrix
    adjacency = jnp.zeros((n_cells, n_cells))
    
    # Fill in top-k connections with correlation values
    for i in range(n_cells):
        neighbors = top_k_indices[i]
        if JAX_AVAILABLE:
            adjacency = adjacency.at[i, neighbors].set(corr_matrix[i, neighbors])
        else:
            adjacency[i, neighbors] = corr_matrix[i, neighbors]
    
    # Make symmetric (take max of A[i,j] and A[j,i])
    adjacency = jnp.maximum(adjacency, adjacency.T)
    
    # Threshold to maintain sparsity
    threshold = 0.3
    adjacency = jnp.where(adjacency >= threshold, adjacency, 0.0)
    
    sparsity = 100 * (1 - jnp.count_nonzero(adjacency) / adjacency.size)
    print(f"✓ Adjacency tensor built: {adjacency.shape}, sparsity: {sparsity:.1f}%")
    
    return adjacency

def vectorized_qac_stabilize(expr_matrix: jnp.ndarray, adjacency: jnp.ndarray, 
                             n_layers: int = 50, reinforce_rate: float = 0.05) -> jnp.ndarray:
    """
    Vectorized QAC stabilization - process ALL genes in parallel.
    
    This is the CORRECT usage:
    - Apply adjacency-based error correction to ALL genes simultaneously
    - Use matrix operations, not loops over genes
    - JAX-compiled for GPU acceleration
    
    Args:
        expr_matrix: (n_cells, n_genes) expression matrix
        adjacency: (n_cells, n_cells) adjacency tensor
        n_layers: Number of QAC error correction layers
        reinforce_rate: Reinforcement rate per layer
        
    Returns:
        Stabilized expression matrix (n_cells, n_genes)
    """
    n_cells, n_genes = expr_matrix.shape
    print(f"\nApplying vectorized QAC stabilization...")
    print(f"  Processing {n_genes} genes in parallel (not sequential loop)")
    print(f"  {n_layers} error correction layers")
    
    # Convert to JAX arrays
    expr_jax = jnp.array(expr_matrix, dtype=jnp.float32)
    adj_jax = jnp.array(adjacency, dtype=jnp.float32)
    
    # Normalize adjacency matrix (row-wise sum to 1)
    row_sums = jnp.sum(adj_jax, axis=1, keepdims=True) + 1e-9
    adj_norm = adj_jax / row_sums
    
    # Apply QAC layers (vectorized across all genes)
    # EXACT formula from stack/tace/core.py stabilize():
    # dissipation = adj @ v / sum(adj, axis=1)
    # v = v - dissipation
    # v = v + reinforce_rate * mean(v)
    stabilized = expr_jax
    
    for layer in range(n_layers):
        if (layer + 1) % 10 == 0:
            print(f"  Layer {layer + 1}/{n_layers}...")
        
        # Dissipation: weighted average of adjacent cells
        # Matrix multiply gives sum_j(adj[i,j] * stabilized[j, g]) for all genes at once
        dissipation = adj_norm @ stabilized
        
        # Error correction: move away from dissipation (EXACT formula from QAC engine)
        stabilized = stabilized - dissipation
        
        # Reinforcement: add back global mean (EXACT formula from QAC engine)
        mean_expr = jnp.mean(stabilized)  # Global mean across all cells and genes
        stabilized = stabilized + reinforce_rate * mean_expr
    
    print(f"✓ Vectorized QAC stabilization complete")
    
    return stabilized

def analyze_stabilization_results(original: np.ndarray, stabilized: np.ndarray, 
                                  gene_names: List[str], alpha_values: np.ndarray) -> List[Dict]:
    """Analyze QAC stabilization results"""
    print("\n" + "="*80)
    print("QAC STABILIZATION ANALYSIS")
    print("="*80)
    
    # Convert back to NumPy for analysis
    original = np.array(original)
    stabilized = np.array(stabilized)
    
    results = []
    
    for i, gene in enumerate(gene_names):
        orig_expr = original[:, i]
        stab_expr = stabilized[:, i]
        
        # Compute change metrics
        change = np.abs(stab_expr - orig_expr)
        relative_change = change.mean() / (np.abs(orig_expr).mean() + 1e-9)
        
        result = {
            'gene': gene,
            'original_mean': float(orig_expr.mean()),
            'stabilized_mean': float(stab_expr.mean()),
            'mean_change': float(change.mean()),
            'max_change': float(change.max()),
            'relative_change': float(relative_change),
            'stability_score': float(1.0 - min(relative_change, 1.0)),  # Clamp to [0, 1]
            'expressing_cells': int((orig_expr > 0).sum()),
            'expressing_pct': float((orig_expr > 0).sum() / len(orig_expr) * 100)
        }
        
        if alpha_values is not None:
            # Alpha correlations
            result['original_alpha_corr'] = float(np.corrcoef(orig_expr, alpha_values)[0, 1])
            result['stabilized_alpha_corr'] = float(np.corrcoef(stab_expr, alpha_values)[0, 1])
            
            # Regime-specific expression
            low_mask = alpha_values < 0.3
            high_mask = alpha_values >= 0.7
            
            result['orig_low_alpha'] = float(orig_expr[low_mask].mean() if low_mask.sum() > 0 else 0)
            result['orig_high_alpha'] = float(orig_expr[high_mask].mean() if high_mask.sum() > 0 else 0)
            result['stab_low_alpha'] = float(stab_expr[low_mask].mean() if low_mask.sum() > 0 else 0)
            result['stab_high_alpha'] = float(stab_expr[high_mask].mean() if high_mask.sum() > 0 else 0)
        
        results.append(result)
    
    # Sort by stability score
    results.sort(key=lambda x: x['stability_score'], reverse=True)
    
    return results

def print_summary(results: List[Dict], alpha_values: np.ndarray):
    """Print analysis summary"""
    print("\n" + "="*80)
    print("TOP 20 QUANTUM-STABLE GENES (Robust Programs)")
    print("="*80)
    print(f"\n{'Rank':<5} {'Gene':<15} {'Stability':<10} {'Alpha Corr':<12} {'Pattern'}")
    print("-"*70)
    
    for i, r in enumerate(results[:20], 1):
        pattern = ""
        if alpha_values is not None and 'original_alpha_corr' in r:
            if r['original_alpha_corr'] > 0.3:
                pattern = "High-α enriched"
            elif r['original_alpha_corr'] < -0.3:
                pattern = "Low-α enriched"
            else:
                pattern = "α-independent"
        
        alpha_corr = r.get('original_alpha_corr', 0)
        print(f"{i:<5} {r['gene']:<15} {r['stability_score']:.4f}   {alpha_corr:>6.4f}       {pattern}")
    
    print("\n" + "="*80)
    print("TOP 20 QUANTUM-LABILE GENES (Dynamic States)")
    print("="*80)
    print(f"\n{'Rank':<5} {'Gene':<15} {'Stability':<10} {'Alpha Corr':<12} {'Pattern'}")
    print("-"*70)
    
    for i, r in enumerate(results[-20:], 1):
        pattern = ""
        if alpha_values is not None and 'original_alpha_corr' in r:
            if r['original_alpha_corr'] > 0.3:
                pattern = "High-α enriched"
            elif r['original_alpha_corr'] < -0.3:
                pattern = "Low-α enriched"
            else:
                pattern = "α-independent"
        
        alpha_corr = r.get('original_alpha_corr', 0)
        print(f"{i:<5} {r['gene']:<15} {r['stability_score']:.4f}   {alpha_corr:>6.4f}       {pattern}")

def save_results(results: List[Dict]):
    """Save results to JSON"""
    output_file = "qac_vectorized_results.json"
    print(f"\nSaving results to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_genes': len(results),
            'method': 'Vectorized QAC + DON-GPU Adjacency Tensor',
            'jax_available': JAX_AVAILABLE,
            'results': results
        }, f, indent=2)
    
    print(f"✓ Saved analysis for {len(results)} genes")

def main():
    """Main entry point"""
    print("="*80)
    print("VECTORIZED QAC GENOMICS ANALYSIS")
    print("DON-GPU Adjacency Tensor + JAX-Accelerated Parallel Stabilization")
    print("="*80)
    
    if not QAC_AVAILABLE:
        print("\n⚠️  QAC Engine not available - cannot proceed")
        return
    
    # Load data
    adata, alpha_values = load_data_with_alpha()
    
    # Get highly variable genes for analysis
    n_genes = 100
    highly_var_genes = adata.var_names[adata.var['highly_variable']][:n_genes]
    expr_matrix = adata[:, highly_var_genes].X
    
    # Convert to dense if sparse
    if hasattr(expr_matrix, 'toarray'):
        expr_matrix = expr_matrix.toarray()
    
    print(f"\nAnalyzing {len(highly_var_genes)} highly variable genes")
    print(f"Expression matrix: {expr_matrix.shape} (cells × genes)")
    
    # Build adjacency tensor (DON-GPU style matrix ops)
    adjacency = build_adjacency_tensor(expr_matrix, k_neighbors=15)
    
    # Apply vectorized QAC stabilization (all genes in parallel)
    stabilized = vectorized_qac_stabilize(expr_matrix, adjacency, n_layers=50, reinforce_rate=0.05)
    
    # Analyze results
    results = analyze_stabilization_results(expr_matrix, stabilized, highly_var_genes, alpha_values)
    
    # Print summary
    print_summary(results, alpha_values)
    
    # Save results
    save_results(results)
    
    print("\n" + "="*80)
    print("VECTORIZED QAC ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
