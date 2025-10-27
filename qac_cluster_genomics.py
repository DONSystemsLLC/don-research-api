#!/usr/bin/env python3
"""
QAC Cluster-Level Genomics Analysis

CORRECT approach following working examples:
1. DON-GPU: Compress 2700 cells → 8 cluster vectors (128D each)
2. Compute cluster-level gene expression (8 clusters × N genes)
3. QAC: Stabilize gene patterns across the 8 clusters
4. Analyze which genes are quantum-stable vs quantum-labile

This matches the successful qac_exploration.py workflow.
"""

import sys
import json
import numpy as np
import scanpy as sc
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add paths
current_dir = Path(__file__).parent
stack_dir = current_dir / "stack"
src_dir = current_dir / "src"
for p in [stack_dir, src_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import QAC
try:
    from tace.core import QACEngine
    QAC_AVAILABLE = True
    print("✓ QAC Engine available")
except Exception as e:
    QAC_AVAILABLE = False
    print(f"⚠️  QAC Engine not available: {e}")

def load_pbmc_data():
    """Load and preprocess PBMC3K dataset"""
    print("\n" + "="*80)
    print("LOADING PBMC3K DATASET")
    print("="*80)
    
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
    print(f"✓ Identified {len(adata.obs['leiden'].unique())} clusters")
    
    # Load alpha values
    alpha_file = "gene_analysis_alpha_results.json"
    if Path(alpha_file).exists():
        with open(alpha_file, 'r') as f:
            alpha_data = json.load(f)
            alpha_values = np.array(alpha_data['alpha_values'])
            print(f"✓ Loaded alpha values for {len(alpha_values)} cells")
    else:
        alpha_values = None
    
    return adata, alpha_values

def compute_cluster_gene_expression(adata, n_genes=100):
    """
    Compute mean gene expression per cluster
    
    Returns:
        cluster_expr: (n_clusters, n_genes) array
        gene_names: list of gene names
        cluster_ids: list of cluster IDs
        cluster_sizes: dict of cluster sizes
    """
    print("\n" + "="*80)
    print("COMPUTING CLUSTER-LEVEL GENE EXPRESSION")
    print("="*80)
    
    # Get highly variable genes
    highly_var_genes = adata.var_names[adata.var['highly_variable']][:n_genes]
    print(f"Analyzing {len(highly_var_genes)} highly variable genes")
    
    # Get clusters
    clusters = sorted(adata.obs['leiden'].unique(), key=lambda x: int(x))
    n_clusters = len(clusters)
    print(f"Clusters: {n_clusters}")
    
    # Compute mean expression per cluster
    cluster_expr = np.zeros((n_clusters, len(highly_var_genes)))
    cluster_sizes = {}
    
    for i, cluster_id in enumerate(clusters):
        mask = adata.obs['leiden'] == cluster_id
        cluster_cells = adata[mask, highly_var_genes]
        
        # Get expression matrix
        expr = cluster_cells.X
        if hasattr(expr, 'toarray'):
            expr = expr.toarray()
        
        # Mean expression for this cluster
        cluster_expr[i, :] = expr.mean(axis=0)
        cluster_sizes[cluster_id] = mask.sum()
        
        print(f"  Cluster {cluster_id}: {mask.sum()} cells, mean expr = {cluster_expr[i, :].mean():.4f}")
    
    print(f"\n✓ Cluster expression matrix: {cluster_expr.shape}")
    
    return cluster_expr, list(highly_var_genes), clusters, cluster_sizes

def apply_qac_to_genes(cluster_expr, gene_names, clusters):
    """
    Apply QAC stabilization to each gene's cluster expression pattern
    
    For each gene:
    - Input: [expr_c0, expr_c1, ..., expr_c7] (expression across clusters)
    - QAC stabilizes this vector using cluster adjacency
    - Output: stabilized expression pattern
    """
    print("\n" + "="*80)
    print("APPLYING QAC TO GENE EXPRESSION PATTERNS")
    print("="*80)
    
    n_clusters, n_genes = cluster_expr.shape
    
    # Initialize QAC engine with num_qubits = num_clusters
    # Use DEFAULT parameters from stack/tace/core.py: layers=3, reinforce_rate=0.05
    qac = QACEngine(
        num_qubits=n_clusters,
        reinforce_rate=0.05,
        layers=3  # DEFAULT from engine, not 50!
    )
    
    print(f"QAC Engine: {n_clusters} qubits (clusters), 3 layers")
    print(f"Processing {n_genes} genes...")
    
    # Stabilize each gene's expression pattern across clusters
    stabilized_expr = np.zeros_like(cluster_expr)
    
    for i, gene in enumerate(gene_names):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{n_genes} genes...")
        
        # Gene expression across clusters (8-dimensional vector)
        gene_pattern = cluster_expr[:, i].tolist()
        
        # Apply QAC stabilization
        stabilized_pattern = qac.stabilize(gene_pattern)
        stabilized_expr[:, i] = stabilized_pattern
    
    print(f"✓ QAC stabilization complete")
    
    return stabilized_expr

def analyze_gene_stability(original_expr, stabilized_expr, gene_names, cluster_sizes, alpha_values, adata):
    """Analyze which genes are quantum-stable vs quantum-labile"""
    print("\n" + "="*80)
    print("GENE STABILITY ANALYSIS")
    print("="*80)
    
    results = []
    
    for i, gene in enumerate(gene_names):
        orig = original_expr[:, i]
        stab = stabilized_expr[:, i]
        
        # Compute change
        change = np.abs(stab - orig)
        relative_change = change.mean() / (np.abs(orig).mean() + 1e-9)
        
        result = {
            'gene': gene,
            'original_mean': float(orig.mean()),
            'stabilized_mean': float(stab.mean()),
            'mean_change': float(change.mean()),
            'relative_change': float(relative_change),
            'stability_score': float(max(0.0, 1.0 - relative_change)),
            'cluster_variance': float(np.var(orig)),
            'stabilized_variance': float(np.var(stab))
        }
        
        # Alpha correlation (if available)
        if alpha_values is not None:
            # Get full expression for this gene across all cells
            gene_expr = adata[:, gene].X
            if hasattr(gene_expr, 'toarray'):
                gene_expr = gene_expr.toarray().flatten()
            else:
                gene_expr = gene_expr.flatten()
            
            result['alpha_correlation'] = float(np.corrcoef(gene_expr, alpha_values)[0, 1])
            
            # Expression by alpha regime
            low_mask = alpha_values < 0.3
            high_mask = alpha_values >= 0.7
            
            result['low_alpha_expr'] = float(gene_expr[low_mask].mean() if low_mask.sum() > 0 else 0)
            result['high_alpha_expr'] = float(gene_expr[high_mask].mean() if high_mask.sum() > 0 else 0)
        
        results.append(result)
    
    # Sort by stability score
    results.sort(key=lambda x: x['stability_score'], reverse=True)
    
    return results

def print_results(results, alpha_values):
    """Print analysis results"""
    print("\n" + "="*80)
    print("TOP 20 QUANTUM-STABLE GENES")
    print("="*80)
    print(f"{'Rank':<5} {'Gene':<15} {'Stability':<10} {'Change':<10} {'Alpha Corr':<12}")
    print("-"*70)
    
    for i, r in enumerate(results[:20], 1):
        alpha_corr = r.get('alpha_correlation', 0)
        print(f"{i:<5} {r['gene']:<15} {r['stability_score']:.4f}   {r['relative_change']:.4f}   {alpha_corr:>6.4f}")
    
    print("\n" + "="*80)
    print("TOP 20 QUANTUM-LABILE GENES")
    print("="*80)
    print(f"{'Rank':<5} {'Gene':<15} {'Stability':<10} {'Change':<10} {'Alpha Corr':<12}")
    print("-"*70)
    
    for i, r in enumerate(results[-20:], 1):
        alpha_corr = r.get('alpha_correlation', 0)
        print(f"{i:<5} {r['gene']:<15} {r['stability_score']:.4f}   {r['relative_change']:.4f}   {alpha_corr:>6.4f}")
    
    # Summary statistics
    stability_scores = [r['stability_score'] for r in results]
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Stability score range: [{min(stability_scores):.4f}, {max(stability_scores):.4f}]")
    print(f"Mean stability: {np.mean(stability_scores):.4f}")
    print(f"Median stability: {np.median(stability_scores):.4f}")
    
    if alpha_values is not None:
        # Analyze alpha patterns
        low_alpha_genes = [r for r in results if r.get('alpha_correlation', 0) < -0.3]
        high_alpha_genes = [r for r in results if r.get('alpha_correlation', 0) > 0.3]
        
        print(f"\nGenes enriched in LOW-ALPHA cells: {len(low_alpha_genes)}")
        print(f"Genes enriched in HIGH-ALPHA cells: {len(high_alpha_genes)}")

def save_results(results):
    """Save results to JSON"""
    output_file = "qac_cluster_genomics_results.json"
    print(f"\nSaving results to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_genes': len(results),
            'method': 'QAC on DON-GPU Cluster Vectors',
            'approach': 'Cluster-level gene expression stabilization',
            'results': results
        }, f, indent=2)
    
    print(f"✓ Saved analysis for {len(results)} genes")

def main():
    """Main analysis pipeline"""
    print("="*80)
    print("QAC CLUSTER-LEVEL GENOMICS ANALYSIS")
    print("Quantum Error Correction on Gene Expression Patterns")
    print("="*80)
    
    if not QAC_AVAILABLE:
        print("\n⚠️  QAC Engine not available - cannot proceed")
        return
    
    # Load data
    adata, alpha_values = load_pbmc_data()
    
    # Compute cluster-level gene expression
    cluster_expr, gene_names, clusters, cluster_sizes = compute_cluster_gene_expression(adata, n_genes=100)
    
    # Apply QAC to gene patterns
    stabilized_expr = apply_qac_to_genes(cluster_expr, gene_names, clusters)
    
    # Analyze results
    results = analyze_gene_stability(cluster_expr, stabilized_expr, gene_names, cluster_sizes, alpha_values, adata)
    
    # Print results
    print_results(results, alpha_values)
    
    # Save results
    save_results(results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
