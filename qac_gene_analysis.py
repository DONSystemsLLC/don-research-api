#!/usr/bin/env python3
"""
QAC Gene Analysis
Apply Quantum Adjacency Code error correction to gene expression patterns
to identify stable biological programs vs transient/noisy states
"""

import sys
import json
import numpy as np
import scanpy as sc
from pathlib import Path
from datetime import datetime

# Add src to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / 'src'))

from don_memory.adapters.don_stack_adapter import DONStackAdapter

# Try to import QAC engine
try:
    stack_dir = current_dir / "stack"
    if str(stack_dir) not in sys.path:
        sys.path.insert(0, str(stack_dir))
    from tace.core import QACEngine
    HAVE_QAC = True
    print("✓ QAC Engine available")
except Exception as e:
    HAVE_QAC = False
    print(f"⚠ QAC Engine not available: {e}")
    print("  Will use fallback analysis")

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
        print("⚠ No alpha values found - run save_alpha_values.py first")
        alpha_values = None
    
    return adata, alpha_values

def apply_qac_to_gene_vectors(adata, alpha_values, top_n_genes=100):
    """
    Apply QAC stabilization to gene expression vectors
    Focus on highly variable genes to find stable biological programs
    """
    print("\n" + "="*80)
    print("QAC GENE EXPRESSION ANALYSIS")
    print("="*80)
    
    # Get highly variable genes for analysis
    highly_var_genes = adata.var_names[adata.var['highly_variable']][:top_n_genes]
    print(f"\nAnalyzing top {len(highly_var_genes)} highly variable genes")
    
    # Extract expression matrix for these genes (cells × genes)
    expr_matrix = adata[:, highly_var_genes].X.toarray() if hasattr(adata.X, 'toarray') else adata[:, highly_var_genes].X
    print(f"Expression matrix shape: {expr_matrix.shape}")
    
    # Add alpha as a feature dimension
    if alpha_values is not None:
        # Normalize alpha to similar scale as expression
        alpha_norm = (alpha_values - alpha_values.mean()) / alpha_values.std()
        # Add as additional dimension
        expr_with_alpha = np.column_stack([expr_matrix, alpha_norm[:, np.newaxis]])
        print(f"✓ Added alpha as feature dimension: {expr_with_alpha.shape}")
    else:
        expr_with_alpha = expr_matrix
    
    if HAVE_QAC:
        print("\nApplying QAC stabilization...")
        return apply_real_qac(expr_with_alpha, highly_var_genes, alpha_values)
    else:
        print("\nApplying fallback analysis (no QAC engine)...")
        return apply_fallback_analysis(expr_with_alpha, highly_var_genes, alpha_values)

def apply_real_qac(expr_matrix, gene_names, alpha_values):
    """Apply real QAC engine to gene expression patterns"""
    n_cells, n_features = expr_matrix.shape
    
    # Initialize QAC engine
    # Number of qubits = number of cells (each cell is a quantum state)
    qac = QACEngine(
        num_qubits=n_cells,
        reinforce_rate=0.05,  # 5% reinforcement per layer
        layers=50              # 50 error correction layers
    )
    
    print(f"QAC Engine initialized: {n_cells} qubits, 50 layers")
    
    # Build adjacency matrix from cell-cell similarities
    # Cells that are similar should be adjacent in the quantum code
    print("Computing cell-cell adjacency...")
    
    # Use correlation between cells as adjacency
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr
    
    # Compute pairwise correlations (cells that express similar genes)
    correlations = np.corrcoef(expr_matrix)
    
    # Convert to adjacency: higher correlation = stronger adjacency
    # Threshold at 0.3 correlation to keep only meaningful connections
    adjacency = (correlations > 0.3).astype(float) * correlations
    np.fill_diagonal(adjacency, 0)  # No self-loops
    
    print(f"✓ Built adjacency matrix: {adjacency.shape}")
    print(f"  Mean connectivity: {adjacency.sum(axis=1).mean():.2f} neighbors/cell")
    
    # Apply QAC stabilization to each gene's expression pattern
    try:
        import jax.numpy as jnp
        qac.base_adj = jnp.array(adjacency, dtype=jnp.float32)
        print("✓ Using JAX for QAC computation")
    except Exception:
        qac.base_adj = adjacency
        print("✓ Using NumPy for QAC computation")
    
    # Stabilize each gene's expression pattern across cells
    stabilized_matrix = np.zeros_like(expr_matrix)
    
    print("\nStabilizing gene expression patterns...")
    for i, gene in enumerate(gene_names):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(gene_names)} genes...")
        
        # Get expression values for this gene across all cells
        gene_expr = expr_matrix[:, i]
        
        # Apply QAC stabilization (error correction)
        stabilized = qac.stabilize(gene_expr.tolist())
        stabilized_matrix[:, i] = np.array(stabilized)
    
    # Add alpha dimension back if it was included
    if expr_matrix.shape[1] > len(gene_names):
        # Last column is alpha
        alpha_stabilized = qac.stabilize(expr_matrix[:, -1].tolist())
        stabilized_matrix = np.column_stack([stabilized_matrix, alpha_stabilized])
    
    print(f"\n✓ Stabilized {len(gene_names)} gene expression patterns")
    
    # Analyze results
    return analyze_qac_results(expr_matrix, stabilized_matrix, gene_names, alpha_values)

def apply_fallback_analysis(expr_matrix, gene_names, alpha_values):
    """Fallback analysis without QAC engine - use statistical stability metrics"""
    print("\nComputing statistical stability metrics...")
    
    results = []
    for i, gene in enumerate(gene_names):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(gene_names)} genes...")
        
        gene_expr = expr_matrix[:, i]
        
        # Compute stability metrics
        stability = {
            'gene': gene,
            'variance': float(np.var(gene_expr)),
            'cv': float(np.std(gene_expr) / (np.mean(gene_expr) + 1e-9)),  # Coefficient of variation
            'sparsity': float((gene_expr > 0).sum() / len(gene_expr)),
            'mean_expr': float(np.mean(gene_expr))
        }
        
        if alpha_values is not None:
            # Correlation with alpha
            stability['alpha_correlation'] = float(np.corrcoef(gene_expr, alpha_values)[0, 1])
            
            # Expression in different alpha regimes
            low_mask = alpha_values < 0.3
            high_mask = alpha_values >= 0.7
            stability['low_alpha_expr'] = float(gene_expr[low_mask].mean() if low_mask.sum() > 0 else 0)
            stability['high_alpha_expr'] = float(gene_expr[high_mask].mean() if high_mask.sum() > 0 else 0)
        
        results.append(stability)
    
    # Sort by variance (higher variance = less stable/more dynamic)
    results.sort(key=lambda x: x['variance'], reverse=True)
    
    print(f"\n✓ Computed stability metrics for {len(gene_names)} genes")
    return results

def analyze_qac_results(original, stabilized, gene_names, alpha_values):
    """Analyze QAC stabilization results"""
    print("\n" + "="*80)
    print("QAC STABILIZATION RESULTS")
    print("="*80)
    
    results = []
    
    for i, gene in enumerate(gene_names):
        orig_expr = original[:, i]
        stab_expr = stabilized[:, i]
        
        # Compute stabilization metrics
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
            # Original correlations
            result['original_alpha_corr'] = float(np.corrcoef(orig_expr, alpha_values)[0, 1])
            result['stabilized_alpha_corr'] = float(np.corrcoef(stab_expr, alpha_values)[0, 1])
            
            # Regime expression
            low_mask = alpha_values < 0.3
            high_mask = alpha_values >= 0.7
            
            result['orig_low_alpha'] = float(orig_expr[low_mask].mean() if low_mask.sum() > 0 else 0)
            result['orig_high_alpha'] = float(orig_expr[high_mask].mean() if high_mask.sum() > 0 else 0)
            result['stab_low_alpha'] = float(stab_expr[low_mask].mean() if low_mask.sum() > 0 else 0)
            result['stab_high_alpha'] = float(stab_expr[high_mask].mean() if high_mask.sum() > 0 else 0)
        
        results.append(result)
    
    # Sort by stability score (higher = more stable/robust biological program)
    results.sort(key=lambda x: x['stability_score'], reverse=True)
    
    return results

def print_summary(results, alpha_values):
    """Print summary of QAC analysis"""
    print("\n" + "="*80)
    print("TOP 20 MOST STABLE GENES (Robust Biological Programs)")
    print("="*80)
    print(f"\n{'Rank':<5} {'Gene':<15} {'Stability':<10} {'Mean Expr':<12} {'Alpha Corr':<12} {'Pattern':<30}")
    print("-"*95)
    
    for i, r in enumerate(results[:20], 1):
        pattern = ""
        if alpha_values is not None:
            if r['original_alpha_corr'] > 0.3:
                pattern = "HIGH-ALPHA enriched (activated)"
            elif r['original_alpha_corr'] < -0.3:
                pattern = "LOW-ALPHA enriched (resting)"
            else:
                pattern = "Alpha-independent"
        
        print(f"{i:<5} {r['gene']:<15} {r['stability_score']:.4f}   {r['original_mean']:.4f}       "
              f"{r.get('original_alpha_corr', 0):.4f}       {pattern:<30}")
    
    print("\n" + "="*80)
    print("TOP 20 MOST DYNAMIC GENES (Transient/Responsive States)")
    print("="*80)
    print(f"\n{'Rank':<5} {'Gene':<15} {'Stability':<10} {'Mean Expr':<12} {'Alpha Corr':<12} {'Pattern':<30}")
    print("-"*95)
    
    for i, r in enumerate(results[-20:], 1):
        pattern = ""
        if alpha_values is not None:
            if r['original_alpha_corr'] > 0.3:
                pattern = "HIGH-ALPHA enriched (activated)"
            elif r['original_alpha_corr'] < -0.3:
                pattern = "LOW-ALPHA enriched (resting)"
            else:
                pattern = "Alpha-independent"
        
        print(f"{i:<5} {r['gene']:<15} {r['stability_score']:.4f}   {r['original_mean']:.4f}       "
              f"{r.get('original_alpha_corr', 0):.4f}       {pattern:<30}")

def save_results(results):
    """Save QAC analysis results"""
    output_file = "qac_gene_analysis_results.json"
    print(f"\nSaving results to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_genes': len(results),
            'qac_engine': 'real' if HAVE_QAC else 'fallback',
            'results': results
        }, f, indent=2)
    
    print(f"✓ Saved QAC analysis for {len(results)} genes")

def main():
    """Main entry point"""
    print("="*80)
    print("QAC GENE EXPRESSION ANALYSIS")
    print("Quantum Adjacency Code Error Correction on Genomics Data")
    print("="*80)
    
    # Load data
    adata, alpha_values = load_data_with_alpha()
    
    # Apply QAC
    results = apply_qac_to_gene_vectors(adata, alpha_values, top_n_genes=100)
    
    # Print summary
    print_summary(results, alpha_values)
    
    # Save results
    save_results(results)
    
    print("\n" + "="*80)
    print("QAC ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
