#!/usr/bin/env python3
"""
Extract and save alpha values from gene_analysis.py computation
This enables the interactive_gene_query.py tool to use precomputed alpha values
"""

import sys
import json
import numpy as np
import scanpy as sc
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / 'src'))

from don_memory.adapters.don_stack_adapter import DONStackAdapter

def main():
    print("="*80)
    print("SAVING ALPHA VALUES FOR INTERACTIVE QUERY TOOL")
    print("="*80)
    
    # Load PBMC3K dataset
    print("\nLoading PBMC3K dataset...")
    adata = sc.read_h5ad("data/pbmc3k.h5ad")
    
    # Standard preprocessing (must match gene_analysis.py)
    print("Preprocessing...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    
    # PCA and clustering
    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_pcs=40)
    sc.tl.leiden(adata, resolution=0.5)
    
    print(f"✓ Loaded {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"✓ Found {len(adata.obs['leiden'].unique())} clusters")
    
    # Initialize DON Stack adapter
    print("\nInitializing DON Stack adapter...")
    adapter = DONStackAdapter()
    
    # Compute alpha for all cells
    print(f"\nComputing alpha for all {adata.n_obs} cells...")
    alpha_values = []
    
    for i in range(adata.n_obs):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{adata.n_obs} cells...")
        
        # Get cell's PCA vector
        cell_pca = adata.obsm['X_pca'][i, :]
        
        # Normalize with DON Stack
        normalized = adapter.normalize(cell_pca)
        
        # Compute tensions (squared differences between adjacent dimensions)
        tensions = np.diff(normalized) ** 2
        
        # Tune alpha using TACE (convert numpy array to list)
        alpha = adapter.tune_alpha(tensions.tolist(), default_alpha=0.5)
        alpha_values.append(float(alpha))
    
    alpha_values = np.array(alpha_values)
    
    print(f"\n✓ Computed alpha for all {len(alpha_values)} cells")
    print(f"\nAlpha Statistics:")
    print(f"  Mean:  {alpha_values.mean():.4f}")
    print(f"  Std:   {alpha_values.std():.4f}")
    print(f"  Range: [{alpha_values.min():.4f}, {alpha_values.max():.4f}]")
    
    # Classify into regimes
    low_alpha = (alpha_values < 0.3).sum()
    mid_alpha = ((alpha_values >= 0.3) & (alpha_values < 0.7)).sum()
    high_alpha = (alpha_values >= 0.7).sum()
    
    print(f"\nAlpha Regime Distribution:")
    print(f"  Low  (<0.3):   {low_alpha:4d} cells ({low_alpha/len(alpha_values)*100:.1f}%)")
    print(f"  Mid  (0.3-0.7): {mid_alpha:4d} cells ({mid_alpha/len(alpha_values)*100:.1f}%)")
    print(f"  High (>0.7):   {high_alpha:4d} cells ({high_alpha/len(alpha_values)*100:.1f}%)")
    
    # Save to JSON
    output_file = "gene_analysis_alpha_results.json"
    print(f"\nSaving alpha values to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump({
            'alpha_values': alpha_values.tolist(),
            'n_cells': len(alpha_values),
            'statistics': {
                'mean': float(alpha_values.mean()),
                'std': float(alpha_values.std()),
                'min': float(alpha_values.min()),
                'max': float(alpha_values.max())
            },
            'regimes': {
                'low': int(low_alpha),
                'mid': int(mid_alpha),
                'high': int(high_alpha)
            }
        }, f, indent=2)
    
    print(f"✓ Saved alpha values for {len(alpha_values)} cells")
    print(f"\nYou can now use: python interactive_gene_query.py")

if __name__ == "__main__":
    main()
