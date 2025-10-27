#!/usr/bin/env python3
"""
Interactive Gene Query Tool
Explores PBMC3K dataset with alpha values computed by gene_analysis.py
Allows text-based gene search to see expression patterns and alpha correlations
"""

import os
import sys
import json
import numpy as np
import scanpy as sc
from pathlib import Path

# Check if we already have the alpha results saved
ALPHA_RESULTS_FILE = "gene_analysis_alpha_results.json"

def load_data():
    """Load PBMC3K dataset with preprocessing"""
    print("Loading PBMC3K dataset...")
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
    print(f"✓ Found {len(adata.obs['leiden'].unique())} clusters")
    
    return adata

def load_alpha_values():
    """Load precomputed alpha values if available"""
    if os.path.exists(ALPHA_RESULTS_FILE):
        print(f"\n✓ Loading precomputed alpha values from {ALPHA_RESULTS_FILE}")
        with open(ALPHA_RESULTS_FILE, 'r') as f:
            data = json.load(f)
            return np.array(data['alpha_values'])
    else:
        print(f"\n⚠ No precomputed alpha values found at {ALPHA_RESULTS_FILE}")
        print("Run gene_analysis.py first to compute alpha for all cells")
        return None

def search_genes(adata, gene_query):
    """Search for genes matching query string"""
    gene_query_upper = gene_query.upper()
    matches = [g for g in adata.var_names if gene_query_upper in g.upper()]
    return sorted(matches)

def get_gene_expression(adata, gene_name):
    """Get expression values for a specific gene"""
    if gene_name not in adata.var_names:
        return None
    
    # Get from raw data (unnormalized counts)
    if adata.raw is not None:
        idx = list(adata.raw.var_names).index(gene_name)
        return adata.raw.X[:, idx].toarray().flatten()
    else:
        idx = list(adata.var_names).index(gene_name)
        return adata.X[:, idx].toarray().flatten()

def analyze_gene_alpha_correlation(adata, gene_name, alpha_values):
    """Analyze correlation between gene expression and alpha"""
    expr = get_gene_expression(adata, gene_name)
    if expr is None:
        return None
    
    # Compute correlation
    correlation = np.corrcoef(expr, alpha_values)[0, 1]
    
    # Expression by alpha regime
    low_alpha_mask = alpha_values < 0.3
    mid_alpha_mask = (alpha_values >= 0.3) & (alpha_values < 0.7)
    high_alpha_mask = alpha_values >= 0.7
    
    low_expr = expr[low_alpha_mask].mean() if low_alpha_mask.sum() > 0 else 0
    mid_expr = expr[mid_alpha_mask].mean() if mid_alpha_mask.sum() > 0 else 0
    high_expr = expr[high_alpha_mask].mean() if high_alpha_mask.sum() > 0 else 0
    
    # Expression by cluster
    cluster_expr = {}
    for cluster_id in sorted(adata.obs['leiden'].unique()):
        cluster_mask = adata.obs['leiden'] == cluster_id
        cluster_expr[cluster_id] = expr[cluster_mask].mean()
    
    return {
        'gene': gene_name,
        'correlation': correlation,
        'mean_expression': expr.mean(),
        'std_expression': expr.std(),
        'expressing_cells': (expr > 0).sum(),
        'expressing_pct': (expr > 0).sum() / len(expr) * 100,
        'alpha_regimes': {
            'low': low_expr,
            'mid': mid_expr,
            'high': high_expr
        },
        'clusters': cluster_expr
    }

def print_gene_report(stats):
    """Print detailed gene expression report"""
    print(f"\n{'='*80}")
    print(f"GENE: {stats['gene']}")
    print(f"{'='*80}")
    
    print(f"\nOverall Expression:")
    print(f"  Mean: {stats['mean_expression']:.4f}")
    print(f"  Std:  {stats['std_expression']:.4f}")
    print(f"  Expressing cells: {stats['expressing_cells']} / {stats['expressing_pct']:.1f}%")
    
    print(f"\nAlpha Correlation:")
    print(f"  Pearson r = {stats['correlation']:.4f}")
    if abs(stats['correlation']) > 0.3:
        direction = "POSITIVE" if stats['correlation'] > 0 else "NEGATIVE"
        strength = "STRONG" if abs(stats['correlation']) > 0.5 else "MODERATE"
        print(f"  ⚠ {strength} {direction} correlation with alpha!")
    
    print(f"\nExpression by Alpha Regime:")
    low = stats['alpha_regimes']['low']
    mid = stats['alpha_regimes']['mid']
    high = stats['alpha_regimes']['high']
    print(f"  Low alpha  (<0.3):  {low:.4f}")
    print(f"  Mid alpha  (0.3-0.7): {mid:.4f}")
    print(f"  High alpha (>0.7):  {high:.4f}")
    
    # Highlight enrichment
    max_regime = max(low, mid, high)
    if max_regime > 0:
        if low == max_regime:
            print(f"  → Enriched in LOW-ALPHA cells (resting T cells)")
        elif high == max_regime:
            print(f"  → Enriched in HIGH-ALPHA cells (activated APCs)")
        else:
            print(f"  → Enriched in MID-ALPHA cells (transitional state)")
    
    print(f"\nExpression by Cluster:")
    for cluster_id, expr in stats['clusters'].items():
        cluster_num = int(cluster_id)
        print(f"  Cluster {cluster_num}: {expr:.4f}")

def interactive_mode(adata, alpha_values):
    """Interactive gene query loop"""
    print("\n" + "="*80)
    print("INTERACTIVE GENE QUERY MODE")
    print("="*80)
    print("\nCommands:")
    print("  - Enter gene name to search (e.g., 'CD3D', 'HLA-DRA')")
    print("  - Enter partial name to see matches (e.g., 'HLA' shows all HLA genes)")
    print("  - 'list' - show all available genes")
    print("  - 'markers' - show known cell type markers")
    print("  - 'quit' or 'exit' - exit the tool")
    print()
    
    known_markers = {
        "T cells": ["CD3D", "CD3E", "CD3G", "IL7R", "IL32", "LTB"],
        "CD4+ T cells": ["CD4", "IL7R", "LTB"],
        "CD8+ T cells": ["CD8A", "CD8B"],
        "B cells": ["CD79A", "CD79B", "MS4A1", "CD19"],
        "Monocytes": ["CD14", "LYZ", "S100A8", "S100A9"],
        "NK cells": ["NKG7", "GNLY", "KLRD1", "KLRF1"],
        "Dendritic cells": ["FCER1A", "CST3", "CLEC10A"],
        "Megakaryocytes": ["PPBP", "PF4", "GP9"],
        "APCs (high alpha)": ["HLA-DRA", "HLA-DRB1", "CD74", "HLA-DPA1", "HLA-DPB1"],
        "T cells (low alpha)": ["CD3D", "IL32", "LTB", "IL7R", "LDHB"]
    }
    
    while True:
        try:
            query = input("\nEnter gene query (or command): ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting. Goodbye!")
                break
            
            if query.lower() == 'list':
                print(f"\nAll available genes ({len(adata.var_names)} total):")
                for i, gene in enumerate(adata.var_names[:100], 1):
                    print(f"  {gene}", end="  ")
                    if i % 8 == 0:
                        print()
                if len(adata.var_names) > 100:
                    print(f"\n  ... and {len(adata.var_names) - 100} more")
                print("\nTip: Use partial search to find specific genes (e.g., 'CD3')")
                continue
            
            if query.lower() == 'markers':
                print("\nKnown Cell Type Markers:")
                for cell_type, genes in known_markers.items():
                    print(f"\n{cell_type}:")
                    print(f"  {', '.join(genes)}")
                continue
            
            # Search for matching genes
            matches = search_genes(adata, query)
            
            if not matches:
                print(f"✗ No genes found matching '{query}'")
                print("Try a partial match (e.g., 'HLA' or 'CD')")
                continue
            
            if len(matches) == 1:
                # Exact or single match - show full analysis
                gene = matches[0]
                if alpha_values is not None:
                    stats = analyze_gene_alpha_correlation(adata, gene, alpha_values)
                    print_gene_report(stats)
                else:
                    expr = get_gene_expression(adata, gene)
                    print(f"\n✓ Found: {gene}")
                    print(f"  Mean expression: {expr.mean():.4f}")
                    print(f"  Expressing cells: {(expr > 0).sum()} / {(expr > 0).sum() / len(expr) * 100:.1f}%")
                    print("\n⚠ Alpha correlation requires precomputed alpha values")
            
            elif len(matches) <= 20:
                # Multiple matches - show list
                print(f"\n✓ Found {len(matches)} matching genes:")
                for gene in matches:
                    expr = get_gene_expression(adata, gene)
                    pct = (expr > 0).sum() / len(expr) * 100
                    print(f"  - {gene:15s} (expressed in {pct:5.1f}% of cells)")
                print("\nEnter full gene name for detailed analysis")
            
            else:
                # Too many matches
                print(f"\n✓ Found {len(matches)} matching genes (showing first 20):")
                for gene in matches[:20]:
                    expr = get_gene_expression(adata, gene)
                    pct = (expr > 0).sum() / len(expr) * 100
                    print(f"  - {gene:15s} (expressed in {pct:5.1f}% of cells)")
                print(f"\n... and {len(matches) - 20} more matches")
                print("Tip: Use more specific search term")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting.")
            break
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main entry point"""
    print("="*80)
    print("PBMC3K INTERACTIVE GENE QUERY TOOL")
    print("="*80)
    
    # Load data
    adata = load_data()
    alpha_values = load_alpha_values()
    
    if alpha_values is not None:
        print(f"✓ Alpha values loaded for {len(alpha_values)} cells")
        print(f"  Alpha range: [{alpha_values.min():.2f}, {alpha_values.max():.2f}]")
        print(f"  Low regime:  {(alpha_values < 0.3).sum()} cells")
        print(f"  Mid regime:  {((alpha_values >= 0.3) & (alpha_values < 0.7)).sum()} cells")
        print(f"  High regime: {(alpha_values >= 0.7).sum()} cells")
    
    # Enter interactive mode
    interactive_mode(adata, alpha_values)

if __name__ == "__main__":
    main()
