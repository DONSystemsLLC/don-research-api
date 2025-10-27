"""
Gene Co-Expression Module Analysis with QAC

Research Question: Does QAC reveal quantum-coherent gene regulatory modules within cells?

Approach:
1. Group genes into 8 functional modules (matching num_qubits=8)
2. Apply QAC to module expression patterns within individual cells
3. Test if low-α vs high-α cells have different module coherence
4. Identify which gene modules show quantum stability

Critical: NO BIAS INJECTION - let the math speak for itself
"""

import numpy as np
import scanpy as sc
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add stack to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "stack"))

from tace.core import QACEngine

# Gene module definitions based on known biological functions
GENE_MODULES = {
    "ribosomal": ["RPL", "RPS"],  # Protein synthesis
    "mitochondrial": ["MT-", "COX", "ATP", "ND"],  # Energy metabolism
    "immune_activation": ["HLA-", "CD74", "B2M"],  # Antigen presentation
    "t_cell_identity": ["CD3", "IL7R", "LDHB"],  # T cell markers
    "inflammatory": ["S100A", "LYZ", "FTL"],  # Inflammation
    "cell_cycle": ["MKI67", "PCNA", "TOP2A"],  # Proliferation
    "transcription": ["JUN", "FOS", "EGR"],  # Gene regulation
    "housekeeping": ["ACTB", "GAPDH", "B2M"]  # Core cellular functions
}

def load_pbmc_data():
    """Load PBMC3K data with TACE alphas"""
    print("Loading PBMC3K dataset with TACE alphas...")
    adata = sc.read_h5ad("data/pbmc3k_with_tace_alpha.h5ad")
    print(f"  Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"  Alpha range: [{adata.obs['alpha_tace'].min():.3f}, {adata.obs['alpha_tace'].max():.3f}]")
    return adata

def assign_genes_to_modules(adata) -> Dict[str, List[str]]:
    """
    Assign genes to functional modules based on gene name patterns
    
    Returns dict of module_name -> list of gene names
    """
    print("\nAssigning genes to functional modules...")
    
    all_genes = adata.var_names.tolist()
    module_assignments = {module: [] for module in GENE_MODULES.keys()}
    unassigned = []
    
    for gene in all_genes:
        assigned = False
        for module_name, patterns in GENE_MODULES.items():
            if any(gene.startswith(pattern) for pattern in patterns):
                module_assignments[module_name].append(gene)
                assigned = True
                break
        
        if not assigned:
            unassigned.append(gene)
    
    # Print module sizes
    print("\nModule assignments:")
    for module_name, genes in module_assignments.items():
        print(f"  {module_name}: {len(genes)} genes")
    print(f"  unassigned: {len(unassigned)} genes")
    
    return module_assignments

def compute_module_expression(adata, module_genes: Dict[str, List[str]]) -> np.ndarray:
    """
    Compute average expression for each module in each cell
    
    Returns: (n_cells, n_modules) array
    """
    print("\nComputing module expression vectors...")
    
    n_cells = adata.n_obs
    module_names = list(module_genes.keys())
    n_modules = len(module_names)
    
    module_expr = np.zeros((n_cells, n_modules))
    
    for i, module_name in enumerate(module_names):
        genes = module_genes[module_name]
        if len(genes) == 0:
            continue
            
        # Find which genes exist in the dataset
        existing_genes = [g for g in genes if g in adata.var_names]
        
        if len(existing_genes) > 0:
            # Average expression across genes in this module
            gene_indices = [adata.var_names.tolist().index(g) for g in existing_genes]
            module_expr[:, i] = adata.X[:, gene_indices].mean(axis=1).A1
    
    print(f"  Module expression matrix: {module_expr.shape}")
    print(f"  Expression range: [{module_expr.min():.3f}, {module_expr.max():.3f}]")
    
    return module_expr, module_names

def apply_qac_to_cell_modules(module_vector: np.ndarray) -> float:
    """
    Apply QAC to a single cell's 8 module expression values
    
    Args:
        module_vector: (8,) array of module expression values
    
    Returns:
        stability_score: QAC stability metric
    """
    # Initialize QAC with DEFAULT parameters (num_qubits=8 matches 8 modules!)
    qac = QACEngine(num_qubits=8)
    
    # QAC expects list of states - treat each module as one quantum state
    # Shape: (8, 1) - 8 states with 1 dimension each
    states = module_vector.reshape(-1, 1)
    
    # Apply QAC stabilization
    stabilized = qac.stabilize(states)
    
    # Compute stability score: variance of stabilized states
    # Lower variance = more stable = modules are quantum-coherent
    stability = -np.var(stabilized)  # Negative so higher = more stable
    
    return stability

def analyze_cells_by_alpha_regime(adata, module_expr: np.ndarray, module_names: List[str]):
    """
    Apply QAC to cells grouped by alpha regime
    Compare module coherence across regimes
    """
    print("\n" + "="*60)
    print("ANALYZING MODULE COHERENCE BY ALPHA REGIME")
    print("="*60)
    
    alphas = adata.obs['alpha_tace'].values
    
    # Define alpha regimes
    low_alpha_mask = alphas < 0.3
    mid_alpha_mask = (alphas >= 0.3) & (alphas < 0.7)
    high_alpha_mask = alphas >= 0.7
    
    print(f"\nAlpha distribution:")
    print(f"  Low-α (<0.3): {low_alpha_mask.sum()} cells")
    print(f"  Mid-α (0.3-0.7): {mid_alpha_mask.sum()} cells")
    print(f"  High-α (>0.7): {high_alpha_mask.sum()} cells")
    
    # Sample cells from each regime for analysis
    np.random.seed(42)
    n_samples = 200  # Sample 200 cells per regime
    
    results = {}
    
    for regime_name, mask in [
        ("low_alpha", low_alpha_mask),
        ("mid_alpha", mid_alpha_mask),
        ("high_alpha", high_alpha_mask)
    ]:
        print(f"\n--- Analyzing {regime_name} cells ---")
        
        # Sample cells
        regime_indices = np.where(mask)[0]
        if len(regime_indices) > n_samples:
            sampled_indices = np.random.choice(regime_indices, n_samples, replace=False)
        else:
            sampled_indices = regime_indices
        
        print(f"  Sampled {len(sampled_indices)} cells")
        
        # Apply QAC to each cell's module vector
        stabilities = []
        for idx in sampled_indices:
            cell_modules = module_expr[idx]
            stability = apply_qac_to_cell_modules(cell_modules)
            stabilities.append(stability)
        
        stabilities = np.array(stabilities)
        
        print(f"  Mean stability: {stabilities.mean():.6f}")
        print(f"  Std stability: {stabilities.std():.6f}")
        print(f"  Range: [{stabilities.min():.6f}, {stabilities.max():.6f}]")
        
        results[regime_name] = {
            "n_cells": len(sampled_indices),
            "mean_stability": float(stabilities.mean()),
            "std_stability": float(stabilities.std()),
            "min_stability": float(stabilities.min()),
            "max_stability": float(stabilities.max()),
            "stabilities": stabilities.tolist()
        }
    
    # Statistical comparison
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON")
    print("="*60)
    
    from scipy.stats import mannwhitneyu, pearsonr
    
    low_stab = np.array(results["low_alpha"]["stabilities"])
    high_stab = np.array(results["high_alpha"]["stabilities"])
    
    # Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(low_stab, high_stab, alternative='two-sided')
    print(f"\nMann-Whitney U test (Low-α vs High-α):")
    print(f"  U-statistic: {u_stat:.2f}")
    print(f"  p-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print("  ✓ SIGNIFICANT difference in module coherence!")
    else:
        print("  ✗ No significant difference")
    
    # Compute correlation with alpha values
    print("\n--- Correlation with Alpha ---")
    
    # Get alpha values for sampled cells
    all_stabilities = []
    all_alphas = []
    
    for regime_name, mask in [
        ("low_alpha", low_alpha_mask),
        ("mid_alpha", mid_alpha_mask),
        ("high_alpha", high_alpha_mask)
    ]:
        regime_indices = np.where(mask)[0]
        if len(regime_indices) > n_samples:
            sampled_indices = np.random.choice(regime_indices, n_samples, replace=False)
        else:
            sampled_indices = regime_indices
        
        for idx in sampled_indices:
            cell_modules = module_expr[idx]
            stability = apply_qac_to_cell_modules(cell_modules)
            all_stabilities.append(stability)
            all_alphas.append(alphas[idx])
    
    all_stabilities = np.array(all_stabilities)
    all_alphas = np.array(all_alphas)
    
    r, p = pearsonr(all_stabilities, all_alphas)
    print(f"  Pearson r = {r:.4f} (p = {p:.6f})")
    
    if abs(r) > 0.1 and p < 0.05:
        if r > 0:
            print("  ✓ Module coherence INCREASES with alpha")
        else:
            print("  ✓ Module coherence DECREASES with alpha")
    else:
        print("  ✗ No correlation with alpha")
    
    results["statistical_tests"] = {
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(p_value),
        "pearson_r": float(r),
        "pearson_p": float(p)
    }
    
    return results

def analyze_module_specific_coherence(adata, module_expr: np.ndarray, module_names: List[str]):
    """
    Analyze which specific modules show quantum coherence
    """
    print("\n" + "="*60)
    print("MODULE-SPECIFIC COHERENCE ANALYSIS")
    print("="*60)
    
    alphas = adata.obs['alpha_tace'].values
    
    # Sample cells across all alpha values
    np.random.seed(42)
    n_samples = 500
    sampled_indices = np.random.choice(adata.n_obs, min(n_samples, adata.n_obs), replace=False)
    
    print(f"\nAnalyzing {len(sampled_indices)} cells")
    
    # For each module, compute variance across cells
    module_variances = {}
    module_alpha_correlations = {}
    
    for i, module_name in enumerate(module_names):
        module_values = module_expr[sampled_indices, i]
        alpha_values = alphas[sampled_indices]
        
        variance = np.var(module_values)
        
        # Correlation with alpha
        from scipy.stats import pearsonr
        r, p = pearsonr(module_values, alpha_values)
        
        module_variances[module_name] = float(variance)
        module_alpha_correlations[module_name] = {
            "pearson_r": float(r),
            "p_value": float(p)
        }
        
        print(f"\n{module_name}:")
        print(f"  Variance: {variance:.6f}")
        print(f"  Alpha correlation: r={r:.4f} (p={p:.6f})")
        if abs(r) > 0.2 and p < 0.05:
            if r > 0:
                print("    → Increases with alpha")
            else:
                print("    → Decreases with alpha")
    
    # Identify most/least variable modules
    sorted_modules = sorted(module_variances.items(), key=lambda x: x[1], reverse=True)
    
    print("\n--- Most Variable Modules (least coherent) ---")
    for module_name, var in sorted_modules[:3]:
        corr = module_alpha_correlations[module_name]
        print(f"  {module_name}: var={var:.6f}, r={corr['pearson_r']:.4f}")
    
    print("\n--- Least Variable Modules (most coherent) ---")
    for module_name, var in sorted_modules[-3:]:
        corr = module_alpha_correlations[module_name]
        print(f"  {module_name}: var={var:.6f}, r={corr['pearson_r']:.4f}")
    
    return {
        "module_variances": module_variances,
        "module_alpha_correlations": module_alpha_correlations
    }

def main():
    """Execute gene co-expression module analysis"""
    
    print("="*70)
    print("GENE CO-EXPRESSION MODULE ANALYSIS WITH QAC")
    print("="*70)
    print("\nResearch Question:")
    print("  Does QAC reveal quantum-coherent gene regulatory modules within cells?")
    print("\nApproach:")
    print("  1. Group genes into 8 functional modules (matching num_qubits=8)")
    print("  2. Compute module expression vectors for each cell")
    print("  3. Apply QAC to module vectors (each module = 1 quantum state)")
    print("  4. Compare module coherence across alpha regimes")
    print("\nCritical: NO BIAS - using DEFAULT QAC parameters, letting math speak")
    print("="*70)
    
    # Load data
    adata = load_pbmc_data()
    
    # Assign genes to modules
    module_genes = assign_genes_to_modules(adata)
    
    # Compute module expression
    module_expr, module_names = compute_module_expression(adata, module_genes)
    
    # Analyze by alpha regime
    regime_results = analyze_cells_by_alpha_regime(adata, module_expr, module_names)
    
    # Analyze module-specific coherence
    module_results = analyze_module_specific_coherence(adata, module_expr, module_names)
    
    # Combine results
    full_results = {
        "research_question": "Does QAC reveal quantum-coherent gene regulatory modules within cells?",
        "methodology": {
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_modules": len(module_names),
            "module_names": module_names,
            "qac_parameters": {
                "num_qubits": 8,
                "layers": 3,
                "reinforce_rate": 0.05,
                "note": "DEFAULT parameters - no bias injection"
            }
        },
        "alpha_regime_analysis": regime_results,
        "module_specific_analysis": module_results
    }
    
    # Save results
    output_path = Path("gene_coexpression_qac_results.json")
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)
    
    print("\n" + "="*70)
    print(f"Results saved to: {output_path}")
    print("="*70)
    
    # Print summary
    print("\nKEY FINDINGS:")
    stat_tests = regime_results["statistical_tests"]
    
    print(f"\n1. Alpha Regime Comparison:")
    print(f"   Low-α stability: {regime_results['low_alpha']['mean_stability']:.6f}")
    print(f"   High-α stability: {regime_results['high_alpha']['mean_stability']:.6f}")
    print(f"   Mann-Whitney p = {stat_tests['mann_whitney_p']:.6f}")
    
    print(f"\n2. Correlation with Alpha:")
    print(f"   Pearson r = {stat_tests['pearson_r']:.4f} (p = {stat_tests['pearson_p']:.6f})")
    
    print(f"\n3. Module Variance Analysis:")
    sorted_vars = sorted(
        module_results["module_variances"].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    print(f"   Most variable: {sorted_vars[0][0]} (var={sorted_vars[0][1]:.6f})")
    print(f"   Least variable: {sorted_vars[-1][0]} (var={sorted_vars[-1][1]:.6f})")
    
    print("\n" + "="*70)
    print("INTERPRETATION:")
    
    if stat_tests['mann_whitney_p'] < 0.05:
        print("\n✓ QAC REVEALS significant differences in module coherence across alpha regimes!")
        print("  → Gene regulatory modules show regime-specific quantum structure")
    else:
        print("\n✗ No significant difference in module coherence across alpha regimes")
        print("  → Module quantum structure may be independent of cellular state")
    
    if abs(stat_tests['pearson_r']) > 0.2 and stat_tests['pearson_p'] < 0.05:
        if stat_tests['pearson_r'] > 0:
            print("\n✓ Module coherence INCREASES with alpha")
            print("  → High-α cells (APCs) have more quantum-coherent gene modules")
        else:
            print("\n✓ Module coherence DECREASES with alpha")
            print("  → Low-α cells (T cells) have more quantum-coherent gene modules")
    else:
        print("\n→ Module coherence shows no strong correlation with alpha")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
