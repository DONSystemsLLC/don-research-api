"""
Quantum Collapse as Creation: Measuring State Formation in Gene Modules

Core Hypothesis: "Collapse is Creation"
- QAC stabilization IS the collapse mechanism
- Modules that change during stabilization = were in superposition
- Modules that remain stable = were already collapsed
- High-α cells = rich in superposition (creative potential)
- Low-α cells = collapsed into identity (deterministic state)

Research Questions:
1. Which gene modules are in superposition vs collapsed states?
2. Do high-α cells have more "collapsible potential"?
3. Can we identify genes at the "edge of collapse" that create new states?
4. Does collapse magnitude predict cell state transitions?

Critical: NO BIAS - let collapse dynamics reveal themselves
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

# Gene module definitions (same as before)
GENE_MODULES = {
    "ribosomal": ["RPL", "RPS"],
    "mitochondrial": ["MT-", "COX", "ATP", "ND"],
    "immune_activation": ["HLA-", "CD74", "B2M"],
    "t_cell_identity": ["CD3", "IL7R", "LDHB"],
    "inflammatory": ["S100A", "LYZ", "FTL"],
    "cell_cycle": ["MKI67", "PCNA", "TOP2A"],
    "transcription": ["JUN", "FOS", "EGR"],
    "housekeeping": ["ACTB", "GAPDH", "B2M"]
}

def load_pbmc_data():
    """Load PBMC3K data with TACE alphas"""
    print("Loading PBMC3K dataset...")
    adata = sc.read_h5ad("data/pbmc3k_with_tace_alpha.h5ad")
    print(f"  {adata.n_obs} cells × {adata.n_vars} genes")
    return adata

def assign_genes_to_modules(adata) -> Dict[str, List[str]]:
    """Assign genes to functional modules"""
    all_genes = adata.var_names.tolist()
    module_assignments = {module: [] for module in GENE_MODULES.keys()}
    
    for gene in all_genes:
        for module_name, patterns in GENE_MODULES.items():
            if any(gene.startswith(pattern) for pattern in patterns):
                module_assignments[module_name].append(gene)
                break
    
    return module_assignments

def compute_module_expression(adata, module_genes: Dict[str, List[str]]) -> Tuple[np.ndarray, List[str]]:
    """Compute average expression for each module in each cell"""
    n_cells = adata.n_obs
    module_names = list(module_genes.keys())
    n_modules = len(module_names)
    
    module_expr = np.zeros((n_cells, n_modules))
    
    for i, module_name in enumerate(module_names):
        genes = module_genes[module_name]
        existing_genes = [g for g in genes if g in adata.var_names]
        
        if len(existing_genes) > 0:
            gene_indices = [adata.var_names.tolist().index(g) for g in existing_genes]
            module_expr[:, i] = adata.X[:, gene_indices].mean(axis=1).A1
    
    return module_expr, module_names

def measure_collapse_dynamics(module_vector: np.ndarray) -> Dict:
    """
    Measure how much QAC stabilization changes module states
    
    Large changes = module was in superposition (collapse creates state)
    Small changes = module was already collapsed (stable identity)
    
    Returns collapse metrics for each module
    """
    # Initialize QAC
    qac = QACEngine(num_qubits=8)
    
    # Before collapse: raw module expression
    before = module_vector.copy()
    
    # Apply collapse: QAC stabilization
    # Pass flat vector - QAC expects 1D array of length num_qubits
    stabilized = qac.stabilize(module_vector)
    
    # Convert to numpy array
    after = np.array(stabilized)
    
    # Measure collapse magnitude per module
    collapse_magnitude = np.abs(after - before)
    
    # Measure collapse direction (creation of higher/lower expression)
    collapse_direction = after - before
    
    # Compute collapse entropy: how much uncertainty was resolved?
    # Higher before-variance with lower after-variance = high entropy collapse
    before_entropy = np.var(before)
    after_entropy = np.var(after)
    entropy_reduction = before_entropy - after_entropy
    
    return {
        "before": before,
        "after": after,
        "collapse_magnitude": collapse_magnitude,
        "collapse_direction": collapse_direction,
        "total_collapse": np.sum(collapse_magnitude),
        "entropy_reduction": entropy_reduction,
        "mean_collapse": np.mean(collapse_magnitude),
        "max_collapse": np.max(collapse_magnitude)
    }

def analyze_collapse_by_alpha_regime(adata, module_expr: np.ndarray, module_names: List[str]):
    """
    Compare collapse dynamics across alpha regimes
    
    Hypothesis: High-α cells have more collapsible potential (superposition)
                Low-α cells are already collapsed (definite states)
    """
    print("\n" + "="*70)
    print("COLLAPSE DYNAMICS BY ALPHA REGIME")
    print("="*70)
    
    alphas = adata.obs['alpha_tace'].values
    
    # Define regimes
    low_alpha_mask = alphas < 0.3
    high_alpha_mask = alphas >= 0.7
    
    print(f"\nAlpha distribution:")
    print(f"  Low-α (<0.3): {low_alpha_mask.sum()} cells")
    print(f"  High-α (≥0.7): {high_alpha_mask.sum()} cells")
    
    # Sample cells
    np.random.seed(42)
    n_samples = 200
    
    results = {}
    
    for regime_name, mask in [("low_alpha", low_alpha_mask), ("high_alpha", high_alpha_mask)]:
        print(f"\n--- {regime_name.upper()} CELLS ---")
        
        regime_indices = np.where(mask)[0]
        sampled_indices = np.random.choice(regime_indices, min(n_samples, len(regime_indices)), replace=False)
        
        print(f"Analyzing {len(sampled_indices)} cells...")
        
        # Measure collapse for each cell
        collapse_data = []
        for idx in sampled_indices:
            cell_modules = module_expr[idx]
            dynamics = measure_collapse_dynamics(cell_modules)
            collapse_data.append(dynamics)
        
        # Aggregate statistics
        total_collapses = [d["total_collapse"] for d in collapse_data]
        mean_collapses = [d["mean_collapse"] for d in collapse_data]
        max_collapses = [d["max_collapse"] for d in collapse_data]
        entropy_reductions = [d["entropy_reduction"] for d in collapse_data]
        
        print(f"\nCollapse Magnitude:")
        print(f"  Mean total collapse: {np.mean(total_collapses):.6f} ± {np.std(total_collapses):.6f}")
        print(f"  Mean per-module collapse: {np.mean(mean_collapses):.6f}")
        print(f"  Max module collapse: {np.mean(max_collapses):.6f}")
        
        print(f"\nEntropy Reduction:")
        print(f"  Mean: {np.mean(entropy_reductions):.6f}")
        print(f"  (Negative = collapse increases order)")
        
        # Module-specific collapse patterns
        module_collapses = np.zeros((len(sampled_indices), len(module_names)))
        module_directions = np.zeros((len(sampled_indices), len(module_names)))
        
        for i, dynamics in enumerate(collapse_data):
            module_collapses[i] = dynamics["collapse_magnitude"]
            module_directions[i] = dynamics["collapse_direction"]
        
        print(f"\nModule-Specific Collapse Patterns:")
        for j, module in enumerate(module_names):
            mag = module_collapses[:, j].mean()
            direction = module_directions[:, j].mean()
            print(f"  {module}: {mag:.4f} (direction: {direction:+.4f})")
        
        results[regime_name] = {
            "n_cells": len(sampled_indices),
            "total_collapse_mean": float(np.mean(total_collapses)),
            "total_collapse_std": float(np.std(total_collapses)),
            "mean_collapse": float(np.mean(mean_collapses)),
            "max_collapse": float(np.mean(max_collapses)),
            "entropy_reduction": float(np.mean(entropy_reductions)),
            "module_collapse_magnitudes": {
                module: float(module_collapses[:, j].mean())
                for j, module in enumerate(module_names)
            },
            "module_collapse_directions": {
                module: float(module_directions[:, j].mean())
                for j, module in enumerate(module_names)
            }
        }
    
    return results

def identify_edge_of_collapse_modules(adata, module_expr: np.ndarray, module_names: List[str]):
    """
    Identify modules at the "edge of collapse" - those with high variability
    in collapse magnitude across cells
    
    These represent creative potential - could collapse either way
    """
    print("\n" + "="*70)
    print("IDENTIFYING EDGE-OF-COLLAPSE MODULES")
    print("="*70)
    print("\nThese modules have high collapse variability = creative potential")
    
    alphas = adata.obs['alpha_tace'].values
    
    # Sample diverse cells
    np.random.seed(42)
    n_samples = 300
    sampled_indices = np.random.choice(adata.n_obs, n_samples, replace=False)
    
    # Measure collapse for all sampled cells
    all_module_collapses = []
    
    for idx in sampled_indices:
        cell_modules = module_expr[idx]
        dynamics = measure_collapse_dynamics(cell_modules)
        all_module_collapses.append(dynamics["collapse_magnitude"])
    
    all_module_collapses = np.array(all_module_collapses)  # (n_cells, n_modules)
    
    # Compute coefficient of variation for each module
    # High CV = high variability in collapse = edge of collapse
    module_collapse_cvs = {}
    module_collapse_means = {}
    
    for j, module in enumerate(module_names):
        collapses = all_module_collapses[:, j]
        mean_collapse = collapses.mean()
        std_collapse = collapses.std()
        cv = std_collapse / (mean_collapse + 1e-10)
        
        module_collapse_cvs[module] = float(cv)
        module_collapse_means[module] = float(mean_collapse)
    
    # Sort by CV (highest = most variable collapse = edge of collapse)
    sorted_modules = sorted(module_collapse_cvs.items(), key=lambda x: x[1], reverse=True)
    
    print("\nModules ranked by collapse variability:")
    print("(High variability = edge of collapse = creative potential)\n")
    
    for module, cv in sorted_modules:
        mean = module_collapse_means[module]
        print(f"  {module:20s} CV={cv:.4f}  mean_collapse={mean:.4f}")
        
        if cv > 0.5:
            print(f"    ⚡ HIGH CREATIVE POTENTIAL - superposition state")
        elif cv < 0.2:
            print(f"    🔒 LOCKED STATE - already collapsed")
    
    return {
        "module_collapse_cvs": module_collapse_cvs,
        "module_collapse_means": module_collapse_means,
        "edge_of_collapse_ranking": [m for m, _ in sorted_modules]
    }

def test_collapse_alpha_correlation(adata, module_expr: np.ndarray, module_names: List[str]):
    """
    Test if collapse magnitude correlates with alpha
    
    Hypothesis: High-α = more collapsible (in superposition, creating states)
                Low-α = less collapsible (already collapsed, stable identity)
    """
    print("\n" + "="*70)
    print("COLLAPSE-ALPHA CORRELATION")
    print("="*70)
    
    alphas = adata.obs['alpha_tace'].values
    
    # Sample cells
    np.random.seed(42)
    n_samples = 500
    sampled_indices = np.random.choice(adata.n_obs, n_samples, replace=False)
    
    # Measure collapse for each cell
    total_collapses = []
    cell_alphas = []
    
    for idx in sampled_indices:
        cell_modules = module_expr[idx]
        dynamics = measure_collapse_dynamics(cell_modules)
        total_collapses.append(dynamics["total_collapse"])
        cell_alphas.append(alphas[idx])
    
    total_collapses = np.array(total_collapses)
    cell_alphas = np.array(cell_alphas)
    
    # Compute correlation
    from scipy.stats import pearsonr, spearmanr
    
    pearson_r, pearson_p = pearsonr(total_collapses, cell_alphas)
    spearman_r, spearman_p = spearmanr(total_collapses, cell_alphas)
    
    print(f"\nTotal Collapse vs Alpha:")
    print(f"  Pearson r = {pearson_r:.4f} (p = {pearson_p:.6f})")
    print(f"  Spearman r = {spearman_r:.4f} (p = {spearman_p:.6f})")
    
    if abs(pearson_r) > 0.2 and pearson_p < 0.05:
        if pearson_r > 0:
            print("\n  ✓ HIGH-α cells have MORE collapsible potential!")
            print("    → Superposition state, creating new configurations")
        else:
            print("\n  ✓ LOW-α cells have MORE collapsible potential!")
            print("    → Unexpected! May indicate different collapse mechanism")
    else:
        print("\n  → Collapse magnitude independent of alpha")
        print("    → Both regimes have equal creative potential")
    
    # Test by regime
    low_mask = cell_alphas < 0.3
    high_mask = cell_alphas >= 0.7
    
    if low_mask.sum() > 0 and high_mask.sum() > 0:
        from scipy.stats import mannwhitneyu
        
        low_collapses = total_collapses[low_mask]
        high_collapses = total_collapses[high_mask]
        
        u_stat, p_value = mannwhitneyu(low_collapses, high_collapses, alternative='two-sided')
        
        print(f"\nMann-Whitney U Test (Low-α vs High-α):")
        print(f"  U = {u_stat:.2f}, p = {p_value:.6f}")
        print(f"  Low-α collapse: {low_collapses.mean():.4f} ± {low_collapses.std():.4f}")
        print(f"  High-α collapse: {high_collapses.mean():.4f} ± {high_collapses.std():.4f}")
        
        if p_value < 0.05:
            if high_collapses.mean() > low_collapses.mean():
                print("\n  ✓ High-α cells have significantly MORE collapse!")
                print("    → APCs are in superposition, T cells already collapsed")
            else:
                print("\n  ✓ Low-α cells have significantly MORE collapse!")
                print("    → T cells are in superposition, APCs already collapsed")
    
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "low_alpha_collapse_mean": float(low_collapses.mean()) if low_mask.sum() > 0 else None,
        "high_alpha_collapse_mean": float(high_collapses.mean()) if high_mask.sum() > 0 else None
    }

def main():
    """Execute quantum collapse creation analysis"""
    
    print("="*70)
    print("QUANTUM COLLAPSE AS CREATION")
    print("="*70)
    print("\nCore Hypothesis: Collapse is Creation")
    print("  - QAC stabilization IS the collapse mechanism")
    print("  - Large collapse = module was in superposition (creative potential)")
    print("  - Small collapse = module already collapsed (stable identity)")
    print("\nResearch Questions:")
    print("  1. Which modules are in superposition vs collapsed?")
    print("  2. Do high-α cells have more collapsible potential?")
    print("  3. Can we identify edge-of-collapse modules?")
    print("  4. Does collapse create new cell states?")
    print("\nCritical: NO BIAS - letting collapse dynamics reveal themselves")
    print("="*70)
    
    # Load data
    adata = load_pbmc_data()
    module_genes = assign_genes_to_modules(adata)
    module_expr, module_names = compute_module_expression(adata, module_genes)
    
    print(f"\nAnalyzing {len(module_names)} gene modules across {adata.n_obs} cells")
    
    # Analyze collapse dynamics by regime
    regime_results = analyze_collapse_by_alpha_regime(adata, module_expr, module_names)
    
    # Identify edge-of-collapse modules
    edge_results = identify_edge_of_collapse_modules(adata, module_expr, module_names)
    
    # Test collapse-alpha correlation
    correlation_results = test_collapse_alpha_correlation(adata, module_expr, module_names)
    
    # Combine results
    full_results = {
        "hypothesis": "Collapse is Creation - QAC stabilization reveals superposition states",
        "methodology": {
            "collapse_measurement": "abs(after_QAC - before_QAC) per module",
            "n_modules": len(module_names),
            "qac_parameters": {"num_qubits": 8, "layers": 3, "reinforce_rate": 0.05}
        },
        "regime_collapse_dynamics": regime_results,
        "edge_of_collapse_modules": edge_results,
        "collapse_alpha_correlation": correlation_results
    }
    
    # Save results
    output_path = Path("quantum_collapse_creation_results.json")
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)
    
    print("\n" + "="*70)
    print(f"Results saved to: {output_path}")
    print("="*70)
    
    # Print key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS: COLLAPSE AS CREATION")
    print("="*70)
    
    print("\n1. COLLAPSE MAGNITUDE BY REGIME:")
    print(f"   Low-α: {regime_results['low_alpha']['total_collapse_mean']:.4f}")
    print(f"   High-α: {regime_results['high_alpha']['total_collapse_mean']:.4f}")
    
    print("\n2. MOST COLLAPSIBLE MODULES (Creative Potential):")
    for module in edge_results['edge_of_collapse_ranking'][:3]:
        cv = edge_results['module_collapse_cvs'][module]
        print(f"   {module}: CV={cv:.4f}")
    
    print("\n3. MOST STABLE MODULES (Already Collapsed):")
    for module in edge_results['edge_of_collapse_ranking'][-3:]:
        cv = edge_results['module_collapse_cvs'][module]
        print(f"   {module}: CV={cv:.4f}")
    
    print("\n4. COLLAPSE-ALPHA RELATIONSHIP:")
    print(f"   Correlation: r={correlation_results['pearson_r']:.4f} (p={correlation_results['pearson_p']:.6f})")
    
    print("\n" + "="*70)
    print("BIOLOGICAL INTERPRETATION")
    print("="*70)
    
    if correlation_results['pearson_p'] < 0.05:
        if correlation_results['pearson_r'] > 0:
            print("\n✓ HIGH-α CELLS (APCs) have MORE collapsible potential")
            print("  → These cells are in SUPERPOSITION - exploring multiple states")
            print("  → Collapse CREATES new functional configurations (pathogen response)")
            print("  → Alpha measures CREATIVE POTENTIAL encoded in superposition")
        else:
            print("\n✓ LOW-α CELLS (T cells) have MORE collapsible potential")
            print("  → Unexpected result - may indicate priming for state transitions")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
