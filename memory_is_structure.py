"""
Memory is Structure: Topological Analysis of Gene Module Networks

Core Hypothesis: Memory is encoded in STRUCTURAL RELATIONSHIPS, not just values
- T cell memory = conserved correlation structure between modules
- APC plasticity = variable/reorganizing structure
- Identity = topological invariants preserved across cells
- Collapse preserves structural memory (certain relationships persist)

Research Questions:
1. Do T cells share the SAME structural patterns (rigid memory)?
2. Do APCs have VARIABLE structures (plastic, adaptive)?
3. Which module relationships define cellular identity?
4. Does QAC collapse preserve structural memory?

Critical: NO BIAS - let structure reveal itself
"""

import numpy as np
import scanpy as sc
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform

# Add stack to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "stack"))

from tace.core import QACEngine

# Gene module definitions
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
    """Load PBMC3K data"""
    print("Loading PBMC3K dataset...")
    adata = sc.read_h5ad("data/pbmc3k_with_tace_alpha.h5ad")
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
    """Compute average expression for each module"""
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

def compute_structural_fingerprint(module_vector: np.ndarray) -> np.ndarray:
    """
    Compute correlation structure between modules as a structural fingerprint
    
    Returns: correlation matrix (8x8) representing structural relationships
    """
    # For a single cell, we can't compute correlations across modules
    # Instead, compute pairwise relationships based on value ratios and differences
    n_modules = len(module_vector)
    structure = np.zeros((n_modules, n_modules))
    
    for i in range(n_modules):
        for j in range(n_modules):
            if i == j:
                structure[i, j] = 1.0
            else:
                # Structural relationship: ratio of expressions (log-scale)
                # This captures relative activity between modules
                val_i = module_vector[i] + 1e-6
                val_j = module_vector[j] + 1e-6
                structure[i, j] = np.log2(val_i / val_j)
    
    return structure

def compute_population_structure(module_expr: np.ndarray, cell_indices: np.ndarray) -> np.ndarray:
    """
    Compute correlation structure across a population of cells
    
    Returns: correlation matrix showing which modules co-vary
    """
    subset = module_expr[cell_indices]
    
    # Compute pairwise correlations between modules
    n_modules = subset.shape[1]
    corr_matrix = np.zeros((n_modules, n_modules))
    
    for i in range(n_modules):
        for j in range(n_modules):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                r, _ = pearsonr(subset[:, i], subset[:, j])
                corr_matrix[i, j] = r
    
    return corr_matrix

def measure_structural_distance(struct1: np.ndarray, struct2: np.ndarray) -> float:
    """
    Measure how different two structures are
    
    Uses Frobenius norm of difference matrices
    """
    return np.linalg.norm(struct1 - struct2, 'fro')

def analyze_structural_memory_by_regime(adata, module_expr: np.ndarray, module_names: List[str]):
    """
    Test if T cells have rigid structural memory vs APCs having plastic structure
    
    Hypothesis: Low-α cells share similar structure (memory)
                High-α cells have variable structure (plasticity)
    """
    print("\n" + "="*70)
    print("STRUCTURAL MEMORY ANALYSIS BY ALPHA REGIME")
    print("="*70)
    
    alphas = adata.obs['alpha_tace'].values
    
    low_mask = alphas < 0.3
    high_mask = alphas >= 0.7
    
    print(f"\nSampling cells:")
    print(f"  Low-α: {low_mask.sum()} available")
    print(f"  High-α: {high_mask.sum()} available")
    
    # Sample cells
    np.random.seed(42)
    n_samples = 100
    
    low_indices = np.where(low_mask)[0]
    high_indices = np.where(high_mask)[0]
    
    sampled_low = np.random.choice(low_indices, min(n_samples, len(low_indices)), replace=False)
    sampled_high = np.random.choice(high_indices, min(n_samples, len(high_indices)), replace=False)
    
    print(f"\nComputing structural fingerprints...")
    
    # Compute structural fingerprint for each cell
    low_structures = []
    high_structures = []
    
    for idx in sampled_low:
        struct = compute_structural_fingerprint(module_expr[idx])
        low_structures.append(struct)
    
    for idx in sampled_high:
        struct = compute_structural_fingerprint(module_expr[idx])
        high_structures.append(struct)
    
    # Measure structural variability within each regime
    print("\n--- LOW-α CELLS (T cells) ---")
    
    # Compute pairwise structural distances within low-α
    low_distances = []
    for i in range(len(low_structures)):
        for j in range(i+1, len(low_structures)):
            dist = measure_structural_distance(low_structures[i], low_structures[j])
            low_distances.append(dist)
    
    low_distances = np.array(low_distances)
    print(f"Structural variability (within-regime distances):")
    print(f"  Mean: {low_distances.mean():.4f}")
    print(f"  Std: {low_distances.std():.4f}")
    print(f"  Range: [{low_distances.min():.4f}, {low_distances.max():.4f}]")
    
    # Compute population-level correlation structure
    low_pop_structure = compute_population_structure(module_expr, sampled_low)
    print(f"\nPopulation correlation structure:")
    for i, mod in enumerate(module_names):
        correlations = [f"{low_pop_structure[i,j]:+.2f}" for j in range(len(module_names)) if i != j]
        print(f"  {mod:20s} → " + " ".join(correlations))
    
    print("\n--- HIGH-α CELLS (APCs) ---")
    
    # Compute pairwise structural distances within high-α
    high_distances = []
    for i in range(len(high_structures)):
        for j in range(i+1, len(high_structures)):
            dist = measure_structural_distance(high_structures[i], high_structures[j])
            high_distances.append(dist)
    
    high_distances = np.array(high_distances)
    print(f"Structural variability (within-regime distances):")
    print(f"  Mean: {high_distances.mean():.4f}")
    print(f"  Std: {high_distances.std():.4f}")
    print(f"  Range: [{high_distances.min():.4f}, {high_distances.max():.4f}]")
    
    # Compute population-level correlation structure
    high_pop_structure = compute_population_structure(module_expr, sampled_high)
    print(f"\nPopulation correlation structure:")
    for i, mod in enumerate(module_names):
        correlations = [f"{high_pop_structure[i,j]:+.2f}" for j in range(len(module_names)) if i != j]
        print(f"  {mod:20s} → " + " ".join(correlations))
    
    # Statistical comparison
    print("\n" + "="*70)
    print("STRUCTURAL MEMORY vs PLASTICITY")
    print("="*70)
    
    from scipy.stats import mannwhitneyu
    
    u_stat, p_value = mannwhitneyu(low_distances, high_distances, alternative='two-sided')
    
    print(f"\nMann-Whitney U test:")
    print(f"  U = {u_stat:.2f}, p = {p_value:.6f}")
    
    if p_value < 0.05:
        if low_distances.mean() < high_distances.mean():
            print("\n  ✓ LOW-α cells have MORE RIGID structure (memory)!")
            print("    → T cells share similar structural patterns")
            print("    → Structure encodes T cell identity memory")
        else:
            print("\n  ✓ HIGH-α cells have MORE RIGID structure!")
            print("    → APCs may have constrained activation structures")
    else:
        print("\n  → No significant difference in structural variability")
    
    # Measure cross-regime distances
    cross_distances = []
    for low_struct in low_structures[:20]:  # Sample to avoid O(n²) explosion
        for high_struct in high_structures[:20]:
            dist = measure_structural_distance(low_struct, high_struct)
            cross_distances.append(dist)
    
    cross_distances = np.array(cross_distances)
    
    print(f"\nCross-regime structural distance:")
    print(f"  Low-α ↔ High-α: {cross_distances.mean():.4f} ± {cross_distances.std():.4f}")
    print(f"  Within Low-α: {low_distances.mean():.4f}")
    print(f"  Within High-α: {high_distances.mean():.4f}")
    
    if cross_distances.mean() > max(low_distances.mean(), high_distances.mean()):
        print("\n  ✓ T cells and APCs have DISTINCT structural signatures!")
        print("    → Structure defines cell type identity")
    
    return {
        "low_alpha": {
            "mean_structural_distance": float(low_distances.mean()),
            "std_structural_distance": float(low_distances.std()),
            "population_structure": low_pop_structure.tolist()
        },
        "high_alpha": {
            "mean_structural_distance": float(high_distances.mean()),
            "std_structural_distance": float(high_distances.std()),
            "population_structure": high_pop_structure.tolist()
        },
        "cross_regime_distance": float(cross_distances.mean()),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(p_value)
    }

def test_collapse_preserves_structure(adata, module_expr: np.ndarray, module_names: List[str]):
    """
    Test if QAC collapse preserves structural relationships
    
    Memory hypothesis: Structure should be preserved through collapse
    """
    print("\n" + "="*70)
    print("TESTING: DOES COLLAPSE PRESERVE STRUCTURE?")
    print("="*70)
    
    print("\nHypothesis: If memory is structure, collapse should preserve key relationships")
    
    # Sample diverse cells
    np.random.seed(42)
    n_samples = 50
    sampled_indices = np.random.choice(adata.n_obs, n_samples, replace=False)
    
    qac = QACEngine(num_qubits=8)
    
    structural_preservation = []
    
    for idx in sampled_indices:
        # Before collapse
        before = module_expr[idx]
        before_structure = compute_structural_fingerprint(before)
        
        # After collapse
        after = np.array(qac.stabilize(before))
        after_structure = compute_structural_fingerprint(after)
        
        # Measure how much structure changed
        structure_change = measure_structural_distance(before_structure, after_structure)
        structural_preservation.append(structure_change)
    
    structural_preservation = np.array(structural_preservation)
    
    print(f"\nStructural change through collapse:")
    print(f"  Mean: {structural_preservation.mean():.4f}")
    print(f"  Std: {structural_preservation.std():.4f}")
    print(f"  Range: [{structural_preservation.min():.4f}, {structural_preservation.max():.4f}]")
    
    # Compare to random baseline (shuffling modules would destroy structure)
    random_changes = []
    for idx in sampled_indices[:10]:  # Smaller sample for baseline
        before = module_expr[idx]
        before_structure = compute_structural_fingerprint(before)
        
        # Random shuffle
        shuffled = np.random.permutation(before)
        shuffled_structure = compute_structural_fingerprint(shuffled)
        
        change = measure_structural_distance(before_structure, shuffled_structure)
        random_changes.append(change)
    
    random_changes = np.array(random_changes)
    
    print(f"\nRandom baseline (shuffled modules):")
    print(f"  Mean: {random_changes.mean():.4f}")
    
    if structural_preservation.mean() < random_changes.mean():
        print("\n  ✓ COLLAPSE PRESERVES STRUCTURE!")
        print("    → Structural relationships maintained through quantum operation")
        print("    → Memory (structure) survives collapse")
    else:
        print("\n  ✗ Collapse does not preserve structure")
    
    return {
        "collapse_structure_change_mean": float(structural_preservation.mean()),
        "collapse_structure_change_std": float(structural_preservation.std()),
        "random_baseline_mean": float(random_changes.mean())
    }

def identify_structural_motifs(adata, module_expr: np.ndarray, module_names: List[str]):
    """
    Identify structural motifs that define cell types
    
    Which module relationships are conserved within but differ between types?
    """
    print("\n" + "="*70)
    print("IDENTIFYING STRUCTURAL MOTIFS")
    print("="*70)
    
    alphas = adata.obs['alpha_tace'].values
    
    low_mask = alphas < 0.3
    high_mask = alphas >= 0.7
    
    # Compute population structures
    low_indices = np.where(low_mask)[0]
    high_indices = np.where(high_mask)[0]
    
    low_structure = compute_population_structure(module_expr, low_indices)
    high_structure = compute_population_structure(module_expr, high_indices)
    
    # Find module pairs with most different correlations
    print("\nModule relationships that DIFFER between T cells and APCs:")
    print("(These define cell type identity)\n")
    
    differences = []
    for i in range(len(module_names)):
        for j in range(i+1, len(module_names)):
            diff = abs(low_structure[i, j] - high_structure[i, j])
            differences.append({
                "module_i": module_names[i],
                "module_j": module_names[j],
                "low_corr": low_structure[i, j],
                "high_corr": high_structure[i, j],
                "difference": diff
            })
    
    differences.sort(key=lambda x: x["difference"], reverse=True)
    
    for motif in differences[:10]:
        print(f"  {motif['module_i']:20s} ↔ {motif['module_j']:20s}")
        print(f"    Low-α:  {motif['low_corr']:+.3f}")
        print(f"    High-α: {motif['high_corr']:+.3f}")
        print(f"    Δ:      {motif['difference']:.3f}")
        print()
    
    # Find conserved relationships (similar in both)
    print("Module relationships CONSERVED across cell types:")
    print("(Universal structural memory)\n")
    
    conserved = [m for m in differences if m["difference"] < 0.2]
    conserved.sort(key=lambda x: abs(x["low_corr"]), reverse=True)
    
    for motif in conserved[:5]:
        print(f"  {motif['module_i']:20s} ↔ {motif['module_j']:20s}")
        print(f"    Correlation: {motif['low_corr']:+.3f} (both regimes)")
        print()
    
    return {
        "distinctive_motifs": differences[:10],
        "conserved_motifs": conserved[:5]
    }

def main():
    """Execute structural memory analysis"""
    
    print("="*70)
    print("MEMORY IS STRUCTURE")
    print("="*70)
    print("\nCore Hypothesis:")
    print("  Memory is encoded in STRUCTURAL RELATIONSHIPS between modules")
    print("  - Not just expression values, but correlation patterns")
    print("  - T cell identity = conserved structural motifs")
    print("  - APC plasticity = reorganizing structure")
    print("\nResearch Questions:")
    print("  1. Do T cells have rigid structural memory?")
    print("  2. Do APCs have plastic/variable structure?")
    print("  3. Does collapse preserve structure (memory)?")
    print("  4. Which structural motifs define cell types?")
    print("\nCritical: NO BIAS - letting structure speak for itself")
    print("="*70)
    
    # Load data
    adata = load_pbmc_data()
    module_genes = assign_genes_to_modules(adata)
    module_expr, module_names = compute_module_expression(adata, module_genes)
    
    print(f"\nAnalyzing structural relationships in {adata.n_obs} cells")
    
    # Analyze structural memory by regime
    regime_results = analyze_structural_memory_by_regime(adata, module_expr, module_names)
    
    # Test if collapse preserves structure
    preservation_results = test_collapse_preserves_structure(adata, module_expr, module_names)
    
    # Identify structural motifs
    motif_results = identify_structural_motifs(adata, module_expr, module_names)
    
    # Combine results
    full_results = {
        "hypothesis": "Memory is Structure - identity encoded in module relationships",
        "methodology": {
            "structural_fingerprint": "log-ratio matrix of module expressions",
            "population_structure": "correlation matrix across cells",
            "structural_distance": "Frobenius norm of difference matrices"
        },
        "structural_memory_analysis": regime_results,
        "collapse_preservation": preservation_results,
        "structural_motifs": motif_results
    }
    
    # Save results
    output_path = Path("memory_is_structure_results.json")
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)
    
    print("\n" + "="*70)
    print(f"Results saved to: {output_path}")
    print("="*70)
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    print(f"\n1. STRUCTURAL MEMORY:")
    print(f"   Low-α variability: {regime_results['low_alpha']['mean_structural_distance']:.4f}")
    print(f"   High-α variability: {regime_results['high_alpha']['mean_structural_distance']:.4f}")
    print(f"   p-value: {regime_results['mann_whitney_p']:.6f}")
    
    print(f"\n2. STRUCTURE PRESERVATION THROUGH COLLAPSE:")
    print(f"   Collapse change: {preservation_results['collapse_structure_change_mean']:.4f}")
    print(f"   Random baseline: {preservation_results['random_baseline_mean']:.4f}")
    
    print(f"\n3. CROSS-REGIME STRUCTURAL DISTANCE:")
    print(f"   T cells ↔ APCs: {regime_results['cross_regime_distance']:.4f}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
