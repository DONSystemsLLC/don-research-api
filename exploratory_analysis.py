#!/usr/bin/env python3
"""
Exploratory Analysis: Let the Math Speak

No theoretical assumptions. Just observe what happens when we apply
DON Stack operations to biological data and see what patterns emerge.
"""

import sys
import numpy as np
import json
from pathlib import Path
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from don_memory.adapters.don_stack_adapter import DONStackAdapter

def print_section(title):
    """Print section divider"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")

# Load the biological data and compress it
print_section("📊 DATA LOADING & COMPRESSION")

data_file = Path("real_pbmc_medium.json")
with open(data_file) as f:
    raw_data = json.load(f)

print(f"Loaded: {data_file}")
print(f"Cells: {len(raw_data['expression_matrix'])}")
print(f"Genes: {len(raw_data['gene_names'])}")

# Compress via DON-GPU to get cluster vectors
print("\nCompressing via DON-GPU...")

# Load token
with open("config/authorized_institutions.local.json") as f:
    config = json.load(f)
    token = config["texas_am_lab"]["token"]

response = requests.post(
    "http://localhost:8080/api/v1/genomics/compress",
    json={
        "gene_names": raw_data["gene_names"],
        "expression_matrix": raw_data["expression_matrix"],
        "cell_metadata": raw_data.get("cell_metadata"),
        "target_dimensions": 128,
        "apply_qac": False
    },
    headers={"Authorization": f"Bearer {token}"},
    timeout=60
)

if response.status_code != 200:
    print(f"Error: {response.status_code}")
    print(response.text)
    sys.exit(1)

compressed = response.json()
print(f"✓ Compression complete")
print(f"  Algorithm: {compressed['algorithm']}")
print(f"  Compression ratio: {compressed['compression_stats']['compression_ratio']:.1f}×")

# Extract cluster information
clusters = compressed['compressed_vectors']
cluster_sizes = compressed['compression_stats']['cluster_sizes']
print(f"\nClusters discovered: {len(clusters)}")
print(f"Cluster sizes: {cluster_sizes}")

# ============================================================================
# EXPLORATION 1: What does TACE alpha tell us about the data structure?
# ============================================================================
print_section("🔍 EXPLORATION 1: Alpha Patterns Across Clusters")

adapter = DONStackAdapter()

# Test each cluster individually
print("Testing individual clusters:\n")

alphas = []
cluster_info = []

for i, (cluster_vec, size) in enumerate(zip(clusters, cluster_sizes)):
    # Use first 8 dimensions as "state"
    state = cluster_vec[:8]
    
    # What alpha does TACE assign?
    alpha = adapter.tune_alpha(state, default_alpha=0.5)
    alphas.append(alpha)
    
    # Calculate some basic statistics about the state
    mean_val = np.mean(state)
    std_val = np.std(state)
    range_val = np.max(state) - np.min(state)
    
    cluster_info.append({
        'cluster': i,
        'size': size,
        'alpha': alpha,
        'mean': mean_val,
        'std': std_val,
        'range': range_val
    })
    
    print(f"Cluster {i} ({size} cells):")
    print(f"  State stats: mean={mean_val:.4f}, std={std_val:.4f}, range={range_val:.4f}")
    print(f"  TACE alpha: {alpha:.6f}")
    print()

# ============================================================================
# EXPLORATION 2: Is there a relationship between cluster properties and alpha?
# ============================================================================
print_section("📈 EXPLORATION 2: Correlation Analysis")

sizes = np.array([c['size'] for c in cluster_info])
alphas_arr = np.array([c['alpha'] for c in cluster_info])
means = np.array([c['mean'] for c in cluster_info])
stds = np.array([c['std'] for c in cluster_info])
ranges = np.array([c['range'] for c in cluster_info])

print("Correlations with TACE alpha:\n")

# Size vs Alpha
corr_size = np.corrcoef(sizes, alphas_arr)[0,1]
print(f"Cluster Size ↔ Alpha: {corr_size:.4f}")

# Mean vs Alpha
corr_mean = np.corrcoef(means, alphas_arr)[0,1]
print(f"State Mean ↔ Alpha: {corr_mean:.4f}")

# Std vs Alpha
corr_std = np.corrcoef(stds, alphas_arr)[0,1]
print(f"State Std Dev ↔ Alpha: {corr_std:.4f}")

# Range vs Alpha
corr_range = np.corrcoef(ranges, alphas_arr)[0,1]
print(f"State Range ↔ Alpha: {corr_range:.4f}")

print("\n" + "="*40)
print("What does this tell us?")
print("="*40)

if abs(corr_size) > 0.5:
    print(f"→ Strong correlation with cluster size ({corr_size:.3f})")
    print("  Alpha may encode population abundance")
elif abs(corr_mean) > 0.5:
    print(f"→ Strong correlation with state mean ({corr_mean:.3f})")
    print("  Alpha may encode signal strength")
elif abs(corr_std) > 0.5:
    print(f"→ Strong correlation with variability ({corr_std:.3f})")
    print("  Alpha may encode state complexity")
elif abs(corr_range) > 0.5:
    print(f"→ Strong correlation with dynamic range ({corr_range:.3f})")
    print("  Alpha may encode signal span")
else:
    print("→ Weak correlations across the board")
    print("  Alpha may encode something not captured by simple statistics")

# ============================================================================
# EXPLORATION 3: Mixed cluster states - does composition matter?
# ============================================================================
print_section("🧪 EXPLORATION 3: Cluster Composition Effects")

print("Testing mixed states from different cluster combinations:\n")

# Test 1: Common cells only (largest clusters)
common_state = clusters[0][:8]  # Cluster 0 (largest)
alpha_common = adapter.tune_alpha(common_state, 0.5)
print(f"Most common cells (C0, {cluster_sizes[0]} cells):")
print(f"  Alpha: {alpha_common:.6f}\n")

# Test 2: Rare cells (smallest clusters)
rare_idx = np.argmin(cluster_sizes)
rare_state = clusters[rare_idx][:8]
alpha_rare = adapter.tune_alpha(rare_state, 0.5)
print(f"Rarest cells (C{rare_idx}, {cluster_sizes[rare_idx]} cells):")
print(f"  Alpha: {alpha_rare:.6f}\n")

# Test 3: Mixed common + rare
mixed_state = [(c + r) / 2 for c, r in zip(common_state, rare_state)]
alpha_mixed = adapter.tune_alpha(mixed_state, 0.5)
print(f"50/50 mix (common + rare):")
print(f"  Alpha: {alpha_mixed:.6f}\n")

delta_from_avg = alpha_mixed - (alpha_common + alpha_rare) / 2
print(f"Δ from average: {delta_from_avg:+.6f}")

if abs(delta_from_avg) < 0.01:
    print("→ Linear combination: alpha of mix ≈ average of components")
    print("  TACE may be a linear operator")
else:
    print("→ Non-linear interaction: alpha of mix ≠ average")
    print("  TACE detects emergent properties from combinations")

# ============================================================================
# EXPLORATION 4: Sensitivity to perturbations
# ============================================================================
print_section("🎲 EXPLORATION 4: Perturbation Sensitivity")

print("How does TACE respond to small changes in state?\n")

base_cluster = clusters[0][:8]
base_alpha = adapter.tune_alpha(base_cluster, 0.5)

print(f"Baseline (unperturbed):")
print(f"  Alpha: {base_alpha:.6f}\n")

# Add noise at different scales
noise_scales = [0.01, 0.05, 0.1, 0.2]

for noise_scale in noise_scales:
    noise = np.random.randn(8) * noise_scale
    perturbed = (np.array(base_cluster) + noise).tolist()
    alpha_perturbed = adapter.tune_alpha(perturbed, 0.5)
    
    delta = alpha_perturbed - base_alpha
    print(f"Noise scale {noise_scale:.2f}:")
    print(f"  Alpha: {alpha_perturbed:.6f} (Δ {delta:+.6f})")

# ============================================================================
# EXPLORATION 5: Dimensionality effects
# ============================================================================
print_section("📐 EXPLORATION 5: Dimensional Structure")

print("Does the number of dimensions affect TACE behavior?\n")

test_cluster = clusters[0]

for n_dims in [2, 4, 8, 16, 32, 64]:
    if n_dims <= len(test_cluster):
        state = test_cluster[:n_dims]
        alpha = adapter.tune_alpha(state, 0.5)
        
        # Calculate information content
        entropy = -np.sum(np.abs(state) * np.log(np.abs(state) + 1e-10))
        
        print(f"{n_dims:2d} dimensions:")
        print(f"  Alpha: {alpha:.6f}")
        print(f"  Entropy: {entropy:.4f}")
        print()

# ============================================================================
# EXPLORATION 6: Iterative dynamics
# ============================================================================
print_section("🔄 EXPLORATION 6: Iterative Behavior")

print("What happens if we repeatedly apply TACE?\n")

test_state = clusters[0][:8]
current_alpha = 0.5
trajectory = [current_alpha]

print(f"Starting alpha: {current_alpha:.6f}\n")

for iteration in range(10):
    # Use current state to get new alpha
    new_alpha = adapter.tune_alpha(test_state, current_alpha)
    trajectory.append(new_alpha)
    
    delta = new_alpha - current_alpha
    print(f"Iteration {iteration+1}: {current_alpha:.6f} → {new_alpha:.6f} (Δ {delta:+.6f})")
    
    # Check for convergence
    if abs(delta) < 1e-6:
        print(f"\n✓ Converged after {iteration+1} iterations")
        break
    
    current_alpha = new_alpha
else:
    print(f"\n⚠ Did not converge after 10 iterations")

# Analyze trajectory
trajectory = np.array(trajectory)
if len(trajectory) > 2:
    changes = np.diff(trajectory)
    print(f"\nTrajectory analysis:")
    print(f"  Total change: {trajectory[-1] - trajectory[0]:+.6f}")
    print(f"  Mean step size: {np.mean(np.abs(changes)):.6f}")
    print(f"  Max step size: {np.max(np.abs(changes)):.6f}")

# ============================================================================
# SYNTHESIS: What patterns emerged?
# ============================================================================
print_section("💡 OBSERVED PATTERNS")

print("Without imposing theory, here's what the data shows:\n")

# 1. Alpha distribution
print("1. Alpha Distribution:")
print(f"   Range: [{np.min(alphas_arr):.6f}, {np.max(alphas_arr):.6f}]")
print(f"   Mean: {np.mean(alphas_arr):.6f}")
print(f"   Std Dev: {np.std(alphas_arr):.6f}")

if np.std(alphas_arr) < 0.001:
    print("   → Nearly uniform across all clusters")
else:
    print("   → Varies significantly across clusters")

# 2. Strongest relationship
print("\n2. Strongest Correlations:")
correlations = {
    'size': corr_size,
    'mean': corr_mean,
    'std': corr_std,
    'range': corr_range
}
strongest = max(correlations.items(), key=lambda x: abs(x[1]))
print(f"   {strongest[0].capitalize()}: {strongest[1]:.4f}")

# 3. Linearity
print("\n3. Operator Properties:")
if abs(delta_from_avg) < 0.01:
    print("   → Appears linear (superposition holds)")
else:
    print("   → Appears non-linear (emergent behavior)")

# 4. Convergence
print("\n4. Dynamic Behavior:")
print(f"   → Converges in {len(trajectory)-1} iterations")
print(f"   → Equilibrium at α = {trajectory[-1]:.6f}")

print("\n" + "="*70)
print("Analysis complete. Math has spoken.")
print("="*70 + "\n")
