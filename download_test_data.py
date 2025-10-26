#!/usr/bin/env python3
"""
Generate realistic single-cell RNA-seq test data for DON Stack QAC endpoints.
Creates synthetic but realistic PBMC-like data with proper biological structure.
"""

import numpy as np
import os
import json

API_TOKEN = os.getenv("DON_API_TOKEN", "demo_token")

def generate_realistic_gene_names(n_genes=2000):
    """Generate realistic gene names with known markers"""
    print("  Generating realistic gene names...")
    
    # Known immune cell marker genes (real gene symbols)
    marker_genes = [
        # T cell markers
        "CD3D", "CD3E", "CD3G", "CD8A", "CD8B", "CD4", "FOXP3", "IL2RA",
        "CTLA4", "PDCD1", "LAG3", "TIM3", "TIGIT", "GZMB", "PRF1",
        
        # B cell markers  
        "CD19", "CD20", "MS4A1", "CD79A", "CD79B", "PAX5", "BCL6", "IGHM",
        "IGHD", "CR2", "FCRLA", "BLK", "BANK1", "FCRL4", "SPIB",
        
        # Monocyte/Macrophage markers
        "CD14", "CD16", "LYZ", "S100A8", "S100A9", "CSF1R", "FCGR3A", 
        "CX3CR1", "CCR2", "ITGAM", "ADGRE1", "MSR1", "MRC1", "ARG1",
        
        # NK cell markers
        "GNLY", "NKG7", "KLRD1", "KLRF1", "NCR1", "FCGR3A", "KIR2DL1",
        "KIR2DL3", "KIR3DL1", "KLRB1", "KLRC1", "KLRC2", "KLRC3",
        
        # Dendritic cell markers
        "CLEC9A", "XCR1", "BATF3", "IRF8", "CLEC4C", "LILRA4", "IRF7",
        "GZMB", "IL3RA", "NRP1", "CLEC10A", "CD1C", "FCER1A",
        
        # Platelet markers
        "PPBP", "PF4", "GP9", "ITGA2B", "ITGB3", "SELP", "TREML1",
        
        # Housekeeping genes
        "GAPDH", "ACTB", "B2M", "HPRT1", "TBP", "GUSB", "RPL32", "YWHAZ",
        "SDHA", "UBC", "RPL13A", "RPS18", "PPIA", "HMBS", "PGK1"
    ]
    
    # Generate remaining gene names in ENSEMBL format
    remaining = n_genes - len(marker_genes)
    ensembl_genes = [f"ENSG{i:08d}" for i in range(remaining)]
    
    # Combine and shuffle
    all_genes = marker_genes + ensembl_genes
    np.random.shuffle(all_genes)
    
    return all_genes[:n_genes]

def generate_realistic_expression_matrix(n_cells, n_genes, seed=42):
    """Generate realistic single-cell expression matrix with biological structure"""
    print(f"  Generating expression matrix ({n_cells} cells × {n_genes} genes)...")
    
    np.random.seed(seed)
    
    # Define cell types and their proportions (realistic PBMC composition)
    cell_types = {
        "T_CD4": 0.35,      # CD4+ T cells  
        "T_CD8": 0.20,      # CD8+ T cells
        "B_cells": 0.15,    # B cells
        "Monocytes": 0.20,  # Monocytes
        "NK_cells": 0.08,   # NK cells
        "Dendritic": 0.02   # Dendritic cells
    }
    
    # Calculate cells per type
    cells_per_type = {}
    cumulative = 0
    for cell_type, prop in cell_types.items():
        n_type_cells = int(n_cells * prop)
        cells_per_type[cell_type] = (cumulative, cumulative + n_type_cells)
        cumulative += n_type_cells
    
    # Initialize expression matrix with sparse, realistic baseline
    # Most genes lowly expressed (negative binomial with high dropout)
    expression = np.random.negative_binomial(n=0.5, p=0.9, size=(n_cells, n_genes)).astype(float)
    
    # Add cell-type specific expression patterns
    genes_per_type = n_genes // len(cell_types)
    
    for i, (cell_type, (start_cell, end_cell)) in enumerate(cells_per_type.items()):
        if end_cell > n_cells:
            end_cell = n_cells
            
        # Each cell type highly expresses certain genes
        start_gene = i * genes_per_type
        end_gene = min((i + 1) * genes_per_type, n_genes)
        
        # Add high expression for marker genes
        expression[start_cell:end_cell, start_gene:end_gene] += np.random.gamma(
            shape=2, scale=3, size=(end_cell - start_cell, end_gene - start_gene)
        )
        
        # Add some shared genes across cell types (housekeeping)
        housekeeping_genes = slice(-50, None)  # Last 50 genes as housekeeping
        expression[start_cell:end_cell, housekeeping_genes] += np.random.gamma(
            shape=1, scale=2, size=(end_cell - start_cell, 50)
        )
    
    # Apply realistic transformations
    # 1. Add batch effects (technical noise)
    n_batches = 3
    batch_size = n_cells // n_batches
    for batch in range(n_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, n_cells)
        batch_effect = np.random.normal(1, 0.1, n_genes)
        expression[start_idx:end_idx] *= batch_effect
    
    # 2. Add dropout (set some expressed genes to 0)
    dropout_rate = 0.3
    dropout_mask = np.random.random((n_cells, n_genes)) < dropout_rate
    expression[dropout_mask] = 0
    
    # 3. Add library size variation
    library_sizes = np.random.lognormal(mean=9, sigma=0.5, size=n_cells)
    for i in range(n_cells):
        expression[i] = expression[i] * library_sizes[i] / np.sum(expression[i])
    
    return expression, list(cells_per_type.keys())

def create_cell_metadata(n_cells, cell_type_assignments, seed=42):
    """Create realistic cell metadata"""
    print("  Creating cell metadata...")
    
    np.random.seed(seed)
    
    # Assign cell types
    cell_types = []
    cells_per_type = len(cell_type_assignments)
    cells_per_assignment = n_cells // cells_per_type
    
    for i, cell_type in enumerate(cell_type_assignments):
        start_idx = i * cells_per_assignment
        end_idx = min((i + 1) * cells_per_assignment, n_cells)
        cell_types.extend([cell_type] * (end_idx - start_idx))
    
    # Add remaining cells to last type
    while len(cell_types) < n_cells:
        cell_types.append(cell_type_assignments[-1])
    
    metadata = {
        "cell_type": cell_types,
        "batch": [f"batch_{np.random.randint(1, 4)}" for _ in range(n_cells)],
        "library_size": np.random.lognormal(9, 0.5, n_cells).tolist(),
        "n_genes_detected": [int(np.random.uniform(800, 2500)) for _ in range(n_cells)],
        "percent_mitochondrial": np.random.uniform(2, 15, n_cells).tolist()
    }
    
    return metadata

def create_test_datasets():
    """Create test datasets of different sizes for QAC testing"""
    print("📊 Creating test datasets...")
    
    # Create data directory
    data_dir = "test_data"
    os.makedirs(data_dir, exist_ok=True)
    
    datasets = {}
    
    # Small dataset (100 cells, 100 genes)
    print("  Creating small dataset (100 cells, 100 genes)...")
    gene_names = generate_realistic_gene_names(100)
    expression_matrix, cell_type_assignments = generate_realistic_expression_matrix(100, 100, seed=42)
    cell_metadata = create_cell_metadata(100, cell_type_assignments, seed=42)
    
    datasets['small'] = {
        'gene_names': gene_names,
        'expression_matrix': expression_matrix.tolist(),
        'cell_metadata': cell_metadata
    }
    
    # Medium dataset (500 cells, 500 genes)
    print("  Creating medium dataset (500 cells, 500 genes)...")
    gene_names = generate_realistic_gene_names(500)
    expression_matrix, cell_type_assignments = generate_realistic_expression_matrix(500, 500, seed=43)
    cell_metadata = create_cell_metadata(500, cell_type_assignments, seed=43)
    
    datasets['medium'] = {
        'gene_names': gene_names,
        'expression_matrix': expression_matrix.tolist(),
        'cell_metadata': cell_metadata
    }
    
    # Large dataset (1000 cells, 1000 genes) 
    print("  Creating large dataset (1000 cells, 1000 genes)...")
    gene_names = generate_realistic_gene_names(1000)
    expression_matrix, cell_type_assignments = generate_realistic_expression_matrix(1000, 1000, seed=44)
    cell_metadata = create_cell_metadata(1000, cell_type_assignments, seed=44)
    
    datasets['large'] = {
        'gene_names': gene_names,
        'expression_matrix': expression_matrix.tolist(),
        'cell_metadata': cell_metadata
    }
    
    # Save datasets
    print("  Saving datasets...")
    for name, data in datasets.items():
        with open(f"test_data/pbmc_{name}.json", 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"✅ Created {len(datasets)} test datasets in test_data/")
    return datasets

def create_qac_test_requests():
    """Create ready-to-use API request files for testing"""
    print("🔬 Creating API test requests...")
    
    # Load the small dataset for quick testing
    with open("test_data/pbmc_small.json", 'r') as f:
        small_data = json.load(f)
    
    # Load medium dataset for more comprehensive testing
    with open("test_data/pbmc_medium.json", 'r') as f:
        medium_data = json.load(f)
    
    # Create requests directory
    requests_dir = "test_data/requests"
    os.makedirs(requests_dir, exist_ok=True)
    
    # 1. Genomics compression tests
    compression_requests = {
        "compression_small_fixed": {
            "data": small_data,
            "compression_target": 8,
            "params": {"mode": "fixed_k", "max_k": 32},
            "seed": 42
        },
        "compression_medium_auto": {
            "data": medium_data,
            "compression_target": 32,
            "params": {"mode": "auto_evr", "evr_target": 0.85, "max_k": 128},
            "seed": 42
        }
    }
    
    # 2. QAC fit tests
    qac_fit_requests = {
        "qac_fit_small": {
            "embedding": small_data["expression_matrix"],
            "params": {
                "engine": "real_qac",
                "k_nn": 5,
                "reinforce_rate": 0.05,
                "layers": 10
            },
            "seed": 42,
            "sync": True
        },
        "qac_fit_medium": {
            "embedding": medium_data["expression_matrix"],
            "params": {
                "engine": "real_qac", 
                "k_nn": 10,
                "reinforce_rate": 0.08,
                "layers": 25
            },
            "seed": 42,
            "sync": False  # Async for larger dataset
        }
    }
    
    # 3. Combined requests (all types)
    all_requests = {**compression_requests, **qac_fit_requests}
    
    # Save request files
    for name, request in all_requests.items():
        filepath = f"{requests_dir}/{name}.json"
        with open(filepath, 'w') as f:
            json.dump(request, f, indent=2)
        print(f"  📝 Created {filepath}")
    
    # Create a test script
    test_script = f"""#!/bin/bash
# DON Stack Research API Test Script
# ================================

echo "🧬 DON Stack Research API Testing"
echo "================================"

# API endpoint
API_BASE="http://localhost:8080/api/v1"
AUTH_HEADER="Authorization: Bearer {API_TOKEN}"

echo ""
echo "1. Testing Health Endpoint..."
curl -s "$API_BASE/health" | python -m json.tool

echo ""
echo "2. Testing Small Dataset Compression..."
curl -X POST "$API_BASE/genomics/compress" \\
    -H "$AUTH_HEADER" \\
    -H "Content-Type: application/json" \\
    -d @{requests_dir}/compression_small_fixed.json \\
    | python -m json.tool

echo ""
echo "3. Testing QAC Fit (Small Dataset)..."
curl -X POST "$API_BASE/quantum/qac/fit" \\
    -H "$AUTH_HEADER" \\
    -H "Content-Type: application/json" \\
    -d @{requests_dir}/qac_fit_small.json \\
    | python -m json.tool

echo ""
echo "4. Testing QAC Models List..."
curl -s "$API_BASE/quantum/qac/models" \\
    -H "$AUTH_HEADER" \\
    | python -m json.tool

echo ""
echo "✅ Testing complete!"
"""
    
    with open(f"{requests_dir}/test_api.sh", 'w') as f:
        f.write(test_script)
    
    # Make executable
    os.chmod(f"{requests_dir}/test_api.sh", 0o755)
    
    print(f"  🚀 Created executable test script: {requests_dir}/test_api.sh")
    return all_requests

if __name__ == "__main__":
    print("🧬 DON Stack Research API Test Data Generator")
    print("=" * 50)
    
    try:
        # Create genomics datasets
        datasets = create_test_datasets()
        
        # Create API test requests
        requests = create_qac_test_requests()
        
        print("\n🎉 Test data generation complete!")
        print("\nAvailable datasets:")
        for name in datasets.keys():
            print(f"  📊 test_data/pbmc_{name}.json")
        
        print(f"\nAPI test requests (test_data/requests/):")
        for name in requests.keys():
            print(f"  📝 {name}.json")
        
        print("\n💡 Quick Start:")
        print("1. Start the API server:")
        print("   source venv/bin/activate && python main.py")
        print("")
        print("2. Run the automated test suite:")
        print("   cd test_data/requests && ./test_api.sh")
        print("")
        print("3. Or test individual endpoints:")
        print("   curl -X POST http://localhost:8080/api/v1/genomics/compress \\")
        print(f"     -H 'Authorization: Bearer {API_TOKEN}' \\")
        print("     -H 'Content-Type: application/json' \\")
        print("     -d @test_data/requests/compression_small_fixed.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()