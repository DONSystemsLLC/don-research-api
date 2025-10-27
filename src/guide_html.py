"""
Comprehensive user guide HTML for /guide endpoint.
Contains detailed tutorials, Swagger UI walkthrough, Bio module docs, and troubleshooting.
"""

GUIDE_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>DON Research API - Complete User Guide | Texas A&M Lab</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; 
            margin: 0; padding: 0; background: #f5f7fb; color: #1a202c; line-height: 1.6;
        }
        header { background: linear-gradient(135deg, #0b3d91 0%, #11203f 100%); color: #fff; padding: 32px 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        header h1 { margin: 0 0 8px 0; font-size: 32px; font-weight: 600; }
        header p { margin: 0; font-size: 16px; opacity: 0.95; }
        nav { background: #fff; padding: 16px 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); position: sticky; top: 0; z-index: 100; }
        nav a { 
            color: #0b3d91; text-decoration: none; margin-right: 24px; font-weight: 500; font-size: 14px; 
            transition: color 0.2s;
        }
        nav a:hover { color: #11203f; text-decoration: underline; }
        main { max-width: 1100px; margin: 0 auto; padding: 40px 24px 60px 24px; }
        section { background: #fff; border-radius: 12px; padding: 32px; margin-bottom: 32px; box-shadow: 0 8px 24px rgba(12, 30, 66, 0.08); }
        h2 { margin-top: 0; color: #0b3d91; font-size: 24px; font-weight: 600; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }
        h3 { color: #2d3748; font-size: 18px; font-weight: 600; margin-top: 24px; margin-bottom: 12px; }
        h4 { color: #4a5568; font-size: 16px; font-weight: 600; margin-top: 16px; margin-bottom: 8px; }
        ol, ul { padding-left: 24px; margin: 12px 0; }
        li { margin-bottom: 8px; }
        code { 
            background: #edf2f7; padding: 3px 8px; border-radius: 4px; font-family: 'Monaco', 'Menlo', monospace; 
            font-size: 13px; color: #c7254e;
        }
        pre { 
            background: #2d3748; color: #e2e8f0; padding: 20px; border-radius: 8px; overflow-x: auto; 
            font-family: 'Monaco', 'Menlo', monospace; font-size: 13px; line-height: 1.5; margin: 16px 0;
        }
        pre code { background: none; padding: 0; color: inherit; }
        table { 
            width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        th { 
            background: #edf2f7; color: #2d3748; font-weight: 600; text-align: left; 
            padding: 12px 16px; border: 1px solid #cbd5e0;
        }
        td { padding: 12px 16px; border: 1px solid #e2e8f0; }
        tr:hover { background: #f7fafc; }
        .badge { 
            display: inline-block; background: #4299e1; color: #fff; padding: 4px 12px; 
            border-radius: 12px; font-size: 12px; font-weight: 600; margin-right: 8px;
        }
        .badge.success { background: #48bb78; }
        .badge.warning { background: #ed8936; }
        .info-box { 
            background: #ebf8ff; border-left: 4px solid #4299e1; padding: 16px 20px; 
            margin: 16px 0; border-radius: 4px;
        }
        .warning-box { 
            background: #fffaf0; border-left: 4px solid #ed8936; padding: 16px 20px; 
            margin: 16px 0; border-radius: 4px;
        }
        .success-box { 
            background: #f0fff4; border-left: 4px solid #48bb78; padding: 16px 20px; 
            margin: 16px 0; border-radius: 4px;
        }
        footer { 
            text-align: center; font-size: 13px; color: #718096; padding: 32px 24px; 
            background: #fff; border-top: 1px solid #e2e8f0;
        }
        a { color: #4299e1; text-decoration: none; transition: color 0.2s; }
        a:hover { color: #2b6cb0; text-decoration: underline; }
        @media (max-width: 768px) {
            header h1 { font-size: 24px; }
            nav a { display: block; margin: 8px 0; }
        }
    </style>
</head>
<body>
    <header>
        <h1>📘 DON Research API - Complete User Guide</h1>
        <p>Comprehensive Tutorial for Texas A&M University Cai Lab</p>
    </header>
    
    <nav>
        <a href="/">← Homepage</a>
        <a href="#overview">Overview</a>
        <a href="#swagger-ui">Swagger UI Tutorial</a>
        <a href="#formats">Data Formats</a>
        <a href="#bio-module">Bio Module</a>
        <a href="#workflows">Complete Workflows</a>
        <a href="#troubleshooting">Troubleshooting</a>
        <a href="/docs">API Docs</a>
    </nav>
    
    <main>
        <!-- Overview Section -->
        <section id="overview">
            <h2>📖 Web Interface Overview</h2>
            
            <h3>What This Guide Covers</h3>
            <p>This comprehensive guide teaches you how to use the DON Research API web interface and its features:</p>
            <ul>
                <li><strong>Homepage Navigation</strong> - Understanding the main documentation page</li>
                <li><strong>Swagger UI</strong> - Interactive API testing in your browser (no code required!)</li>
                <li><strong>Data Formats</strong> - H5AD files, GEO accessions, URLs, gene lists, text queries</li>
                <li><strong>Bio Module</strong> - ResoTrace integration for advanced workflows</li>
                <li><strong>Complete Workflows</strong> - Step-by-step examples from start to finish</li>
                <li><strong>Troubleshooting</strong> - Solutions for common errors</li>
            </ul>
            
            <div class="info-box">
                <strong>🎯 Quick Links:</strong><br>
                • <strong>Homepage:</strong> <a href="/">https://don-research.onrender.com/</a> - Quick reference docs<br>
                • <strong>This Guide:</strong> <a href="/guide">https://don-research.onrender.com/guide</a> - Detailed tutorials<br>
                • <strong>Swagger UI:</strong> <a href="/docs">https://don-research.onrender.com/docs</a> - Interactive testing
            </div>
            
            <h3>Who Should Use This Guide?</h3>
            <p>This guide is designed for Texas A&M researchers who:</p>
            <ul>
                <li>✅ Have received their API token (via secure email)</li>
                <li>✅ Want to test endpoints in their browser before writing code</li>
                <li>✅ Need detailed explanations of Bio module features</li>
                <li>✅ Want complete workflow examples for common tasks</li>
                <li>✅ Need troubleshooting help for API errors</li>
            </ul>
        </section>
        
        <!-- Swagger UI Tutorial -->
        <section id="swagger-ui">
            <h2>🔍 Swagger UI Tutorial: Test APIs in Your Browser</h2>
            
            <h3>What is Swagger UI?</h3>
            <p><strong>Swagger UI</strong> is an interactive tool that lets you <strong>test API endpoints directly in your web browser</strong> without writing any code. It's perfect for:</p>
            <ul>
                <li>✅ Learning how endpoints work before writing scripts</li>
                <li>✅ Validating your API token</li>
                <li>✅ Testing with small datasets</li>
                <li>✅ Debugging API calls</li>
                <li>✅ Seeing real-time request/response data</li>
            </ul>
            
            <h3>Accessing Swagger UI</h3>
            <ol>
                <li>Open your web browser (Chrome, Firefox, Safari, or Edge)</li>
                <li>Navigate to: <strong><a href="/docs" target="_blank">https://don-research.onrender.com/docs</a></strong></li>
                <li>You'll see a list of all available API endpoints organized by category</li>
            </ol>
            
            <h3>Step-by-Step: Testing Your First Endpoint</h3>
            
            <h4>Step 1: Authenticate with Your Token</h4>
            <ol>
                <li>Look for the <strong>green "Authorize" button</strong> at the top right of the page</li>
                <li>Click it to open the authentication dialog</li>
                <li>In the "Value" field, enter: <code>Bearer your-tamu-token-here</code>
                    <ul>
                        <li>⚠️ <strong>Important:</strong> Include the word "Bearer " (with a space) before your token</li>
                        <li>Example: <code>Bearer tamu_cai_lab_2025_HkRs17sgvbjnQax2KzD1iqYcHWbAs5xvZZ2ApKptWuc</code></li>
                    </ul>
                </li>
                <li>Click <strong>"Authorize"</strong> button</li>
                <li>Click <strong>"Close"</strong> to return to the main page</li>
            </ol>
            
            <div class="success-box">
                <strong>✓ Success Indicator:</strong> The padlock icons next to endpoints change from open 🔓 to closed 🔒
            </div>
            
            <h4>Step 2: Select an Endpoint to Test</h4>
            <p>Endpoints are organized into sections. For your first test, try the <strong>health check</strong>:</p>
            <ol>
                <li>Scroll down to find <span class="badge success">GET</span> <code>/health</code></li>
                <li>Click on it to expand the endpoint details</li>
            </ol>
            
            <h4>Step 3: Execute the Request</h4>
            <ol>
                <li>Click the <strong>"Try it out"</strong> button (top right of the endpoint section)</li>
                <li>Click the <strong>"Execute"</strong> button (blue button)</li>
                <li>Wait a moment for the response...</li>
            </ol>
            
            <h4>Step 4: View the Results</h4>
            <p>After execution, you'll see three important sections:</p>
            
            <p><strong>Request URL:</strong></p>
            <pre><code>https://don-research.onrender.com/health</code></pre>
            
            <p><strong>Response Body:</strong> (should look like this)</p>
            <pre><code>{
  "status": "ok",
  "timestamp": "2025-10-27T15:30:45.123Z",
  "database": "connected",
  "don_stack": {
    "mode": "real",
    "don_gpu": true,
    "tace": true,
    "qac": true
  }
}</code></pre>
            
            <p><strong>Response Headers:</strong></p>
            <pre><code>X-Trace-ID: tamu_20251027_abc123def
content-type: application/json</code></pre>
            
            <h3>Testing File Upload Endpoints</h3>
            
            <h4>Example: Build Feature Vectors</h4>
            <ol>
                <li>Find <span class="badge warning">POST</span> <code>/api/v1/genomics/vectors/build</code></li>
                <li>Click to expand → Click <strong>"Try it out"</strong></li>
                <li><strong>file parameter:</strong> Click "Choose File" → Select your <code>.h5ad</code> file</li>
                <li><strong>mode parameter:</strong> Select <code>cluster</code> from dropdown</li>
                <li>Click <strong>"Execute"</strong></li>
                <li>View response with vector counts and preview data</li>
            </ol>
            
            <div class="warning-box">
                <strong>⚠️ Best Practices:</strong><br>
                • Use <strong>small datasets</strong> (< 5,000 cells) in Swagger UI<br>
                • For large files, use Python scripts instead<br>
                • Don't upload sensitive/unpublished data via browser<br>
                • Copy the <strong>curl command</strong> shown for use in scripts
            </div>
            
            <h3>Understanding Response Codes</h3>
            <table>
                <tr>
                    <th>Code</th>
                    <th>Meaning</th>
                    <th>Action</th>
                </tr>
                <tr>
                    <td><span class="badge success">200</span></td>
                    <td>Success</td>
                    <td>Request completed successfully</td>
                </tr>
                <tr>
                    <td><span class="badge warning">400</span></td>
                    <td>Bad Request</td>
                    <td>Check parameter format (e.g., file extension)</td>
                </tr>
                <tr>
                    <td><span class="badge warning">401</span></td>
                    <td>Unauthorized</td>
                    <td>Check token format (must include "Bearer ")</td>
                </tr>
                <tr>
                    <td><span class="badge warning">429</span></td>
                    <td>Rate Limit Exceeded</td>
                    <td>Wait 1 hour or use fewer requests</td>
                </tr>
                <tr>
                    <td><span class="badge warning">500</span></td>
                    <td>Server Error</td>
                    <td>Contact support with trace_id</td>
                </tr>
            </table>
        </section>
        
        <!-- Data Formats Section -->
        <section id="formats">
            <h2>📁 Supported Data Formats</h2>
            
            <h3>Format Overview</h3>
            <p>The DON Research API supports multiple input formats, not just H5AD files:</p>
            
            <table>
                <tr>
                    <th>Format</th>
                    <th>Example</th>
                    <th>Use Case</th>
                    <th>Supported Endpoints</th>
                </tr>
                <tr>
                    <td><strong>H5AD Files</strong></td>
                    <td><code>pbmc3k.h5ad</code></td>
                    <td>Direct upload of single-cell data (AnnData format)</td>
                    <td>All genomics + Bio module</td>
                </tr>
                <tr>
                    <td><strong>GEO Accessions</strong></td>
                    <td><code>GSE12345</code></td>
                    <td>Auto-download from NCBI GEO database</td>
                    <td><code>/load</code></td>
                </tr>
                <tr>
                    <td><strong>Direct URLs</strong></td>
                    <td><code>https://example.com/data.h5ad</code></td>
                    <td>Download from external HTTP/HTTPS sources</td>
                    <td><code>/load</code></td>
                </tr>
                <tr>
                    <td><strong>Gene Lists (JSON)</strong></td>
                    <td><code>["CD3E", "CD8A", "CD4"]</code></td>
                    <td>Encode cell type marker genes as query vectors</td>
                    <td><code>/query/encode</code></td>
                </tr>
                <tr>
                    <td><strong>Text Queries</strong></td>
                    <td><code>"T cell markers in PBMC"</code></td>
                    <td>Natural language biological queries</td>
                    <td><code>/query/encode</code></td>
                </tr>
            </table>
            
            <div class="info-box">
                <strong>📌 Important:</strong><br>
                • <strong>Bio Module</strong> endpoints require <strong>H5AD files only</strong> (for ResoTrace compatibility)<br>
                • <strong>Genomics</strong> endpoints support all formats above<br>
                • GEO accessions and URLs are automatically downloaded and converted to H5AD
            </div>
            
            <h3>Format-Specific Usage Examples</h3>
            
            <h4>1. H5AD Files (Most Common)</h4>
            <pre><code>import requests

# Upload H5AD file directly
with open("pbmc_3k.h5ad", "rb") as f:
    files = {"file": ("pbmc_3k.h5ad", f, "application/octet-stream")}
    response = requests.post(
        "https://don-research.onrender.com/api/v1/genomics/vectors/build",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files=files,
        data={"mode": "cluster"}
    )
print(response.json())</code></pre>
            
            <h4>2. GEO Accessions</h4>
            <pre><code># Automatically downloads from NCBI GEO
data = {"accession_or_path": "GSE12345"}
response = requests.post(
    "https://don-research.onrender.com/api/v1/genomics/load",
    headers={"Authorization": f"Bearer {TOKEN}"},
    data=data
)
h5ad_path = response.json()["h5ad_path"]
print(f"Downloaded to: {h5ad_path}")</code></pre>
            
            <h4>3. Direct URLs</h4>
            <pre><code># Download from any HTTP/HTTPS source
data = {"accession_or_path": "https://example.com/dataset.h5ad"}
response = requests.post(
    "https://don-research.onrender.com/api/v1/genomics/load",
    headers={"Authorization": f"Bearer {TOKEN}"},
    data=data
)
h5ad_path = response.json()["h5ad_path"]</code></pre>
            
            <h4>4. Gene Lists (JSON)</h4>
            <pre><code>import json

# T cell marker genes
gene_list = ["CD3E", "CD8A", "CD4", "IL7R", "CCR7"]
data = {"gene_list_json": json.dumps(gene_list)}

response = requests.post(
    "https://don-research.onrender.com/api/v1/genomics/query/encode",
    headers={"Authorization": f"Bearer {TOKEN}"},
    data=data
)
query_vector = response.json()["psi"]  # 128-dimensional vector</code></pre>
            
            <h4>5. Text Queries</h4>
            <pre><code># Natural language query
data = {
    "text": "T cell markers in PBMC tissue",
    "cell_type": "T cell",
    "tissue": "PBMC"
}

response = requests.post(
    "https://don-research.onrender.com/api/v1/genomics/query/encode",
    headers={"Authorization": f"Bearer {TOKEN}"},
    data=data
)
query_vector = response.json()["psi"]</code></pre>
            
            <h3>When to Use Each Format</h3>
            <ul>
                <li><strong>H5AD files:</strong> When you have preprocessed single-cell data locally</li>
                <li><strong>GEO accessions:</strong> When referencing published datasets (e.g., "GSE12345" from papers)</li>
                <li><strong>URLs:</strong> When data is hosted externally (collaborator's server, cloud storage)</li>
                <li><strong>Gene lists:</strong> When searching for specific cell types by marker genes</li>
                <li><strong>Text queries:</strong> When exploring data without knowing exact gene names</li>
            </ul>
        </section>
        
        <!-- Bio Module Section -->
        <section id="bio-module">
            <h2>🧬 Bio Module: ResoTrace Integration</h2>
            
            <h3>Overview</h3>
            <p>The Bio Module provides advanced single-cell analysis workflows optimized for <strong>ResoTrace integration</strong>. Key capabilities:</p>
            <ul>
                <li>✅ <strong>Export Artifacts:</strong> Convert H5AD → ResoTrace collapse maps</li>
                <li>✅ <strong>Signal Sync:</strong> Compare pipeline runs for reproducibility</li>
                <li>✅ <strong>Parasite Detection:</strong> QC for ambient RNA, doublets, batch effects</li>
                <li>✅ <strong>Evolution Report:</strong> Track pipeline stability over parameter changes</li>
            </ul>
            
            <h3>Sync vs Async Execution Modes</h3>
            <p><strong>Every Bio endpoint supports two execution modes:</strong></p>
            
            <table>
                <tr>
                    <th>Mode</th>
                    <th>When to Use</th>
                    <th>Response Time</th>
                    <th>Best For</th>
                </tr>
                <tr>
                    <td><code>sync=true</code></td>
                    <td>Small datasets (< 5K cells)</td>
                    <td>Immediate (< 30s)</td>
                    <td>Quick validation, exploratory analysis, Swagger UI testing</td>
                </tr>
                <tr>
                    <td><code>sync=false</code></td>
                    <td>Large datasets (> 10K cells)</td>
                    <td>Background job</td>
                    <td>Production pipelines, batch processing, automated workflows</td>
                </tr>
            </table>
            
            <h3>Feature 1: Export Artifacts</h3>
            <p><span class="badge warning">POST</span> <code>/api/v1/bio/export-artifacts</code></p>
            
            <p><strong>What it does:</strong></p>
            <ul>
                <li>Converts <code>.h5ad</code> files into ResoTrace-compatible formats</li>
                <li>Generates collapse maps (cluster graph structure)</li>
                <li>Exports cell-level vector collections (128D embeddings)</li>
                <li>Includes PAGA connectivity (if available)</li>
            </ul>
            
            <p><strong>Required parameters:</strong></p>
            <ul>
                <li><code>file</code>: H5AD file upload</li>
                <li><code>cluster_key</code>: Column in <code>adata.obs</code> with cluster labels (e.g., "leiden")</li>
                <li><code>latent_key</code>: Embedding in <code>adata.obsm</code> (e.g., "X_umap", "X_pca")</li>
            </ul>
            
            <p><strong>Example (Synchronous):</strong></p>
            <pre><code>with open("pbmc_3k.h5ad", "rb") as f:
    files = {"file": ("pbmc.h5ad", f, "application/octet-stream")}
    data = {
        "cluster_key": "leiden",
        "latent_key": "X_umap",
        "sync": "true",
        "project_id": "cai_lab_pbmc_study",
        "user_id": "researcher_001"
    }
    
    response = requests.post(
        "https://don-research.onrender.com/api/v1/bio/export-artifacts",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files=files,
        data=data
    )

result = response.json()
print(f"✓ Exported {result['nodes']} clusters")
print(f"✓ {result['vectors']} cell vectors")
print(f"✓ Trace ID: {result.get('trace_id')}")</code></pre>
            
            <h3>Feature 2: Parasite Detector (QC)</h3>
            <p><span class="badge warning">POST</span> <code>/api/v1/bio/qc/parasite-detect</code></p>
            
            <p><strong>What it does:</strong></p>
            <ul>
                <li>Flags low-quality cells ("parasites")</li>
                <li>Detects: Ambient RNA, doublets, batch effects</li>
                <li>Returns per-cell boolean flags</li>
                <li>Computes overall contamination score</li>
            </ul>
            
            <p><strong>Recommended Actions:</strong></p>
            <table>
                <tr>
                    <th>Parasite Score</th>
                    <th>Quality</th>
                    <th>Action</th>
                </tr>
                <tr>
                    <td>0-5%</td>
                    <td>Excellent</td>
                    <td>Proceed without filtering</td>
                </tr>
                <tr>
                    <td>5-15%</td>
                    <td>Good</td>
                    <td>Minor filtering recommended</td>
                </tr>
                <tr>
                    <td>15-30%</td>
                    <td>Moderate</td>
                    <td>Filter flagged cells</td>
                </tr>
                <tr>
                    <td>> 30%</td>
                    <td>Poor</td>
                    <td>Review QC pipeline</td>
                </tr>
            </table>
            
            <p>For complete Bio module documentation, see the <a href="/">homepage</a> or <a href="/docs">Swagger UI</a>.</p>
        </section>
        
        <!-- Complete Workflows -->
        <section id="workflows">
            <h2>🔬 Complete Workflow Examples</h2>
            
            <h3>Workflow 1: Cell Type Discovery with T Cells</h3>
            <p><strong>Goal:</strong> Identify T cell clusters in PBMC dataset using marker genes</p>
            
            <pre><code>import requests
import json

API_URL = "https://don-research.onrender.com"
TOKEN = "your-tamu-token-here"
headers = {"Authorization": f"Bearer {TOKEN}"}

# Step 1: Build cluster vectors
print("Step 1: Building vectors...")
with open("pbmc_3k.h5ad", "rb") as f:
    files = {"file": ("pbmc_3k.h5ad", f, "application/octet-stream")}
    response = requests.post(
        f"{API_URL}/api/v1/genomics/vectors/build",
        headers=headers,
        files=files,
        data={"mode": "cluster"}
    )

vectors_result = response.json()
jsonl_path = vectors_result["jsonl"]
print(f"✓ Built {vectors_result['count']} cluster vectors")

# Step 2: Encode T cell query
print("\\nStep 2: Encoding T cell markers...")
t_cell_genes = ["CD3E", "CD8A", "CD4", "IL7R"]
query_data = {"gene_list_json": json.dumps(t_cell_genes)}

response = requests.post(
    f"{API_URL}/api/v1/genomics/query/encode",
    headers=headers,
    data=query_data
)
query_vector = response.json()["psi"]
print("✓ Encoded query vector (128 dimensions)")

# Step 3: Search for matching clusters
print("\\nStep 3: Searching for T cell-like clusters...")
search_data = {
    "jsonl_path": jsonl_path,
    "psi": json.dumps(query_vector),
    "k": 5
}
response = requests.post(
    f"{API_URL}/api/v1/genomics/vectors/search",
    headers=headers,
    data=search_data
)

results = response.json()["hits"]
print(f"\\n✓ Top 5 T cell-like clusters:")
for i, hit in enumerate(results, 1):
    cluster_id = hit['meta']['cluster']
    distance = hit['distance']
    cells = hit['meta']['cells']
    print(f"{i}. Cluster {cluster_id}: distance={distance:.4f}, cells={cells}")</code></pre>
            
            <h3>Workflow 2: QC Pipeline with Parasite Detection</h3>
            <p><strong>Goal:</strong> Clean dataset by detecting and removing low-quality cells</p>
            
            <pre><code>import requests
import scanpy as sc
import numpy as np

# Step 1: Detect parasites
print("Step 1: Detecting QC parasites...")
with open("pbmc_raw.h5ad", "rb") as f:
    files = {"file": ("pbmc_raw.h5ad", f, "application/octet-stream")}
    data = {
        "cluster_key": "leiden",
        "batch_key": "sample",
        "sync": "true"
    }
    
    response = requests.post(
        f"{API_URL}/api/v1/bio/qc/parasite-detect",
        headers=headers,
        files=files,
        data=data
    )

result = response.json()
flags = result["flags"]
parasite_score = result["parasite_score"]
print(f"✓ Parasite score: {parasite_score:.1f}%")

# Step 2: Filter flagged cells
print("\\nStep 2: Filtering flagged cells...")
adata = sc.read_h5ad("pbmc_raw.h5ad")
adata = adata[~np.array(flags), :]
adata.write_h5ad("pbmc_cleaned.h5ad")
print(f"✓ Saved {adata.n_obs} clean cells")</code></pre>
        </section>
        
        <!-- Troubleshooting Section -->
        <section id="troubleshooting">
            <h2>🔧 Troubleshooting Common Errors</h2>
            
            <h3>Error 401: Authentication Failed</h3>
            <div class="warning-box">
                <strong>Error:</strong> <code>{"detail": "Invalid or missing token"}</code><br><br>
                <strong>Solutions:</strong>
                <ul>
                    <li>✅ Verify token format includes "Bearer " prefix</li>
                    <li>✅ Check for extra whitespace in token</li>
                    <li>✅ Confirm token hasn't expired (valid 1 year)</li>
                </ul>
            </div>
            
            <h3>Error 400: File Upload Failed</h3>
            <div class="warning-box">
                <strong>Error:</strong> <code>{"detail": "Expected .h5ad file"}</code><br><br>
                <strong>Solutions:</strong>
                <ul>
                    <li>✅ Verify file has <code>.h5ad</code> extension</li>
                    <li>✅ Validate AnnData format: <code>sc.read_h5ad("file.h5ad")</code></li>
                    <li>✅ Check file size < 500MB</li>
                </ul>
            </div>
            
            <h3>Error 429: Rate Limit Exceeded</h3>
            <div class="warning-box">
                <strong>Error:</strong> <code>{"detail": "Rate limit exceeded"}</code><br><br>
                <strong>Solutions:</strong>
                <ul>
                    <li>✅ Wait 1 hour for rate limit reset (1,000 req/hour)</li>
                    <li>✅ Implement exponential backoff in scripts</li>
                    <li>✅ Use cluster mode instead of cell mode</li>
                    <li>✅ Contact support for higher limits if needed</li>
                </ul>
            </div>
            
            <h3>Contact Support</h3>
            <p>When reporting issues, include:</p>
            <ol>
                <li>Institution: Texas A&M University (Cai Lab)</li>
                <li>API endpoint and method (e.g., POST /vectors/build)</li>
                <li>Full error message (JSON response)</li>
                <li>Trace ID from response header (<code>X-Trace-ID</code>)</li>
                <li>Dataset description (cells, genes, file size)</li>
            </ol>
            
            <p><strong>Email:</strong> support@donsystems.com | <strong>Response time:</strong> < 24 hours</p>
        </section>
    </main>
    
    <footer>
        <p>&copy; 2024-2025 DON Systems LLC. Proprietary technology.</p>
        <p><a href="/">← Back to Homepage</a> | <a href="/docs">API Docs</a> | support@donsystems.com</p>
    </footer>
</body>
</html>"""
