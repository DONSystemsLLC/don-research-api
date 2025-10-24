# DON Stack Research API - Demo System Documentation

## Overview

The DON Stack Research API demo system is a comprehensive demonstration framework that showcases the quantum-enhanced genomics capabilities of the DON (Distributed Order Network) Stack. The system provides multiple demonstration scenarios tailored for different audiences, from quick technical overviews to detailed business presentations.

## Demo Architecture

### Directory Structure

```
demos/
├── demo_launcher.py          # Main interactive demo launcher
├── quick/                    # Quick demonstrations (2-5 min)
│   ├── stack_health_demo.py     # System verification
│   ├── basic_compression_demo.py # Basic genomics compression
│   └── quantum_vs_classical_demo.py # Comparative analysis
├── technical/               # Technical deep-dives (7-15 min)
│   ├── don_gpu_deep_dive.py     # DON-GPU fractal clustering
│   ├── qac_error_correction_demo.py # Quantum error correction
│   ├── tace_temporal_demo.py     # Temporal control systems
│   └── full_pipeline_demo.py    # Complete workflow
├── business/               # Business/investor presentations (10-15 min)
│   ├── roi_performance_demo.py   # ROI and performance metrics
│   ├── competitive_analysis_demo.py # Market positioning
│   └── market_applications_demo.py # Industry applications
└── visualization/          # Interactive demonstrations (15-20 min)
    └── realtime_dashboard_demo.py # Live performance monitoring
```

## Quick Start Guide

### Prerequisites

1. **DON Stack API Server**: Must be running on `localhost:8080`
   ```bash
   cd /path/to/don-research-api
   python main.py
   ```

2. **Dependencies**: Core Python packages
   ```bash
   pip install numpy requests json pathlib
   ```

3. **Test Data**: At least one genomics dataset
   - `real_pbmc_medium_correct.json` (preferred)
   - `test_data/pbmc_small.json` (fallback)

4. **Authentication**: Valid research institution token
   - Default demo token: `demo_token`
   - Custom tokens available via `research@donsystems.com`

### Running Demos

**Interactive Launcher** (Recommended):
```bash
cd demos
python demo_launcher.py
```

**Direct Demo Execution**:
```bash
python demos/quick/stack_health_demo.py
python demos/quick/basic_compression_demo.py
python demos/business/roi_performance_demo.py
```

## Demo Categories

### 1. Quick Demonstrations (2-5 minutes)

#### Stack Health Demo
- **Purpose**: System verification and troubleshooting
- **Audience**: Technical users, pre-demo setup
- **Features**:
  - API server connectivity
  - DON Stack adapter status
  - Authentication verification
  - Performance baseline testing

#### Basic Compression Demo
- **Purpose**: Introduction to DON-GPU capabilities
- **Audience**: First-time users, general audiences
- **Features**:
  - Real genomics data processing
  - Compression ratio demonstration
  - Biological significance analysis
  - Performance metrics

#### Quantum vs Classical Demo
- **Purpose**: Comparative advantage demonstration
- **Audience**: Technical stakeholders, researchers
- **Features**:
  - Side-by-side PCA comparison
  - Quality metrics analysis
  - Performance benchmarking
  - Technical advantages summary

### 2. Technical Deep-Dives (7-15 minutes)

#### DON-GPU Deep Dive
- **Purpose**: Comprehensive fractal clustering analysis
- **Audience**: Engineers, technical decision makers
- **Features**:
  - Multi-scale compression testing
  - Performance scaling analysis
  - Algorithm complexity evaluation
  - Technical architecture overview

#### QAC Error Correction Demo
- **Purpose**: Quantum error correction showcase
- **Audience**: Quantum computing specialists
- **Features**:
  - Adjacency-based stabilization
  - Error correction efficiency
  - Coherence time improvements
  - Quantum advantage validation

#### Full Pipeline Demo
- **Purpose**: End-to-end workflow demonstration
- **Audience**: System integrators, architects
- **Features**:
  - DON-GPU → QAC → TACE integration
  - Real-time feedback control
  - Complete processing pipeline
  - System orchestration

### 3. Business Presentations (10-15 minutes)

#### ROI Performance Demo
- **Purpose**: Financial justification and business case
- **Audience**: Executives, investors, procurement
- **Features**:
  - Market size and opportunity
  - Cost-benefit analysis
  - Payback period calculations
  - Competitive positioning
  - Investment projections

#### Competitive Analysis Demo
- **Purpose**: Market differentiation
- **Audience**: Business development, sales
- **Features**:
  - Competitor comparison matrix
  - Technology advantages
  - Market positioning
  - Strategic partnerships

#### Market Applications Demo
- **Purpose**: Industry use cases
- **Audience**: Industry specialists, customers
- **Features**:
  - Pharmaceutical applications
  - Academic research scenarios
  - Clinical diagnostics
  - Personalized medicine

### 4. Visualization Demos (15-20 minutes)

#### Real-time Dashboard
- **Purpose**: Interactive performance monitoring
- **Audience**: Operations teams, technical managers
- **Features**:
  - Live processing metrics
  - Resource utilization
  - Performance trends
  - System health monitoring

## Customization Guide

### Creating Custom Demos

1. **Choose Demo Category**:
   - `quick/` for short demonstrations
   - `technical/` for detailed analysis
   - `business/` for executive presentations
   - `visualization/` for interactive demos

2. **Demo Template**:
   ```python
   #!/usr/bin/env python3
   """
   Custom Demo Title
   ================
   Description of demo purpose and audience
   """
   
   import sys
   from pathlib import Path
   
   # Add project root to path
   project_root = Path(__file__).parent.parent.parent
   sys.path.insert(0, str(project_root))
   
   def run_custom_demo() -> bool:
       """Execute the custom demonstration"""
       
       print("🎯 CUSTOM DEMO TITLE")
       print("=" * 30)
       
       # Demo implementation here
       
       return True
   
   if __name__ == "__main__":
       run_custom_demo()
   ```

3. **Integration with Launcher**:
   Add demo to `demo_launcher.py` menu and routing function.

### Dataset Customization

#### Using Real Data
```python
def load_custom_dataset():
    data_file = project_root / "custom_data.json"
    with open(data_file, 'r') as f:
        expression_matrix = json.load(f)
    
    return {
        "data": {
            "expression_matrix": expression_matrix,
            "gene_names": [f"Gene_{i}" for i in range(len(expression_matrix[0]))]
        },
        "compression_target": 8,
        "seed": 42
    }
```

#### Synthetic Data Generation
```python
def generate_synthetic_data(n_cells=50, n_genes=200):
    np.random.seed(42)
    expression_matrix = []
    
    for cell in range(n_cells):
        # Custom expression pattern logic
        expression = np.random.lognormal(1.0, 0.5, n_genes)
        expression_matrix.append(expression.tolist())
    
    return expression_matrix
```

### API Request Customization

#### Authentication
```python
headers = {
    "Authorization": "Bearer your_institution_token",
    "Content-Type": "application/json"
}
```

#### Request Parameters
```python
request_data = {
    "data": {
        "expression_matrix": expression_matrix,
        "gene_names": gene_names,
        "cell_metadata": optional_metadata
    },
    "compression_target": 8,      # Target dimensions
    "seed": 42,                   # Reproducibility
    "stabilize": True,            # Quantum stabilization
    "params": {                   # Advanced parameters
        "mode": "auto_evr",       # Compression mode
        "evr_target": 0.95,       # Explained variance ratio
        "max_k": 64               # Maximum dimensions
    }
}
```

## Troubleshooting

### Common Issues

1. **API Server Not Running**
   ```
   ❌ API Server connection failed
   💡 Start server with: python main.py
   ```

2. **DON Stack Import Errors**
   ```
   ⚠️ DON Stack: Fallback mode (limited functionality)
   ```
   - Check DON Stack mode: `export DON_STACK_MODE=internal`
   - Verify stack modules in `src/` directory

3. **Authentication Failures**
   ```
   ❌ Authentication failed: Status 401
   ```
   - Verify token in `AUTHORIZED_INSTITUTIONS`
   - Check rate limits and usage

4. **Missing Test Data**
   ```
   ❌ Test Data: Only 0/3 datasets found
   ```
   - Download test datasets
   - Use synthetic data fallback

### System Diagnostics

Run comprehensive system check:
```bash
python demos/demo_launcher.py
# Select option 15: View System Diagnostics
```

### Performance Issues

1. **Slow Processing**
   - Reduce dataset size for demos
   - Check system resources
   - Verify DON Stack mode

2. **Memory Issues**
   - Use smaller datasets
   - Implement data streaming
   - Monitor system resources

## Advanced Configuration

### DON Stack Modes

#### Internal Mode (Default)
```bash
export DON_STACK_MODE=internal
```
- Direct Python calls to stack modules
- Fastest performance
- No network overhead

#### HTTP Mode (Microservices)
```bash
export DON_STACK_MODE=http
export DON_GPU_ENDPOINT=http://127.0.0.1:8001
export TACE_ENDPOINT=http://127.0.0.1:8002
```
- Distributed architecture
- Service isolation
- Requires running microservices

### Performance Tuning

#### Dataset Optimization
- Use appropriate dataset sizes for demo duration
- Pre-process large datasets
- Implement data caching

#### Request Optimization
- Batch multiple requests
- Use appropriate compression targets
- Enable stabilization selectively

### Security Considerations

1. **Token Management**
   - Rotate demonstration tokens regularly
   - Use institution-specific tokens
   - Monitor usage and rate limits

2. **Data Protection**
   - Use synthetic data for public demos
   - Anonymize real datasets
   - Implement proper access controls

## Demo Best Practices

### Presentation Guidelines

1. **Know Your Audience**
   - Technical: Focus on algorithms and performance
   - Business: Emphasize ROI and competitive advantage
   - Academic: Highlight research applications

2. **Demo Flow**
   - Start with health check
   - Build complexity gradually
   - End with key takeaways

3. **Interactive Elements**
   - Allow audience questions
   - Customize parameters live
   - Show real-time results

### Technical Recommendations

1. **Pre-Demo Setup**
   - Run stack health demo first
   - Verify all prerequisites
   - Prepare backup scenarios

2. **Error Handling**
   - Have fallback demonstrations ready
   - Explain system status clearly
   - Use synthetic data if needed

3. **Performance Monitoring**
   - Track demo execution times
   - Monitor system resources
   - Log any issues for improvement

## Contact and Support

- **Technical Questions**: `research@donsystems.com`
- **Demo Customization**: Contact development team
- **Business Inquiries**: `business@donsystems.com`
- **Documentation Updates**: Submit issues or PRs

## Future Enhancements

### Planned Features

1. **Interactive Visualizations**
   - Real-time performance charts
   - 3D genomics data visualization
   - Network topology diagrams

2. **Extended Demos**
   - Multi-modal data integration
   - Longitudinal analysis workflows
   - Cross-platform comparisons

3. **Automation**
   - Automated demo orchestration
   - Performance regression testing
   - Continuous demo validation

### Contributing

To contribute new demos or improvements:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-demo`
3. Follow existing demo patterns
4. Add comprehensive documentation
5. Test across different scenarios
6. Submit pull request

---

*This documentation is maintained alongside the DON Stack Research API development. For the latest updates, refer to the project repository.*