#!/usr/bin/env python3
"""
TAMU Package Verification Script

Tests that all components are working correctly before sending to Dr. Cai.
Run this to verify the package integrity.
"""

import sys
import os
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if Path(filepath).exists():
        size = Path(filepath).stat().st_size / 1024  # KB
        print(f"  ✅ {description}: {filepath} ({size:.1f} KB)")
        return True
    else:
        print(f"  ❌ MISSING: {description}: {filepath}")
        return False

def verify_package():
    """Verify TAMU collaboration package contents."""
    
    print("="*70)
    print("TAMU COLLABORATION PACKAGE - VERIFICATION")
    print("="*70)
    
    pkg_dir = Path("TAMU_COLLABORATION_PACKAGE")
    
    if not pkg_dir.exists():
        print("\n❌ ERROR: TAMU_COLLABORATION_PACKAGE directory not found!")
        return False
    
    all_ok = True
    
    # Check documentation
    print("\n📄 Documentation Files:")
    all_ok &= check_file_exists(pkg_dir / "README.md", "Main README")
    all_ok &= check_file_exists(pkg_dir / "TAMU_API_USAGE_GUIDE.md", "API Usage Guide")
    all_ok &= check_file_exists(pkg_dir / "TAMU_EXECUTIVE_SUMMARY.md", "Executive Summary")
    all_ok &= check_file_exists(pkg_dir / "TAMU_PACKAGE_INDEX.md", "Package Index")
    all_ok &= check_file_exists(pkg_dir / "TAMU_QUICK_START.md", "Quick Start")
    
    # Check scripts
    print("\n🐍 Python Scripts:")
    all_ok &= check_file_exists(pkg_dir / "don_research_client.py", "API Client Library")
    all_ok &= check_file_exists(pkg_dir / "tamu_gene_coexpression_qac.py", "Gene Coexpression Analysis")
    
    # Check dependencies
    print("\n📦 Dependencies:")
    all_ok &= check_file_exists(pkg_dir / "requirements.txt", "Requirements File")
    
    # Check data
    print("\n💾 Data Files:")
    all_ok &= check_file_exists(pkg_dir / "data" / "pbmc3k_with_tace_alpha.h5ad", "PBMC3K Dataset")
    
    # Check results
    print("\n📊 Result Files:")
    all_ok &= check_file_exists(pkg_dir / "gene_coexpression_qac_results.json", "Discovery 1 Results")
    all_ok &= check_file_exists(pkg_dir / "quantum_collapse_creation_results.json", "Discovery 2 Results")
    all_ok &= check_file_exists(pkg_dir / "memory_is_structure_results.json", "Discovery 3 Results")
    
    # Check figures
    print("\n🖼️  Figure Files:")
    all_ok &= check_file_exists(pkg_dir / "tamu_figures" / "tamu_summary_figure.png", "Summary Figure")
    all_ok &= check_file_exists(pkg_dir / "tamu_figures" / "tamu_medical_applications.png", "Medical Applications")
    all_ok &= check_file_exists(pkg_dir / "tamu_figures" / "tamu_collaboration_roadmap.png", "Collaboration Roadmap")
    
    # Test Python imports
    print("\n🔬 Testing Python Imports:")
    try:
        sys.path.insert(0, str(pkg_dir))
        from don_research_client import DonResearchClient, QACParams
        print("  ✅ don_research_client imports successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import don_research_client: {e}")
        all_ok = False
    
    # Check for __pycache__ (should be excluded from zip)
    pycache = list(pkg_dir.rglob("__pycache__"))
    if pycache:
        print(f"\n⚠️  WARNING: Found {len(pycache)} __pycache__ directories (should be excluded from zip)")
    else:
        print("\n✅ No __pycache__ directories (good)")
    
    # Summary
    print("\n" + "="*70)
    if all_ok:
        print("✅ VERIFICATION PASSED - Package is ready to send!")
    else:
        print("❌ VERIFICATION FAILED - Fix errors before sending")
    print("="*70)
    
    return all_ok

if __name__ == "__main__":
    success = verify_package()
    sys.exit(0 if success else 1)
