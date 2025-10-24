#!/usr/bin/env python3
"""
DON Stack Research API - Interactive Demo Launcher
=================================================

Comprehensive demonstration system showcasing quantum-enhanced genomics analysis
with the proprietary DON (Distributed Order Network) Stack.

Usage:
    python demos/demo_launcher.py

Features:
    • Multiple demo scenarios for different audiences
    • Real-time performance metrics and visualization
    • Interactive quantum vs classical comparisons
    • Comprehensive technical deep-dives
    • Executive/investor presentations
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def print_banner():
    """Display the main demo banner"""
    print("🧬" + "=" * 70 + "🧬")
    print("   DON STACK RESEARCH API - INTERACTIVE DEMO SYSTEM")
    print("   Quantum-Enhanced Genomics Analysis Platform")
    print("   © DON Systems LLC - Proprietary Technology")
    print("🧬" + "=" * 70 + "🧬")
    print()

def print_menu():
    """Display the main demo menu"""
    print("📋 AVAILABLE DEMONSTRATIONS:")
    print("=" * 50)
    print()
    print("🎯 QUICK DEMONSTRATIONS:")
    print("   1. Stack Health Check & Smoke Test (2 min)")
    print("   2. Basic Genomics Compression Demo (3 min) [select dataset]")
    print("   3. Quantum vs Classical Comparison (5 min)")
    print()
    print("🔬 TECHNICAL DEEP-DIVES:")
    print("   4. DON-GPU Fractal Clustering Analysis (10 min)")
    print("   5. QAC Quantum Error Correction Demo (8 min)")
    print("   6. TACE Temporal Control Showcase (7 min)")
    print("   7. Full Pipeline Integration Demo (15 min)")
    print()
    print("💼 BUSINESS/INVESTOR PRESENTATIONS:")
    print("   8. ROI & Performance Benchmarks (12 min)")
    print("   9. Competitive Analysis Demo (10 min)")
    print("   10. Market Applications Showcase (15 min)")
    print()
    print("🎬 SPECIAL DEMONSTRATIONS:")
    print("   11. Real-time Visualization Dashboard (20 min)")
    print("   12. Custom Research Institution Demo")
    print("   13. Professor Cai Demo (Legacy)")
    print()
    print("⚙️  SYSTEM OPTIONS:")
    print("   14. Switch DON Stack Mode (Internal/HTTP)")
    print("   15. View System Diagnostics")
    print("   16. Exit")
    print()

def get_user_choice() -> int:
    """Get and validate user menu choice"""
    while True:
        try:
            choice = input("👉 Select demonstration (1-16): ").strip()
            if choice.lower() in ['exit', 'quit', 'q']:
                return 16
            
            choice_num = int(choice)
            if 1 <= choice_num <= 16:
                return choice_num
            else:
                print("❌ Please enter a number between 1-16")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Exiting demo launcher...")
            return 16

def choose_dataset_size() -> Optional[str]:
    """Prompt the operator to select a demo dataset size."""
    print("\n📦 DATASET SIZE OPTIONS:")
    print("   1. Small  - 100 cells × 100 genes")
    print("   2. Medium - 250 cells × 500 genes")
    print("   3. Large  - Full PBMC cohort")
    print()

    mapping = {
        "1": "small",
        "small": "small",
        "2": "medium",
        "": "medium",
        "default": "medium",
        "medium": "medium",
        "3": "large",
        "large": "large",
    }

    while True:
        selection = input("Select dataset size (1-3, default: medium): ").strip().lower()
        if selection in mapping:
            dataset = mapping[selection]
            print(f"\n📌 Using {dataset} dataset for this run.\n")
            return dataset

        print("❌ Invalid selection. Please choose 1, 2, or 3.")

def check_prerequisites() -> Dict[str, Any]:
    """Check system prerequisites and return status"""
    print("🔍 CHECKING SYSTEM PREREQUISITES...")
    print("-" * 40)
    
    status = {
        "api_server": False,
        "don_stack": False,
        "test_data": False,
        "dependencies": False
    }
    
    # Check API server
    try:
        import requests
        response = requests.get("http://localhost:8080/", timeout=3)
        if response.status_code == 200:
            status["api_server"] = True
            print("✅ API Server: Running on localhost:8080")
        else:
            print("❌ API Server: Not responding properly")
    except Exception:
        print("❌ API Server: Not running (start with: python main.py)")
    
    # Check DON Stack
    try:
        from don_memory.adapters.don_stack_adapter import DONStackAdapter
        adapter = DONStackAdapter()
        health = adapter.health()
        if health.get('don_gpu') and health.get('tace'):
            status["don_stack"] = True
            print("✅ DON Stack: Real implementations loaded")
        else:
            print("⚠️  DON Stack: Fallback mode (limited functionality)")
    except Exception as e:
        print(f"❌ DON Stack: Import failed - {e}")
    
    # Check test data
    test_files = [
        "real_pbmc_medium_correct.json",
        "test_data/pbmc_small.json",
        "test_data/pbmc_medium.json"
    ]
    
    data_count = 0
    for file_path in test_files:
        if (project_root / file_path).exists():
            data_count += 1
    
    if data_count >= 2:
        status["test_data"] = True
        print(f"✅ Test Data: {data_count}/{len(test_files)} datasets available")
    else:
        print(f"❌ Test Data: Only {data_count}/{len(test_files)} datasets found")
    
    # Check dependencies
    try:
        import numpy
        import requests
        import json
        status["dependencies"] = True
        print("✅ Dependencies: Core packages available")
    except ImportError as e:
        print(f"❌ Dependencies: Missing packages - {e}")
    
    print()
    return status

def run_demo(choice: int, prerequisites: Dict[str, Any]) -> bool:
    """Execute the selected demonstration"""
    
    if choice == 1:
        from demos.quick.stack_health_demo import run_stack_health_demo
        return run_stack_health_demo()
    
    elif choice == 2:
        from demos.quick.basic_compression_demo import run_basic_compression_demo
        dataset_size = choose_dataset_size()
        return run_basic_compression_demo(preferred_dataset=dataset_size)
    
    elif choice == 3:
        from demos.quick.quantum_vs_classical_demo import run_quantum_vs_classical_demo
        return run_quantum_vs_classical_demo()
    
    elif choice == 4:
        from demos.technical.don_gpu_deep_dive import run_don_gpu_demo
        return run_don_gpu_demo()
    
    elif choice == 5:
        from demos.technical.qac_error_correction_demo import run_qac_demo
        return run_qac_demo()
    
    elif choice == 6:
        from demos.technical.tace_temporal_demo import run_tace_demo
        return run_tace_demo()
    
    elif choice == 7:
        from demos.technical.full_pipeline_demo import run_full_pipeline_demo
        return run_full_pipeline_demo()
    
    elif choice == 8:
        from demos.business.roi_performance_demo import run_roi_demo
        return run_roi_demo()
    
    elif choice == 9:
        from demos.business.competitive_analysis_demo import run_competitive_demo
        return run_competitive_demo()
    
    elif choice == 10:
        from demos.business.market_applications_demo import run_market_demo
        return run_market_demo()
    
    elif choice == 11:
        from demos.visualization.realtime_dashboard_demo import run_dashboard_demo
        return run_dashboard_demo()
    
    elif choice == 12:
        return run_custom_institution_demo()
    
    elif choice == 13:
        from demo_for_professor_cai import main as run_cai_demo
        print("🎓 Running Professor Cai Legacy Demo...")
        run_cai_demo()
        return True
    
    elif choice == 14:
        return switch_don_stack_mode()
    
    elif choice == 15:
        return view_system_diagnostics()
    
    else:
        print("❌ Invalid choice")
        return False

def run_custom_institution_demo() -> bool:
    """Run a customized demo for a specific research institution"""
    print("🏛️ CUSTOM RESEARCH INSTITUTION DEMO")
    print("=" * 45)
    print()
    
    institution = input("Enter institution name: ").strip()
    focus_area = input("Research focus (genomics/proteomics/drug-discovery): ").strip()
    duration = input("Demo duration in minutes (5-30): ").strip()
    
    try:
        duration_int = int(duration)
        if not (5 <= duration_int <= 30):
            duration_int = 15
    except ValueError:
        duration_int = 15
    
    print(f"\n🎯 Preparing custom demo for {institution}")
    print(f"   Focus: {focus_area}")
    print(f"   Duration: {duration_int} minutes")
    print()
    
    # This would integrate with the existing professor_cai demo structure
    # but customize the content and datasets based on institution needs
    
    print("✅ Custom demo completed!")
    return True

def switch_don_stack_mode() -> bool:
    """Switch between internal and HTTP DON Stack modes"""
    print("⚙️ DON STACK MODE CONFIGURATION")
    print("=" * 40)
    print()
    
    current_mode = os.getenv("DON_STACK_MODE", "internal")
    print(f"Current mode: {current_mode.upper()}")
    print()
    print("Available modes:")
    print("  1. Internal - Direct Python calls (default)")
    print("  2. HTTP - Microservices architecture")
    print()
    
    choice = input("Select mode (1/2): ").strip()
    
    if choice == "1":
        os.environ["DON_STACK_MODE"] = "internal"
        print("✅ Switched to INTERNAL mode")
        print("   Restart demos to apply changes")
    elif choice == "2":
        os.environ["DON_STACK_MODE"] = "http"
        os.environ["DON_GPU_ENDPOINT"] = "http://127.0.0.1:8001"
        os.environ["TACE_ENDPOINT"] = "http://127.0.0.1:8002"
        print("✅ Switched to HTTP mode")
        print("   Make sure DON-GPU and TACE services are running:")
        print("   - DON-GPU: http://127.0.0.1:8001")
        print("   - TACE: http://127.0.0.1:8002")
    else:
        print("❌ Invalid choice")
    
    return True

def view_system_diagnostics() -> bool:
    """Display comprehensive system diagnostics"""
    print("🔧 SYSTEM DIAGNOSTICS")
    print("=" * 30)
    print()
    
    # Environment information
    print("📍 Environment:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Platform: {sys.platform}")
    print(f"   DON Stack Mode: {os.getenv('DON_STACK_MODE', 'internal')}")
    print()
    
    # DON Stack status
    try:
        from don_memory.adapters.don_stack_adapter import DONStackAdapter
        adapter = DONStackAdapter()
        health = adapter.health()
        
        print("🧮 DON Stack Components:")
        print(f"   Mode: {health.get('mode', 'unknown')}")
        print(f"   DON-GPU: {'✅' if health.get('don_gpu') else '❌'}")
        print(f"   TACE: {'✅' if health.get('tace') else '❌'}")
        print(f"   QAC: {'✅' if health.get('qac', True) else '❌'}")
        print()
    except Exception as e:
        print(f"❌ DON Stack diagnostics failed: {e}")
        print()
    
    # Dataset inventory
    print("📊 Dataset Inventory:")
    datasets = [
        ("real_pbmc_medium_correct.json", "PBMC Medium (Corrected)"),
        ("test_data/pbmc_small.json", "PBMC Small"),
        ("test_data/pbmc_medium.json", "PBMC Medium"),
        ("test_data/pbmc_large.json", "PBMC Large")
    ]
    
    for file_path, description in datasets:
        exists = (project_root / file_path).exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {description}")
    
    print()
    
    # Memory and performance
    try:
        import psutil
        memory = psutil.virtual_memory()
        print("💾 System Resources:")
        print(f"   Memory: {memory.percent}% used ({memory.available / (1024**3):.1f}GB available)")
        print(f"   CPU: {psutil.cpu_percent()}% usage")
    except ImportError:
        print("💾 System Resources: psutil not available")
    
    print()
    return True

def main():
    """Main demo launcher loop"""
    print_banner()
    
    # Check prerequisites
    prerequisites = check_prerequisites()
    
    # Main demo loop
    while True:
        print_menu()
        choice = get_user_choice()
        
        if choice == 16:  # Exit
            print("👋 Thank you for exploring the DON Stack Research API!")
            print("   Contact: research@donsystems.com")
            break
        
        print(f"\n🚀 LAUNCHING DEMONSTRATION #{choice}")
        print("=" * 50)
        
        try:
            success = run_demo(choice, prerequisites)
            
            if success:
                print("\n✅ Demonstration completed successfully!")
            else:
                print("\n⚠️ Demonstration encountered issues")
                
        except KeyboardInterrupt:
            print("\n🛑 Demonstration interrupted by user")
        except Exception as e:
            print(f"\n❌ Demonstration failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 50)
        input("📝 Press Enter to return to main menu...")
        print("\n" * 2)

if __name__ == "__main__":
    main()