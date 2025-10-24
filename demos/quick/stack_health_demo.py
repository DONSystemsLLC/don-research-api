#!/usr/bin/env python3
"""
Stack Health Demo - Quick system verification
============================================

Validates DON Stack components and API availability.
Perfect for troubleshooting and pre-demo setup verification.
"""

import sys
import time
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

def run_stack_health_demo() -> bool:
    """Execute the stack health demonstration"""
    
    print("🏥 DON STACK HEALTH VERIFICATION")
    print("=" * 45)
    print("Comprehensive system check for demos and production use")
    print()
    
    success = True
    
    # 1. API Server Health
    print("1️⃣ API SERVER STATUS")
    print("-" * 25)
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server: {data['status']}")
            print(f"✅ Service: {data['service']}")
            print(f"✅ DON Stack: {data['don_stack_status']}")
            print(f"✅ Version: {data['version']}")
        else:
            print(f"❌ API Server error: Status {response.status_code}")
            success = False
    except Exception as e:
        print(f"❌ API Server connection failed: {e}")
        print("   💡 Start server with: python main.py")
        success = False
    
    print()
    
    # 2. DON Stack Adapter
    print("2️⃣ DON STACK ADAPTER")
    print("-" * 25)
    try:
        from don_memory.adapters.don_stack_adapter import DONStackAdapter
        adapter = DONStackAdapter()
        health = adapter.health()
        
        print(f"✅ Adapter loaded: {health['mode']} mode")
        print(f"✅ DON-GPU: {'Available' if health['don_gpu'] else 'Fallback'}")
        print(f"✅ TACE: {'Available' if health['tace'] else 'Fallback'}")
        
        # Quick adapter test
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = adapter.normalize(test_data)
        print(f"✅ Adapter test: {len(test_data)}→{len(normalized)} dimensions")
        
    except Exception as e:
        print(f"❌ DON Stack Adapter failed: {e}")
        success = False
    
    print()
    
    # 3. API Health Endpoint
    print("3️⃣ DETAILED API HEALTH")
    print("-" * 25)
    try:
        response = requests.get("http://localhost:8080/api/v1/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            
            print("DON Stack Components:")
            don_stack = health.get('don_stack', {})
            print(f"   Mode: {don_stack.get('mode', 'unknown')}")
            print(f"   DON-GPU: {'✅' if don_stack.get('don_gpu') else '❌'}")
            print(f"   TACE: {'✅' if don_stack.get('tace') else '❌'}")
            print(f"   QAC: {'✅' if don_stack.get('qac') else '❌'}")
            
            print("QAC Configuration:")
            qac = health.get('qac', {})
            engines = qac.get('supported_engines', [])
            print(f"   Engines: {', '.join(engines)}")
            print(f"   Default: {qac.get('default_engine', 'unknown')}")
            
        else:
            print(f"❌ Health endpoint error: Status {response.status_code}")
            success = False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        success = False
    
    print()
    
    # 4. Authentication Test
    print("4️⃣ AUTHENTICATION TEST")
    print("-" * 25)
    try:
        headers = {"Authorization": "Bearer demo_token"}
        response = requests.get("http://localhost:8080/api/v1/usage", headers=headers, timeout=5)
        
        if response.status_code == 200:
            usage = response.json()
            print(f"✅ Auth successful: {usage['institution']}")
            print(f"✅ Rate limit: {usage['requests_used']}/{usage.get('rate_limit', 'Unknown')}")
        else:
            print(f"❌ Authentication failed: Status {response.status_code}")
            success = False
            
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        success = False
    
    print()
    
    # 5. Performance Baseline
    print("5️⃣ PERFORMANCE BASELINE")
    print("-" * 25)
    try:
        import numpy as np
        
        # Test small compression request
        test_request = {
            "data": {
                "gene_names": [f"Gene_{i}" for i in range(10)],
                "expression_matrix": np.random.rand(5, 10).tolist()
            },
            "compression_target": 4,
            "seed": 42
        }
        
        headers = {
            "Authorization": "Bearer demo_token",
            "Content-Type": "application/json"
        }
        
        start_time = time.perf_counter()
        response = requests.post(
            "http://localhost:8080/api/v1/genomics/compress",
            json=test_request,
            headers=headers,
            timeout=10
        )
        elapsed = (time.perf_counter() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            stats = result.get('compression_stats', {})
            print(f"✅ Test compression: {elapsed:.1f}ms")
            print(f"✅ Compression: {stats.get('original_dimensions')}→{stats.get('compressed_dimensions')}")
            print(f"✅ Algorithm: {result.get('algorithm', 'Unknown')}")
        else:
            print(f"❌ Compression test failed: Status {response.status_code}")
            success = False
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        success = False
    
    print()
    
    # Summary
    print("📊 HEALTH CHECK SUMMARY")
    print("=" * 30)
    if success:
        print("✅ All systems operational - Ready for demonstrations!")
        print("🚀 You can now run any demo scenario safely")
    else:
        print("⚠️ Some issues detected - Check the errors above")
        print("💡 Common fixes:")
        print("   • Start API server: python main.py")
        print("   • Check DON Stack mode: export DON_STACK_MODE=internal")
        print("   • Verify test data files exist")
    
    return success

if __name__ == "__main__":
    run_stack_health_demo()