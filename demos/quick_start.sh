#!/bin/bash
# DON Stack Demo Quick Start Script
# =================================
# 
# This script helps you quickly set up and launch the DON Stack demo system.
# It checks prerequisites, starts the API server if needed, and launches demos.

set -e  # Exit on error

echo "🧬 DON Stack Research API - Demo Quick Start"
echo "==========================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is in use
port_in_use() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

# Check Python
echo "🐍 Checking Python installation..."
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found. Please install Python 3.7+ and try again."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo "✅ Python $PYTHON_VERSION found"

# Check dependencies
echo ""
echo "📦 Checking dependencies..."
cd "$PROJECT_ROOT"

if [ -f "requirements.txt" ]; then
    echo "Installing/updating dependencies..."
    $PYTHON_CMD -m pip install -r requirements.txt --quiet
    echo "✅ Dependencies updated"
else
    echo "⚠️  No requirements.txt found, checking core dependencies..."
    $PYTHON_CMD -c "import numpy, requests, json, pathlib" 2>/dev/null && echo "✅ Core dependencies available" || {
        echo "❌ Missing dependencies. Installing numpy and requests..."
        $PYTHON_CMD -m pip install numpy requests
    }
fi

# Check API server
echo ""
echo "🚀 Checking API server..."
if port_in_use 8080; then
    echo "✅ API server already running on port 8080"
    SERVER_RUNNING=true
else
    echo "📡 Starting API server..."
    # Start server in background
    cd "$PROJECT_ROOT"
    $PYTHON_CMD main.py &
    SERVER_PID=$!
    SERVER_RUNNING=false
    
    # Wait for server to start
    echo "Waiting for server to initialize..."
    for i in {1..10}; do
        if port_in_use 8080; then
            echo "✅ API server started successfully (PID: $SERVER_PID)"
            SERVER_RUNNING=true
            break
        fi
        sleep 1
        echo -n "."
    done
    
    if [ "$SERVER_RUNNING" = false ]; then
        echo "❌ Failed to start API server. Please check the logs."
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
fi

# Test API connectivity
echo ""
echo "🔌 Testing API connectivity..."
if curl -s http://localhost:8080/ >/dev/null 2>&1; then
    echo "✅ API server responding"
else
    echo "❌ API server not responding. Please check manually."
    exit 1
fi

# Check test data
echo ""
echo "📊 Checking test data..."
TEST_DATA_COUNT=0

for file in "real_pbmc_medium_correct.json" "test_data/pbmc_small.json" "test_data/pbmc_medium.json"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        ((TEST_DATA_COUNT++))
    fi
done

if [ $TEST_DATA_COUNT -gt 0 ]; then
    echo "✅ Test data available ($TEST_DATA_COUNT datasets found)"
else
    echo "⚠️  No test data found - demos will use synthetic data"
fi

# Launch options
echo ""
echo "🎯 Demo Launch Options:"
echo "======================"
echo ""
echo "1. Interactive Demo Launcher (Recommended)"
echo "2. Quick Health Check"
echo "3. Basic Compression Demo"
echo "4. Quantum vs Classical Demo"
echo "5. Business ROI Demo"
echo "6. Technical Deep Dive"
echo "7. View Available Demos"
echo "8. Exit"
echo ""

while true; do
    read -p "👉 Select option (1-8): " choice
    echo ""
    
    case $choice in
        1)
            echo "🚀 Launching Interactive Demo Launcher..."
            cd "$PROJECT_ROOT/demos"
            $PYTHON_CMD demo_launcher.py
            break
            ;;
        2)
            echo "🏥 Running Health Check Demo..."
            cd "$PROJECT_ROOT"
            $PYTHON_CMD demos/quick/stack_health_demo.py
            break
            ;;
        3)
            echo "🧬 Running Basic Compression Demo..."
            cd "$PROJECT_ROOT"
            $PYTHON_CMD demos/quick/basic_compression_demo.py
            break
            ;;
        4)
            echo "🔀 Running Quantum vs Classical Demo..."
            cd "$PROJECT_ROOT"
            $PYTHON_CMD demos/quick/quantum_vs_classical_demo.py
            break
            ;;
        5)
            echo "💼 Running Business ROI Demo..."
            cd "$PROJECT_ROOT"
            $PYTHON_CMD demos/business/roi_performance_demo.py
            break
            ;;
        6)
            echo "🧮 Running Technical Deep Dive..."
            cd "$PROJECT_ROOT"
            $PYTHON_CMD demos/technical/don_gpu_deep_dive.py
            break
            ;;
        7)
            echo "📋 Available Demo Categories:"
            echo "============================"
            echo ""
            echo "Quick Demos (2-5 min):"
            echo "  • Stack Health Check"
            echo "  • Basic Compression"
            echo "  • Quantum vs Classical"
            echo ""
            echo "Technical Deep-Dives (7-15 min):"
            echo "  • DON-GPU Fractal Clustering"
            echo "  • QAC Error Correction"
            echo "  • TACE Temporal Control"
            echo "  • Full Pipeline Integration"
            echo ""
            echo "Business Presentations (10-15 min):"
            echo "  • ROI & Performance"
            echo "  • Competitive Analysis"
            echo "  • Market Applications"
            echo ""
            echo "Interactive Demos (15-20 min):"
            echo "  • Real-time Dashboard"
            echo "  • Custom Institution Demos"
            echo ""
            continue
            ;;
        8)
            echo "👋 Exiting demo launcher..."
            break
            ;;
        *)
            echo "❌ Invalid option. Please select 1-8."
            continue
            ;;
    esac
done

# Cleanup function
cleanup() {
    if [ "$SERVER_RUNNING" = false ] && [ ! -z "$SERVER_PID" ]; then
        echo ""
        echo "🧹 Stopping API server..."
        kill $SERVER_PID 2>/dev/null || true
        echo "✅ Cleanup completed"
    fi
}

# Set trap for cleanup on script exit
trap cleanup EXIT

echo ""
echo "🎉 Demo session completed!"
echo ""
echo "📞 Need help? Contact:"
echo "   Technical: research@donsystems.com"
echo "   Business: business@donsystems.com"
echo ""
echo "🔗 Learn more: https://donsystems.com"