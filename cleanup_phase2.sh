#!/bin/bash

# Energy Forecast Repository Cleanup - Phase 2
# Clean Python cache and temporary files

set -e

echo "=========================================="
echo "Energy Forecast Cleanup - Phase 2"
echo "Cleaning Python cache and temp files"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo "Error: Please run this script from the energy-forecast root directory"
    exit 1
fi

echo ""
echo "Step 1: Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ __pycache__ directories removed"

echo ""
echo "Step 2: Removing .pyc files..."
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "  ✓ .pyc files removed"

echo ""
echo "Step 3: Removing .pyo files..."
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "  ✓ .pyo files removed"

echo ""
echo "Step 4: Removing .pyd files..."
find . -type f -name "*.pyd" -delete 2>/dev/null || true
echo "  ✓ .pyd files removed"

echo ""
echo "Step 5: Removing logs directory if exists..."
if [ -d "logs" ]; then
    rm -rf logs/
    echo "  ✓ Logs directory removed"
else
    echo "  ✓ No logs directory found"
fi

echo ""
echo "Step 6: Checking for .DS_Store files (macOS)..."
find . -name ".DS_Store" -delete 2>/dev/null || true
echo "  ✓ .DS_Store files removed"

echo ""
echo "=========================================="
echo "Phase 2 Complete!"
echo "=========================================="
echo ""
echo "Python cache cleaned. Next steps:"
echo "1. Run: bash cleanup_phase3.sh (to create .gitignore)"
echo "2. Review changes: git status"
echo ""