#!/bin/bash

# Energy Forecast Repository - Complete Cleanup
# Master script to run all cleanup phases

set -e

echo "=========================================="
echo "Energy Forecast Repository Cleanup"
echo "Complete Automated Cleanup Process"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Create backup branch"
echo "  2. Remove temporary and duplicate files"
echo "  3. Clean Python cache"
echo "  4. Create comprehensive .gitignore"
echo "  5. Verify the cleanup"
echo ""
echo "Your current work will be saved in 'cleanup-backup' branch"
echo ""

read -p "Continue with cleanup? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo "Error: Please run this script from the energy-forecast root directory"
    exit 1
fi

echo ""
echo "Starting cleanup process..."
echo ""

# Phase 1: Remove temporary and duplicate files
echo "=========================================="
echo "Running Phase 1: File Removal"
echo "=========================================="
bash cleanup_phase1.sh

# Phase 2: Clean Python cache
echo ""
echo "=========================================="
echo "Running Phase 2: Python Cache Cleanup"
echo "=========================================="
bash cleanup_phase2.sh

# Phase 3: Create .gitignore
echo ""
echo "=========================================="
echo "Running Phase 3: .gitignore Creation"
echo "=========================================="
bash cleanup_phase3.sh

# Phase 4: Verification
echo ""
echo "=========================================="
echo "Running Phase 4: Verification"
echo "=========================================="
bash cleanup_phase4.sh

echo ""
echo "=========================================="
echo "✓ ALL CLEANUP PHASES COMPLETE!"
echo "=========================================="
echo ""
echo "Repository has been cleaned up successfully!"
echo ""
echo "IMPORTANT: Before committing, please:"
echo "1. Review changes: git status"
echo "2. Test your code still works:"
echo "   cd ~/portfolio-project/energy-forecast"
echo "   python scripts/demo_neural_network.py"
echo ""
echo "If everything works:"
echo "3. Stage all changes: git add -A"
echo "4. Commit: git commit -m 'Clean up repository structure for portfolio'"
echo "5. Push: git push origin repo-cleanup"
echo ""