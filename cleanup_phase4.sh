#!/bin/bash

# Energy Forecast Repository Cleanup - Phase 4
# Final verification and commit preparation

set -e

echo "=========================================="
echo "Energy Forecast Cleanup - Phase 4"
echo "Final verification and commit"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo "Error: Please run this script from the energy-forecast root directory"
    exit 1
fi

echo ""
echo "Step 1: Verifying directory structure..."
echo ""

# Check essential directories exist
directories=("src" "scripts" "notebooks" "deployment" "docs" "tests" "config")
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir/ exists"
    else
        echo "  ✗ Warning: $dir/ not found"
    fi
done

echo ""
echo "Step 2: Checking for remaining problematic files..."
echo ""

# Check for files that shouldn't be there
problematic_files=0

if ls *.zip 2>/dev/null; then
    echo "  ✗ Warning: ZIP files still present"
    problematic_files=$((problematic_files + 1))
else
    echo "  ✓ No ZIP files"
fi

if ls *.pdf 2>/dev/null; then
    echo "  ✗ Warning: PDF files still present"
    problematic_files=$((problematic_files + 1))
else
    echo "  ✓ No PDF files"
fi

if find . -name "__pycache__" 2>/dev/null | grep -q .; then
    echo "  ✗ Warning: __pycache__ directories still present"
    problematic_files=$((problematic_files + 1))
else
    echo "  ✓ No __pycache__ directories"
fi

if ls *copy.py *Copy.py 2>/dev/null; then
    echo "  ✗ Warning: Copy files still present"
    problematic_files=$((problematic_files + 1))
else
    echo "  ✓ No copy files"
fi

echo ""
echo "Step 3: Creating .gitkeep files for empty directories..."
touch data/processed/.gitkeep 2>/dev/null || true
touch data/external/.gitkeep 2>/dev/null || true
touch models/baseline/.gitkeep 2>/dev/null || true
touch models/advanced/.gitkeep 2>/dev/null || true
touch models/production/.gitkeep 2>/dev/null || true
echo "  ✓ .gitkeep files created"

echo ""
echo "Step 4: Git status summary..."
echo ""
git status --short | head -20
echo ""
echo "  (Showing first 20 changes, run 'git status' for full list)"

echo ""
echo "Step 5: Counting changes..."
added=$(git status --short | grep "^??" | wc -l)
modified=$(git status --short | grep "^ M" | wc -l)
deleted=$(git status --short | grep "^ D" | wc -l)

echo "  • New files: $added"
echo "  • Modified files: $modified"
echo "  • Deleted files: $deleted"

echo ""
echo "=========================================="
if [ $problematic_files -eq 0 ]; then
    echo "✓ Phase 4 Complete - Repository is clean!"
else
    echo "⚠ Phase 4 Complete with warnings"
    echo "  Please review and fix the warnings above"
fi
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review all changes: git status"
echo "2. Review specific files: git diff"
echo "3. Stage changes: git add -A"
echo "4. Commit: git commit -m 'Clean up repository structure'"
echo "5. Test that everything works:"
echo "   - Run tests: pytest tests/"
echo "   - Try training: python scripts/train.py"
echo "   - Try evaluation: python scripts/evaluate.py"
echo ""
echo "Once verified:"
echo "6. Push to remote: git push origin repo-cleanup"
echo "7. Create pull request to merge into main"
echo ""