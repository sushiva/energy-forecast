#!/bin/bash

# Energy Forecast Repository Cleanup - DRY RUN (Simple Version)
# Shows what will be deleted without actually deleting anything

echo "=========================================="
echo "Energy Forecast Cleanup - DRY RUN"
echo "Preview of files to be removed"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo "Error: Please run this script from the energy-forecast root directory"
    exit 1
fi

echo ""
echo "ZIP FILES TO BE REMOVED:"
find . -maxdepth 1 -name "*.zip" -type f 2>/dev/null || echo "  None found"

echo ""
echo "PDF FILES TO BE REMOVED:"
find . -maxdepth 1 -name "*.pdf" -type f 2>/dev/null || echo "  None found"

echo ""
echo "DUPLICATE/COPY FILES TO BE REMOVED:"
find . -maxdepth 1 -name "*copy.py" -type f 2>/dev/null
find . -maxdepth 1 -name "*Copy.py" -type f 2>/dev/null
find . -maxdepth 1 -name "Advanced Model Summary" -type f 2>/dev/null

echo ""
echo "FOLDERS TO BE REMOVED:"
for folder in energy-forecast-docker-document feature-engineering-verification neural-network-summary tiny-examples experiments; do
    if [ -d "$folder" ]; then
        size=$(du -sh "$folder" 2>/dev/null | cut -f1)
        files=$(find "$folder" -type f 2>/dev/null | wc -l)
        echo "  $folder/ ($size, $files files)"
    fi
done

echo ""
echo "__pycache__ DIRECTORIES:"
pycache_count=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
echo "  Found: $pycache_count directories"

echo ""
echo "=========================================="
echo "This was a DRY RUN - no files were deleted"
echo "=========================================="
echo ""
echo "To proceed with cleanup, run:"
echo "  bash run_complete_cleanup.sh"
echo ""