#!/bin/bash

# Energy Forecast Repository Cleanup - Phase 1
# Removes temporary, duplicate, and unnecessary files

set -e  # Exit on error

echo "=========================================="
echo "Energy Forecast Cleanup - Phase 1"
echo "Removing temporary and duplicate files"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo "Error: Please run this script from the energy-forecast root directory"
    exit 1
fi

# Create backup branch first
echo ""
echo "Step 1: Creating backup branch..."
git checkout -b cleanup-backup 2>/dev/null || echo "Backup branch already exists"
git checkout main 2>/dev/null || git checkout master 2>/dev/null

# Create cleanup branch
echo ""
echo "Step 2: Creating cleanup branch..."
git checkout -b repo-cleanup 2>/dev/null || git checkout repo-cleanup

echo ""
echo "Step 3: Removing ZIP files..."
rm -f energy-forecast-docker-document.zip
rm -f energy-forecast-with-modified-SHAP.zip
rm -f feature-engineering-verification.zip
rm -f FInal-SHAP-Files.zip
rm -f neural-network-summary.zip
rm -f neural-nw-complete-instructions.zip
rm -f SHAP-dashboard-changes.zip
echo "  ✓ ZIP files removed"

echo ""
echo "Step 4: Removing PDF files..."
rm -f FEATURE_ENGINEERING_INTERVIEW_GUIDE.md.pdf
rm -f FEATURE_ENGINEERING_SUMMARY.md.pdf
rm -f FEATURE_ENGINEERING_SUMMARY_New.md.pdf
rm -f neural_network_summary.md.pdf
rm -f neural_network_summary.pdf
rm -f README.md.pdf
echo "  ✓ PDF files removed"

echo ""
echo "Step 5: Removing duplicate/copy files..."
rm -f "setup_project_structure copy.py"
rm -f "train_and_save_for_shap copy.py"
rm -f "interactive_shap_app copy.py"
rm -f "Advanced Model Summary"
rm -f FINAL_INTERACTIVE_SUMMARY.txt
rm -f SESSION_SUMMARY.md
echo "  ✓ Duplicate files removed"

echo ""
echo "Step 6: Removing old experiment files..."
rm -f compare_models.py
rm -f diagnose.py
rm -f find_features.py
rm -f test_baseline.py
rm -f verify_features.py
rm -f visualize_comparison.py
echo "  ✓ Old experiment files removed"

echo ""
echo "Step 7: Removing multiple SHAP variations..."
rm -f demo_shap_visualizations.py
rm -f shap_visualizations.py
rm -f simpleshape.py
rm -f simple_shap.py
rm -f simplest_shap.py
echo "  ✓ SHAP variations removed"

echo ""
echo "Step 8: Removing old gradio apps..."
rm -f gradio_app_DEPLOYMENT_READY.py
rm -f gradio_app_enhanced.py
rm -f gradio_app_with_shap.py
rm -f interactive_shap_app.py
echo "  ✓ Old gradio apps removed"

echo ""
echo "Step 9: Removing training scripts from root..."
rm -f train_and_save_for_deploy_shap.py
echo "  ✓ Training scripts removed"

echo ""
echo "Step 10: Removing blog/documentation drafts..."
rm -f blog_post_comparison.md
rm -f BLOG_POST_OUTLINE.md
rm -f ADVANCED_MODELS_RATIONALE.md
rm -f VARIANCE_MEAN_EXPLANATION.md
rm -f PIPELINE_GUIDE.md
rm -f HF_DEPLOYMENT_GUIDE.md
rm -f INTERACTIVE_APP_GUIDE.md
echo "  ✓ Documentation drafts removed"

echo ""
echo "Step 11: Removing duplicate folders..."
rm -rf energy-forecast-docker-document/
rm -rf feature-engineering-verification/
rm -rf neural-network-summary/
rm -rf tiny-examples/
rm -rf experiments/
echo "  ✓ Duplicate folders removed"

echo ""
echo "Step 12: Removing duplicate deployment files from root..."
rm -f docker-compose.yml
rm -f Dockerfile
echo "  ✓ Duplicate deployment files removed"

echo ""
echo "Step 13: Cleaning deployment/api folder..."
cd deployment/api
rm -f app-old.py
rm -f app-gemini.py
rm -f app-gradio.py
rm -f app-streamlit.py
rm -f app_final_working.py
# Keep only app.py and test_api.py
cd ../..
echo "  ✓ Old API files removed"

echo ""
echo "=========================================="
echo "Phase 1 Complete!"
echo "=========================================="
echo ""
echo "Files removed. Next steps:"
echo "1. Run: bash cleanup_phase2.sh (to clean Python cache)"
echo "2. Review changes: git status"
echo "3. Test that code still works"
echo ""