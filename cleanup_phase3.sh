#!/bin/bash

# Energy Forecast Repository Cleanup - Phase 3
# Create comprehensive .gitignore file

set -e

echo "=========================================="
echo "Energy Forecast Cleanup - Phase 3"
echo "Creating comprehensive .gitignore"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo "Error: Please run this script from the energy-forecast root directory"
    exit 1
fi

echo ""
echo "Creating .gitignore file..."

cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/
.venv/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.project
.pydevproject
.settings/

# OS specific
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/
*.log.*

# Project specific - temporary files
*.zip
*.tar.gz
*.pdf
temp/
tmp/
*copy.py
*Copy.py
*_old.py
*-old.py
*_backup.py

# Model files (large files - use Git LFS if needed)
*.pkl
*.h5
*.joblib
*.pt
*.pth
*.onnx

# Data files
data/raw/*
!data/raw/.gitkeep
!data/raw/energy-efficiency-data.csv
data/processed/*
!data/processed/.gitkeep
data/external/*
!data/external/.gitkeep

# Experiments and temporary analysis
experiments/
scratch/
sandbox/

# Jupyter Notebook checkpoints
.ipynb_checkpoints/
*/.ipynb_checkpoints/*

# Documentation builds
docs/_build/
docs/_static/
docs/_templates/

# Session files
.session/
*.session

# Environment variables
.env
.env.local
.env.*.local

# Cloud deployment credentials
*.pem
*.key
*.crt
aws_credentials
gcp_credentials.json

# Container
.dockerignore

# MacOS
*.dmg

# Test outputs
test_output/
test_results/
EOF

echo "  ✓ .gitignore created"

echo ""
echo "=========================================="
echo "Phase 3 Complete!"
echo "=========================================="
echo ""
echo ".gitignore created. Next steps:"
echo "1. Review the .gitignore file"
echo "2. Run: git status (to see what will be tracked)"
echo "3. Run: bash cleanup_phase4.sh (final verification)"
echo ""