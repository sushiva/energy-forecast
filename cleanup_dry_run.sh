# Copy all scripts to your repository
cd ~/portfolio-project/energy-forecast

# Make scripts executable
chmod +x cleanup_*.sh run_complete_cleanup.sh

# Run dry-run to see what will be deleted (SAFE - doesn't delete anything)
bash cleanup_dry_run.sh