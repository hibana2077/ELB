#!/bin/bash

# Stop on error
set -e

echo "========================================="
echo "       ELB Experiment Runner"
echo "========================================="

# 1. Install Dependencies
echo "[1/5] Installing dependencies..."
pip install -r requirements.txt

# 2. Setup Environment
# Add current directory to PYTHONPATH so python can find 'src'
export PYTHONPATH=$PYTHONPATH:.

# 3. Create Output Directory
mkdir -p results

# 4. Run Experiments
echo "----------------------------------------------------------------"
echo "[2/5] Running Toy Experiment..."
python experiments/toy_experiment.py --out_dir ./results --n_samples 1000

echo "----------------------------------------------------------------"
echo "[3/5] Running Real Data Experiment (GSE2034)..."
# Pass --no_download if you want to force synthetic data, 
# otherwise it attempts to download from GEO.
python experiments/real_data_experiment.py --gse_id GSE2034 --out_dir ./results --pca_dim 20

echo "----------------------------------------------------------------"
echo "[4/5] Running Robustness Experiment..."
python experiments/robustness.py --out_dir ./results

# 5. Finalize
echo "----------------------------------------------------------------"
echo "[5/5] Processing Results..."

# Display summaries
echo "Experiment Summaries:"
find results -name "summary.json" -type f -print0 | xargs -0 -I {} sh -c 'echo "File: {}"; cat "{}"; echo "\n"'

# Zip results
zip -r elb_results.zip results
echo "Results zipped to: elb_results.zip"
