# ELB Project

Energy Landscape-Based Subclass Discovery.

## Structure
- `src/`: Core logic (Models, ELB algorithm, Data loading).
- `experiments/`: Scripts for Toy, Real Data, and Robustness experiments.
- `ELB_Colab_Runner.ipynb`: Jupyter Notebook to run experiments on Google Colab.

## How to run on Google Colab

1. **Upload**: Upload this entire folder (or zip it and unzip it) to your Google Drive or the Colab instance.
2. **Open**: Open `ELB_Colab_Runner.ipynb`.
3. **Run**: Execute the cells in order.
   - It will install dependencies from `requirements.txt`.
   - It will run the 3 experiments defined in `experiments/`.
   - It will generate `summary.json` files in the `results/` folder.
   - It will zip the results for you to download.

## Experiments

1. **Toy Experiment**: Verifies the landscape logic on synthetic data.
2. **Real Data**: Runs on GSE2034 (or synthetic fallback).
3. **Robustness**: Bootstrap analysis of basin stability.