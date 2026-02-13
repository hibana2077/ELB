import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import save_summary, get_base_args, set_seed
from src.data_loader import get_synthetic_data
from src.models import SimpleRealNVP, train_model
from src.elb import landscape_analysis
from src.analysis import compute_clustering_metrics

def main():
    parser = get_base_args()
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # 1. Data
    print("Generating Toy Data...")
    X, y_true = get_synthetic_data(n_samples=args.n_samples, n_clusters=3)
    
    # Normalize X for better training
    X = (X - X.mean(0)) / X.std(0)
    
    # 2. Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleRealNVP(input_dim=2, hidden_dim=64, n_layers=6).to(device)
    
    # 3. Train
    loss_history = train_model(model, X, epochs=200, lr=0.005, device=device)
    
    # 4. ELB Analysis
    print("Performing ELB Analysis...")
    elb, labels_pred, minima = landscape_analysis(model, X, device=device)
    
    # 5. Metrics
    metrics = compute_clustering_metrics(X, y_true, labels_pred)
    metrics['final_loss'] = loss_history[-1]
    
    print("\nResults:")
    print(metrics)
    
    # 6. Save
    exp_dir = os.path.join(args.out_dir, "toy_experiment")
    save_summary(exp_dir, metrics, args)
    
    # Optional: Plot
    # plt.figure(figsize=(10,5))
    # plt.scatter(X[:,0], X[:,1], c=labels_pred, cmap='tab10', alpha=0.5)
    # plt.scatter(minima[:,0], minima[:,1], c='red', marker='x', s=100, label='Minima')
    # plt.title("ELB Basins")
    # plt.savefig(os.path.join(exp_dir, " basins.png"))

if __name__ == "__main__":
    main()
