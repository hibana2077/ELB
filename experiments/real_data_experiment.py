import sys
import os
import torch
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import save_summary, get_base_args, set_seed
from src.data_loader import load_geo_dataset, preprocess_expression
from src.models import SimpleRealNVP, train_model
from src.elb import landscape_analysis
from src.analysis import compute_clustering_metrics, survival_analysis

def parse_survival_data(metadata_df):
    """
    Attempt to parse clinical data for survival analysis.
    Very specific to dataset formats.
    """
    # Placeholder: Look for common keywords
    # This is non-trivial for generic GEO datasets
    return None, None

def main():
    parser = get_base_args()
    parser.add_argument("--gse_id", type=str, default="GSE2034")
    parser.add_argument("--pca_dim", type=int, default=20)
    parser.add_argument("--no_download", action="store_true", help="Use synthetic data if download fails")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # 1. Data
    df, metadata = None, None
    if not args.no_download:
        df, metadata = load_geo_dataset(args.gse_id)
        
    if df is None:
        print("Using synthetic high-dim data (Fallback)...")
        from sklearn.datasets import make_classification
        X_raw, y_true = make_classification(n_samples=200, n_features=1000, n_informative=50, n_classes=3)
        # Mock dataframe structure
        df = pd.DataFrame(X_raw.T) # Genes x Samples usually
    else:
        # Preprocess assumes Samples x Genes or Genes x Samples?
        # data_loader.preprocess_expression assumes df.values.T -> (Samples, Genes)
        # Usually GEO df is Genes (rows) x Samples (cols)
        pass

    # 2. Preprocess
    print("Preprocessing...")
    if isinstance(df, pd.DataFrame):
         # df is Genes x Samples. preprocess expects it and transposes inside.
         X, pca_model = preprocess_expression(df, n_pca=args.pca_dim)
    else:
         # Already numpy array from synthetic
         X = df
         
    print(f"Data shape: {X.shape}")
    
    # 3. Model (RealNVP)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Input dim is pca_dim
    input_dim = X.shape[1]
    model = SimpleRealNVP(input_dim=input_dim, hidden_dim=64, n_layers=8).to(device)
    
    # 4. Train
    loss_history = train_model(model, X, epochs=300, lr=0.001, device=device)
    
    # 5. ELB
    print("Performing ELB Analysis...")
    elb, labels_pred, minima = landscape_analysis(model, X, device=device)
    
    # 6. Metrics
    metrics = compute_clustering_metrics(X, None, labels_pred) # No Ground Truth for generic GEO
    metrics['final_loss'] = loss_history[-1]
    metrics['n_minima'] = len(minima)
    
    # Survival analysis if possible (skipped for generic implementation)
    
    print("\nResults:")
    print(metrics)
    
    # 7. Save
    exp_dir = os.path.join(args.out_dir, f"{args.gse_id}_experiment")
    save_summary(exp_dir, metrics, args)

if __name__ == "__main__":
    main()
