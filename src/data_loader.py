import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.decomposition import PCA

try:
    import GEOparse
except ImportError:
    GEOparse = None

def load_geo_dataset(gse_id, destdir="./geo_data"):
    if GEOparse is None:
        raise ImportError("GEOparse not installed.")
        
    print(f"Loading {gse_id}...")
    try:
        gse = GEOparse.get_GEO(geo=gse_id, destdir=destdir)
    except Exception as e:
        print(f"Failed to download {gse_id}: {e}")
        return None, None

    # Extract expression matrix
    # GEOparse usually puts data in gse.pivot_samples(value_col_name) if it's GSMs
    # Often for GSE series, gse.gpls[platform].table or similar.
    # But usually `gse.pivot_samples('VALUE')` works for simple cases
    
    # Try basic pivoting
    try:
        df = gse.pivot_samples("VALUE")
    except:
        # Fallback for some datasets that might be simpler
        print("Pivot failed, trying raw columns if available...")
        # This is highly dependent on specific file structure.
        # For demo, we return dummy if fail.
        return None, None
        
    metadata = gse.metadata
    return df, metadata

def preprocess_expression(df, log2=True, quantile_norm=True, n_pca=20):
    """
    Standard microarray preprocessing pipeline.
    1. Log2 transform (if not done)
    2. Quantile Normalization
    3. PCA
    """
    data = df.values.T # (Samples, Genes)
    
    # Handle NaN
    if np.isnan(data).any():
        print("Filling NaNs with 0")
        data = np.nan_to_num(data)
    
    # Check if log needed (simple heuristic: max value > 20 probably raw intensity)
    if log2 and np.max(data) > 20: 
        print("Applying Log2 transform")
        data = np.log2(data + 1)
        
    if quantile_norm:
        print("Applying Quantile Normalization")
        qt = QuantileTransformer(output_distribution='normal')
        data = qt.fit_transform(data)
        
    # PCA
    pca_model = None
    if n_pca:
        print(f"Applying PCA (n={n_pca})")
        pca_model = PCA(n_components=n_pca)
        data = pca_model.fit_transform(data)
        
    return data, pca_model

def get_synthetic_data(n_samples=500, n_features=2, n_clusters=3):
    """Gaussian Mixture + Ridge for Toy Experiment"""
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
    
    # Add some noise/ridge structure manually?
    # make_blobs is enough for basic basin testing
    return X, y

