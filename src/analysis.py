import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
except ImportError:
    KaplanMeierFitter = None

def compute_clustering_metrics(X, labels_true, labels_pred):
    metrics = {}
    if labels_true is not None:
        metrics['ARI'] = adjusted_rand_score(labels_true, labels_pred)
        metrics['NMI'] = normalized_mutual_info_score(labels_true, labels_pred)
    
    if len(np.unique(labels_pred)) > 1:
        metrics['Silhouette'] = silhouette_score(X, labels_pred)
    else:
        metrics['Silhouette'] = -1.0
        
    metrics['n_clusters'] = len(np.unique(labels_pred))
    return metrics

def survival_analysis(T, E, labels):
    """
    T: Time
    E: Event (dead/alive 1/0)
    labels: Cluster assignment
    """
    results = {}
    if KaplanMeierFitter is None:
        print("lifelines not installed, skipping survival analysis")
        return results

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {'p_value': 1.0}
    
    # Pairwise Log-Rank or Multivariate
    # Simple: Just return p-value of multivariate log-rank test if possible
    # or just perform test between all groups
    
    try:
        from lifelines.statistics import multivariate_logrank_test
        result = multivariate_logrank_test(T, labels, E)
        results['p_value'] = result.p_value
    except Exception as e:
        print(f"Survival analysis failed: {e}")
        results['p_value'] = None
        
    return results
