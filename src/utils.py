import torch
import numpy as np

def compute_softdtw(D, gamma=1.0):
    """
    Computes SoftDTW cost matrix and distance.
    D: (B, N, M) cost matrix
    gamma: smoothing parameter
    Returns:
        cost: (B, ) soft-dtw distance
        R: (B, N+1, M+1) accumulated cost matrix (optional, for debugging/visualizing)
    """
    B, N, M = D.shape
    device = D.device
    
    # Initialize R with infinity
    R = torch.full((B, N + 1, M + 1), float('inf'), device=device)
    R[:, 0, 0] = 0
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            # Costs from neighbors: i-1,j-1 (match), i-1,j (insert), i,j-1 (delete)
            # Standard DTW usually allows these 3 moves.
            r0 = R[:, i-1, j-1]
            r1 = R[:, i-1, j]
            r2 = R[:, i, j-1]
            
            # Softmin of (r0, r1, r2)
            # softmin(x) = -gamma * log(sum(exp(-x/gamma)))
            vals = torch.stack([r0, r1, r2], dim=1)
            
            # efficient softmin with stability
            min_val, _ = torch.min(vals, dim=1)
            # If gamma is very small, behave like min
            if gamma < 1e-4:
                soft_v = min_val
            else:
                soft_v = min_val - gamma * torch.log(torch.sum(torch.exp(-(vals - min_val.unsqueeze(1)) / gamma), dim=1))
            
            R[:, i, j] = D[:, i-1, j-1] + soft_v
            
    return R[:, N, M]

def get_alignment_matrix(D, gamma=1.0):
    """
    Computes the alignment matrix (probability of path passing through i,j).
    This is the gradient of SoftDTW(D) with respect to D.
    """
    D_ = D.clone().detach().requires_grad_(True)
    loss = compute_softdtw(D_, gamma).sum()
    loss.backward()
    return D_.grad

def generate_synthetic_data(num_samples=100, seq_len=16, vocab_size=10):
    """
    Generates X, Y sequences. 
    Y is X with some noise/shifts.
    Returns: X, Y, GT_Align_Mask
    """
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    # Y is just X for now (perfect match preferred), or slight mutation
    # To test alignment, let's make Y a shifted version of X by 1
    Y = torch.roll(X, shifts=1, dims=1)
    
    # Ground truth alignment: X[i] matches Y[i+1] (circular)
    # i.e., Align matrix has 1s at (i, (i+1)%L)
    GT = torch.zeros((num_samples, seq_len, seq_len))
    for b in range(num_samples):
        for i in range(seq_len):
            GT[b, i, (i+1)%seq_len] = 1.0
            
    return X, Y, GT
