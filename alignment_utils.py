import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_min(x, gamma):
    # Soft-min operator: -gamma * log(sum(exp(-x/gamma)))
    # For numerical stability, use log-sum-exp trick
    return -gamma * torch.logsumexp(-x / gamma, dim=-1)

class SoftDTW(nn.Module):
    def __init__(self, gamma=1.0):
        super(SoftDTW, self).__init__()
        self.gamma = gamma

    def forward(self, D):
        """
        D: Distance matrix of shape (batch_size, N, M)
        Returns: strict soft-dtw cost
        """
        B, N, M = D.shape
        device = D.device
        
        # Initialize DP table
        # We generally pad with infinity, but for soft-min, we initialize carefully.
        # R[b, i, j] is the soft-dtw distance to reach cell (i, j)
        
        # R needs to be (B, N+1, M+1)
        # Initialization:
        # R[:, 0, 0] = 0
        # R[:, 0, j] = infinity for j > 0
        # R[:, i, 0] = infinity for i > 0
        
        R = torch.full((B, N + 1, M + 1), float('inf'), device=device)
        R[:, 0, 0] = 0
        
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                # Costs from left, top, diagonal
                r0 = R[:, i-1, j-1] # match
                r1 = R[:, i-1, j]   # insertion
                r2 = R[:, i, j-1]   # deletion
                
                # Soft min of previous paths + current cost
                # soft_min(a, b, c) = -gamma * log(exp(-a/g) + exp(-b/g) + exp(-c/g))
                
                stacked = torch.stack([r0, r1, r2], dim=1)
                soft_prev = -self.gamma * torch.logsumexp(-stacked / self.gamma, dim=1)
                
                R[:, i, j] = D[:, i-1, j-1] + soft_prev
                
        return R[:, N, M]

    def get_alignment_matrix(self, D):
        """
        Compute the gradient of the loss w.r.t D, which corresponds to the alignment matrix (expected counts).
        """
        # We need to retain grad on D if it's not a leaf, but here we usually pass a computation graph.
        # However, to get the specific gradient for the alignment matrix visualization/metric,
        # we can use autograd.grad.
        
        # Ensure D requires grad
        D_ = D.clone().detach().requires_grad_(True)
        loss = self.forward(D_)
        # We want the gradient of the sum of losses (or mean) w.r.t D_
        grad = torch.autograd.grad(loss.sum(), D_, create_graph=False)[0]
        return grad

def generate_synthetic_data(batch_size=10, len_x=20, len_y=20, vocab_size=5):
    """
    Generates random sequences and a 'true' alignment based on simple matching.
    """
    # Random integer sequences
    seq_x = torch.randint(0, vocab_size, (batch_size, len_x))
    seq_y = torch.randint(0, vocab_size, (batch_size, len_y))
    
    # Ground truth: 1 if symbols match, 0 otherwise (Simplistic "true" alignment for toy exp)
    # In real bio, this would be structural alignment. 
    # Here we define "True Alignment" as the identity match where x[i] == y[j]
    # This acts as our oracle target.
    
    true_alignment = (seq_x.unsqueeze(2) == seq_y.unsqueeze(1)).float()
    
    return seq_x, seq_y, true_alignment

def pairwise_distance_matrix(seq_x, seq_y, embedding):
    """
    Computes distance matrix based on embedding euclidean distance.
    seq_x: (B, N)
    seq_y: (B, M)
    embedding: (V, D)
    """
    emb_x = embedding(seq_x) # (B, N, D)
    emb_y = embedding(seq_y) # (B, M, D)
    
    # Simple Euclidean distance
    # shape: (B, N, M)
    dist = torch.cdist(emb_x, emb_y, p=2)
    return dist
