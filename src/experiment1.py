import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from utils import compute_softdtw, get_alignment_matrix, generate_synthetic_data

# Configuration
SEQ_LEN = 16
VOCAB_SIZE = 20
EMBED_DIM = 16
BATCH_SIZE = 16 # Small batch for toy experiment
EPOCHS = 50
LEARNING_RATE = 0.01
GAMMA = 0.1 # SoftDTW temperature

class ToyModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, x, y):
        x_emb = self.embedding(x) # B, L, E
        y_emb = self.embedding(y) # B, L, E
        
        # Compute pairwise distance matrix (Squared Euclidean)
        # x: B, N, E; y: B, M, E
        # D_ij = ||x_i - y_j||^2
        x_norm = (x_emb**2).sum(-1).view(x_emb.shape[0], x_emb.shape[1], 1)
        y_norm = (y_emb**2).sum(-1).view(y_emb.shape[0], 1, y_emb.shape[1])
        dist = x_norm + y_norm - 2.0 * torch.bmm(x_emb, y_emb.transpose(1, 2))
        return dist, x_emb, y_emb

def entropy(probs):
    # probs: B, N, M
    # Normalize to get valid distribution if needed, but alignment matrix row-sum is roughly 1? 
    # Actually SoftDTW gradient is not strictly a probability distribution in the same way as Softmax (it sums to path counts).
    # But usually we normalize it or treat it as soft assignment.
    # We'll normalize row-wise for entropy calculation.
    row_sum = probs.sum(dim=-1, keepdim=True) + 1e-9
    p = probs / row_sum
    return -(p * torch.log(p + 1e-9)).sum(dim=-1).mean().item()

def run_experiment_task(task_type):
    print(f"Starting Experiment 1 - Task: {task_type}")
    
    # 1. Setup Data
    # Static dataset for reproducibility
    torch.manual_seed(42)
    X, Y, GT_Align = generate_synthetic_data(num_samples=100, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
    
    # Create simple labels for 'Unrelated' task (e.g., is sum of tokens even?)
    if task_type == 'unrelated':
        X_labels = (X.sum(dim=1) % 2).long()
    
    model = ToyModel(VOCAB_SIZE, EMBED_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {
        'epoch': [],
        'entropy': [],
        'recovery_rate': [],
        'grad_cosine': [],
        'task_loss': [],
        'align_loss': []
    }
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Forward
        D, x_emb, y_emb = model(X, Y)
        
        # Alignment Loss (SoftDTW)
        # We want to minimize distance between aligned X and Y
        # For 'Unrelated' task, strictly optimizing Alignment might not effectively align if embeddings degenerate.
        # But we explicitly add SoftDTW to the loss.
        l_align_val = compute_softdtw(D, gamma=GAMMA).mean()
        
        # Get Alignment Matrix (Gradient of SoftDTW w.r.t D)
        # We need this for Task 1 (using alignment) or just for metrics.
        # To make "Alignment" differentiable for Task 1, strictly speaking we should use differentiable alignment.
        # But `compute_softdtw` only returns distance. 
        # For Task 1, we often define Reconstruction = (SoftDTW_Grad * Y).
        # We can simulate this by taking grad. But double backward is tricky/slow.
        # Simplified: Use Softmax(-D/gamma) as a proxy for alignment in Task 1 forward pass?
        # NO, the prompt asks for "Alignment Collapse... under Multi-Objective".
        # Let's assume Task use a PROXY alignment or just independent features, 
        # BUT we track the *actual* SoftDTW alignment quality.
        
        # Let's derive A = Softmax(-D) for Task 1 usage to be differentiable easily
        # Note: SoftDTW alignment is different from Softmax attention.
        # But for the sake of a "toy experiment" demonstrating conflict:
        # Task 1 uses Softmax Attention (Local).
        # Alignment Loss uses SoftDTW (Global).
        # Conflict: Local vs Global alignment preference.
        
        # Task Loss
        l_task_val = torch.zeros((), dtype=l_align_val.dtype, device=l_align_val.device)
        
        if task_type == 'precise':
            # Task: Reconstruct X using Softmax Attention on Y
            attn = torch.softmax(-D / GAMMA, dim=-1)
            fused = torch.bmm(attn, y_emb)
            l_task_val = torch.nn.functional.mse_loss(fused, x_emb)
            
        elif task_type == 'coarse':
            # Task: Match global means
            l_task_val = torch.nn.functional.mse_loss(x_emb.mean(1), y_emb.mean(1))
            
        elif task_type == 'unrelated':
            # Task: Classify X sum parity
            # Simple linear probe on mean embedding (detached to only train embedding)
            # Actually, we want gradients to flow to embedding to potentially conflict.
            # So we create a small head on the fly
             # (In a real script this would be in model, but here is hacked for brevity)
            pred = x_emb.mean(1).sum(1) # Dummy projection
            # We need a learnable head to make it realistic? 
            # Let's just use sum of coords as a proxy for feature extraction
            l_task_val = torch.nn.functional.mse_loss(x_emb.mean(1).sum(1), X_labels.to(x_emb.device).float())

        # --- Metrics & Measurements ---
        
        # 1. Alignment Matrix (SoftDTW Gradient)
        # We specifically want to check the structure of the SoftDTW alignment
        # So we re-compute the grad w.r.t D just for inspection.
        # This requires gradient calculation (even though D is detached, the operations need tracking).
        with torch.enable_grad():
            D_detached = D.detach().requires_grad_(True)
            dummy_loss = compute_softdtw(D_detached, gamma=GAMMA).mean()
            dummy_loss.backward()
            align_matrix = D_detached.grad
            
        with torch.no_grad():
            # 2. Entropy
            ent = entropy(align_matrix)
            
            # 3. Recovery Rate (vs GT)
            # Simple argmax check
            recovered_map = align_matrix.argmax(dim=-1) # B, N
            # GT is B, N, N (one-hot). GT index for each i is (i+1)%L
            gt_indices = torch.tensor([(i + 1) % SEQ_LEN for i in range(SEQ_LEN)]).expand(len(X), -1)
            acc = (recovered_map == gt_indices).float().mean().item()
            
            # 4. Gradient Cosine Similarity
            # We need grad of l_align vs l_task on embedding parameters.
            # Use autograd.grad so we don't consume/free the graph before the real optimizer backward.

        emb_params = tuple(model.embedding.parameters())

        grad_a_list = torch.autograd.grad(
            l_align_val,
            emb_params,
            retain_graph=True,
            allow_unused=True,
        )
        grad_t_list = torch.autograd.grad(
            l_task_val,
            emb_params,
            retain_graph=True,
            allow_unused=True,
        )

        flat_a = []
        flat_t = []
        for p, ga, gt in zip(emb_params, grad_a_list, grad_t_list):
            if ga is None:
                ga = torch.zeros_like(p)
            if gt is None:
                gt = torch.zeros_like(p)
            flat_a.append(ga.reshape(-1))
            flat_t.append(gt.reshape(-1))

        grad_a = torch.cat(flat_a)
        grad_t = torch.cat(flat_t)

        if grad_a.norm() > 0 and grad_t.norm() > 0:
            cos_sim = (torch.dot(grad_a, grad_t) / (grad_a.norm() * grad_t.norm())).item()
        else:
            cos_sim = 0.0

        # Total Loss + optimizer step (single backward per epoch)
        loss = l_align_val + l_task_val
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log
        history['epoch'].append(epoch)
        history['entropy'].append(ent)
        history['recovery_rate'].append(acc)
        history['grad_cosine'].append(cos_sim)
        history['task_loss'].append(l_task_val.item())
        history['align_loss'].append(l_align_val.item())
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: TaskL={l_task_val.item():.4f}, AlignL={l_align_val.item():.4f}, Rec={acc:.2f}, Cos={cos_sim:.2f}")

    return history

def main():
    results = {}
    for task in ['precise', 'coarse', 'unrelated']:
        results[task] = run_experiment_task(task)
        
    with open('exp1_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Experiment 1 generated exp1_summary.json")

if __name__ == '__main__':
    main()
