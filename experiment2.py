import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import time
from alignment_utils import SoftDTW

def run_experiment_2():
    print("Starting Experiment 2: Adversarial Consistency Regularization")
    
    # Configuration
    batch_size = 16
    seq_len = 20
    vocab_size = 20
    emb_dim = 16
    epochs = 200
    lambda_reg = 0.5 # Regularization strength
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data Setup
    # Template Y has a fixed structure (contact map)
    # Query X has a ground truth structure
    # True alignment is diagonal (identity)
    # The task is to predict X's structure by aligning to Y.
    # Pred_Structure_X = P @ Structure_Y @ P.T
    
    # 1. Generate Structures
    # Random distance matrices pattern? Or just random SPSD matrices.
    # Let's use simple band diagonal matrices to simulate polymer chains.
    def make_contact_map(L):
        # Band diagonal
        x = torch.arange(L).float().unsqueeze(0) - torch.arange(L).float().unsqueeze(1)
        contacts = torch.exp(-0.1 * x**2) # Gaussian decay
        return contacts.to(device)
    
    Y_structure = make_contact_map(seq_len)
    X_structure_true = Y_structure.clone() # Same structure if perfectly aligned
    
    # Make batch
    Y_struct_batch = Y_structure.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    X_struct_batch = X_structure_true.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    
    # X and Y sequences (Randomly initialized, model learns simply to align them)
    # If the model aligns X_i to Y_i, it gets perfect structure prediction.
    # But if it aligns X_i to Y_j where j!=i, structure prediction is wrong (unless structure is invariant).
    # Since structure is band-diagonal (position dependent), alignment matters.
    
    fixed_X = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    fixed_Y = fixed_X.clone().to(device) # Perfect identity alignment is the goal
    
    # Models to compare
    modes = ['baseline', 'consistency']
    results = {}
    
    for mode in modes:
        print(f"\nRunning training mode: {mode}")
        
        # Model: Simple Embedding
        embedding = nn.Embedding(vocab_size, emb_dim).to(device)
        optimizer = optim.Adam(embedding.parameters(), lr=0.01)
        
        # Two SoftDTW layers with different temperatures for consistency check
        # Main for prediction (moderate temp for differentiability)
        tau_main = 0.5
        soft_dtw_main = SoftDTW().to(device)
        soft_dtw_main.gamma = tau_main
        
        # Auxiliary for regularization (different temp)
        tau_aux = 2.0
        soft_dtw_aux = SoftDTW().to(device)
        soft_dtw_aux.gamma = tau_aux
        
        history = {
            "epoch": [],
            "downstream_loss": [],
            "reg_loss": [],
            "alignment_deviation": [], # Deviation from diagonal
            "alignment_entropy": []
        }
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            
            # 1. Compute Alignment Probabilities (P)
            emb_x = embedding(fixed_X)
            emb_y = embedding(fixed_Y)
            dist_matrix = torch.cdist(emb_x, emb_y)
            
            # Helper to get P differentiable
            # We use the gradient trick: P = grad_D(SoftDTW(D))
            # But double backprop requirements?
            # SoftDTW forward is differentiable. The gradient of SoftDTW w.r.t D IS the alignment matrix.
            # However, we usually don't have second derivatives of SoftDTW implemented efficiently for 'grad of grad'.
            # A common approximation in these experiments:
            # P approx SoftMax(-D/tau)? No, that's local.
            # P approx SoftDTW backward.
            
            # For this toy experiment, we can use a simpler Soft-Alignment mechanism 
            # if SoftDTW second derivative is issue. 
            # Or we simply use the "Alignment" defined by Softmax of -Distance/Temperature 
            # which is "Attention" (local alignment)?
            # The prompt implies SoftDTW. 
            # Let's stick to SoftDTW. Pytorch autograd handles 2nd derivative if operations are differntiated.
            # My manual forward pass uses basic torch ops, so it SHOULD support double grad.
            
            # Main Branch (tau1)
            # We need P1 to compute downstream structure.
            # P1 = grad(SoftDTW(dist)) wrt dist.
            # To perform backprop through P1, we need create_graph=True.
            
            # Step 1: Forward SoftDTW
            #   L_dtw = SoftDTW(D)
            #   P = dL/dD
            
            #   Pred = P @ Struct_Y @ P.T (approx mapping)
            #   L_down = MSE(Pred, Struct_X)
            
            # This requires Hessian-vector products or double grad through SoftDTW.
            # My 'alignment_utils' naive implementation supports this because it's just torch ops!
            
            # To differentiate w.r.t dist_matrix and backprop to embedding, we need dist_matrix to be part of the graph.
            # And we need to differentiate the SoftDTW op twice (once to get P, once to train P).
            
            # Make sure we retain gradients on intermediate dist_matrix if needed for debugging or if autograd needs it to be leaf-like?
            # No, autograd creates graph. P depends on D. D depends on Emb.
            # P = grad(SoftDTW(D), D).
            # We must use create_graph=True in grad call.
            
            # dist_matrix already has requires_grad=True because it comes from embedding.
            loss_dtw_main = soft_dtw_main(dist_matrix).sum()
            
            # Get P1 (Alignment Matrix)
            # create_graph=True allows gradients to flow back through P1 to dist_matrix -> embedding
            # Note: dist_matrix is not a leaf, so we can't accumulate .grad on it directly unless we retain_grad(), 
            # but autograd.grad handles non-leaf inputs fine as long as they are in the graph.
            P1 = torch.autograd.grad(loss_dtw_main, dist_matrix, create_graph=True)[0]
            
            # Normalize P1 to make it a valid soft alignment matrix (rows sum to ~1)
            # SoftDTW marginals are not strictly probability distributions, they are expected counts.
            # For visualization and downstream usage as "Attention", we often normalize.
            # Or use as is if the scale is appropriate. SoftDTW gradients effectively are marginals.
            
            P1_norm = P1 / (P1.sum(dim=-1, keepdim=True) + 1e-9)
            
            # 2. Downstream Prediction
            # Map Y structure to X space
            # If X matches Y, P ~ Identity.
            # Pred = P @ Y_struct @ P.T
            # Dimensions: (B, L, L)
            # Use normalized P1 to keep scale
            pred_structure = torch.matmul(torch.matmul(P1_norm, Y_struct_batch), P1_norm.transpose(1, 2))
            
            # Downstream Loss (MSE)
            loss_downstream = nn.MSELoss()(pred_structure, X_struct_batch)
            
            total_loss = loss_downstream
            reg_val = torch.tensor(0.0).to(device)
            
            # 3. Consistency Regularization
            if mode == 'consistency':
                # Branch 2 (tau2)
                loss_dtw_aux = soft_dtw_aux(dist_matrix).sum()
                # Again, use create_graph=True if we want gradients of reg loss back to dist_matrix
                P2 = torch.autograd.grad(loss_dtw_aux, dist_matrix, create_graph=True)[0]
                
                # KL Divergence between P1 and P2
                # Treat rows of P as categorical distributions (they align x_i to y_j)
                # P is (B, N, M). Normalize rows?
                # SoftDTW marginals sum to expected visits, not necessarily 1 (can be >1 for loops, but here N=M so roughly 1).
                # Normalizing for KL:
                
                eps = 1e-9
                # P1_norm is already computed
                P2_norm = P2 / (P2.sum(dim=-1, keepdim=True) + eps)
                
                # KL(P1 || P2) = sum P1 log (P1 / P2)
                kl_loss = (P1_norm * (torch.log(P1_norm + eps) - torch.log(P2_norm + eps))).sum(dim=-1).mean()
                
                reg_val = lambda_reg * kl_loss
                total_loss += reg_val
            
            # Backward
            total_loss.backward()
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                # Deviation from diagonal (biological prior)
                # We assume diagonal is true physical alignment
                # Weight P by distance from diagonal |i-j|
                indices = torch.arange(seq_len).to(device)
                i_grid, j_grid = torch.meshgrid(indices, indices, indexing='ij')
                dist_from_diag = torch.abs(i_grid - j_grid).float()
                
                deviation = (P1 * dist_from_diag).sum() / P1.sum()
                
                entropy = -(P1_norm * torch.log(P1_norm + eps)).sum(dim=-1).mean() if mode == 'consistency' else 0.0
                
                history["epoch"].append(epoch)
                history["downstream_loss"].append(loss_downstream.item())
                history["reg_loss"].append(reg_val.item())
                history["alignment_deviation"].append(deviation.item())
                history["alignment_entropy"].append(entropy if mode == 'consistency' else 0.0)

            if epoch % 20 == 0:
                print(f"    Epoch {epoch}: Loss {loss_downstream.item():.4f}, Reg {reg_val.item():.4f}, Dev {deviation.item():.2f}")

        results[mode] = history
        print(f"  Completed {mode} in {time.time() - start_time:.2f}s")

    # Save summary
    with open('exp2_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nExperiment 2 Complete. Results saved to exp2_summary.json")

if __name__ == "__main__":
    run_experiment_2()
