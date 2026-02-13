import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import time
import sys

# Import our custom alignment module (assuming it's in the same directory)
from alignment_utils import SoftDTW

def run_experiment_1():
    print("Starting Experiment 1: Temperature-Induced Grokking in Soft Alignment")
    
    # Configuration
    batch_size = 32
    seq_len = 20
    vocab_size = 20
    emb_dim = 16
    epochs = 200
    temperatures = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = {}
    
    for tau in temperatures:
        print(f"\nRunning training with temperature (tau) = {tau}")
        
        # Reset model for each temperature
        embedding = nn.Embedding(vocab_size, emb_dim).to(device)
        optimizer = optim.Adam(embedding.parameters(), lr=0.01)
        
        # SoftDTW module with current temperature
        # Note: In SoftDTW, gamma acts as temperature. strict soft-min: -gamma * logsumexp(-x/gamma)
        # Higher gamma = smoother (higher temp). Lower gamma = harder (lower temp).
        soft_dtw = SoftDTW().to(device)
        # We manually handle the gamma in the forward pass or patching the class
        soft_dtw.gamma = tau
        
        # Metrics storage
        history = {
            "epoch": [],
            "loss": [],
            "alignment_accuracy": [],
            "alignment_entropy": [],
            "effective_rank": []
        }
        
        # Fixed synthetic dataset for consistency across temps
        # Task: Learn to align mutated sequences.
        # X is random. Y is X with 20% mutations.
        # The true alignment is still the diagonal (assuming mutations are substitutions, no indels for simplicity).
        # We want the model to learn that (i, i) is the correct alignment despite the noise.
        # This requires learning the similarity of mutated tokens if we were learning a matrix, 
        # but here we are learning embeddings.
        # If we use random embeddings, d(A, B) is random.
        # If we optimize SoftDTW, the model will try to move A and B closer if they often appear at the same position.
        # This is a valid "learning" task.
        
        fixed_X = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        # Create Y as mutated version of X
        mutation_prob = 0.2
        mask = torch.rand_like(fixed_X.float()) < mutation_prob
        noise = torch.randint(0, vocab_size, fixed_X.shape).to(device)
        fixed_Y = torch.where(mask, noise, fixed_X)
        
        # True alignment is diagonal (since we only did substitutions)
        true_alignment = torch.eye(seq_len).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        
        start_time = time.time()
        print(f"  Training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            
            # 1. Compute Distance Matrix
            emb_x = embedding(fixed_X) 
            emb_y = embedding(fixed_Y)
            dist_matrix = torch.cdist(emb_x, emb_y, p=2) # (B, L, L)
            
            # 2. Compute SoftDTW loss
            # We add a "regularization" that forces the model to actually learn the alignment
            # by penalizing the distance matrix on the diagonal? 
            # No, standard SoftDTW loss minimization is:
            # Min SoftDTW(D). 
            # If D_ii is small, SoftDTW is small.
            # So the model will learn to minimize distance between aligned pairs.
            loss_val = soft_dtw.forward(dist_matrix) 
            loss = loss_val.mean()
            
            loss.backward()
            optimizer.step()
            
            # 3. Measurements
            if epoch % 10 == 0 or epoch == epochs:
                with torch.no_grad():
                    # Recompute for analysis (clean graph)
                    dist_matrix_grad = torch.cdist(embedding(fixed_X), embedding(fixed_Y), p=2)
                    alignment_matrix = soft_dtw.get_alignment_matrix(dist_matrix_grad) # (B, L, L)
                    
                    # Accuracy: Trace / Sum
                    # Only diagonal elements count
                    trace = torch.einsum('bii->b', alignment_matrix)
                    total_mass = alignment_matrix.sum(dim=(1,2))
                    acc = (trace / total_mass).mean().item()
                    
                    # Entropy of alignment path (from flattened matrix)
                    # Normalize P
                    P_flat = alignment_matrix.view(batch_size, -1)
                    P_sum = P_flat.sum(dim=1, keepdim=True) + 1e-9
                    P_norm = P_flat / P_sum
                    # entropy = -sum(p log p)
                    entropy = -(P_norm * torch.log(P_norm + 1e-9)).sum(dim=1).mean().item()
                    
                    # Measurement of Effective Rank of Embedding Matrix
                    W = embedding.weight
                    s = torch.linalg.svdvals(W)
                    s = s / s.sum()
                    effective_rank = torch.exp(-(s * torch.log(s + 1e-9)).sum()).item()
                    
                    history["epoch"].append(epoch)
                    history["loss"].append(loss.item())
                    history["alignment_accuracy"].append(acc)
                    history["alignment_entropy"].append(entropy)
                    history["effective_rank"].append(effective_rank)
                
                if epoch % 20 == 0:
                     print(f"    Epoch {epoch}: Loss {loss.item():.4f}, Acc {acc:.4f}, Rank {effective_rank:.2f}")

        results[str(tau)] = history
        print(f"  Completed tau={tau} in {time.time() - start_time:.2f}s")


    # Save summary
    with open('exp1_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nExperiment 1 Complete. Results saved to exp1_summary.json")

if __name__ == "__main__":
    run_experiment_1()
