import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from utils import compute_softdtw, get_alignment_matrix

# Configurations
N = 20
M = 20
TAUS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
NOISE_REPEATS = 20
NOISE_STD = 0.1

def generate_cost_matrix(c_type):
    C = torch.zeros((1, N, M))
    for i in range(N):
        for j in range(M):
            if c_type == 'unimodal':
                # Single valley along diagonal
                C[0, i, j] = abs(i - j) ** 2
            elif c_type == 'multimodal':
                # Two valleys: One on diagonal, one shifted by 5
                v1 = abs(i - j) ** 2
                v2 = abs(i - j - 5) ** 2
                C[0, i, j] = min(v1, v2)
            elif c_type == 'flat':
                # Random noise
                torch.manual_seed(i*M + j) # Deterministic random
                C[0, i, j] = torch.rand(1).item()
                
    # Normalize for fair comparison
    C = (C - C.min()) / (C.max() - C.min() + 1e-9)
    return C

def compute_metrics(C, tau):
    # 1. Path Entropy using Alignment Matrix (Gradient)
    # Alignment matrix P_ij is probability of path passing through (i,j)
    # P = grad_C SoftDTW(C)
    grad = get_alignment_matrix(C, gamma=tau)
    
    # Entropy of the path distribution?
    # The alignment matrix is NOT a probability distribution over a single variable,
    # but a marginal distribution. 
    # Global Path entropy is -Sum P(path) log P(path).
    # SoftDTW value is -gamma * log(Sum exp(-Cost/gamma)) -> Free Energy.
    # Entropy = (Expected Cost - Soft Min Cost) / gamma.
    # Expected Cost = <P, C>
    
    soft_val = compute_softdtw(C, gamma=tau)
    expected_cost = (grad * C).sum()
    
    # Shannon Entropy of the distribution over paths
    # H = (Expected Energy - Free Energy) / T
    # See "Smoothed analysis of alignments" or standard physics deriv.
    path_entropy = (expected_cost - soft_val) / tau
    
    # 2. Effective Path Count
    eff_count = torch.exp(path_entropy).item()
    
    # 3. SNR of Gradient
    # Perturb C multiple times
    grads = []
    for _ in range(NOISE_REPEATS):
        noise = torch.randn_like(C) * NOISE_STD
        C_noisy = C + noise
        g = get_alignment_matrix(C_noisy, gamma=tau)
        grads.append(g)
        
    grads = torch.stack(grads) # K, 1, N, M
    g_mean = grads.mean(dim=0)
    g_std = grads.std(dim=0) + 1e-9
    
    # Average SNR across the matrix (focusing on active regions?)
    # Just mean SNR
    snr = (g_mean.abs() / g_std).mean().item()
    
    # 4. Optimal Gap
    # Difference between Soft Cost and Hard Cost (approx by min possible)
    # Hard cost is roughly SoftDTW with very small tau
    hard_val = compute_softdtw(C, gamma=1e-5)
    gap = (soft_val - hard_val).item()
    
    return {
        'tau': tau,
        'entropy': path_entropy.item(),
        'eff_path_count': eff_count,
        'snr': snr,
        'optimality_gap': gap
    }

def run_experiment_2():
    print("Starting Experiment 2...")
    results = {}
    
    for c_type in ['unimodal', 'multimodal', 'flat']:
        print(f"Processing landscape: {c_type}")
        C = generate_cost_matrix(c_type)
        
        type_res = []
        for tau in TAUS:
            m = compute_metrics(C, tau)
            type_res.append(m)
            print(f"  Tau={tau}: H={m['entropy']:.2f}, SNR={m['snr']:.2f}, Count={m['eff_path_count']:.2f}")
            
        results[c_type] = type_res
        
    with open('exp2_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Experiment 2 generated exp2_summary.json")

if __name__ == '__main__':
    run_experiment_2()
