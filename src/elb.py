import torch
import numpy as np
import warnings

class EnergyLandscape:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.minima = None
        self.labels = None

    def potential(self, x):
        """Returns E(x) and grad E(x)."""
        x_in = x.clone().detach().requires_grad_(True).to(self.device)
        E = self.model.energy(x_in)
        grad = torch.autograd.grad(E.sum(), x_in)[0]
        return E.detach(), grad

    def flow_step(self, x, step_size=1e-2):
        """Performs one step of gradient flow (gradient descent)."""
        _, grad = self.potential(x)
        return x - step_size * grad

    def find_minima(self, initial_points, lr=0.1, max_steps=1000, tol=1e-4):
        """
        Run gradient flow from initial_points to find critical points (minima).
        """
        print(f"Finding minima from {len(initial_points)} points...")
        current = torch.FloatTensor(initial_points).to(self.device)
        
        # Simple Gradient Descent Loop
        for i in range(max_steps):
            E, grad = self.potential(current)
            grad_norm = grad.norm(dim=1).mean().item()
            
            current = current - lr * grad
            
            if grad_norm < tol and i > 50:
                print(f"Converged at step {i} with mean grad norm {grad_norm:.5f}")
                break
                
        # Cluster the results to get unique minima
        final_points = current.detach().cpu().numpy()
        
        # Simple clustering based on distance
        unique_minima = []
        counts = []
        
        # Greedy clustering
        indices = np.arange(len(final_points))
        mask = np.ones(len(final_points), dtype=bool)
        
        labels = -1 * np.ones(len(final_points), dtype=int)
        
        cluster_id = 0
        eps = 0.5 # Distance threshold
        
        for i in range(len(final_points)):
            if not mask[i]: continue
            
            # Create new cluster
            center = final_points[i]
            dists = np.linalg.norm(final_points - center, axis=1)
            
            params_close = dists < eps
            
            # Refine center
            center = final_points[params_close].mean(axis=0)
            
            # Recalculate
            dists = np.linalg.norm(final_points - center, axis=1)
            params_close = dists < eps
            
            # Mark as processed
            mask[params_close] = False
            labels[params_close] = cluster_id
            
            unique_minima.append(center)
            counts.append(params_close.sum())
            cluster_id += 1
            
        self.minima = np.array(unique_minima)
        print(f"Found {len(self.minima)} unique minima.")
        return self.minima, labels

    def assign_basins(self, data, steps=1000, lr=0.1):
        """
        Assigns data points to basins based on gradient flow.
        Returns the cluster labels for each data point.
        """
        if self.minima is None:
            # If no minima found yet, find them using the data itself
            self.find_minima(data, max_steps=steps, lr=lr)
        
        # Just flow the data points and match to nearest minima
        print("Assigning basins...")
        current = torch.FloatTensor(data).to(self.device)
        for i in range(steps):
             E, grad = self.potential(current)
             current = current - lr * grad
             if i % 100 == 0:
                 pass # print(f"Flow step {i}")
        
        final_pos = current.detach().cpu().numpy()
        labels = np.zeros(len(data))
        
        for i, pt in enumerate(final_pos):
            dists = np.linalg.norm(self.minima - pt, axis=1)
            labels[i] = np.argmin(dists)
            
        self.labels = labels
        return labels

def landscape_analysis(model, data, device='cpu'):
    elb = EnergyLandscape(model, device)
    # 1. Flow to find minima
    minima, _ = elb.find_minima(data) # Use data as starting points
    # 2. Assign basins
    labels = elb.assign_basins(data)
    
    return elb, labels, minima
