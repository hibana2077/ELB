import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np

class RealNVPContext(nn.Module):
    """
    Simple RealNVP implementation for density estimation.
    E(x) = -log p(x)
    """
    def __init__(self, input_dim, hidden_dim=64, n_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.n_layers = n_layers
        
        self.masks = []
        for i in range(n_layers):
            mask = torch.arange(input_dim) % 2 == i % 2
            self.masks.append(mask.float())
            
        self.nets = nn.ModuleList()
        self.scales = nn.ModuleList()
        for i in range(n_layers):
            self.nets.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, input_dim),
            ))
            # Learnable scale parameter
            self.scales.append(nn.Parameter(torch.zeros(input_dim)))

        self.prior = D.MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))

    def forward(self, x):
        log_det_J = 0
        z = x
        
        for i in range(self.n_layers):
            mask = self.masks[i].to(x.device)
            z_masked = z * mask
            
            s = self.nets[i](z_masked) * (1 - mask)
            t = self.nets[i](z_masked) * (1 - mask) # Using same net for shift for simplicity, or separate
            # Actually, standard realNVP usually splits. Let's start simple.
            # Simplified coupling:
            # y_A = x_A
            # y_B = x_B * exp(s(x_A)) + t(x_A)
            
            # Re-implement standard affine coupling layer logic cleaner
            pass

        return z, log_det_J

class AffineCoupling(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask):
        super().__init__()
        self.mask = mask
        # Net predicts scale and shift
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim * 2) # s and t
        )
        
    def forward(self, x):
        mask = self.mask.to(x.device)
        x_masked = x * mask
        
        st = self.net(x_masked)
        s, t = st.chunk(2, dim=1)
        
        # Tanh scaling to avoid effective explosion
        s = torch.tanh(s) * (1 - mask)
        t = t * (1 - mask)
        
        # y = x * exp(s) + t  (applied where mask is 0)
        # where mask is 1, s=0, t=0 => y = x
        y = x * torch.exp(s) + t
        
        log_det_J = s.sum(dim=1)
        return y, log_det_J

    def inverse(self, y):
        mask = self.mask.to(y.device)
        y_masked = y * mask # x_A = y_A
        
        st = self.net(y_masked)
        s, t = st.chunk(2, dim=1)
        
        s = torch.tanh(s) * (1 - mask)
        t = t * (1 - mask)
        
        x = (y - t) * torch.exp(-s)
        return x

class SimpleRealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(n_layers):
            mask = torch.zeros(input_dim)
            if i % 2 == 0:
                mask[::2] = 1
            else:
                mask[1::2] = 1
            self.layers.append(AffineCoupling(input_dim, hidden_dim, mask))
            
        self.prior = D.MultivariateNormal(torch.zeros(input_dim), torch.eye(input_dim))

    def forward(self, x):
        log_det_J_sum = 0
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_J_sum = log_det_J_sum + log_det
        return z, log_det_J_sum

    def inverse(self, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

    def log_prob(self, x):
        if x.device != self.prior.loc.device:
            self.prior.loc = self.prior.loc.to(x.device)
            self.prior.covariance_matrix = self.prior.covariance_matrix.to(x.device)
            self.prior._unbroadcasted_scale_tril = self.prior._unbroadcasted_scale_tril.to(x.device)

        z, log_det_J = self.forward(x)
        log_pz = self.prior.log_prob(z)
        return log_pz + log_det_J
    
    def energy(self, x):
        return -self.log_prob(x)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
    

def train_model(model, data, epochs=100, lr=1e-3, batch_size=64, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.to(device)
    loss_history = []
    
    print(f"Training {model.__class__.__name__}...")
    for epoch in range(epochs):
        total_loss = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            
            if isinstance(model, SimpleRealNVP):
                loss = -model.log_prob(batch_x).mean()
            elif isinstance(model, AutoEncoder):
                recon, _ = model(batch_x)
                loss = nn.MSELoss()(recon, batch_x)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            
        avg_loss = total_loss / len(data)
        loss_history.append(avg_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
            
    return loss_history
