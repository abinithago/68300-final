import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaptiveCorruption(nn.Module):
    def __init__(self, input_dim, alpha=1.0, tau=0.5):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        
    def compute_uncertainty(self, x, model, num_samples=5):
        """Compute per-pixel uncertainty using Monte Carlo dropout"""
        uncertainties = []
        model.train()  # Enable dropout
        for _ in range(num_samples):
            pred = model(x)
            uncertainties.append(pred)
        model.eval()
        
        uncertainties = torch.stack(uncertainties)
        return torch.var(uncertainties, dim=0)
    
    def get_corruption_mask(self, uncertainty):
        """Generate corruption mask based on uncertainty"""
        # Compute corruption probabilities
        p = torch.sigmoid(self.alpha * (uncertainty - self.tau))
        
        # Sample mask using Gumbel-Softmax for differentiability
        if self.training:
            # During training, use Gumbel-Softmax relaxation
            uniform = torch.rand_like(p)
            gumbel = -torch.log(-torch.log(uniform + 1e-10) + 1e-10)
            mask = torch.sigmoid((torch.log(p + 1e-10) - torch.log(1 - p + 1e-10) + gumbel) / 0.1)
        else:
            # During inference, use hard thresholding
            mask = (p > 0.5).float()
            
        return mask

class AdaptiveDiffusionModel(nn.Module):
    def __init__(self, base_model, input_dim, alpha=1.0, tau=0.5, lambda_reg=0.1):
        super().__init__()
        self.base_model = base_model
        self.corruption = AdaptiveCorruption(input_dim, alpha, tau)
        self.lambda_reg = lambda_reg
        
    def forward(self, x, corrupted_x):
        # Compute uncertainty
        uncertainty = self.corruption.compute_uncertainty(corrupted_x, self.base_model)
        
        # Get corruption mask
        mask = self.corruption.get_corruption_mask(uncertainty)
        
        # Apply corruption
        noise = torch.randn_like(corrupted_x)
        further_corrupted = mask * corrupted_x + (1 - mask) * noise
        
        # Reconstruct
        reconstruction = self.base_model(further_corrupted)
        
        return reconstruction, mask
    
    def compute_loss(self, x, corrupted_x, reconstruction, mask):
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x)
        
        # Regularization on mask (encourage sparsity)
        reg_loss = self.lambda_reg * torch.mean(mask)
        
        return recon_loss + reg_loss 