import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.adaptive_diffusion import AdaptiveDiffusionModel
from tqdm import tqdm
import numpy as np

def train_adaptive_diffusion(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for x, corrupted_x in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            x = x.to(device)
            corrupted_x = corrupted_x.to(device)
            
            # Forward pass
            reconstruction, mask = model(x, corrupted_x)
            
            # Compute loss
            loss = model.compute_loss(x, corrupted_x, reconstruction, mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, corrupted_x in val_loader:
                x = x.to(device)
                corrupted_x = corrupted_x.to(device)
                
                reconstruction, mask = model(x, corrupted_x)
                loss = model.compute_loss(x, corrupted_x, reconstruction, mask)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

if __name__ == '__main__':
    # Example usage
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    
    # Load dataset
    train_dataset = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
    val_dataset = MNIST(root='./data', train=False, transform=ToTensor())
    
    # Create corrupted versions (example: random noise)
    def create_corrupted(x):
        noise = torch.randn_like(x) * 0.2
        return x + noise
    
    train_dataset = [(x, create_corrupted(x)) for x, _ in train_dataset]
    val_dataset = [(x, create_corrupted(x)) for x, _ in val_dataset]
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Create base model (example: simple UNet)
    class BaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 2, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 1, 3, padding=1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x = self.encoder(x)
            return self.decoder(x)
    
    base_model = BaseModel()
    model = AdaptiveDiffusionModel(base_model, input_dim=28*28)
    
    # Train the model
    trained_model = train_adaptive_diffusion(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        learning_rate=1e-3
    ) 