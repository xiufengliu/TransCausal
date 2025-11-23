import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.synthetic import generate_lorenz96, generate_ground_truth_lorenz96
from src.models.transformer import CausalTransformer
from src.models.causal_testing import CausalTesting
from sklearn.linear_model import Lasso

# --- Helper Functions ---

def train_causalformer_ci(X, config):
    """
    Train CausalFormer using Channel-Independent strategy.
    X: (B, T, N)
    Returns: Transformed X (B, T, N)
    """
    B, T, N = X.shape
    
    # Channel Independence: Reshape to (B*N, T, 1)
    X_reshaped = X.reshape(B * N, T, 1)
    
    # Model Config
    d_model = config.get('d_model', 64)
    nhead = config.get('nhead', 4)
    num_layers = config.get('num_layers', 2)
    lambda_sparse = config.get('lambda_sparse', 0.01)
    lr = config.get('lr', 0.001)
    epochs = config.get('epochs', 10)
    batch_size = config.get('batch_size', 32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    model = CausalTransformer(
        input_dim=1,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        output_dim=1,
        dropout=0.1
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_tensor = torch.FloatTensor(X_reshaped).to(device)
    
    model.train()
    for epoch in range(epochs):
        # Simple batching
        indices = torch.randperm(X_tensor.size(0))
        for i in range(0, X_tensor.size(0), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = X_tensor[batch_idx] # (Batch, T, 1)
            
            if batch_x.size(0) < 2: continue
            
            input_seq = batch_x[:, :-1, :]
            target_seq = batch_x[:, 1:, :]
            
            # Noise Injection / Denoising Objective
            if config.get('noise_injection', False):
                noise_level = config.get('noise_level', 0.05)
                noise = torch.randn_like(input_seq) * noise_level
                input_seq = input_seq + noise
            
            optimizer.zero_grad()
            output, _, attn_weights, _ = model(input_seq)
            
            # Prediction Loss
            pred_loss = criterion(output, target_seq)
            
            # Sparsity Loss (L1 on attention weights)
            sparsity_loss = 0
            for attn in attn_weights:
                sparsity_loss += torch.mean(torch.abs(attn))
                
            loss = pred_loss + lambda_sparse * sparsity_loss
            
            loss.backward()
            optimizer.step()
            
    # Inference (Transform data)
    model.eval()
    with torch.no_grad():
        # Process in batches to avoid OOM
        outputs = []
        for i in range(0, X_tensor.size(0), batch_size):
            batch_x = X_tensor[i:i+batch_size]
            input_seq = batch_x[:, :-1, :] # (Batch, T-1, 1)
            _, features, _, skip_out = model(input_seq) # (Batch, T-1, D)
            
            # Feature Selection: Select feature with highest variance
            # Here we do it per batch, but ideally should be global.
            # For simplicity/speed in this script, we take the mean across D (or first PC)
            # Taking mean is safer than max var per batch (which might switch indices)
            # Let's take the mean feature
            feat_reduced = torch.mean(features, dim=2, keepdim=True) # (Batch, T-1, 1)
            
            # Feature Skip Connection: Add linear skip output to features
            # We scale down the deep features to prioritize the linear signal (skip_out)
            # This helps in linear regimes (VAR) where deep features might be noisy
            feat_reduced = 0.1 * feat_reduced + skip_out
            
            # Pad the first time step
            pad = torch.zeros(feat_reduced.size(0), 1, 1).to(device)
            out_padded = torch.cat([pad, feat_reduced], dim=1)
            outputs.append(out_padded)
            
        X_transformed = torch.cat(outputs, dim=0) # (B*N, T, 1)
        
    # Reshape back to (B, T, N)
    X_out = X_transformed.reshape(B, T, N).cpu().numpy()
    return X_out

def evaluate_f1(pred_graph, true_graph):
    pred_flat = pred_graph.flatten()
    true_flat = true_graph.flatten()
    tp = np.sum((pred_flat == 1) & (true_flat == 1))
    fp = np.sum((pred_flat == 1) & (true_flat == 0))
    fn = np.sum((pred_flat == 0) & (true_flat == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1



def run_demo():
    print("Running TransCausal Demo on Lorenz-96...")
    
    # 1. Generate Data
    N, T = 10, 500
    print(f"Generating synthetic data (N={N}, T={T})...")
    data = generate_lorenz96(N=N, T=T, seed=42)
    true_graph = generate_ground_truth_lorenz96(N)
    
    # Add batch dimension
    data_batch = data[np.newaxis, :, :]
    
    # 2. Train TransCausal
    print("Training TransCausal model...")
    # Enable noise injection for robustness as per paper
    config = {
        'epochs': 10, 
        'noise_injection': True, 
        'noise_level': 0.1,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2
    }
    X_trans = train_causalformer_ci(data_batch, config)
    
    # 3. Causal Discovery (PCMCI)
    print("Running causal discovery (PCMCI)...")
    pcmci = CausalTesting(max_lag=3)
    _, _, p_vals = pcmci.run_pcmci(X_trans[0])
    
    # 4. Evaluate
    pred_graph = (p_vals < 0.05).astype(int)
    f1 = evaluate_f1(pred_graph, true_graph)
    
    print(f"\nDemo Results:")
    print(f"TransCausal F1 Score: {f1:.4f}")
    print("Demo completed successfully!")

if __name__ == "__main__":
    run_demo()
