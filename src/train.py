import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.models.transformer import CausalTransformer

def train_transformer_model(X_train, X_val, config):
    """
    Train the Causal Transformer for next-step prediction.
    X_train: (B, T, N)
    X_val: (B, T, N)
    """
    input_dim = X_train.shape[2]
    output_dim = input_dim # Predicting the next step of all variables
    
    model = CausalTransformer(
        input_dim=input_dim,
        d_model=config.get('d_model', 64),
        nhead=config.get('nhead', 4),
        num_layers=config.get('num_layers', 2),
        output_dim=output_dim,
        dropout=config.get('dropout', 0.1)
    )
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    
    # Training loop
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 32)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Simple batching
        indices = torch.randperm(X_train_t.size(0))
        for i in range(0, X_train_t.size(0), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = X_train_t[batch_idx]
            
            # Predict next step: Input x[0:T-1], Target x[1:T]
            # Ideally we use a sliding window approach, but for simplicity here:
            # We'll just predict x[t+1] from history up to t.
            
            # For sequence to sequence training:
            # Input: Batch[:, :-1, :]
            # Target: Batch[:, 1:, :]
            
            input_seq = batch_x[:, :-1, :]
            target_seq = batch_x[:, 1:, :]
            
            optimizer.zero_grad()
            output, _, _, _ = model(input_seq)
            
            loss = criterion(output, target_seq)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_input = X_val_t[:, :-1, :]
            val_target = X_val_t[:, 1:, :]
            val_out, _, _, _ = model(val_input)
            val_loss = criterion(val_out, val_target).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    model.load_state_dict(best_model_state)
    return model

def extract_features(model, X):
    """
    Run inference to get features.
    X: (B, T, N)
    Returns: Features (B, T-1, d_model)
    """
    device = next(model.parameters()).device
    X_t = torch.FloatTensor(X).to(device)
    
    model.eval()
    with torch.no_grad():
        # Input is up to T-1 to match training alignment
        input_seq = X_t[:, :-1, :]
        _, features, attn_weights, _ = model(input_seq)
        
    return features.cpu().numpy(), attn_weights
