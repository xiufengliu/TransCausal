"""
TransCausal: Causality-Aware Transformer for Time Series Causal Discovery

This module implements the core Transformer architecture with three novel components:
1. Dilated temporal convolutions with exponentially increasing receptive fields
2. Causal self-attention with strict temporal masking
3. Linear skip connections initialized to identity for autoregressive priors

The architecture combines standard Transformer components (multi-head attention, 
feed-forward networks) with causal discovery-specific innovations.

Implementation follows standard PyTorch patterns for Transformer architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # mask is (T, T), scores is (B, H, T, T)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        
        return out, attn_weights

class DilatedConvLayer(nn.Module):
    def __init__(self, d_model, dilation=1, kernel_size=3):
        super().__init__()
        # Causal padding
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=self.padding)
        
    def forward(self, x):
        # x: (B, T, C) -> (B, C, T) for Conv1d
        x = x.transpose(1, 2)
        out = self.conv(x)
        # Remove future padding
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = out.transpose(1, 2)
        return out

class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dilation=1, dropout=0.1):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = DilatedConvLayer(d_model, dilation=dilation)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Attention block
        attn_out, attn_weights = self.self_attn(x, mask)
        x = self.norm1(x + attn_out)
        
        # Conv block
        conv_out = self.conv(x)
        x = self.norm2(x + conv_out)
        
        # FFN block
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        return x, attn_weights

class CausalTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 5000, d_model)) # Fixed max len
        
        self.layers = nn.ModuleList([
            CausalTransformerBlock(d_model, nhead, dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, output_dim)
        
        # Linear Skip Connection for autoregressive dynamics
        self.linear_skip = nn.Linear(input_dim, output_dim)
        # Initialize to Identity/Random Walk (strong prior for time series)
        nn.init.constant_(self.linear_skip.weight, 1.0)
        nn.init.constant_(self.linear_skip.bias, 0.0)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Linear skip path
        skip_out = self.linear_skip(x)
        
        x = self.input_proj(x)
        if T > 5000:
             raise ValueError("Sequence length exceeds positional encoding limit")
        x = x + self.pos_encoder[:, :T, :]
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T)).to(x.device)
        
        all_attn_weights = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            all_attn_weights.append(attn)
            
        features = x
        out = self.output_proj(x)
        
        # Combine Transformer output with Linear Skip
        out = out + skip_out
        
        return out, features, all_attn_weights, skip_out
