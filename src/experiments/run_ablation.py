"""
Ablation Study for TransCausal
Tests the contribution of each component:
1. PCMCI (baseline)
2. Transformer + PCMCI (TPCMCI - without causal loss)
3. TransCausal (full model with causal loss)
4. Ablations: -Dilated Conv, -Causal Mask, -Causal Loss
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
from src.models.causal_testing import CausalTesting
from src.models.transformer import CausalTransformer
from src.train import train_transformer_model, extract_features
import torch

def load_dataset(name):
    """Load a dataset for ablation study"""
    path = f'data/real_world/{name}.csv'
    if not os.path.exists(path):
        return None, None, None
    
    df = pd.read_csv(path)
    data = df.values
    
    if len(data) == 0:
        return None, None, None
    
    T, N = data.shape
    return data, T, N

def run_ablation_variant(data, variant_name, config):
    """
    Run a specific ablation variant
    
    Variants:
    - 'PCMCI': Vanilla PCMCI (no Transformer)
    - 'TPCMCI': Transformer + PCMCI (no causal loss)
    - 'TransCausal': Full model (with causal loss)
    - 'No_Dilated': TransCausal without dilated convolutions
    - 'No_Causal_Mask': TransCausal without causal attention mask
    - 'No_Causal_Loss': TransCausal without causal regularization
    """
    T, N = data.shape
    
    # Configuration
    max_lag = config.get('max_lag', 3)
    alpha = config.get('alpha', 0.05)
    
    if variant_name == 'PCMCI':
        # Vanilla PCMCI - no Transformer
        results, _, p_vals = pcmci.run_pcmci(data)
        graph = (p_vals < alpha).astype(int)
        
    elif variant_name == 'TPCMCI':
        # Transformer + PCMCI (no causal loss during training)
        d_model = N
        nhead = 1
        model = CausalTransformer(d_model=d_model, nhead=nhead, num_layers=2, 
                                   dim_feedforward=128, max_seq_len=T)
        
        # Train WITHOUT causal loss (use_causal_loss=False)
        trained_model = train_transformer_model(
            data, model, epochs=10, lr=0.01, batch_size=1, 
            use_causal_loss=False  # Key difference
        )
        
        # Extract features
        features = extract_features(data, trained_model)
        
        # Run PCMCI on features
        pcmci = CausalTesting(max_lag=max_lag, alpha=alpha)
        results, _, _ = pcmci.run_pcmci(features)
        
    elif variant_name == 'TransCausal':
        # Full TransCausal with causal loss
        d_model = N
        nhead = 1
        model = CausalTransformer(d_model=d_model, nhead=nhead, num_layers=2,
                                   dim_feedforward=128, max_seq_len=T)
        
        # Train WITH causal loss
        trained_model = train_transformer_model(
            data, model, epochs=10, lr=0.01, batch_size=1,
            use_causal_loss=True  # Key difference
        )
        
        features = extract_features(data, trained_model)
        pcmci = CausalTesting(max_lag=max_lag, alpha=alpha)
        results, _, _ = pcmci.run_pcmci(features)
        
    elif variant_name == 'No_Dilated':
        # TransCausal without dilated convolutions
        # (Would need to modify CausalTransformer - use standard conv)
        # For now, same as TransCausal (placeholder)
        d_model = N
        nhead = 1
        model = CausalTransformer(d_model=d_model, nhead=nhead, num_layers=2,
                                   dim_feedforward=128, max_seq_len=T,
                                   use_dilated_conv=False)  # Disable dilated conv
        
        trained_model = train_transformer_model(
            data, model, epochs=10, lr=0.01, batch_size=1, use_causal_loss=True
        )
        
        features = extract_features(data, trained_model)
        pcmci = CausalTesting(max_lag=max_lag, alpha=alpha)
        results, _, _ = pcmci.run_pcmci(features)
        
    elif variant_name == 'No_Causal_Mask':
        # TransCausal without causal attention mask
        d_model = N
        nhead = 1
        model = CausalTransformer(d_model=d_model, nhead=nhead, num_layers=2,
                                   dim_feedforward=128, max_seq_len=T,
                                   use_causal_mask=False)  # Disable causal mask
        
        trained_model = train_transformer_model(
            data, model, epochs=10, lr=0.01, batch_size=1, use_causal_loss=True
        )
        
        features = extract_features(data, trained_model)
        pcmci = CausalTesting(max_lag=max_lag, alpha=alpha)
        results, _, _ = pcmci.run_pcmci(features)
        
    elif variant_name == 'No_Causal_Loss':
        # Same as TPCMCI
        return run_ablation_variant(data, 'TPCMCI', config)
    
    else:
        raise ValueError(f"Unknown variant: {variant_name}")
    
    # Calculate metrics
    num_edges = np.sum(graph != 0)
    possible_edges = N * N * max_lag
    density = num_edges / possible_edges if possible_edges > 0 else 0
    
    return {
        'variant': variant_name,
        'num_edges': int(num_edges),
        'density': float(density),
        'graph': graph.tolist() if isinstance(graph, np.ndarray) else graph
    }

def run_ablation_study():
    """Run complete ablation study on selected datasets"""
    
    print("=" * 80)
    print("TransCausal ABLATION STUDY")
    print("=" * 80)
    
    # Select representative datasets for ablation
    test_datasets = ['air_quality', 'climate', 'stocks', 'ETTh1', 'ecg']
    
    # Ablation variants
    variants = [
        'PCMCI',           # Baseline
        'TPCMCI',          # Transformer without causal loss
        'TransCausal',          # Full model
        # 'No_Dilated',      # Without dilated convolutions
        # 'No_Causal_Mask',  # Without causal mask
    ]
    
    config = {
        'max_lag': 3,
        'alpha': 0.05
    }
    
    all_results = []
    
    for dataset_name in test_datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")
        
        data, T, N = load_dataset(dataset_name)
        
        if data is None:
            print(f"Skipping {dataset_name}: not found")
            continue
        
        print(f"Shape: N={N}, T={T}")
        
        dataset_results = {
            'dataset': dataset_name,
            'N': N,
            'T': T,
            'variants': {}
        }
        
        for variant in variants:
            print(f"\nRunning {variant}...")
            try:
                result = run_ablation_variant(data, variant, config)
                dataset_results['variants'][variant] = result
                print(f"  ✓ {variant}: {result['num_edges']} edges, density={result['density']:.4f}")
            except Exception as e:
                print(f"  ✗ {variant} failed: {e}")
                dataset_results['variants'][variant] = {'error': str(e)}
        
        all_results.append(dataset_results)
    
    # Save results
    with open('ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    
    # Compute averages
    variant_stats = {v: {'edges': [], 'density': []} for v in variants}
    
    for ds_result in all_results:
        for variant, result in ds_result['variants'].items():
            if 'error' not in result:
                variant_stats[variant]['edges'].append(result['num_edges'])
                variant_stats[variant]['density'].append(result['density'])
    
    print(f"\n{'Variant':<20s} {'Avg Edges':>12s} {'Avg Density':>12s} {'Improvement':>12s}")
    print("-" * 80)
    
    baseline_edges = np.mean(variant_stats['PCMCI']['edges']) if variant_stats['PCMCI']['edges'] else 0
    
    for variant in variants:
        if variant_stats[variant]['edges']:
            avg_edges = np.mean(variant_stats[variant]['edges'])
            avg_density = np.mean(variant_stats[variant]['density'])
            improvement = ((avg_edges - baseline_edges) / baseline_edges * 100) if baseline_edges > 0 else 0
            print(f"{variant:<20s} {avg_edges:>12.2f} {avg_density:>12.4f} {improvement:>11.1f}%")
    
    print("\n✓ Ablation study completed. Results saved to ablation_results.json")
    
    return all_results

if __name__ == "__main__":
    run_ablation_study()
