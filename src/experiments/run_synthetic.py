import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from src.data.synthetic import generate_lorenz96, generate_mackey_glass_coupled
from src.models.transformer import CausalTransformer
from src.train import train_transformer_model, extract_features
from src.models.causal_testing import CausalTesting

def get_ground_truth_lorenz96(N):
    # x[i] is influenced by x[i-1], x[i-2], x[i+1]
    # So parents of i are (i-1)%N, (i-2)%N, (i+1)%N
    adj = np.zeros((N, N))
    for i in range(N):
        adj[(i-1)%N, i] = 1
        adj[(i-2)%N, i] = 1
        adj[(i+1)%N, i] = 1
    return adj

def evaluate_graph(predicted_adj, true_adj):
    # Flatten
    pred = predicted_adj.flatten()
    true = true_adj.flatten()
    
    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    shd = fp + fn
    
    return precision, recall, f1, shd

def run_experiment(dataset_name='lorenz96', N=10, T=500, seed=42):
    # print(f"Running experiment: {dataset_name}, N={N}, T={T}, Seed={seed}")
    
    # 1. Data Generation
    if dataset_name == 'lorenz96':
        data = generate_lorenz96(N=N, T=T, seed=seed)
        true_adj = get_ground_truth_lorenz96(N)
    elif dataset_name == 'mackey_glass':
        data, true_adj = generate_mackey_glass_coupled(N=N, T=T, seed=seed)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
        
    # Normalize
    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
    
    # Prepare for Transformer (B, T, N)
    X_train = data[np.newaxis, :, :]
    
    # 2. Train Transformer
    # Use N features to map to N variables
    config = {
        'd_model': N, 
        'nhead': 1, # Use 1 head to ensure divisibility with any N
        'num_layers': 2,
        'epochs': 20,
        'lr': 0.01,
        'batch_size': 1
    }
    
    # print("Training Transformer...")
    model = train_transformer_model(X_train, X_train, config)
    
    # 3. Extract Features
    # print("Extracting features...")
    features_tensor, _ = extract_features(model, X_train)
    features = features_tensor[0] # (T-1, N)
    
    # 4. Run TransCausal
    # print("Running TransCausal...")
    pcmci = CausalTesting(max_lag=3, alpha=0.05)
    results, val_matrix, p_values = pcmci.run_pcmci(features)
    
    # 5. Construct Predicted Graph
    predicted_adj = np.zeros((N, N))
    for (i, j, lag, corr, p_val) in results:
        if i != j: 
             predicted_adj[i, j] = 1
             
    # 6. Evaluate
    prec, recall, f1, shd = evaluate_graph(predicted_adj, true_adj)
    
    return {
        'dataset': dataset_name,
        'N': N,
        'T': T,
        'seed': seed,
        'precision': prec,
        'recall': recall,
        'f1': f1,
        'shd': shd
    }

if __name__ == "__main__":
    results = []
    
    # Experiment 1: Varying T
    print("Experiment 1: Varying Sample Size T")
    Ns = [10]
    Ts = [200, 500, 1000]
    Seeds = [42, 43, 44]
    
    for ds in ['lorenz96', 'mackey_glass']:
        for N in Ns:
            for T in Ts:
                metrics = {'precision': [], 'recall': [], 'f1': [], 'shd': []}
                for s in tqdm(Seeds, desc=f"{ds} N={N} T={T}"):
                    res = run_experiment(ds, N, T, s)
                    metrics['precision'].append(res['precision'])
                    metrics['recall'].append(res['recall'])
                    metrics['f1'].append(res['f1'])
                    metrics['shd'].append(res['shd'])
                
                avg_res = {k: np.mean(v) for k, v in metrics.items()}
                std_res = {k: np.std(v) for k, v in metrics.items()}
                
                print(f"Dataset: {ds}, T: {T} -> F1: {avg_res['f1']:.4f} +/- {std_res['f1']:.4f}")
                
                results.append({
                    'experiment': 'varying_T',
                    'dataset': ds,
                    'N': N,
                    'T': T,
                    'metrics_mean': avg_res,
                    'metrics_std': std_res
                })
                
    # Experiment 2: Varying N
    print("\nExperiment 2: Varying Dimension N")
    Ns = [5, 10, 20]
    T = 500
    
    for ds in ['lorenz96', 'mackey_glass']:
        for N in Ns:
            metrics = {'precision': [], 'recall': [], 'f1': [], 'shd': []}
            for s in tqdm(Seeds, desc=f"{ds} N={N} T={T}"):
                res = run_experiment(ds, N, T, s)
                metrics['precision'].append(res['precision'])
                metrics['recall'].append(res['recall'])
                metrics['f1'].append(res['f1'])
                metrics['shd'].append(res['shd'])
            
            avg_res = {k: np.mean(v) for k, v in metrics.items()}
            std_res = {k: np.std(v) for k, v in metrics.items()}
            
            print(f"Dataset: {ds}, N: {N} -> F1: {avg_res['f1']:.4f} +/- {std_res['f1']:.4f}")
            
            results.append({
                'experiment': 'varying_N',
                'dataset': ds,
                'N': N,
                'T': T,
                'metrics_mean': avg_res,
                'metrics_std': std_res
            })
        
    # Save results
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
