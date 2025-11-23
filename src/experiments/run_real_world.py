import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json
import warnings

from src.models.causal_testing import CausalTesting
from src.train import train_transformer_model, extract_features

warnings.filterwarnings("ignore")

# Define 20 baselines (Conceptual placeholders for this implementation)
# In a real scenario, you'd import these from other libraries or implement them
BASELINES = [
    "PCMCI", "TransCausal", "Granger", "Lasso", "Ridge", "ElasticNet",
    "VarLiNGAM", "DirectLiNGAM", "DYNOTEARS", "NOTEARS",
    "TiMINO", "TCDF", "Nod", "GVAR", "NeuralGC",
    "CCM", "TransferEntropy", "Momentum", "SRU", "CUTS+"
]

def load_real_data(dataset_name):
    path = f"data/real_world/{dataset_name}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    return df.select_dtypes(include=[np.number]).values

def run_baseline(baseline, data):
    """
    Mock function to run baseline algorithms.
    Returns a random adjacency matrix for demonstration.
    """
    N = data.shape[1]
    # Simulate runtime and result
    adj = np.random.randint(0, 2, (N, N))
    np.fill_diagonal(adj, 0)
    return adj

def evaluate_consistency(predicted_adj):
    """
    Since we don't have ground truth for real data usually,
    we can evaluate stability/consistency or sparsity.
    Here we just return density as a dummy metric.
    """
    return np.mean(predicted_adj)

def run_real_world_experiment(dataset_name):
    print(f"Running experiment on {dataset_name}...")
    
    try:
        data = load_real_data(dataset_name)
    except Exception as e:
        print(f"Skipping {dataset_name}: {e}")
        return None

    # Normalize
    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
    T, N = data.shape
    
    if N == 0 or T == 0:
        print(f"Skipping {dataset_name}: Empty data shape {data.shape}")
        return None

    # Limit T for speed
    if T > 1000:
        data = data[:1000]
        T = 1000
        
    # Train TransCausal (Our Method)
    X_train = data[np.newaxis, :, :]
    
    # Ensure d_model is valid
    d_model = max(N, 1) 
    
    config = {
        'd_model': d_model, 
        'nhead': 1,
        'num_layers': 2,
        'epochs': 10, # Fast training
        'lr': 0.005,
        'batch_size': 1
    }
    
    # 1. Train Transformer
    model = train_transformer_model(X_train, X_train, config)
    features_tensor, _ = extract_features(model, X_train)
    features = features_tensor[0]
    
    # 2. Run TransCausal
    pcmci = CausalTesting(max_lag=3, alpha=0.05)
    results, _, _ = pcmci.run_pcmci(features)
    
    pcmci_plus_adj = np.zeros((N, N))
    for (i, j, lag, corr, p_val) in results:
        if i != j: 
             pcmci_plus_adj[i, j] = 1
             
    # Run Baselines
    from src.experiments.baselines import run_baseline_algo
    
    baseline_results = {}
    for b in BASELINES:
        print(f"Running baseline: {b}")
        if b == "TransCausal":
            res = pcmci_plus_adj
        elif b == "PCMCI":
            # Run PCMCI without transformer features (raw data)
            # Using our CausalTesting class but on raw data
            pcmci_raw = CausalTesting(max_lag=3, alpha=0.05)
            # Raw data requires reshaping or adapting if it expects features
            # Our run_pcmci expects (T, N)
            r_res, _, _ = pcmci_raw.run_pcmci(data)
            res = np.zeros((N, N))
            for (i, j, lag, corr, p_val) in r_res:
                if i != j: res[i, j] = 1
        else:
            res = run_baseline_algo(b, data)
        
        density = evaluate_consistency(res)
        baseline_results[b] = {
            'density': density,
            'num_edges': int(np.sum(res))
        }
        
    return {
        'dataset': dataset_name,
        'N': N,
        'T': T,
        'results': baseline_results
    }

if __name__ == "__main__":
    datasets = [
        # Original 10
        "air_quality", "energy", "climate", "bike_hourly", "traffic",
        "stocks", "ETTh1", "ETTm1", "solar", "weather",
        # New 10
        "ecg", "electricity", "pollution", "covid", "sales",
        "web_traffic", "power_demand", "water_quality", "nasdaq", "forex"
    ]
    
    all_results = []
    
    for ds in datasets:
        res = run_real_world_experiment(ds)
        if res:
            all_results.append(res)
            
    with open('real_world_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
        
    print("Real-world experiments completed.")
