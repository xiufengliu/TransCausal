import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from src.models.causal_testing import CausalTesting
from src.train import train_transformer_model, extract_features
from src.experiments.baselines import run_granger


def load_climate(max_rows: int = 1500):
    # Prefer the smaller derived file if available
    candidates = [
        Path("data/real_world/climate.csv"),
        Path("data/real_world/jena_climate_2009_2016.csv"),
    ]
    for c in candidates:
        if c.exists():
            df = pd.read_csv(c)
            # keep numeric
            df = df.select_dtypes(include=[np.number])
            if len(df) > max_rows:
                df = df.iloc[:max_rows]
            return df.values
    raise FileNotFoundError("Climate dataset not found")


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def run_pcmci(features: np.ndarray, max_lag: int = 3, alpha: float = 0.05):
    pcmci = CausalTesting(max_lag=max_lag, alpha=alpha)
    results, _, _ = pcmci.run_pcmci(features)
    n = features.shape[1]
    adj = np.zeros((n, n))
    for (i, j, lag, _, p) in results:
        if i != j:
            adj[i, j] = 1
    return adj


def run_transcausal(data: np.ndarray, max_lag: int = 3):
    T, N = data.shape
    X = data[np.newaxis, :, :]
    cfg = {
        "d_model": max(N, 8),
        "nhead": 1,
        "num_layers": 2,
        "epochs": 20,
        "lr": 0.01,
        "batch_size": 1,
    }
    model = train_transformer_model(X, X, cfg)
    feats, _ = extract_features(model, X)
    feats = feats[0]
    return run_pcmci(feats, max_lag=max_lag)


def summarize_adj(adj: np.ndarray) -> Dict[str, float]:
    density = adj.mean()
    num_edges = int(adj.sum())
    return {"density": float(density), "num_edges": num_edges}


def main():
    data = load_climate()
    data = standardize(data)

    # TransCausal: Transformer features -> PCMCI
    trans_adj = run_transcausal(data, max_lag=3)

    # PCMCI on raw data
    pcmci_adj = run_pcmci(data, max_lag=3)

    # Granger
    granger_adj = run_granger(data, max_lag=3)

    results = {
        "dataset": "climate",
        "shape": list(data.shape),
        "TransCausal": summarize_adj(trans_adj),
        "PCMCI": summarize_adj(pcmci_adj),
        "Granger": summarize_adj(granger_adj),
    }

    with open("climate_case_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save adjacency matrices for optional plotting
    np.save("climate_trans_adj.npy", trans_adj)
    np.save("climate_pcmci_adj.npy", pcmci_adj)
    np.save("climate_granger_adj.npy", granger_adj)


if __name__ == "__main__":
    main()
