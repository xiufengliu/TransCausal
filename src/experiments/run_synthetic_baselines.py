import json
import numpy as np
import torch
from tqdm import tqdm

from src.data.synthetic import generate_lorenz96, generate_mackey_glass_coupled
from src.models.causal_testing import CausalTesting
from src.train import train_transformer_model, extract_features
from src.experiments.baselines import run_granger, run_lasso, run_ridge


def get_ground_truth_lorenz96(n_vars: int) -> np.ndarray:
    """Parents of i are (i-1)%N, (i-2)%N, (i+1)%N."""
    adj = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        adj[(i - 1) % n_vars, i] = 1
        adj[(i - 2) % n_vars, i] = 1
        adj[(i + 1) % n_vars, i] = 1
    return adj


def evaluate_graph(predicted_adj: np.ndarray, true_adj: np.ndarray):
    pred = predicted_adj.flatten()
    true = true_adj.flatten()
    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    shd = fp + fn
    return precision, recall, f1, shd


def prepare_data(dataset: str, n_vars: int, length: int, seed: int):
    if dataset == "lorenz96":
        data = generate_lorenz96(N=n_vars, T=length, seed=seed)
        gt = get_ground_truth_lorenz96(n_vars)
    elif dataset == "mackey_glass":
        data, gt = generate_mackey_glass_coupled(N=n_vars, T=length, seed=seed)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
    return data, gt


def run_transcausal(data: np.ndarray, max_lag: int = 3):
    """Transformer feature learning + TransCausal on learned features."""
    bsz, length, n_vars = 1, data.shape[0], data.shape[1]
    X = data[np.newaxis, :, :]
    cfg = {
        "d_model": max(n_vars, 8),
        "nhead": 1,
        "num_layers": 2,
        "epochs": 20,
        "lr": 0.01,
        "batch_size": 1,
    }
    model = train_transformer_model(X, X, cfg)
    device = next(model.parameters()).device
    X_t = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        input_seq = X_t[:, :-1, :]
        out, _, _, _ = model(input_seq)
    # Use model outputs aligned with variables as features
    feats = out.cpu().numpy()[0]

    pcmci = CausalTesting(max_lag=max_lag, alpha=0.05)
    results, _, _ = pcmci.run_pcmci(feats)
    adj = np.zeros((n_vars, n_vars))
    for (i, j, lag, _, _) in results:
        if i != j:
            adj[i, j] = 1
    return adj


def run_pcmci_raw(data: np.ndarray, max_lag: int = 3):
    pcmci = CausalTesting(max_lag=max_lag, alpha=0.05)
    results, _, _ = pcmci.run_pcmci(data)
    n_vars = data.shape[1]
    adj = np.zeros((n_vars, n_vars))
    for (i, j, lag, _, _) in results:
        if i != j:
            adj[i, j] = 1
    return adj


def run_suite(dataset: str, n_vars: int, length: int, seeds):
    methods = {
        "TransCausal": run_transcausal,
        "PCMCI-raw": run_pcmci_raw,
        "Granger": run_granger,
        "Lasso": run_lasso,
        "Ridge": run_ridge,
    }

    metrics = {m: [] for m in methods}
    for s in seeds:
        data, gt = prepare_data(dataset, n_vars, length, s)
        for name, fn in methods.items():
            adj = fn(data)
            prec, rec, f1, shd = evaluate_graph(adj, gt)
            metrics[name].append(
                {"precision": prec, "recall": rec, "f1": f1, "shd": shd}
            )

    summary = {}
    for name, vals in metrics.items():
        arr = {k: np.array([v[k] for v in vals]) for k in vals[0]}
        summary[name] = {
            "mean": {k: float(arr[k].mean()) for k in arr},
            "std": {k: float(arr[k].std()) for k in arr},
        }
    return summary


def main():
    seeds = [42, 43, 44]
    configs = [
        ("lorenz96", 5, 500),
        ("lorenz96", 10, 500),
        ("mackey_glass", 5, 500),
        ("mackey_glass", 10, 500),
    ]
    results = []
    for ds, n, t in configs:
        print(f"Running {ds} N={n} T={t}")
        res = run_suite(ds, n, t, seeds)
        results.append({"dataset": ds, "N": n, "T": t, "results": res})

    with open("synthetic_baseline_comparison.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
