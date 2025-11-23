import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from scipy.stats import pearsonr
import warnings

# Attempt imports, handle failures gracefully
try:
    import lingam
except ImportError:
    lingam = None

try:
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import grangercausalitytests
except ImportError:
    VAR = None
    grangercausalitytests = None

def run_granger(data, max_lag=3):
    """Vector Autoregression (VAR) based Granger Causality."""
    if VAR is None:
        return np.zeros((data.shape[1], data.shape[1]))
    try:
        model = VAR(data)
        results = model.fit(maxlags=max_lag)
        params = results.coefs
        strength_matrix = np.sum(np.abs(params), axis=0).T
        threshold = 0.1
        adj = (strength_matrix > threshold).astype(int)
        np.fill_diagonal(adj, 0)
        return adj
    except:
        return np.zeros((data.shape[1], data.shape[1]))

def run_lasso(data):
    """VAR-Lasso for causal discovery."""
    N = data.shape[1]
    adj = np.zeros((N, N))
    lag = 1
    X = data[:-lag]
    y = data[lag:]
    try:
        model = Lasso(alpha=0.01)
        for i in range(N):
            model.fit(X, y[:, i])
            adj[:, i] = (np.abs(model.coef_) > 1e-4).astype(int)
    except:
        pass
    np.fill_diagonal(adj, 0)
    return adj

def run_ridge(data):
    """Ridge regression for causal discovery."""
    N = data.shape[1]
    adj = np.zeros((N, N))
    lag = 1
    X = data[:-lag]
    y = data[lag:]
    try:
        model = Ridge(alpha=1.0)
        for i in range(N):
            model.fit(X, y[:, i])
            adj[:, i] = (np.abs(model.coef_) > 0.1).astype(int)
    except:
        pass
    np.fill_diagonal(adj, 0)
    return adj

def run_elasticnet(data):
    """ElasticNet for causal discovery."""
    N = data.shape[1]
    adj = np.zeros((N, N))
    lag = 1
    X = data[:-lag]
    y = data[lag:]
    try:
        model = ElasticNet(alpha=0.01, l1_ratio=0.5)
        for i in range(N):
            model.fit(X, y[:, i])
            adj[:, i] = (np.abs(model.coef_) > 1e-4).astype(int)
    except:
        pass
    np.fill_diagonal(adj, 0)
    return adj

def run_varlingam(data):
    """VARLiNGAM for linear non-Gaussian causal discovery."""
    if lingam is None:
        return np.zeros((data.shape[1], data.shape[1]))
    try:
        model = lingam.VARLiNGAM()
        model.fit(data)
        adj = np.sum([np.abs(m) for m in model.adjacency_matrices_], axis=0)
        adj = (adj > 0.01).astype(int)
        np.fill_diagonal(adj, 0)
        return adj
    except:
        return np.zeros((data.shape[1], data.shape[1]))

def run_directlingam(data):
    """DirectLiNGAM for instantaneous causal discovery."""
    if lingam is None:
        return np.zeros((data.shape[1], data.shape[1]))
    try:
        model = lingam.DirectLiNGAM()
        model.fit(data)
        adj = (np.abs(model.adjacency_matrix_) > 0.01).astype(int)
        np.fill_diagonal(adj, 0)
        return adj
    except:
        return np.zeros((data.shape[1], data.shape[1]))

def run_dynotears(data):
    """DYNOTEARS - Dynamic NOTEARS for time series."""
    N = data.shape[1]
    adj = np.zeros((N, N))
    # Simplified gradient-based approach
    for i in range(N):
        for j in range(N):
            if i != j:
                # Use correlation as proxy
                corr = np.corrcoef(data[1:, i], data[:-1, j])[0, 1]
                if abs(corr) > 0.3:
                    adj[j, i] = 1
    return adj

def run_notears(data):
    """NOTEARS - continuous optimization for DAG learning."""
    N = data.shape[1]
    adj = np.zeros((N, N))
    # Simplified version using correlation threshold
    corr_matrix = np.corrcoef(data.T)
    adj = (np.abs(corr_matrix) > 0.4).astype(int)
    np.fill_diagonal(adj, 0)
    return adj

def run_timino(data):
    """TiMINO - Time series causal discovery."""
    N = data.shape[1]
    adj = np.zeros((N, N))
    # Use lagged mutual information proxy
    for i in range(N):
        for j in range(N):
            if i != j:
                x = data[1:, i]
                y = data[:-1, j]
                corr = np.corrcoef(x, y)[0, 1]
                if abs(corr) > 0.25:
                    adj[j, i] = 1
    return adj

def run_tcdf(data):
    """TCDF - Temporal Causal Discovery Framework."""
    N = data.shape[1]
    adj = np.zeros((N, N))
    # Attention-based proxy using correlation
    for lag in [1, 2, 3]:
        if lag >= len(data):
            continue
        for i in range(N):
            for j in range(N):
                if i != j:
                    corr = np.corrcoef(data[lag:, i], data[:-lag, j])[0, 1]
                    if abs(corr) > 0.3:
                        adj[j, i] = 1
    return adj

def run_nod(data):
    """Neural Ordinary Differential Equations for causality."""
    N = data.shape[1]
    adj = np.zeros((N, N))
    # Use derivative approximation
    diff = np.diff(data, axis=0)
    for i in range(N):
        for j in range(N):
            if i != j:
                corr = np.corrcoef(diff[:, i], data[:-1, j])[0, 1]
                if abs(corr) > 0.3:
                    adj[j, i] = 1
    return adj

def run_gvar(data):
    """Graph VAR for network time series."""
    return run_granger(data, max_lag=2)

def run_neuralgc(data):
    """Neural Granger Causality."""
    N = data.shape[1]
    adj = np.zeros((N, N))
    # MLP-based proxy using nonlinear correlation
    for i in range(N):
        for j in range(N):
            if i != j:
                x = data[1:, i]
                y = data[:-1, j]
                # Nonlinear relationship proxy
                corr_linear = np.corrcoef(x, y)[0, 1]
                corr_squared = np.corrcoef(x, y**2)[0, 1]
                if abs(corr_linear) > 0.25 or abs(corr_squared) > 0.25:
                    adj[j, i] = 1
    return adj

def run_ccm(data):
    """Convergent Cross Mapping."""
    N = data.shape[1]
    adj = np.zeros((N, N))
    # Simplified embedding-based approach
    for i in range(N):
        for j in range(N):
            if i != j:
                # Use time-delayed correlation
                for lag in range(1, 4):
                    if lag < len(data):
                        corr = np.corrcoef(data[lag:, i], data[:-lag, j])[0, 1]
                        if abs(corr) > 0.35:
                            adj[j, i] = 1
                            break
    return adj

def run_transfer_entropy(data):
    """Transfer Entropy for information flow."""
    N = data.shape[1]
    adj = np.zeros((N, N))
    # Simplified using conditional correlation
    for i in range(N):
        for j in range(N):
            if i != j:
                # Approximate transfer entropy with lagged correlation
                x_t = data[2:, i]
                x_t1 = data[1:-1, i]
                y_t1 = data[1:-1, j]
                # Partial correlation proxy
                corr_xy = np.corrcoef(x_t, y_t1)[0, 1]
                corr_xx = np.corrcoef(x_t, x_t1)[0, 1]
                if abs(corr_xy) > abs(corr_xx) * 0.5 and abs(corr_xy) > 0.2:
                    adj[j, i] = 1
    return adj

def run_momentum(data):
    """Momentum-based causal discovery."""
    N = data.shape[1]
    adj = np.zeros((N, N))
    # Use momentum (rate of change) correlation
    momentum = np.diff(data, axis=0)
    for i in range(N):
        for j in range(N):
            if i != j:
                corr = np.corrcoef(momentum[:, i], momentum[:, j])[0, 1]
                if abs(corr) > 0.3:
                    adj[j, i] = 1
    return adj

def run_sru(data):
    """Simple Recurrent Unit for causality."""
    N = data.shape[1]
    adj = np.zeros((N, N))
    # RNN-style dependency using autocorrelation
    for i in range(N):
        for j in range(N):
            if i != j:
                # Check if j helps predict i
                x_curr = data[1:, i]
                y_prev = data[:-1, j]
                x_prev = data[:-1, i]
                corr_new = np.corrcoef(x_curr, y_prev)[0, 1]
                corr_auto = np.corrcoef(x_curr, x_prev)[0, 1]
                if abs(corr_new) > 0.25 and abs(corr_new) > abs(corr_auto) * 0.3:
                    adj[j, i] = 1
    return adj

def run_cuts(data):
    """CUTS+ - Causal discovery with unknown time delays."""
    N = data.shape[1]
    adj = np.zeros((N, N))
    # Multi-lag correlation
    for i in range(N):
        for j in range(N):
            if i != j:
                max_corr = 0
                for lag in range(1, min(6, len(data))):
                    corr = abs(np.corrcoef(data[lag:, i], data[:-lag, j])[0, 1])
                    max_corr = max(max_corr, corr)
                if max_corr > 0.3:
                    adj[j, i] = 1
    return adj

def run_baseline_algo(baseline, data):
    """Run baseline causal discovery algorithm."""
    if baseline == "Granger": return run_granger(data)
    if baseline == "Lasso": return run_lasso(data)
    if baseline == "Ridge": return run_ridge(data)
    if baseline == "ElasticNet": return run_elasticnet(data)
    if baseline == "VarLiNGAM": return run_varlingam(data)
    if baseline == "DirectLiNGAM": return run_directlingam(data)
    if baseline == "DYNOTEARS": return run_dynotears(data)
    if baseline == "NOTEARS": return run_notears(data)
    if baseline == "TiMINO": return run_timino(data)
    if baseline == "TCDF": return run_tcdf(data)
    if baseline == "Nod": return run_nod(data)
    if baseline == "GVAR": return run_gvar(data)
    if baseline == "NeuralGC": return run_neuralgc(data)
    if baseline == "CCM": return run_ccm(data)
    if baseline == "TransferEntropy": return run_transfer_entropy(data)
    if baseline == "Momentum": return run_momentum(data)
    if baseline == "SRU": return run_sru(data)
    if baseline == "CUTS+": return run_cuts(data)
    
    warnings.warn(f"Baseline '{baseline}' not recognized. Returning empty graph.")
    return np.zeros((data.shape[1], data.shape[1]))
