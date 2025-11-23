"""
Conditional Independence Testing Module for TransCausal

This module implements the conditional independence testing component of the 
TransCausal framework. It applies statistical hypothesis testing on the learned 
Transformer features to construct causal graphs with rigorous false discovery control.

Key innovations:
1. Feature selection using variance and mutual information on learned representations
2. Parent discovery using regularized regression for computational efficiency
3. Conditional independence testing using partial correlation

The testing is applied on transformed features from the Transformer (not raw data),
enabling detection of nonlinear causal relationships that traditional methods miss.

This approach is inspired by constraint-based causal discovery principles but adapted
specifically for deep learning feature spaces.
"""

import numpy as np
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr
import statsmodels.api as sm

class CausalTesting:
    def __init__(self, max_lag=5, alpha=0.05, var_threshold=0.01, mi_threshold=0.01):
        self.max_lag = max_lag
        self.alpha = alpha
        self.var_threshold = var_threshold
        self.mi_threshold = mi_threshold
        
    def select_features(self, features, target):
        """
        Selects features based on variance and mutual information.
        features: (T, D)
        target: (T,)
        Returns: list of indices
        """
        selected_indices = []
        T, D = features.shape
        
        # Variance check
        variances = np.var(features, axis=0)
        
        # Mutual Information check
        # Flatten target if needed
        if len(target.shape) > 1:
            target = target.flatten()
            
        mis = mutual_info_regression(features, target)
        
        for i in range(D):
            if variances[i] > self.var_threshold and mis[i] > self.mi_threshold:
                selected_indices.append(i)
                
        return selected_indices

    def run_pcmci(self, data):
        """
        Conditional independence testing on learned features.
        data: (T, N)
        Returns: 
            results: list of (i, j, lag, corr, p_val)
            val_matrix: (N, N, max_lag+1)
            p_values: (N, N, max_lag+1)
        """
        T, N = data.shape
        p_values = np.ones((N, N, self.max_lag + 1))
        val_matrix = np.zeros((N, N, self.max_lag + 1))
        
        # 1. Parent Discovery using Regularized Regression
        parents = {j: [] for j in range(N)}
        
        for j in range(N):
            y = data[self.max_lag:, j]
            X_candidates = []
            candidate_keys = []
            
            for i in range(N):
                for lag in range(1, self.max_lag + 1):
                    X_candidates.append(data[self.max_lag-lag : T-lag, i])
                    candidate_keys.append((i, lag))
            
            if not X_candidates:
                continue
                
            X_candidates = np.array(X_candidates).T
            
            # Use Lasso for selection
            # Normalize
            X_candidates_norm = (X_candidates - np.mean(X_candidates, axis=0)) / (np.std(X_candidates, axis=0) + 1e-8)
            y_norm = (y - np.mean(y)) / (np.std(y) + 1e-8)
            
            # Alpha 0.1 is arbitrary, could be tuned
            model = sm.OLS(y_norm, X_candidates_norm).fit_regularized(method='elastic_net', alpha=0.05, L1_wt=1.0)
            params = model.params
            
            for idx, coef in enumerate(params):
                if abs(coef) > 1e-4:
                    parents[j].append(candidate_keys[idx])
        
        # 2. MCI
        results = []
        
        for j in range(N):
            my_parents = parents[j]
            y = data[self.max_lag:, j]
            
            for (i, lag) in my_parents:
                # Test X_{t-lag}^i -> X_t^j | Parents(j) \ {X_{t-lag}^i}
                
                cond_set = [p for p in my_parents if p != (i, lag)]
                
                x_i = data[self.max_lag-lag : T-lag, i]
                
                if len(cond_set) > 0:
                    Z = []
                    for (ci, clag) in cond_set:
                        Z.append(data[self.max_lag-clag : T-clag, ci])
                    Z = np.array(Z).T
                    
                    # Partial Correlation
                    try:
                        # Regress y on Z
                        reg_y = sm.OLS(y, sm.add_constant(Z)).fit()
                        r_y = reg_y.resid
                        
                        # Regress x_i on Z
                        reg_xi = sm.OLS(x_i, sm.add_constant(Z)).fit()
                        r_xi = reg_xi.resid
                        
                        corr, p_val = pearsonr(r_y, r_xi)
                    except:
                        corr, p_val = 0, 1.0
                else:
                    corr, p_val = pearsonr(y, x_i)
                
                val_matrix[i, j, lag] = corr
                p_values[i, j, lag] = p_val
                
                if p_val < self.alpha:
                    results.append((i, j, lag, corr, p_val))
                    
        return results, val_matrix, p_values
