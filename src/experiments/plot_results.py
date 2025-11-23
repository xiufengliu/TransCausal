import json
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(results_file='experiment_results.json'):
    with open(results_file, 'r') as f:
        results = json.load(f)
        
    # Separate by dataset
    datasets = set([r['dataset'] for r in results])
    
    # 1. Varying T Plot
    for ds in datasets:
        ds_results_T = [r for r in results if r['dataset'] == ds and r.get('experiment') == 'varying_T']
        
        if not ds_results_T:
            continue
            
        Ts = sorted(list(set([r['T'] for r in ds_results_T])))
        
        f1_means = []
        f1_stds = []
        shd_means = []
        shd_stds = []
        
        for T in Ts:
            res = next(r for r in ds_results_T if r['T'] == T)
            f1_means.append(res['metrics_mean']['f1'])
            f1_stds.append(res['metrics_std']['f1'])
            shd_means.append(res['metrics_mean']['shd'])
            shd_stds.append(res['metrics_std']['shd'])
            
        # Plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.errorbar(Ts, f1_means, yerr=f1_stds, capsize=5, marker='o', label='TransCausal')
        plt.xlabel('Sample Size (T)')
        plt.ylabel('F1 Score')
        plt.title(f'F1 vs Sample Size ({ds})')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.errorbar(Ts, shd_means, yerr=shd_stds, capsize=5, marker='o', color='orange', label='TransCausal')
        plt.xlabel('Sample Size (T)')
        plt.ylabel('SHD')
        plt.title(f'SHD vs Sample Size ({ds})')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results_T_{ds}.png')
        print(f"Saved plot to results_T_{ds}.png")

    # 2. Varying N Plot
    for ds in datasets:
        ds_results_N = [r for r in results if r['dataset'] == ds and r.get('experiment') == 'varying_N']
        
        if not ds_results_N:
            continue
            
        Ns = sorted(list(set([r['N'] for r in ds_results_N])))
        
        f1_means = []
        f1_stds = []
        shd_means = []
        shd_stds = []
        
        for N in Ns:
            res = next(r for r in ds_results_N if r['N'] == N)
            f1_means.append(res['metrics_mean']['f1'])
            f1_stds.append(res['metrics_std']['f1'])
            shd_means.append(res['metrics_mean']['shd'])
            shd_stds.append(res['metrics_std']['shd'])
            
        # Plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.errorbar(Ns, f1_means, yerr=f1_stds, capsize=5, marker='o', label='TransCausal')
        plt.xlabel('Number of Variables (N)')
        plt.ylabel('F1 Score')
        plt.title(f'F1 vs Dimensions ({ds})')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.errorbar(Ns, shd_means, yerr=shd_stds, capsize=5, marker='o', color='orange', label='TransCausal')
        plt.xlabel('Number of Variables (N)')
        plt.ylabel('SHD')
        plt.title(f'SHD vs Dimensions ({ds})')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results_N_{ds}.png')
        print(f"Saved plot to results_N_{ds}.png")

if __name__ == "__main__":
    plot_metrics()
