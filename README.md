# TransCausal: Deep Learning Meets Conditional Independence Testing for Time Series Causal Discovery

TransCausal is a novel framework that integrates causality-aware Transformer architectures with conditional independence testing for robust and scalable time series causal discovery. It combines the representational power of deep learning to capture nonlinear dependencies with the statistical rigor of conditional independence tests to control false discoveries.

## Novel Contributions

TransCausal addresses fundamental limitations in time series causal discovery through three key architectural innovations:

### 1. Causality-Aware Transformer Architecture
- **Dilated Temporal Convolutions**: Exponentially expanding receptive fields (dilation rates: 1, 2, 4, ...) to capture multi-scale temporal dependencies without the quadratic complexity of full attention.
- **Causal Attention Masking**: Strict enforcement of temporal ordering through lower-triangular attention masks, preventing information leakage from future timesteps.
- **Channel-Independent Processing**: Each time series variable is processed independently through the Transformer, enabling linear scalability with dimensionality (O(N) instead of O(N²)).

### 2. Dual-Objective Training Strategy
- **Predictive Loss**: Standard autoregressive forecasting objective to learn meaningful temporal representations.
- **Sparsity Regularization**: L1 penalty on learned features to encourage discovery of sparse causal graphs, preventing overfitting and false discoveries.
- **Linear Skip Connections**: Identity-initialized linear path that preserves autoregressive dynamics, crucial for robustness on linear and noisy systems.

### 3. Hybrid Deep Learning + Statistical Testing
- **Feature Transformation**: Transformer learns nonlinear feature representations that make causal relationships more apparent.
- **Conditional Independence Testing**: PCMCI applied on transformed features (not raw data) for rigorous false discovery control.
- **Noise Injection Training**: Gaussian noise augmentation during training (denoising objective) improves robustness to real-world measurement noise.

### Key Advantages
- **Nonlinearity**: Captures complex dependencies that linear methods (Granger, VAR) miss.
- **Statistical Rigor**: Controls false positives better than pure neural approaches (Neural GC, TCDF).
- **Scalability**: Linear complexity in dimensionality, handles high-dimensional systems (N > 50).
- **Robustness**: Maintains performance under noise (σ up to 1.0) through skip connections and denoising.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/xiufengliu/TransCausal.git
   cd TransCausal
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Demo
We provide a demo script `demo.py` that runs TransCausal on synthetic data (Lorenz-96) to demonstrate its capabilities.

```bash
python demo.py
```

This script will:
1. Generate synthetic Lorenz-96 data.
2. Train the TransCausal model.
3. Perform causal discovery using the trained model.
4. Evaluate performance (F1 score) against ground truth.

### Using TransCausal in Your Code

```python
from src.models.transformer import CausalTransformer
from src.models.causal_testing import CausalTesting
# ... (import other necessary modules)

# 1. Train CausalTransformer
# ... (see demo.py for training loop example)

# 2. Run Causal Discovery
causal_test = CausalTesting(max_lag=3)
results = causal_test.run_pcmci(transformed_data)
```

## License
This project is licensed under the MIT License.
