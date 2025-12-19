# torchfde: Differentiable Neural FDE Solvers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch library for solving and training neural fractional-order differential equations (FDEs) with GPU support and memory-efficient adjoint backpropagation.

## Overview

Neural Fractional Differential Equations (FDEs) generalize classical differential operators from integer to real orders, enabling more accurate modeling of systems with memory effects, anomalous diffusion, and complex dynamics. This approach was first applied to GNNs in "[Unleashing the Potential of Fractional Calculus in Graph Neural Networks](https://ml4physicalsciences.github.io/2023/files/NeurIPS_ML4PS_2023_242.pdf)" (NeurIPS ML4PS Workshop, 2023) and has since been generalized to a wide range of domains.

This library solves neural FDEs of the form:

$$D^\beta \mathbf{z}(t) = f(t, \mathbf{z}(t); \boldsymbol{\theta}), \quad \mathbf{z}(0) = \mathbf{z}_0, \quad 0 < \beta \leq 1$$

where $D^\beta$ denotes the Caputo or Riemann–Liouville fractional derivative. When $\beta = 1$, this reduces to a standard first-order ODE.

__Roadmap:__ 
- Support for fast FFT and additional operators (e.g., tempered fractional derivatives) is coming soon!

### Key Features

- **Multiple solvers**: Adams-Bashforth predictor, L1 scheme, Grünwald-Letnikov, Product Trapezoidal
- **Adjoint backpropagation**: Memory-efficient training by solving an augmented FDE backward in time
- **Multi-term FDEs**: Support for distributed-order and multi-term fractional equations
- **Trajectory output**: Option to return full solution trajectory at all time points
- **GPU acceleration**: Full CUDA support for all solvers
- **PyTorch integration**: Seamless use with `nn.Module` and autograd

## Installation
```bash
pip install git+https://github.com/kangqiyu/torchfde.git
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- NumPy

## Quick Start
```python
import torch
from torchfde import fdeint

# Define the dynamics: dz/dt^β = -z
def func(t, z):
    return -z

# Initial condition and parameters
z0 = torch.tensor([1.0, 0.5])
beta = torch.tensor(0.9)  # Fractional order

# Solve from t=0 to t=5
z_final = fdeint(func, z0, beta, t=5.0, step_size=0.01, method='predictor')
print(f"z(5) = {z_final}")
```

## Usage

### Basic Integration
```python
from torchfde import fdeint

z_final = fdeint(
    func,           # Right-hand side function f(t, z)
    z0,             # Initial condition
    beta,           # Fractional order (scalar or tensor)
    t=T,            # Terminal time
    step_size=h,    # Integration step size
    method='l1'     # Solver method
)
```

### Returning Full Trajectory

For non-adjoint methods, you can return the full solution trajectory at all time points using the `return_history` option:
```python
from torchfde import fdeint

# Return full trajectory instead of just final value
trajectory = fdeint(
    func, z0, beta, 
    t=T, 
    step_size=h, 
    method='predictor',
    options={'return_history': True}
)

# trajectory shape: (N_timesteps, *z0.shape)
# For T=1.0 and step_size=0.1, N_timesteps = 11
print(f"Trajectory shape: {trajectory.shape}")  # e.g., (11, 2)
print(f"z(0) = {trajectory[0]}")
print(f"z(T) = {trajectory[-1]}")
```

For tuple-valued states (e.g., neural FDE blocks with multiple outputs):
```python
# If z0 = (tensor1, tensor2), the output is a tuple of trajectories
trajectory = fdeint(func, z0, beta, t=T, step_size=h, method='gl', 
                    options={'return_history': True})

# trajectory = (traj_component1, traj_component2)
# Each component has shape (N_timesteps, *component_shape)
print(f"Component 0 shape: {trajectory[0].shape}")  # (N_timesteps, *tensor1.shape)
print(f"Component 1 shape: {trajectory[1].shape}")  # (N_timesteps, *tensor2.shape)
```

> **Note**: The `return_history` option is only available for non-adjoint methods (direct backpropagation). Adjoint methods compute gradients by solving backward in time and do not store the forward trajectory.

### With Adjoint Backpropagation

For memory-efficient training of neural FDEs:
```python
from torchfde import fdeint_adjoint as fdeint

# Gradients computed via adjoint method)
z_final = fdeint(func, z0, beta, t=T, step_size=h, method='gl-f')
loss = loss_fn(z_final, target)
loss.backward()  # Memory-efficient backward pass
```

### Neural FDE Layer
```python
import torch.nn as nn
from torchfde import fdeint

class NeuralFDE(nn.Module):
    def __init__(self, dim, beta=0.9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, dim)
        )
        self.beta = beta
    
    def forward(self, t, z):
        return self.net(z)

class FDEBlock(nn.Module):
    def __init__(self, dim, beta=0.9, T=1.0, step_size=0.1, return_history=False):
        super().__init__()
        self.func = NeuralFDE(dim, beta)
        self.beta = beta
        self.T = T
        self.step_size = step_size
        self.return_history = return_history
    
    def forward(self, z0):
        return fdeint(
            self.func, z0, 
            torch.tensor(self.beta), 
            t=self.T, 
            step_size=self.step_size,
            method='predictor',
            options={'return_history': self.return_history}
        )
```

### Multi-Term FDEs

For distributed-order or multi-term fractional equations:

$$\sum_{i=1}^{n} c_i D^{\beta_i} z(t) = f(t, z)$$
```python
z_final = fdeint(
    func, z0, 
    beta=torch.tensor([0.3, 0.7, 0.9]),  # Multiple orders
    t=T, 
    step_size=h,
    method='glmulti',
    options={
        'multi_coefficient': torch.tensor([0.2, 0.3, 0.5]),  # Coefficients c_i
    }
)
```

## Available Methods

### Direct Methods (Standard Backpropagation)

| Method | Description | Use Case |
|--------|-------------|----------|
| `'predictor'` | Adams-Bashforth (Euler) predictor | Caputo formulation |
| `'l1'` | L1 scheme | Caputo formulation |
| `'gl'` | Grünwald-Letnikov | Riemann-Liouville formulation |
| `'trap'` | Product Trapezoidal | Riemann-Liouville formulation |
| `'glmulti'` | Multi-term GL method | Distributed-order FDEs |

### Adjoint Methods (Memory-Efficient)

Append `-f` (fixed-point) or `-o` (one-step) to the base method name:

| Method | Description |
|--------|-------------|
| `'predictor-f'`, `'predictor-o'` | Adams-Bashforth with adjoint |
| `'l1-f'`, `'l1-o'` | L1 scheme with adjoint |
| `'gl-f'`, `'gl-o'` | Grünwald-Letnikov with adjoint |
| `'trap-f'`, `'trap-o'` | Product Trapezoidal with adjoint |

### Choosing a Method

- **General use**: Start with `'gl'` or `'l1'`
- **Memory-constrained**: Use adjoint variants (`'-f'` or `'-o'`)
- **Multi-term/distributed-order**: Use `'glmulti'`
- **Need full trajectory**: Use direct methods with `return_history=True`

## Options

Additional options can be passed via the `options` dictionary:
```python
options = {
    'memory': 100,              # Truncate history to last 100 steps (-1 for full history)
    'return_history': True,     # Return full trajectory (non-adjoint methods only)
    'multi_coefficient': coeffs, # For multi-term FDEs
}

z_final = fdeint(func, z0, beta, t=T, step_size=h, method='predictor', options=options)
```

### Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `memory` | int | -1 | Maximum history length for short-memory approximation. Use -1 for full history. |
| `return_history` | bool | False | If True, return full trajectory at all time points. Only available for non-adjoint methods. |
| `multi_coefficient` | Tensor | None | Coefficients for multi-term FDEs (required for `'glmulti'` method). |

## Examples

See the [`examples`](./examples) directory for complete examples:

- **MNIST Classification**: `examples/fde_mnist.py` - Image classification with FDE-based neural networks
- **Graph Neural Networks**: Integration with GNN architectures ([FROND](https://github.com/zknus/ICLR2024-FROND))  

### Running MNIST Example
```bash
# Standard backpropagation
python examples/fde_mnist.py --method predictor --beta 0.9 --T 1 --step_size 0.1

# With adjoint backpropagation (memory-efficient)
python examples/fde_mnist.py --method predictor-f --adjoint True --beta 0.9 --T 1 --step_size 0.1

# Multi-term FDE
python examples/fde_mnist.py --method glmulti --multi_beta 0.3 0.7 0.9 --multi_coefficient 0.2 0.3 0.5

# Return full trajectory (non-adjoint only)
python examples/test_tuple.py --method gl --return_history True --beta 0.9 --T 1 --step_size 0.1
```

## Mathematical Background

See [fractional_ode_solver.pdf](pdf/fractional_ode_solver.pdf) for details.

## Citation

If you use this library in your research, please cite:
```bibtex
@inproceedings{KanLiZha:C25,
    author = {Qiyu Kang and Xuhao Li and Kai Zhao and Wenjun Cui and Yanan Zhao 
              and Weihua Deng and Wee Peng Tay},
    title = {Efficient Training of Neural Fractional-Order Differential Equation 
             via Adjoint Backpropagation},
    booktitle = {Proc. AAAI Conference on Artificial Intelligence},
    year = {2025},
    address = {Philadelphia, USA},
}

@inproceedings{KanZhaDin:C24,
    author = {Qiyu Kang and Kai Zhao and Qinxu Ding and Feng Ji and Xuhao Li 
              and Wenfei Liang and Yang Song and Wee Peng Tay},
    title = {Unleashing the Potential of Fractional Calculus in Graph Neural 
             Networks with {FROND}},
    booktitle = {Proc. International Conference on Learning Representations},
    year = {2024},
    address = {Vienna, Austria}
}

@inproceedings{ZhaKanJi:C24,
    author = {Kai Zhao and Qiyu Kang and Feng Ji and Xuhao Li and Qinxu Ding and Yanan Zhao and Wenfei Liang and Wee Peng Tay},
    title = {Distributed-Order Fractional Graph Operating Network},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2024},
    address = {Vancouver, Canada},
}
```

## Other Related Work

- W. Cui, Q. Kang, X. Li, K. Zhao, W. P. Tay, W. Deng, and Y. Li, "[Neural variable-order fractional differential equation networks](https://github.com/cuiwjTech/AAAI2025_NvoFDE)," in Proc. AAAI Conference on Artificial Intelligence (AAAI), Philadelphia, USA, Feb. 2025.
- Q. Kang, K. Zhao, Y. Song, Y. Xie, Y. Zhao, S. Wang, R. She, and W. P. Tay, "[Coupling graph neural networks with fractional order continuous dynamics: A robustness study](https://arxiv.org/pdf/2401.04331)," Proc. AAAI Conference on Artificial Intelligence, Vancouver, Canada, Feb. 2024.

## License

MIT License - see [LICENSE](LICENSE) for details.