# torchfde: Differentiable FDE Solvers in PyTorch


## Introduction

Fractional-order differential equations (FDEs) extend traditional differential operators by generalizing their order from integers to real numbers. This enhanced flexibility enables more accurate modeling of complex dynamic systems and intricate structures across various domains.

The dynamics of the system state are modeled by the following neural FDE:

$${ }_{\text {left }} D_C^\beta \mathbf{z}(t)=f(t, \mathbf{z}(t) ; \boldsymbol{\theta}), \quad 0<\beta \leq 1$$

where $\mathbf{z}(0)=\mathbf{z}_0$. When $\beta=1$, the equation simplifies to a standard first-order system.

- $f$: Trainable fractional derivatives of the hidden state
- $\mathbf{z}(t)$: System hidden state

- $\mathbf{z}(0)=\mathbf{z}_0$: Initial condition
- $T$: Predetermined terminal time

The terminal state $\mathbf{z}(T)$ is used for downstream tasks such as classification or regression.

__In this library, we propose a scalable adjoint backpropagation method for training neural FDEs by solving an augmented FDE backward in time, which substantially reduces memory requirements__. 

As the solvers are implemented in PyTorch, algorithms in this repository are fully supported to run on the GPU.


*Note: The preliminary version (no adjoint backpropagation) of this library is at https://github.com/zknus/torchfde*

## Installation
```bash
pip install git+https://github.com/kangqiyu/torchfde.git
```

## Examples

See the  [`examples`](./examples) directory.

## Basic Usage

```

from torchfde import fdeint_adjoint as fdeint
# Uncomment the following lines if not using adjoint backpropagation
# from torchfde import fdeint

out = fdeint(odefunc, z_0, beta, T, step_size, method)
```
#### Keyword arguments:

- `T`: Scalar terminal time for integration.
-  `method`: Integration methods supported:
    * adjoint method: `'predictor-f'`, `'predictor-o'`, `'gl-f'`, `'gl-o'`, `'trap-f'`, `'trap-o'`
    * direct method:  `'predictor'`, `'corrector'`, `'implicitl1'`, `'gl'`, `'trap'`  

## References

Q. Kang, X. Li, K. Zhao, W. Cui, Y. Zhao, W. Deng, and W. P. Tay, “[Efficient training of neural fractional-order differential equation via adjoint backpropagation](https://arxiv.org/abs/2503.16666),” in Proc. AAAI Conference on Artificial Intelligence (AAAI), Philadelphia, USA, Feb. 2025.
```
@INPROCEEDINGS{KanLiZha:C25,
	author = {Qiyu Kang and Xuhao Li and Kai Zhao and Wenjun Cui and Yanan Zhao and Weihua Deng and Wee Peng Tay},
	title = {Efficient Training of Neural Fractional-Order Differential Equation via Adjoint Backpropagation},
	booktitle = {Proc. AAAI Conference on Artificial Intelligence},
	month = {Feb.},
	year = {2025},
	address = {Philadelphia, USA},
}
```
Q. Kang, K. Zhao, Q. Ding, F. Ji, X. Li, W. Liang, Y. Song, and W. P. Tay, “[Unleashing the potential of fractional calculus in graph neural networks with FROND](https://openreview.net/forum?id=wcka3bd7P4),” Proc. International Conference on Learning Representations (ICLR), Vienna, Austria, May 2024, __Spotlight__.
```
@INPROCEEDINGS{KanZhaDin:C24,
    author = {Qiyu Kang and Kai Zhao and Qinxu Ding and Feng Ji and Xuhao Li and Wenfei Liang and Yang Song and Wee Peng Tay},
    title={Unleashing the Potential of Fractional Calculus in Graph Neural Networks with {FROND}},
    booktitle={Proc. International Conference on Learning Representations},
    year={2024},
    address = {Vienna, Austria},
}
```
#### Related Works

- W. Cui, Q. Kang, X. Li, K. Zhao, W. P. Tay, W. Deng, and Y. Li, “[Neural variable-order fractional differential equation networks](https://github.com/cuiwjTech/AAAI2025_NvoFDE),” in Proc. AAAI Conference on Artificial Intelligence (AAAI), Philadelphia, USA, Feb. 2025.
- K. Zhao, X. Li, Q. Kang, F. Ji, Q. Ding, Y. Zhao, W. Liang, and W. P. Tay, “[Distributed-order fractional graph operating network](https://arxiv.org/pdf/2411.05274)," Advances in Neural Information Processing Systems (NeurIPS), Vancouver, Canada, Dec. 2024, __Spotlight__.
- Q. Kang, K. Zhao, Y. Song, Y. Xie, Y. Zhao, S. Wang, R. She, and W. P. Tay, “[Coupling graph neural networks with fractional order continuous dynamics: A robustness study](https://arxiv.org/pdf/2401.04331),” Proc. AAAI Conference on Artificial Intelligence, Vancouver, Canada, Feb. 2024.
