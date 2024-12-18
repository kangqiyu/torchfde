# torchfde: Differentiable FDE Solvers in PyTorch

The preliminary version (no adjoint backpropagation) of this library is at https://github.com/zknus/torchfde

Fractional-order differential equations (FDEs) extend the concept of traditional differential operators by generalizing their order from integers to real numbers, enhancing flexibility in modeling complex dynamic systems and intricate structures. 

Recent progress at the intersection of FDEs and deep learning has catalyzed a new wave of innovative models and applications, demonstrating the potential to address challenges such as graph representation learning. Examples:

1. Q. Kang, K. Zhao, Q. Ding, F. Ji, X. Li, W. Liang, Y. Song, and W. P. Tay, “[Unleashing the potential of fractional calculus in graph neural networks with FROND](https://openreview.net/forum?id=wcka3bd7P4),” Proc. International Conference on Learning Representations (ICLR), Vienna, Austria, May 2024, __Spotlight__.

2. Q. Kang, K. Zhao, Y. Song, Y. Xie, Y. Zhao, S. Wang, R. She, and W. P. Tay, “[Coupling graph neural networks with fractional order continuous dynamics: A robustness study](https://arxiv.org/pdf/2401.04331),” Proc. AAAI Conference on Artificial Intelligence, Vancouver, Canada, Feb. 2024.
   
3. W. Cui, Q. Kang, X. Li, K. Zhao, W. P. Tay, W. Deng, and Y. Li, “Neural variable-order fractional differential equation networks,” in Proc. AAAI Conference on Artificial Intelligence (AAAI), Philadelphia, USA, Feb. 2025.
   
4. K. Zhao, X. Li, Q. Kang†, F. Ji, Q. Ding, Y. Zhao, W. Liang, and W. P. Tay, “[Distributed-order fractional graph operating network](https://arxiv.org/pdf/2411.05274)," Advances in Neural Information Processing Systems (NeurIPS), Vancouver, Canada, Dec. 2024, __Spotlight__.


However, training neural FDEs has primarily relied on direct differentiation through forward-pass operations in FDE solvers, leading to increased memory usage and computational complexity, particularly in large-scale applications. 

___To address these challenges, in this library, we propose a scalable adjoint backpropagation method for training neural FDEs by solving an augmented FDE backward in time, which substantially reduces memory requirements___. 

As the solvers are implemented in PyTorch, algorithms in this repository are fully supported to run on the GPU.

The code is coming soon.

# References

Q. Kang, X. Li, K. Zhao, W. Cui, Y. Zhao, W. Deng, and W. P. Tay, “Efficient training of neural fractional-order differential equation via adjoint backpropagation,” in Proc. AAAI Conference on Artificial Intelligence (AAAI), Philadelphia, USA, Feb. 2025.
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
Q. Kang, K. Zhao, Q. Ding, F. Ji, X. Li, W. Liang, Y. Song, and W. P. Tay, “Unleashing the potential of fractional calculus in graph neural networks with FROND,” in Proc. International Conference on Learning Representations (ICLR), Vienna, Austria, May 2024, spotlight.
```
@INPROCEEDINGS{KanZhaDin:C24,
    author = {Qiyu Kang and Kai Zhao and Qinxu Ding and Feng Ji and Xuhao Li and Wenfei Liang and Yang Song and Wee Peng Tay},
    title={Unleashing the Potential of Fractional Calculus in Graph Neural Networks with {FROND}},
    booktitle={Proc. International Conference on Learning Representations},
    year={2024},
    address = {Vienna, Austria},
}
```

 

