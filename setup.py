from setuptools import setup, find_packages

setup(
    name="torchfde",
    author="Qiyu Kang",
    author_email="qiyukang@ustc.edu.cn",
    description="Differentiable FDE solvers in PyTorch with the adjoint backpropagation technique.",
    url="https://github.com/kangqiyu/torchfde",
    version="0.0.1",
    packages=find_packages(), 
    install_requires=[],
    python_requires=">=3.7",
)
