#!/usr/bin/env python
"""
MNIST Classification with Fractional Differential Equation Networks

This example demonstrates training a neural network where the feature extraction
layer is replaced by a fractional differential equation block. The FDE layer
captures long-range dependencies through its non-local memory structure.

Requirements:
    - torch
    - torchvision
    - torchfde (custom FDE solver library)

Usage:
    # Basic training with predictor method
    python fde_mnist.py --method gl --beta 0.9 --T 2 --step_size 0.1

    # Memory-efficient adjoint backpropagation
    python fde_mnist.py --method gl-f --adjoint True --beta 0.9 --T 2 --step_size 0.1

    # With truncated memory (for efficiency)
    python fde_mnist.py --method gl --beta 0.9 --T 2 --memory 50

    # Multi-term FDE
    python fde_mnist.py --method glmulti \\
        --multi_beta 0.3 0.7 0.9 \\
        --multi_coefficient 0.2 0.3 0.5 \\
        --learn_coefficient

"""

import os
import argparse
import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FDEConfig:
    """Configuration for the FDE solver."""
    beta: float = 0.9
    T: float = 5.0
    step_size: float = 0.1
    method: str = 'predictor'
    memory: int = -1  # -1 for full history
    return_history: bool = False

    # Multi-term FDE settings
    multi_beta: Optional[List[float]] = None
    multi_coefficient: Optional[List[float]] = None
    learn_coefficient: bool = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train FDE-based neural network on MNIST',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Network architecture
    parser.add_argument('--network', type=str, default='odenet',
                        choices=['resnet', 'odenet'],
                        help='Network architecture')
    parser.add_argument('--downsampling-method', type=str, default='conv',
                        choices=['conv', 'res'],
                        help='Downsampling method before FDE block')

    # Training parameters
    parser.add_argument('--nepochs', type=int, default=160,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=256,
                        help='Test/evaluation batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--data_aug', action='store_true', default=True,
                        help='Enable data augmentation')
    parser.add_argument('--no_data_aug', action='store_false', dest='data_aug',
                        help='Disable data augmentation')

    # FDE solver parameters
    parser.add_argument('--method', type=str, default='predictor',
                        choices=[
                            # Adjoint methods
                            'predictor-f', 'predictor-o',
                            'l1-f', 'l1-o',
                            'gl-f', 'gl-o',
                            'trap-f', 'trap-o',
                            # Direct methods
                            'predictor', 'l1', 'gl', 'trap', 'glmulti'
                        ],
                        help='FDE solver method')
    parser.add_argument('--adjoint', action='store_true', default=False,
                        help='Use adjoint backpropagation')
    parser.add_argument('--no_adjoint', action='store_false', dest='adjoint',
                        help='Use direct backpropagation')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='Fractional order (0 < beta <= 1)')
    parser.add_argument('--T', type=float, default=2.0,
                        help='Integration terminal time')
    parser.add_argument('--step_size', type=float, default=0.1,
                        help='Integration step size')
    parser.add_argument('--memory', type=int, default=-1,
                        help='Memory length for history truncation (-1 for full history)')
    parser.add_argument('--return_history', action='store_true', default=False,
                        help='Return full trajectory history from FDE solver')

    # Multi-term FDE parameters
    parser.add_argument('--multi_beta', type=float, nargs='+', default=None,
                        help='Fractional orders for multi-term FDE')
    parser.add_argument('--multi_coefficient', type=float, nargs='+', default=None,
                        help='Coefficients for multi-term FDE')
    parser.add_argument('--learn_coefficient', action='store_true',
                        help='Make multi-term coefficients learnable')

    # System settings
    parser.add_argument('--save', type=str, default='./exp_mnist',
                        help='Directory to save logs and checkpoints')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    return parser.parse_args()


# =============================================================================
# Neural Network Components
# =============================================================================

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


def norm(dim: int) -> nn.GroupNorm:
    """Group normalization layer."""
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    """Residual block with pre-activation."""

    def __init__(self, inplanes: int, planes: int,
                 stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.relu(self.norm2(out))
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):
    """Convolution that concatenates time as an extra channel."""

    def __init__(self, dim_in: int, dim_out: int, ksize: int = 3,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 groups: int = 1, bias: bool = True, transpose: bool = False):
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out,
            kernel_size=ksize, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Broadcast time to spatial dimensions
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], dim=1)
        return self._layer(ttx)


class ODEFunc(nn.Module):
    """
    Neural network defining the right-hand side of the FDE.

    Implements f(t, z) for the equation D^β z = f(t, z).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.norm1(x))
        out = self.conv1(t, out)
        out = self.relu(self.norm2(out))
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEFuncDeep(nn.Module):
    """
    Deeper ODEFunc: 4 convolutional layers with residual connection.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = norm(dim)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.conv3 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm4 = norm(dim)
        self.conv4 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm5 = norm(dim)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # First block
        out = self.act(self.norm1(x))
        out = self.conv1(t, out)
        out = self.act(self.norm2(out))
        out = self.dropout(out)
        out = self.conv2(t, out)

        # Residual connection
        residual = out

        # Second block
        out = self.act(self.norm3(out))
        out = self.conv3(t, out)
        out = self.act(self.norm4(out))
        out = self.dropout(out)
        out = self.conv4(t, out)

        # Add residual
        out = out + residual
        out = self.norm5(out)

        return out


class FDEBlock(nn.Module):
    """
    Fractional Differential Equation block.

    Replaces standard neural network layers with an FDE integration.
    The block solves D^β z = f(t, z) from t=0 to t=T.
    """

    def __init__(self, odefunc: nn.Module, fde_config: FDEConfig, fdeint_solver):
        super().__init__()
        self.odefunc = odefunc
        self.fde_config = fde_config
        self.fdeint_solver = fdeint_solver

        # Setup multi-term parameters if specified
        self._setup_multi_term()

    def _setup_multi_term(self):
        """Initialize multi-term FDE parameters."""
        cfg = self.fde_config

        if cfg.multi_coefficient is not None:
            coeff_tensor = torch.tensor(cfg.multi_coefficient, dtype=torch.float32)
            beta_tensor = torch.tensor(cfg.multi_beta, dtype=torch.float32)

            if cfg.learn_coefficient:
                self.multi_coefficient = nn.Parameter(coeff_tensor)
            else:
                self.register_buffer('multi_coefficient', coeff_tensor)

            self.register_buffer('multi_beta', beta_tensor)
        else:
            self.multi_coefficient = None
            self.multi_beta = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.fde_config
        options = {
            'memory': cfg.memory,
            'return_history': cfg.return_history
        }

        # Determine beta and coefficients
        if self.multi_coefficient is not None:
            beta = self.multi_beta.to(x.device)
            options['multi_coefficient'] = self.multi_coefficient.to(x.device)
        else:
            beta = torch.tensor(cfg.beta, device=x.device, dtype=x.dtype)

        # Integrate the FDE
        out = self.fdeint_solver(
            self.odefunc, x, beta,
            t=cfg.T,
            step_size=cfg.step_size,
            method=cfg.method,
            options=options
        )

        return out

    def extra_repr(self) -> str:
        cfg = self.fde_config
        base_repr = f"beta={cfg.beta}, T={cfg.T}, step_size={cfg.step_size}, method='{cfg.method}'"
        base_repr += f", memory={cfg.memory}, return_history={cfg.return_history}"
        if self.multi_coefficient is not None:
            base_repr = f"multi_term=True, beta={list(cfg.multi_beta)}, " + base_repr
        return base_repr


class Flatten(nn.Module):
    """Flatten spatial dimensions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


# =============================================================================
# Data Loading
# =============================================================================

def get_mnist_loaders(data_aug: bool = True, batch_size: int = 128,
                      test_batch_size: int = 1000) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create MNIST data loaders.

    Returns:
        Tuple of (train_loader, test_loader, train_eval_loader)
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4) if data_aug else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root='data/mnist', train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.MNIST(
        root='data/mnist', train=False, download=True, transform=transform_test
    )
    train_eval_dataset = datasets.MNIST(
        root='data/mnist', train=True, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, drop_last=True
    )
    # Note: drop_last=False for test loaders to evaluate on all samples
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=2, drop_last=False
    )
    train_eval_loader = DataLoader(
        train_eval_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=2, drop_last=False
    )

    return train_loader, test_loader, train_eval_loader


# =============================================================================
# Training Utilities
# =============================================================================

def inf_generator(iterable):
    """Infinite iterator over a dataloader."""
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def get_lr_scheduler(initial_lr: float, batch_size: int, batches_per_epoch: int,
                     boundary_epochs: Tuple[int, ...],
                     decay_rates: Tuple[float, ...]) -> Callable[[int], float]:
    """
    Create a step learning rate scheduler.

    Returns a function that maps iteration number to learning rate.
    """
    # Scale LR by batch size
    scaled_lr = initial_lr * batch_size / 128

    boundaries = [batches_per_epoch * epoch for epoch in boundary_epochs]
    lrs = [scaled_lr * decay for decay in decay_rates]

    def lr_fn(iteration: int) -> float:
        for boundary, lr in zip(boundaries, lrs):
            if iteration < boundary:
                return lr
        return lrs[-1]

    return lr_fn


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, dataloader: DataLoader,
                      device: torch.device) -> float:
    """Compute classification accuracy."""
    model.eval()
    correct = 0
    total = 0

    for x, y in dataloader:
        x = x.to(device)
        logits = model(x)
        predictions = logits.argmax(dim=1).cpu()
        correct += (predictions == y).sum().item()
        total += y.size(0)

    model.train()
    return correct / total


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Model Construction
# =============================================================================

def build_model(network_type: str, downsampling: str,
                fde_config: FDEConfig, fdeint_solver, dim: int = 64) -> nn.Module:
    """
    Build the complete model.

    Architecture:
        Input -> Downsampling -> Feature Extraction -> Classification
    """
    # Downsampling layers
    if downsampling == 'conv':
        downsampling_layers = [
            nn.Conv2d(1, dim, 3, 1),
            norm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            norm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 4, 2, 1),
        ]
    else:  # 'res'
        downsampling_layers = [
            nn.Conv2d(1, dim, 3, 1),
            ResBlock(dim, dim, stride=2, downsample=conv1x1(dim, dim, 2)),
            ResBlock(dim, dim, stride=2, downsample=conv1x1(dim, dim, 2)),
        ]

    # Feature extraction layers
    if network_type == 'odenet':
        feature_layers = [FDEBlock(ODEFuncDeep(dim), fde_config, fdeint_solver)]
    else:  # 'resnet'
        feature_layers = [ResBlock(dim, dim) for _ in range(6)]

    # Classification head
    fc_layers = [
        norm(dim),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Linear(dim, 10)
    ]

    return nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers)


# =============================================================================
# Logging
# =============================================================================

def setup_logger(logpath: str, filepath: str, debug: bool = False) -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger('fde_mnist')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # File handler
    file_handler = logging.FileHandler(logpath, mode='a')
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    # Log the source file
    logger.info(f"Source: {filepath}")

    return logger


# =============================================================================
# Main Training Loop
# =============================================================================

def train(args):
    """Main training function."""
    # Setup directories and logging
    os.makedirs(args.save, exist_ok=True)
    logger = setup_logger(
        logpath=os.path.join(args.save, 'training.log'),
        filepath=os.path.abspath(__file__),
        debug=args.debug
    )

    # Device setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Import FDE integrator based on adjoint setting
    if args.adjoint:
        logger.info("Using adjoint backpropagation")
        from torchfde import fdeint_adjoint as fdeint_solver
    else:
        logger.info("Using direct backpropagation")
        from torchfde import fdeint as fdeint_solver

    # Create FDE configuration
    fde_config = FDEConfig(
        beta=args.beta,
        T=args.T,
        step_size=args.step_size,
        method=args.method,
        memory=args.memory,
        return_history=args.return_history,
        multi_beta=args.multi_beta,
        multi_coefficient=args.multi_coefficient,
        learn_coefficient=args.learn_coefficient,
    )

    # Log configuration
    logger.info(f"FDE Config: beta={fde_config.beta}, T={fde_config.T}, "
                f"step_size={fde_config.step_size}, method='{fde_config.method}', "
                f"memory={fde_config.memory}, return_history={fde_config.return_history}")
    if fde_config.multi_beta:
        logger.info(f"Multi-term: beta={fde_config.multi_beta}, "
                    f"coefficients={fde_config.multi_coefficient}, "
                    f"learnable={fde_config.learn_coefficient}")

    # Build model
    model = build_model(
        network_type=args.network,
        downsampling=args.downsampling_method,
        fde_config=fde_config,
        fdeint_solver=fdeint_solver
    ).to(device)

    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Total parameters: {count_parameters(model):,}")

    # Data loaders
    train_loader, test_loader, train_eval_loader = get_mnist_loaders(
        data_aug=args.data_aug,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size
    )
    batches_per_epoch = len(train_loader)
    logger.info(f"Training samples: {len(train_loader.dataset)}, "
                f"batches/epoch: {batches_per_epoch}")

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    lr_fn = get_lr_scheduler(
        initial_lr=args.lr,
        batch_size=args.batch_size,
        batches_per_epoch=batches_per_epoch,
        boundary_epochs=(60, 100, 140),
        decay_rates=(1.0, 0.1, 0.01, 0.001)
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    data_gen = inf_generator(train_loader)
    best_acc = 0.0

    logger.info("Starting training...")
    epoch_start_time = time.time()

    for iteration in range(args.nepochs * batches_per_epoch):
        # Update learning rate
        lr = lr_fn(iteration)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Training step
        optimizer.zero_grad()
        x, y = next(data_gen)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # End of epoch evaluation
        if (iteration + 1) % batches_per_epoch == 0:
            epoch = (iteration + 1) // batches_per_epoch
            epoch_time = time.time() - epoch_start_time

            # Evaluate
            train_acc = evaluate_accuracy(model, train_eval_loader, device)
            test_acc = evaluate_accuracy(model, test_loader, device)

            # Track best
            if test_acc > best_acc:
                best_acc = test_acc

            logger.info(
                f"Epoch {epoch:03d} | "
                f"Time {epoch_time:.1f}s | "
                f"LR {lr:.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Test Acc {test_acc:.4f} | "
                f"Best {best_acc:.4f}"
            )

            epoch_start_time = time.time()

    logger.info(f"Training complete. Best test accuracy: {best_acc:.4f}")
    return best_acc


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    args = parse_args()
    train(args)