import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')

parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--save', type=str, default='./exp_mnist')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--method', type=str, choices=
['gl', 'trap'], default='gl')

# parameters for the FDE solver
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--step_size', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--T', type=float, default=1.0)

# New argument for testing return_history
parser.add_argument('--return_history', type=eval, default=False, choices=[True, False])

args = parser.parse_args()

if args.adjoint:
    print("using adjoint backpropagation")
    from torchfde import fdeint_adjoint as fdeint
else:
    print("not using adjoint backpropagation")
    from torchfde import fdeint


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        # self.nfe = 0

    def forward(self, t, x):
        # self.nfe += 1
        x = x[0]
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return (out, out)


class ODEBlock(nn.Module):

    def __init__(self, odefunc, return_history=False):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.return_history = return_history
        self._print_shape_once = True  # Flag to print shape only once

    def forward(self, x):
        out = fdeint(self.odefunc, x, torch.tensor(args.beta), t=args.T,
                     step_size=args.step_size, method=args.method,
                     options={"corrector_step": 5, "return_history": self.return_history})

        if self.return_history:
            # out is a tuple of stacked tensors: (traj_component1, traj_component2)
            # Each has shape (N_timesteps, batch, channels, height, width)

            # Print shape info (only once to avoid cluttering output)
            if self._print_shape_once:
                print("\n" + "=" * 60)
                print("FDE return_history=True output shapes:")
                print(f"  Output type: {type(out)}")
                print(f"  Number of components: {len(out)}")
                for i, comp in enumerate(out):
                    print(f"  Component {i} shape: {comp.shape}")
                    print(f"    - N_timesteps: {comp.shape[0]}")
                    print(f"    - Batch size: {comp.shape[1]}")
                    print(f"    - Channels: {comp.shape[2]}")
                    print(f"    - Height x Width: {comp.shape[3]} x {comp.shape[4]}")
                print("=" * 60 + "\n")
                self._print_shape_once = False

            # Extract the second component at the final time step for classification
            # out[1] is the second component trajectory, shape (N, batch, C, H, W)
            # out[1][-1] is the final time step, shape (batch, C, H, W)
            final_output = (out[0][-1], out[1][-1])
            return final_output
        else:
            # out is a tuple: (component1, component2) at final time
            # Each has shape (batch, channels, height, width)

            # Print shape info (only once)
            if self._print_shape_once:
                print("\n" + "=" * 60)
                print("FDE return_history=False output shapes:")
                print(f"  Output type: {type(out)}")
                print(f"  Number of components: {len(out)}")
                for i, comp in enumerate(out):
                    print(f"  Component {i} shape: {comp.shape}")
                print("=" * 60 + "\n")
                self._print_shape_once = False

            return out


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class Downsampling(nn.Module):

    def __init__(self):
        super(Downsampling, self).__init__()
        self.downsampling_layers = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        )

    def forward(self, x):
        out = self.downsampling_layers(x)
        return (out, out)


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.Sequential(
            norm(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        # x is a tuple (component1, component2)
        # Use the second component for classification
        out = self.decoder_layers(x[1])
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='data/mnist', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='data/mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


if __name__ == '__main__':

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    is_odenet = args.network == 'odenet'

    downsampling_layers = Downsampling()
    # Pass return_history flag to ODEBlock
    feature_layers = [ODEBlock(ODEfunc(64), return_history=args.return_history)] if is_odenet else [ResBlock(64, 64) for
                                                                                                    _ in range(6)]
    fc_layers = Decoder()

    model = nn.Sequential(downsampling_layers, *feature_layers, fc_layers).to(device)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader, train_eval_loader = get_mnist_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_acc = 0
    print(
        f"beta={args.beta:.2f}, T={args.T:.1f}, step_size={args.step_size:.2f}, method='{args.method}', return_history={args.return_history}")

    for itr in range(args.nepochs * batches_per_epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        if itr % batches_per_epoch == 0 and itr > 0:
            with torch.no_grad():
                train_acc = accuracy(model, train_eval_loader)
                val_acc = accuracy(model, test_loader)

                print("Epoch {:04d} |Train Acc  {:.4f}| Test Acc {:.4f}".format(itr // batches_per_epoch, train_acc,
                                                                                val_acc))
