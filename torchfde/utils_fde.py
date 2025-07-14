import torch
import warnings
from . import config


def _check_inputs(func, y0, t, step_size, method, beta, SOLVERS):
    if method not in SOLVERS:
        raise ValueError('Invalid method "{}". Must be one of {}'.format(method,
                                                                         '{"' + '", "'.join(SOLVERS.keys()) + '"}.'))
    if _is_tuple(y0):
        device = y0[0].device
    else:
        device = y0.device
    # check t is a float tensor, if not  convert it to one
    if not isinstance(t, torch.Tensor):
            # check t is a float tensor, if not  convert it to one
            t = torch.tensor(t, dtype=torch.float32, device=device)
        # print("t converted to tensor")
    else:
        t = t.to(device)

    # check t is > 0 else raise error
    if not (t > 0).all():
        raise ValueError("t must be > 0")
    # ~Backward compatibility

    # # Add perturb argument to func.
    # func = _PerturbFunc(func)

    # check beta is a float tensor, if not  convert it to one
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=torch.float32, device=device)
        # print("beta converted to tensor")
    else:
        beta = beta.to(device)
    # check beta is > 0 else raise error
    if not (beta > 0).all():
        raise ValueError("beta must be > 0")
    # check beta is <= 1 else raise warning
    if not (beta <= 1).all():
        warnings.warn("beta should be <= 1 for the initial value problem")

    # check stepsize is a float tensor, if not  convert it to one
    if not isinstance(step_size, torch.Tensor):
        step_size = torch.tensor(step_size, dtype=torch.float32, device=device)
        # print("step_size converted to tensor")
    else:
        step_size = step_size.to(device)
    # check step_size is > 0 else raise error
    if not (step_size > 0).all():
        raise ValueError("step_size must be > 0")

    # check step_size is <= t else raise error
    if not (step_size < t).all():
        raise ValueError("step_size must be < t")

    # print(t,step_size)
    # tspan = torch.arange(0,t,step_size)

    num_steps = int((t - 0) / step_size) + 1  # plus one to include 't' itself
    # Generate tspan
    tspan = torch.linspace(0, t, num_steps)

    # Initialize tracking variables
    shapes = None
    tensor_input = False

    # Handle different tensor modes
    if config.TENSOR_MODE == 'concat':
        # CONCAT MODE: Flatten and concatenate tuple elements into a single tensor
        is_tuple = not isinstance(y0, torch.Tensor)
        if is_tuple:
            # Case 1: y0 is tuple - flatten each element and concatenate
            assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
            shapes = [y0_.shape for y0_ in y0]
            y0 = torch.cat([y0_.reshape(-1) for y0_ in y0])
            func = _TupleFunc(func, shapes)
        else:
            # Case 2: y0 is already a tensor - just mark it
            assert isinstance(y0, torch.Tensor), 'y0 must be either a torch.Tensor or a tuple'
            tensor_input = True
        assert torch.is_tensor(y0), 'should be a (concatenate) tensor'
    else:
        # NON-CONCAT MODE: Keep original structure, just check if tensor
        if torch.is_tensor(y0):
            tensor_input = True

    # Convert single tensors to tuple format for unified processing
    if torch.is_tensor(y0):
        y0 = (y0, )
        func = _Tensor2TupleFunc(func) # Wrap func to handle tensor-to-tuple conversion

    # Final validation: ensure y0 is a tuple of tensors
    assert isinstance(y0, tuple), 'y0 must be a tuple'

    for y0_ in y0:
        assert torch.is_tensor(y0_), 'each element must be a torch.Tensor but received {}'.format(type(y0_))

    return shapes, tensor_input, func, y0, tspan, method, beta


def _check_timelike(name, timelike, can_grad):
    assert isinstance(timelike, torch.Tensor), '{} must be a torch.Tensor'.format(name)
    _assert_floating(name, timelike)
    assert timelike.ndimension() == 1, "{} must be one dimensional".format(name)
    if not can_grad:
        assert not timelike.requires_grad, "{} cannot require gradient".format(name)
    # diff = timelike[1:] > timelike[:-1]
    # assert diff.all() or (~diff).all(), '{} must be strictly increasing or decreasing'.format(name)


def _assert_floating(name, t):
    if not torch.is_floating_point(t):
        raise TypeError('`{}` must be a floating point Tensor but is a {}'.format(name, t.type()))
class _ReverseFunc(torch.nn.Module):
    def __init__(self, base_func, mul=1.0):
        super(_ReverseFunc, self).__init__()
        self.base_func = base_func
        self.mul = mul

    def forward(self, t, y):
        return self.mul * self.base_func(-t, y)

def _assert_increasing(name, t):
    assert (t[1:] > t[:-1]).all(), '{} must be strictly increasing or decreasing'.format(name)


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def _flatten_convert_none_to_zeros(sequence, like_sequence):
    flat = [
        p.contiguous().view(-1) if p is not None else torch.zeros_like(q).view(-1)
        for p, q in zip(sequence, like_sequence)
    ]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])

def _is_tuple(x):
    return isinstance(x, tuple)


def _tuple_map(fn, *args):
    """Apply fn to each tuple element."""
    if _is_tuple(args[0]):
        return tuple(fn(*[arg[i] for arg in args]) for i in range(len(args[0])))
    return fn(*args)


def _add(a, b):
    if _is_tuple(a) and _is_tuple(b):
        return tuple(a_i + b_i for a_i, b_i in zip(a, b))
    return a + b

def _minus(a, b):
    if _is_tuple(a) and _is_tuple(b):
        return tuple(a_i - b_i for a_i, b_i in zip(a, b))
    return a - b

def _multiply(a, b):
    if _is_tuple(b):
        return tuple(a * b_i for b_i in b)
    return a * b


def _clone(y):
    if _is_tuple(y):
        return tuple(y_i.clone() for y_i in y)
    return y.clone()

class ReversedListView:
    def __init__(self, original_list):
        self.original = original_list

    def __getitem__(self, i):
        return self.original[-1 - i]

    def __len__(self):
        return len(self.original)


def _flat_to_shape(tensor, length, shapes):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + shape.numel()
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_list.append(tensor[..., total:next_total].view((*length, *shape)))
        total = next_total
    return tuple(tensor_list)

class _TupleFunc(torch.nn.Module):
    def __init__(self, base_func, shapes):
        super(_TupleFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def forward(self, t, y):
        f = self.base_func(t, _flat_to_shape(y, (), self.shapes))
        return torch.cat([f_.reshape(-1) for f_ in f])

class _Tensor2TupleFunc(torch.nn.Module):
    # func = lambda t, y: (_base_nontuple_func_(t, y[0]),)
    def __init__(self, base_func):
        super(_Tensor2TupleFunc, self).__init__()
        self.base_func = base_func

    def forward(self, t, y):
        assert len(y) == 1, 'should be a length one tuple'
        f = self.base_func(t, y[0])
        return (f, )


def _check_inputs_tensorinput(func, y0, t, step_size, method, beta, SOLVERS):
    if method not in SOLVERS:
        raise ValueError('Invalid method "{}". Must be one of {}'.format(method,
                                                                         '{"' + '", "'.join(SOLVERS.keys()) + '"}.'))

    # check t is a float tensor, if not  convert it to one
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32, device=y0.device)
        # print("t converted to tensor")
    else:
        t = t.to(y0.device)
    # check t is > 0 else raise error
    if not (t > 0).all():
        raise ValueError("t must be > 0")
    # ~Backward compatibility

    # # Add perturb argument to func.
    # func = _PerturbFunc(func)

    # check beta is a float tensor, if not  convert it to one
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=torch.float32, device=y0.device)
        # print("beta converted to tensor")
    else:
        beta = beta.to(y0.device)
    # check beta is > 0 else raise error
    if not (beta > 0).all():
        raise ValueError("beta must be > 0")
    # check beta is <= 1 else raise warning
    if not (beta <= 1).all():
        warnings.warn("beta should be <= 1 for the initial value problem")

    # check stepsize is a float tensor, if not  convert it to one
    if not isinstance(step_size, torch.Tensor):
        step_size = torch.tensor(step_size, dtype=torch.float32, device=y0.device)
        # print("step_size converted to tensor")
    else:
        step_size = step_size.to(y0.device)
    # check step_size is > 0 else raise error
    if not (step_size > 0).all():
        raise ValueError("step_size must be > 0")

    # check step_size is <= t else raise error
    if not (step_size < t).all():
        raise ValueError("step_size must be < t")

    # print(t,step_size)
    # tspan = torch.arange(0,t,step_size)

    num_steps = int((t - 0) / step_size) + 1  # plus one to include 't' itself
    # Generate tspan
    tspan = torch.linspace(0, t, num_steps)

    return func, y0, tspan, method, beta