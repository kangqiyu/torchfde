from .utils_fde import _check_inputs, _flat_to_shape
from .caputo_solver import predictor_corrector
from .explicit_solver import predictor, l1solver, glmethod, product_trap, glmethod_multiterm
from . import config
import torch
from typing import List, Any

# from .utils_fde import _check_inputs, _flat_to_shape
# from .caputo_solver import predictor,l1solver, predictor_corrector
# from .riemann_liouville_solver import glmethod, product_trap, glmethod_multiterm
# from . import config
# import torch


SOLVERS = {"predictor":predictor,
          "corrector":predictor_corrector,
           "l1":l1solver,
           "gl":glmethod,
           "trap":product_trap,
           "glmulti":glmethod_multiterm,
}

def fdeint(func, y0, beta, t, step_size, method, options=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
```
        D^(\beta)_t = func(t, y), y(t[0]) = y0
```
    where y is a Tensor or tuple of Tensors of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Function that maps a scalar Tensor `t` and a Tensor holding the state `y`
            into a Tensor of state derivatives with respect to time. Optionally, `y`
            can also be a tuple of Tensors.
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. Optionally, `y0`
            can also be a tuple of Tensors.
        t: float torch tensor, the integrate terminate time. The default initial time point is set as 0.
        step_size: float torch tensor, the step size of the integrate method.
        method: optional string indicating the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.
            Supported options:
                - memory: int, maximum history length (-1 for full memory)
                - return_history: bool, if True return full trajectory instead of final value
                - multi_coefficient: Tensor, weights for multi-term methods

    Returns:
        If return_history=False (default):
            y: Tensor or tuple of Tensors (with the same shape as y0) at time tspan[-1]
        If return_history=True:
            y: Full trajectory at all time points
               - If y0 is a Tensor: shape (N, *y0.shape)
               - If y0 is a tuple: tuple of Tensors, each with shape (N, *y0_i.shape)
               where N is the number of time points.

    Raises:
        ValueError: if an invalid `method` is provided.
    """
    shapes, tensor_input, func, y0, tspan, method, beta = _check_inputs(
        func, y0, t, step_size, method, beta, SOLVERS
    )
    if options is None:
        options = {}

    solution = SOLVERS[method](func=func, y0=y0, beta=beta, tspan=tspan, **options)

    return_history = options.get('return_history', False)

    if return_history:
        # solution is a list of tuples: [y_0, y_1, ..., y_N-1]
        # Each y_k is a tuple of tensors (after _check_inputs processing)

        if config.TENSOR_MODE == 'concat' and not tensor_input:
            # CONCAT MODE with tuple input: reshape each element back to original structure
            assert shapes is not None, 'for tuple, we need to provide shapes'
            solution = [_flat_to_shape(y[0], (), shapes) for y in solution]

        # Stack the trajectory (handles both tensor and tuple cases)
        solution = _stack_trajectory(solution, tensor_input)

    else:
        # return_history=False: solution is final value only (tuple of tensors)
        if config.TENSOR_MODE == 'concat':
            # CONCAT MODE
            if not tensor_input:
                # Original input was tuple - reshape back
                assert shapes is not None, 'for tuple, we need to provide shapes'
                solution = solution[0]
                solution = _flat_to_shape(solution, (), shapes)
            else:
                # Original input was tensor
                solution = solution[0]
        else:
            # LOOP MODE
            if tensor_input:
                solution = solution[0]

    # Validate output type matches input type
    if tensor_input:
        assert torch.is_tensor(solution)
    else:
        assert isinstance(solution, tuple)

    return solution


def _stack_trajectory(trajectory: List[Any], tensor_input: bool) -> Any:
    """
    Stack a list of y values into a single tensor or tuple of tensors.

    Args:
        trajectory: List of y values (each is a tuple of Tensors after _check_inputs)
        tensor_input: Whether the original input was a tensor (vs tuple)

    Returns:
        If tensor_input=True: stacked Tensor of shape (N, *y.shape)
        If tensor_input=False: tuple of stacked Tensors, each of shape (N, *y_i.shape)
    """
    # After _check_inputs, trajectory elements are always tuples
    n_components = len(trajectory[0])

    if tensor_input:
        # Original was single tensor wrapped as (tensor,)
        # Stack and return single tensor
        assert n_components == 1, "tensor_input expects single component"
        return torch.stack([y[0] for y in trajectory], dim=0)
    else:
        # Original was tuple - stack each component
        return tuple(
            torch.stack([y[i] for y in trajectory], dim=0)
            for i in range(n_components)
        )