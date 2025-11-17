from .utils_fde import _check_inputs, _flat_to_shape
from .caputo_solver import predictor,l1solver, predictor_corrector
from .riemann_liouville_solver import glmethod, product_trap
from . import config
import torch

SOLVERS = {"predictor":predictor,
          "corrector":predictor_corrector,
           "l1":l1solver,
           "gl":glmethod,
           "trap":product_trap
}

def fdeint(func,y0,beta,t,step_size,method,options=None):
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
          t: float torch tensor,the integrate terminate time. the default initial time point is set as 0.
          step_size: float torch tensor, the step size of the integrate method.
          method: optional string indicating the integration method to use.
          options: optional dict of configuring options for the indicated integration
              method. Can only be provided if a `method` is explicitly set.
      Returns:
          y: Tensor, where the first dimension corresponds to different
              time points. Contains the solved value of y for each desired time point in
              `t`, with the initial value `y0` being the first element along the first
              dimension.

      Raises:
          ValueError: if an invalid `method` is provided.
      """
    shapes, tensor_input, func, y0, tspan, method, beta = _check_inputs(func, y0, t, step_size, method, beta, SOLVERS)
    if options is None:
        options = {}
    solution = SOLVERS[method](func=func, y0=y0, beta=beta, tspan=tspan, **options)

    # Post-process solution based on tensor mode and input type
    if config.TENSOR_MODE == 'concat':
        # CONCAT MODE: Solution needs to be reshaped back to original structure
        if not tensor_input:
            # Case 1: Original input was tuple - reshape flattened solution back to tuple of original shapes
             #asset shapes is not None
            assert shapes is not None, 'for tuple, we need to provide shapes'
            solution = solution[0]
            solution = _flat_to_shape(solution, (), shapes)
        else:
            # Case 2: Original input was tensor - just extract the solution
            solution = solution[0]
    else:
        # NON-CONCAT MODE: Only extract solution if original input was a single tensor
        if tensor_input:
            solution = solution[0]
        # (If input was tuple, solution remains as-is)

    # Validate output type matches input type
    if tensor_input:
        assert torch.is_tensor(solution)
    else:
        assert isinstance(solution, tuple)

    return solution
