import torch
import math
from functools import partial
from typing import Callable, Any, Dict, Optional, Tuple
from .utils_fde import _flatten, _flatten_convert_none_to_zeros, _check_inputs, _flat_to_shape
from .utils_fde import _addmul_inplace, _mul_inplace, _minusmul_inplace
from .utils_fde import _is_tuple, _clone, _add, _multiply, _minus, ReversedListView


class FractionalSolverConfig:
    """Configuration for fractional differential equation solvers"""

    def __init__(self, beta: float, tspan: torch.Tensor, y0: Any, options: Dict):
        self.N = len(tspan)
        self.h = (tspan[-1] - tspan[0]) / (self.N - 1)
        self.beta = beta
        self.one_minus_beta = 1 - beta
        self.tspan = tspan
        self.y0 = y0
        self.device = y0[0].device if _is_tuple(y0) else y0.device
        self.options = options

        # Common precomputed values
        self.h_beta = torch.pow(self.h, beta)
        self.gamma_beta = 1 / math.gamma(beta)
        self.gamma_2_minus_beta = math.gamma(2 - beta)
        self.h_alpha_gamma = self.h_beta * self.gamma_2_minus_beta


def get_memory_range(k: int, options: Dict) -> Tuple[int, int]:
    """Compute memory range for index k"""
    if 'memory' not in options or options['memory'] == -1:
        memory_length = k + 1
    else:
        memory_length = min(options['memory'], k + 1)
        assert memory_length > 0, "memory must be greater than 0"

    start_idx = max(0, k + 1 - memory_length)
    return start_idx, memory_length


def compute_convolution(start_idx: int, end_idx: int, weights: torch.Tensor,
                        history: list, y0: Any) -> Any:
    """Compute weighted sum over history"""
    convolution_sum = None
    for j in range(start_idx, end_idx + 1):
        local_idx = j - start_idx
        term = _multiply(weights[local_idx], history[j])
        if convolution_sum is None:
            convolution_sum = term
        else:
            convolution_sum = _add(convolution_sum, term)
    return convolution_sum


# Weight computation strategies
def adams_bashforth_weights(k: int, start_idx: int, config: FractionalSolverConfig) -> torch.Tensor:
    """Compute Adams-Bashforth weights"""
    j_vals = torch.arange(start_idx, k + 1, dtype=torch.float32, device=config.device).unsqueeze(1)
    h_beta_over_beta = config.h_beta / config.beta
    return h_beta_over_beta * (torch.pow(k + 1 - j_vals, config.beta) -
                               torch.pow(k - j_vals, config.beta))


def l1_weights(k: int, start_idx: int, config: FractionalSolverConfig) -> torch.Tensor:
    """Compute L1 method weights"""
    j_vals = torch.arange(start_idx, k + 1, dtype=torch.float32, device=config.device)

    kjp2 = torch.pow(k + 2 - j_vals, config.one_minus_beta)
    kjp1 = torch.pow(k + 1 - j_vals, config.one_minus_beta)
    kj = torch.pow(k - j_vals, config.one_minus_beta)
    c_j_k = kjp2 - 2 * kjp1 + kj

    if start_idx == 0:
        c_j_k[0] = -(torch.pow(torch.tensor(k + 1, dtype=torch.float32, device=config.device),
                               config.one_minus_beta) -
                     torch.pow(torch.tensor(k, dtype=torch.float32, device=config.device),
                               config.one_minus_beta))
    return c_j_k


def grunwald_letnikov_weights(k: int, start_idx: int, config: FractionalSolverConfig,
                              c: torch.Tensor) -> torch.Tensor:
    """Compute Grünwald-Letnikov weights"""
    weights = torch.zeros(k + 1 - start_idx, dtype=torch.float64, device=config.device)
    for j in range(start_idx, k + 1):
        weights[j - start_idx] = c[k + 1 - j]
    return weights


def product_trap_weights(k: int, start_idx: int, config: FractionalSolverConfig) -> torch.Tensor:
    """Compute Product Trapezoidal weights"""
    j_vals = torch.arange(start_idx, k + 1, dtype=torch.float32, device=config.device)

    kjp2 = torch.pow(k + 2 - j_vals, config.one_minus_beta)
    kj = torch.pow(k - j_vals, config.one_minus_beta)
    kjp1 = torch.pow(k + 1 - j_vals, config.one_minus_beta)

    A_j_kp1 = kjp2 + kj - 2 * kjp1

    if start_idx == 0:
        k_power = torch.pow(torch.tensor(k, dtype=torch.float32, device=config.device),
                            config.one_minus_beta)
        kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=torch.float32, device=config.device),
                                  -config.beta)
        A_j_kp1[0] = k_power - (k + config.beta) * kp1_neg_alpha

    return A_j_kp1


# F-term computation strategies
def adams_f_term(f_k: Any, config: FractionalSolverConfig) -> Any:
    """Adams-Bashforth doesn't use f_term in its update"""
    return None  # Not used in adams_update


def l1_f_term(f_k: Any, config: FractionalSolverConfig) -> Any:
    """L1 method f-term: h^α * Γ(2-α) * f(t_k, y_k)"""
    return _multiply(config.h_alpha_gamma, f_k)


def gl_f_term(f_k: Any, config: FractionalSolverConfig) -> Any:
    """Grünwald-Letnikov f-term: h^α * f(t_k, y_k)"""
    return _multiply(config.h_beta, f_k)


def product_trap_f_term(f_k: Any, config: FractionalSolverConfig) -> Any:
    """Product Trapezoidal f-term: same as L1"""
    return _multiply(config.h_alpha_gamma, f_k)


# Update strategies
def adams_update(f_term: Any, conv_sum: Any, config: FractionalSolverConfig) -> Any:
    """Adams-Bashforth update step"""
    return _add(config.y0, _multiply(config.gamma_beta, conv_sum))


def l1_update(f_term: Any, conv_sum: Any, config: FractionalSolverConfig) -> Any:
    """L1 method update step"""
    return _add(f_term, _multiply(-1, conv_sum))


def gl_update(f_term: Any, conv_sum: Any, config: FractionalSolverConfig) -> Any:
    """Grünwald-Letnikov update step"""
    return _minus(f_term, conv_sum)


# Generic solver framework
def fractional_solver(func: Callable,
                      y0: Any,
                      beta: float,
                      tspan: torch.Tensor,
                      weight_fn: Callable,
                      update_fn: Callable,
                      f_term_fn: Callable,
                      store_y_history: bool = False,
                      precompute_fn: Optional[Callable] = None,
                      **options) -> Any:
    """
    Generic fractional differential equation solver framework

    Args:
        func: RHS function f(t, y)
        y0: Initial condition
        beta: Fractional order
        tspan: Time points
        weight_fn: Function to compute weights (k, start_idx, config, *args) -> weights
        update_fn: Function to compute update (f_term, conv_sum, config) -> y_next
        f_term_fn: Function to compute f-term (f_k, config) -> f_term
        store_y_history: Whether to store y values (True) or f values (False)
        precompute_fn: Optional function to precompute values needed by weight_fn
        **options: Additional options including memory limit
    """
    config = FractionalSolverConfig(beta, tspan, y0, options)

    y_current = _clone(y0)
    history = []

    # Run precomputation if needed
    precompute_args = precompute_fn(config) if precompute_fn else ()

    # Initial history setup
    if store_y_history:
        history.append(y0)

    for k in range(config.N - 1):
        t_k = config.tspan[k]
        f_k = func(t_k, y_current)

        # Store f(t_k, y_k) if using f-history
        if not store_y_history:
            history.append(f_k)

        # Get memory range
        start_idx, _ = get_memory_range(k, options)

        # Compute weights
        if precompute_args:
            weights = weight_fn(k, start_idx, config, *precompute_args)
        else:
            weights = weight_fn(k, start_idx, config)

        # Compute convolution
        conv_sum = compute_convolution(start_idx, k, weights, history, y0)

        # Compute f-term using method-specific function
        f_term = f_term_fn(f_k, config)

        # Update y
        y_current = update_fn(f_term, conv_sum, config)

        # Store y_{k+1} if using y-history
        if store_y_history:
            history.append(y_current)

    del history
    return y_current


# Precompute function for Grünwald-Letnikov
def gl_precompute(config: FractionalSolverConfig) -> Tuple[torch.Tensor]:
    """Precompute GL coefficients"""
    c = torch.zeros(config.N + 1, dtype=torch.float64, device=config.device)
    c[0] = 1
    for j in range(1, config.N + 1):
        c[j] = (1 - (1 + config.beta) / j) * c[j - 1]
    return (c,)


# Public API functions using the generic solver
def predictor(func, y0, beta, tspan, **options):
    """Use one-step Adams-Bashforth (Euler) method to integrate Caputo equation
        D^beta y(t) = f(t,y)
        Args:
          beta: fractional exponent in the range (0,1)
          f: callable(y,t) returning a numpy array of shape (d,)
             Vector-valued function to define the right hand side of the system
          y0: N-D Tensor or tuple of Tensors giving the initial state vector y(t==0)
          tspan (array): The sequence of time points for which to solve for y.
            These must be equally spaced, e.g. np.arange(0,10,0.005)
            tspan[0] is the initial time corresponding to the initial state y0.
        Returns:
          y: Tensor or tuple of Tensors with the same structure as y0
             With the initial value y0 in the first row
        """
    return fractional_solver(
        func, y0, beta, tspan,
        weight_fn=adams_bashforth_weights,
        update_fn=adams_update,
        f_term_fn=adams_f_term,
        store_y_history=False,
        **options
    )


def l1solver(func, y0, beta, tspan, **options):
    """Use L1 method to integrate Caputo equation
        D^beta y(t) = f(t,y)
        Args:
          beta: fractional exponent in the range (0,1)
          func: callable(y,t) returning a numpy array of shape (d,)
             Vector-valued function to define the right hand side of the system
          y0: N-D Tensor or tuple of Tensors giving the initial state vector y(t==0)
          tspan (array): The sequence of time points for which to solve for y.
            These must be equally spaced, e.g. np.arange(0,10,0.005)
            tspan[0] is the initial time corresponding to the initial state y0.
        Returns:
          y: Tensor or tuple of Tensors with the same structure as y0
             With the initial value y0 in the first row
        """
    return fractional_solver(
        func, y0, beta, tspan,
        weight_fn=l1_weights,
        update_fn=l1_update,
        f_term_fn=l1_f_term,
        store_y_history=True,
        **options
    )


def glmethod(func, y0, beta, tspan, **options):
    """Use Grünwald-Letnikov method to integrate RL fractional equation
        D^beta y(t) = f(t,y)
        Args:
          beta: fractional exponent in the range (0,1)
          func: callable(y,t) returning a numpy array of shape (d,)
             Vector-valued function to define the right hand side of the system
          y0: N-D Tensor or tuple of Tensors giving the initial state vector y(t==0)
          tspan (array): The sequence of time points for which to solve for y.
            These must be equally spaced, e.g. np.arange(0,10,0.005)
            tspan[0] is the initial time corresponding to the initial state y0.
        Returns:
          y: Tensor or tuple of Tensors with the same structure as y0
             With the initial value y0 in the first row
        """
    return fractional_solver(
        func, y0, beta, tspan,
        weight_fn=grunwald_letnikov_weights,
        update_fn=gl_update,
        f_term_fn=gl_f_term,
        store_y_history=True,
        precompute_fn=gl_precompute,
        **options
    )


def product_trap(func, y0, beta, tspan, **options):
    """Use Product Trapezoidal method to integrate fractional equation
       D^beta y(t) = f(t,y)
        Args:
          beta: fractional exponent in the range (0,1)
          func: callable(y,t) returning a numpy array of shape (d,)
             Vector-valued function to define the right hand side of the system
          y0: N-D Tensor or tuple of Tensors giving the initial state vector y(t==0)
          tspan (array): The sequence of time points for which to solve for y.
            These must be equally spaced, e.g. np.arange(0,10,0.005)
            tspan[0] is the initial time corresponding to the initial state y0.
        Returns:
          y: Tensor or tuple of Tensors with the same structure as y0
             With the initial value y0 in the first row
        """
    return fractional_solver(
        func, y0, beta, tspan,
        weight_fn=product_trap_weights,
        update_fn=l1_update,  # Same update as L1
        f_term_fn=product_trap_f_term,  # Same f-term as L1
        store_y_history=True,
        **options
    )


