"""
Unified Fractional Differential Equation Solvers for

Caputo formulation:
predictor
l1solver

Riemann-Liouville formulation:
glmethod
glmethod_multiterm
product_trap

All solvers follow the same mathematical pattern:
    y_{k+1} = initial_term + scale * Σ w_j * history[j]

where the differences lie in:
    - What we store in history (y values or f values)
    - How we compute weights w_j
    - The initial term and scale factor
"""

import torch
import math
from functools import partial
from typing import Callable, Any, Dict, Optional, Tuple, Union, List
from .utils_fde import _flatten, _flatten_convert_none_to_zeros, _check_inputs, _flat_to_shape
from .utils_fde import _addmul_inplace, _mul_inplace, _minusmul_inplace
from .utils_fde import _is_tuple, _clone, _add, _multiply, _minus, ReversedListView
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SolverConfig:
    """Immutable configuration computed once at solver initialization."""
    N: int                      # Number of time points
    h: torch.Tensor            # Step size
    beta: Any                   # Fractional order: float for single-term, Tensor for multi-term
    device: torch.device
    dtype: torch.dtype

    # Precomputed constants for single-term solvers (None for multi-term)
    h_beta: Optional[torch.Tensor] = None
    gamma_beta: Optional[float] = None
    gamma_2_minus_beta: Optional[float] = None
    h_beta_over_beta_gamma: Optional[torch.Tensor] = None
    h_beta_gamma_2_minus_beta: Optional[torch.Tensor] = None

    @classmethod
    def from_tspan(
        cls,
        y0: Any,
        beta: Any,
        tspan: torch.Tensor,
    ) -> 'SolverConfig':
        """Create configuration from time span and initial condition."""
        device = y0[0].device if _is_tuple(y0) else y0.device
        dtype = y0[0].dtype if _is_tuple(y0) else y0.dtype

        N = len(tspan)
        h = (tspan[-1] - tspan[0]) / (N - 1)

        # Check if beta is multi-term (tensor with multiple elements)
        is_multiterm = isinstance(beta, torch.Tensor) and beta.numel() > 1

        if is_multiterm:
            # Multi-term case: don't precompute single-term constants
            return cls(
                N=N,
                h=h,
                beta=beta,
                device=device,
                dtype=dtype,
            )
        else:
            # Single-term case: extract scalar value and precompute constants
            beta_val = beta.item() if isinstance(beta, torch.Tensor) else beta
            h_beta = torch.pow(h, beta_val)
            gamma_beta = math.gamma(beta_val)
            gamma_2_minus_beta = math.gamma(2 - beta_val)

            return cls(
                N=N,
                h=h,
                beta=beta_val,
                device=device,
                dtype=dtype,
                h_beta=h_beta,
                gamma_beta=gamma_beta,
                gamma_2_minus_beta=gamma_2_minus_beta,
                h_beta_over_beta_gamma=h_beta / (beta_val * gamma_beta),
                h_beta_gamma_2_minus_beta=h_beta * gamma_2_minus_beta,
            )


def get_memory_bounds(k: int, memory: int) -> Tuple[int, int]:
    """
    Compute the range of history indices to use.

    Args:
        k: Current time step index
        memory: Maximum history length (-1 for unlimited)

    Returns:
        (start_idx, end_idx) inclusive bounds for history access
    """
    if memory == -1:
        return 0, k
    memory_length = min(memory, k + 1)
    start_idx = max(0, k + 1 - memory_length)
    return start_idx, k


class FractionalMethod(ABC):
    """
    Abstract base class for fractional differential equation methods.

    Each method must define:
        - Whether it stores y-history or f-history
        - How to compute weights
        - How to combine terms into the update
    """

    @property
    @abstractmethod
    def stores_y_history(self) -> bool:
        """True if method stores y values, False if it stores f values."""
        pass

    @abstractmethod
    def compute_weights(self, k: int, start_idx: int, config: SolverConfig) -> torch.Tensor:
        """
        Compute weights for indices [start_idx, k].

        Returns tensor of shape (k - start_idx + 1,) or (k - start_idx + 1, 1)
        """
        pass

    @abstractmethod
    def compute_update(
        self,
        y0: Any,
        f_k: Any,
        convolution: Any,
        config: SolverConfig
    ) -> Any:
        """
        Compute y_{k+1} from the convolution sum and current f value.
        """
        pass

    def initialize(self, config: SolverConfig) -> None:
        """Optional hook for method-specific precomputation."""
        pass


class AdamsBashforth(FractionalMethod):
    """
    One-step Adams-Bashforth (Euler) predictor for Caputo equations.

    Reference: "Detailed error analysis for a fractional Adams method"
               Diethelm et al., Numerical Algorithms 36(1), 2004

    Update formula:
        y_{k+1} = y_0 + (h^β / (β·Γ(β))) * Σ b_j * f(t_j, y_j)

    where b_j = (k+1-j)^β - (k-j)^β
    """

    @property
    def stores_y_history(self) -> bool:
        return False  # Stores f-values

    def compute_weights(self, k: int, start_idx: int, config: SolverConfig) -> torch.Tensor:
        j_vals = torch.arange(start_idx, k + 1, dtype=config.dtype, device=config.device)
        # b_j = (k+1-j)^β - (k-j)^β
        weights = (
            torch.pow(k + 1 - j_vals, config.beta) -
            torch.pow(k - j_vals, config.beta)
        )
        return config.h_beta_over_beta_gamma * weights

    def compute_update(self, y0: Any, f_k: Any, convolution: Any, config: SolverConfig) -> Any:
        # y_{k+1} = y_0 + convolution (weights already include h^β/(β·Γ(β)))
        return _add(y0, convolution)


class L1Method(FractionalMethod):
    """
    L1 scheme for Caputo equations.

    Reference: "A compact finite difference scheme for fractional sub-diffusion equations"
               Gao & Sun, J. Comp. Physics 230(3), 2011

    Update formula:
        y_{k+1} = h^β·Γ(2-β)·f(t_k,y_k) - Σ c_j^(k) * y_j

    where c_j^(k) has special formula for j=0.
    """

    @property
    def stores_y_history(self) -> bool:
        return True

    def compute_weights(self, k: int, start_idx: int, config: SolverConfig) -> torch.Tensor:
        one_minus_beta = 1 - config.beta
        j_vals = torch.arange(start_idx, k + 1, dtype=config.dtype, device=config.device)

        # General formula: c_j = (k+2-j)^{1-β} - 2(k+1-j)^{1-β} + (k-j)^{1-β}
        kjp2 = torch.pow(k + 2 - j_vals, one_minus_beta)
        kjp1 = torch.pow(k + 1 - j_vals, one_minus_beta)
        kj = torch.pow(k - j_vals, one_minus_beta)
        weights = kjp2 - 2 * kjp1 + kj

        # Special case for j=0: c_0 = -[(k+1)^{1-β} - k^{1-β}]
        if start_idx == 0:
            weights[0] = -(
                torch.pow(torch.tensor(k + 1, dtype=config.dtype, device=config.device), one_minus_beta) -
                torch.pow(torch.tensor(k, dtype=config.dtype, device=config.device), one_minus_beta)
            )

        return weights

    def compute_update(self, y0: Any, f_k: Any, convolution: Any, config: SolverConfig) -> Any:
        # y_{k+1} = h^β·Γ(2-β)·f_k - convolution
        f_term = _multiply(config.h_beta_gamma_2_minus_beta, f_k)
        return _minus(f_term, convolution)


class GrunwaldLetnikov(FractionalMethod):
    """
    Grünwald-Letnikov method for Riemann-Liouville equations.

    Reference: "Determining chaotic behavior in fractional-order finance system"
               Xu et al., Nonlinear Dynamics 94(2), 2018

    Update formula:
        y_{k+1} = h^β·f(t_k,y_k) - Σ c_{k+1-j} * y_j

    where c_j = (1 - (1+β)/j) * c_{j-1}, c_0 = 1
    """

    def __init__(self):
        self._c: Optional[torch.Tensor] = None

    @property
    def stores_y_history(self) -> bool:
        return True

    def initialize(self, config: SolverConfig) -> None:
        """Precompute GL coefficients."""
        c = torch.zeros(config.N + 1, dtype=config.dtype, device=config.device)
        c[0] = 1
        for j in range(1, config.N + 1):
            c[j] = (1 - (1 + config.beta) / j) * c[j - 1]
        self._c = c

    def compute_weights(self, k: int, start_idx: int, config: SolverConfig) -> torch.Tensor:
        # Weight for y_j is c_{k+1-j}
        indices = torch.arange(k + 1 - start_idx, 0, -1, dtype=torch.long, device=config.device)
        return self._c[indices]

    def compute_update(self, y0: Any, f_k: Any, convolution: Any, config: SolverConfig) -> Any:
        # y_{k+1} = h^β·f_k - convolution
        f_term = _multiply(config.h_beta, f_k)
        return _minus(f_term, convolution)


class GrunwaldLetnikovMultiterm(FractionalMethod):
    """
    Grünwald-Letnikov method for multi-term Riemann-Liouville equations.

    Solves: Σ w_j D^{β_j} y(t) = f(t,y)

    Uses aggregated weights: c̃_m = Σ w_j * h^{-β_j} * c_m^{(β_j)}

    Reference: Based on distributed GL scheme where aggregated weights
               encode the multi-term structure.

    Update formula:
        y_{k+1} = (1/c̃_0) * (f(t_k,y_k) - Σ c̃_{k+1-j} * y_j)
    """

    def __init__(self, coefficient: torch.Tensor):
        """
        Args:
            coefficient: Tensor of weights w_j for each fractional term.
                        Must have the same length as beta tensor.
                        Already converted to correct device/dtype by caller.
        """
        self._coefficient = coefficient
        self._c_tilde: Optional[torch.Tensor] = None
        self._inv_c_tilde_0: Optional[torch.Tensor] = None

    @property
    def stores_y_history(self) -> bool:
        return True

    def initialize(self, config: SolverConfig) -> None:
        """Precompute aggregated GL coefficients for all terms."""
        beta = config.beta
        coefficient = self._coefficient

        # beta is already a tensor from glmethod_multiterm
        # coefficient is already a tensor with correct device/dtype from glmethod_multiterm
        n_terms = len(beta)

        # Compute GL coefficients c[j, m] = c_m^{(β_j)} for each fractional order
        # c_0 = 1, c_m = (1 - (1+β)/m) * c_{m-1}
        c = torch.zeros(n_terms, config.N + 1, dtype=config.dtype, device=config.device)
        c[:, 0] = 1.0
        for m in range(1, config.N + 1):
            # Vectorized over all terms: c[j,m] = (1 - (1+β_j)/m) * c[j,m-1]
            c[:, m] = (1 - (1 + beta) / m) * c[:, m - 1]

        # Compute h^{-β_j} for each term
        h_neg_power = torch.pow(config.h, -beta)  # Shape: (n_terms,)

        # Compute aggregated weights: c̃_m = Σ w_j * h^{-β_j} * c_m^{(β_j)}
        weighted_h = coefficient * h_neg_power  # Shape: (n_terms,)
        self._c_tilde = torch.sum(weighted_h.unsqueeze(1) * c, dim=0)  # Shape: (N+1,)

        # Store inverse of c̃_0 for scaling in update
        self._inv_c_tilde_0 = 1.0 / self._c_tilde[0]

    def compute_weights(self, k: int, start_idx: int, config: SolverConfig) -> torch.Tensor:
        # Weight for y_j is c̃_{k+1-j}
        # For j from start_idx to k: indices are k+1-start_idx down to 1
        indices = torch.arange(k + 1 - start_idx, 0, -1, dtype=torch.long, device=config.device)
        return self._c_tilde[indices]

    def compute_update(self, y0: Any, f_k: Any, convolution: Any, config: SolverConfig) -> Any:
        # y_{k+1} = (1/c̃_0) * (f_k - convolution)
        diff = _minus(f_k, convolution)
        return _multiply(self._inv_c_tilde_0, diff)


class ProductTrapezoidal(FractionalMethod):
    """
    Product Trapezoidal method for fractional equations.

    Reference: Baleanu, Diethelm, Scalas, Trujillo
               "Fractional Calculus: Models and Numerical Methods"
               World Scientific, 2012

    Update formula:
        y_{k+1} = h^β·Γ(2-β)·f(t_k,y_k) - Σ A_{j,k+1} * y_j

    where A_{j,k+1} has special formula for j=0.
    """

    @property
    def stores_y_history(self) -> bool:
        return True

    def compute_weights(self, k: int, start_idx: int, config: SolverConfig) -> torch.Tensor:
        one_minus_beta = 1 - config.beta
        j_vals = torch.arange(start_idx, k + 1, dtype=config.dtype, device=config.device)

        # General formula: A_{j,k+1} = (k+2-j)^{1-β} + (k-j)^{1-β} - 2(k+1-j)^{1-β}
        kjp2 = torch.pow(k + 2 - j_vals, one_minus_beta)
        kj = torch.pow(k - j_vals, one_minus_beta)
        kjp1 = torch.pow(k + 1 - j_vals, one_minus_beta)
        weights = kjp2 + kj - 2 * kjp1

        # Special case for j=0: A_{0,k+1} = k^{1-β} - (k+β)(k+1)^{-β}
        if start_idx == 0:
            k_power = torch.pow(
                torch.tensor(k, dtype=config.dtype, device=config.device),
                one_minus_beta
            )
            kp1_neg_beta = torch.pow(
                torch.tensor(k + 1, dtype=config.dtype, device=config.device),
                -config.beta
            )
            weights[0] = k_power - (k + config.beta) * kp1_neg_beta

        return weights

    def compute_update(self, y0: Any, f_k: Any, convolution: Any, config: SolverConfig) -> Any:
        # y_{k+1} = h^β·Γ(2-β)·f_k - convolution
        f_term = _multiply(config.h_beta_gamma_2_minus_beta, f_k)
        return _minus(f_term, convolution)


def compute_convolution(
    weights: torch.Tensor,
    history: list,
    start_idx: int
) -> Any:
    """
    Compute weighted sum: Σ weights[i] * history[start_idx + i]

    Handles both tensor and tuple-of-tensor cases.
    """
    result = None
    for i, j in enumerate(range(start_idx, start_idx + len(weights))):
        term = _multiply(weights[i], history[j])
        if result is None:
            result = term
        else:
            result = _add(result, term)
    return result


def _stack_trajectory(trajectory: List[Any]) -> Any:
    """
    Stack a list of y values into a single tensor or tuple of tensors.

    Args:
        trajectory: List of y values (each is a Tensor or tuple of Tensors)

    Returns:
        If y is a Tensor: stacked Tensor of shape (N, *y.shape)
        If y is a tuple: tuple of stacked Tensors, each of shape (N, *y_i.shape)
    """
    if _is_tuple(trajectory[0]):
        # Tuple case: stack each component separately
        n_components = len(trajectory[0])
        return tuple(
            torch.stack([y[i] for y in trajectory], dim=0)
            for i in range(n_components)
        )
    else:
        # Single tensor case
        return torch.stack(trajectory, dim=0)


def solve(
    func: Callable[[torch.Tensor, Any], Any],
    y0: Any,
    beta: Any,
    tspan: torch.Tensor,
    method: FractionalMethod,
    memory: int = -1,
    return_history: bool = False,
) -> Any:
    """
    Solve a fractional differential equation D^β y(t) = f(t, y).

    Args:
        func: Right-hand side function f(t, y)
        y0: Initial condition y(0)
        beta: Fractional order in (0, 1), or tensor of orders for multi-term
        tspan: Equally-spaced time points starting at t=0
        method: FractionalMethod instance defining the numerical scheme
        memory: Maximum history length (-1 for full history)
        return_history: If True, return the full trajectory at all time points.
                       If False (default), return only the final value.

    Returns:
        If return_history=False:
            y: Solution at tspan[-1] (Tensor or tuple of Tensors)
        If return_history=True:
            y: Full trajectory at all time points
               - If y0 is a Tensor: shape (N, *y0.shape)
               - If y0 is a tuple: tuple of Tensors, each with shape (N, *y0_i.shape)
    """
    config = SolverConfig.from_tspan(y0, beta, tspan)
    method.initialize(config)

    y_current = _clone(y0)
    history = [y0] if method.stores_y_history else []

    # Track full trajectory if requested
    if return_history:
        trajectory = [y0,]  # Start with y0

    for k in range(config.N - 1):
        t_k = tspan[k]
        f_k = func(t_k, y_current)

        # Store f(t_k, y_k) if method uses f-history
        if not method.stores_y_history:
            history.append(f_k)

        # Compute memory bounds
        start_idx, end_idx = get_memory_bounds(k, memory)

        # Compute weights and convolution
        weights = method.compute_weights(k, start_idx, config)
        convolution = compute_convolution(weights, history, start_idx)

        # Update solution
        y_current = method.compute_update(y0, f_k, convolution, config)

        # Store y_{k+1} if method uses y-history
        if method.stores_y_history:
            history.append(y_current)

            # Track trajectory if requested
        if return_history:
            trajectory.append(y_current)

    if return_history:
        return trajectory  # Return raw list, let fdeint handle stacking
    else:
        return y_current

# ============================================================================
# Public API - Drop-in replacements for original functions
# ============================================================================

"""
    Args:
      func: returning f(t,y), with the same shape as y0
      y0: Tensor or tuple of Tensors, giving the initial state vector y(t==0)
      beta: fractional order in the range (0,1)
      tspan (array): The sequence of time points for which to solve for y.
        These must be equally spaced, e.g. np.arange(0,10,0.005)
        tspan[0] is the initial time corresponding to the initial state y0.
    Returns:
      y: Tensor or tuple of Tensors (with the same shape as y0) at time tspan[-1]
         If return_history=True, returns full trajectory with shape (N, *y0.shape)

"""
def predictor(func, y0, beta, tspan, **options):
    """
    Adams-Bashforth (Euler) predictor for Caputo equation D^β y(t) = f(t,y).

    Reference: Diethelm et al., "Detailed error analysis for a fractional
    Adams method", Numerical Algorithms 36(1), 2004.
    """
    memory = options.get('memory', -1)
    return_history = options.get('return_history', False)
    return solve(func, y0, beta, tspan, AdamsBashforth(),
                 memory=memory, return_history=return_history)


def l1solver(func, y0, beta, tspan, **options):
    """
    L1 method for Caputo equation D^β y(t) = f(t,y).

    Reference: Gao & Sun, "A compact finite difference scheme for fractional
    sub-diffusion equations", J. Comp. Physics 230(3), 2011.
    """
    memory = options.get('memory', -1)
    return_history = options.get('return_history', False)
    return solve(func, y0, beta, tspan, L1Method(),
                 memory=memory, return_history=return_history)


def glmethod(func, y0, beta, tspan, **options):
    """
    Grünwald-Letnikov method for Riemann-Liouville equation D^β y(t) = f(t,y).

    Reference: Xu et al., "Determining chaotic behavior in fractional-order
    finance system", Nonlinear Dynamics 94(2), 2018.
    """
    memory = options.get('memory', -1)
    return_history = options.get('return_history', False)
    return solve(func, y0, beta, tspan, GrunwaldLetnikov(),
                 memory=memory, return_history=return_history)


def glmethod_multiterm(func, y0, beta, tspan, **options):
    """
    Grünwald-Letnikov method for multi-term Riemann-Liouville equation
    Σ w_j D^{β_j} y(t) = f(t,y).

    Based on the distributed GL scheme where the aggregated weights encode
    the multi-term structure through: c̃_m = Σ w_j h^{-β_j} c_m^{(β_j)}

    Args:
        func: returning f(t,y), with the same shape as y0
        y0: Tensor or tuple of Tensors, giving the initial state vector y(t==0)
        beta: Tensor of fractional orders, each in the range (0,1), shape (n_terms,)
        tspan: The sequence of time points for which to solve for y.
            These must be equally spaced, e.g. np.arange(0,10,0.005)
            tspan[0] is the initial time corresponding to the initial state y0.
        **options:
            multi_coefficient: Tensor of weights w_j for each term, shape (n_terms,).
                              Optional. If not provided, defaults to all ones.
            memory: int, optional. If specified, use short-memory approximation
                    with the given memory length M. Default is -1 (full memory).
            return_history: bool, optional. If True, return the full trajectory
                           at all time points. Default is False.

    Returns:
        y: Tensor or tuple of Tensors (with the same shape as y0) at time tspan[-1]
           If return_history=True, returns full trajectory with shape (N, *y0.shape)
    """
    memory = options.get('memory', -1)
    return_history = options.get('return_history', False)

    # Get device/dtype from y0
    device = y0[0].device if _is_tuple(y0) else y0.device
    dtype = y0[0].dtype if _is_tuple(y0) else y0.dtype

    # Ensure beta is a tensor
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=dtype, device=device)
    else:
        beta = beta.to(device=device, dtype=dtype)

    n_terms = len(beta)

    # Handle multi_coefficient: default to all ones if not provided
    if 'multi_coefficient' in options and options['multi_coefficient'] is not None:
        coefficient = options['multi_coefficient']
        if not isinstance(coefficient, torch.Tensor):
            coefficient = torch.tensor(coefficient, dtype=dtype, device=device)
        else:
            coefficient = coefficient.to(device=device, dtype=dtype)
        # Validate length matches beta
        assert len(coefficient) == n_terms, (
            f"multi_coefficient length ({len(coefficient)}) must match "
            f"beta length ({n_terms})"
        )
    else:
        # Default to all ones
        coefficient = torch.ones(n_terms, dtype=dtype, device=device)

    return solve(
        func, y0, beta, tspan,
        GrunwaldLetnikovMultiterm(coefficient),
        memory=memory,
        return_history=return_history
    )


def product_trap(func, y0, beta, tspan, **options):
    """
    Product Trapezoidal method for fractional equation D^β y(t) = f(t,y).

    Reference: Baleanu et al., "Fractional Calculus: Models and Numerical
    Methods", World Scientific, 2012.
    """
    memory = options.get('memory', -1)
    return_history = options.get('return_history', False)
    return solve(func, y0, beta, tspan, ProductTrapezoidal(),
                 memory=memory, return_history=return_history)