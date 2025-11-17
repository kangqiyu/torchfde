import torch
import math
from .utils_fde import _is_tuple, _clone, _add, _multiply, _minus
from . import config

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
    N = len(tspan)
    h = (tspan[N - 1] - tspan[0]) / (N - 1)

    # Get device from y0 (handle both tensor and tuple cases)
    if _is_tuple(y0):
        device = y0[0].device
    else:
        device = y0.device

    c = torch.zeros(N + 1, dtype=torch.float64, device=device)
    c[0] = 1
    for j in range(1, N + 1):
        c[j] = (1 - (1 + beta) / j) * c[j - 1]
    h_power = torch.pow(h, beta)

    y_current = _clone(y0)
    y_history = [y_current]  # y_history[j] stores y_j

    for k in range(N - 1):
        # Current time point t_k
        t_k = tspan[k]

        # Evaluate f(t_k, y_k) - needed for computing y_{k+1}
        f_k = func(t_k, y_current)

        # Initialize accumulator for convolution sum
        # if _is_tuple(y0):
        #     convolution_sum = tuple(0 for comp in y0)#tuple(torch.zeros_like(comp) for comp in y0)
        # else:
        #     convolution_sum = 0
        convolution_sum = None

        # Determine memory range
        if 'memory' not in options or options['memory'] == -1:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(options['memory'], k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Accumulate: Σ c_{k+1-j} * y_j for j from start_idx to k
        for j in range(start_idx, k + 1):
            coefficient_idx = k + 1 - j  # Index for c coefficient
            if convolution_sum is None:
                convolution_sum = _multiply(c[coefficient_idx], y_history[j])
            else:
                convolution_sum = _add(convolution_sum, _multiply(c[coefficient_idx], y_history[j]))

        # Compute y_{k+1} = h^α * f(t_k, y_k) - convolution_sum
        f_h_term = _multiply(h_power, f_k)
        y_current = _minus(f_h_term, convolution_sum)  # This is now y_{k+1}

        # Store y_{k+1} in history
        y_history.append(y_current)

    del y_history
    return y_current


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
    N = len(tspan)
    h = (tspan[-1] - tspan[0]) / (N - 1)
    h_alpha_gamma = torch.pow(h, beta) * math.gamma(2 - beta)
    one_minus_beta = 1 - beta

    # Get device from y0 (handle both tensor and tuple cases)
    if _is_tuple(y0):
        device = y0[0].device
    else:
        device = y0.device

    y_current = _clone(y0)
    yhistory = [y0]  # Store y0 as the first element

    for k in range(N - 1):
        # Current time point t_k
        t_k = tspan[k]

        # Evaluate f(t_k, y_k)
        f_k = func(t_k, y_current)

        # Determine memory range
        if 'memory' not in options or options['memory'] == -1:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(options['memory'], k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Vectorized computation of A_{j,k+1} weights for indices from start_idx to k
        j_vals = torch.arange(start_idx, k + 1, dtype=torch.float32, device=device)

        # Compute A_{j,k+1} for all j values
        # For j=0: A_{0,k+1} = k^{1-α} - (k+α)(k+1)^{-α}
        # For 1≤j≤k: A_{j,k+1} = (k+2-j)^{1-α} + (k-j)^{1-α} - 2(k+1-j)^{1-α}

        # Compute the general formula for j >= 1
        kjp2 = torch.pow(k + 2 - j_vals, one_minus_beta)
        kj = torch.pow(k - j_vals, one_minus_beta)
        kjp1 = torch.pow(k + 1 - j_vals, one_minus_beta)

        # General formula for j >= 1
        A_j_kp1 = kjp2 + kj - 2 * kjp1

        # Special handling for j=0 if it's in the range
        if start_idx == 0:
            k_power = torch.pow(torch.tensor(k, dtype=torch.float32, device=device), one_minus_beta)
            kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=torch.float32, device=device), -beta)
            A_j_kp1[0] = k_power - (k + beta) * kp1_neg_alpha

        # Initialize accumulator for the sum
        convolution_sum = None

        # Accumulate: sum from j=start_idx to k
        for j in range(start_idx, k + 1):
            local_idx = j - start_idx  # Index into A_j_kp1 array
            if convolution_sum is None:
                convolution_sum = _multiply(A_j_kp1[local_idx], yhistory[j])
            else:
                convolution_sum = _add(convolution_sum, _multiply(A_j_kp1[local_idx], yhistory[j]))

        # Compute y_{k+1} = Γ(2-α) * h^α * f(t_k, y_k) - sum
        f_term = _multiply(h_alpha_gamma, f_k)
        y_current = _add(f_term, _multiply(-1, convolution_sum))

        # Store y_{k+1} in history
        yhistory.append(y_current)

    # Release memory
    del yhistory
    return y_current
