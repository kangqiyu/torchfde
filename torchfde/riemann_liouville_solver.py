import torch
import math
from .utils_fde import _is_tuple, _clone, _add, _multiply, _minus
from . import config

def glmethod(func, y0, beta, tspan, **options):
    """Use Grünwald-Letnikov method to integrate RL fractional equation
        D^beta y(t) = f(t,y)
        From the paper "Determining the chaotic behavior in a fractional-order
        finance system with negative parameters, Nonlinear Dynamics 94.2 (2018): 1303-1317."

        Args:
          func: returning f(t,y), with the same shape as y0
          y0: Tensor or tuple of Tensors, giving the initial state vector y(t==0)
          beta: fractional order in the range (0,1)
          tspan (array): The sequence of time points for which to solve for y.
            These must be equally spaced, e.g. np.arange(0,10,0.005)
            tspan[0] is the initial time corresponding to the initial state y0.
        Returns:
          y: Tensor or tuple of Tensors (with the same shape as y0) at time tspan[-1]
        """

    N = len(tspan)
    h = (tspan[N - 1] - tspan[0]) / (N - 1)

    # Get device from y0 (handle both tensor and tuple cases)
    device = y0[0].device if _is_tuple(y0) else y0.device
    dtype = y0[0].dtype if _is_tuple(y0) else y0.dtype

    c = torch.zeros(N + 1, dtype=dtype, device=device)
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
    """Use Product Trapezoidal method to integrate RL fractional equation
        D^beta y(t) = f(t,y)
        From the book "D. Baleanu, K. Diethelm, E. Scalas, and J. J. Trujillo,
        Fractional calculus: models and numerical methods. World Scientific, 2012, vol. 3."

        Args:
          func: returning f(t,y), with the same shape as y0
          y0: Tensor or tuple of Tensors, giving the initial state vector y(t==0)
          beta: fractional order in the range (0,1)
          tspan (array): The sequence of time points for which to solve for y.
            These must be equally spaced, e.g. np.arange(0,10,0.005)
            tspan[0] is the initial time corresponding to the initial state y0.
        Returns:
          y: Tensor or tuple of Tensors (with the same shape as y0) at time tspan[-1]
        """

    N = len(tspan)
    h = (tspan[-1] - tspan[0]) / (N - 1)
    h_alpha_gamma = torch.pow(h, beta) * math.gamma(2 - beta)
    one_minus_beta = 1 - beta

    # Get device from y0 (handle both tensor and tuple cases)
    device = y0[0].device if _is_tuple(y0) else y0.device
    dtype = y0[0].dtype if _is_tuple(y0) else y0.dtype

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
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)

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
            k_power = torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_beta)
            kp1_neg_alpha = torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), -beta)
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


def glmethod_multiterm(func, y0, beta, tspan, **options):
    """Use Grünwald-Letnikov method to integrate multi-term RL fractional equation
        Σ w_j D^{β_j} y(t) = f(t,y)

    Based on the distributed GL scheme where the aggregated weights encode
    the multi-term structure through: c̃_m = Σ w_j h^{-α_j} c_m^{(α_j)}

    Args:
        func: returning f(t,y), with the same shape as y0
        y0: Tensor or tuple of Tensors, giving the initial state vector y(t==0)
        beta: Tensor of fractional orders, each in the range (0,1), shape (n_terms,)
        coefficient: Tensor of weights w_j for each term, shape (n_terms,)
        tspan (array): The sequence of time points for which to solve for y.
            These must be equally spaced, e.g. np.arange(0,10,0.005)
            tspan[0] is the initial time corresponding to the initial state y0.
        **options:
            memory: int, optional. If specified, use short-memory approximation
                    with the given memory length M. Default is -1 (full memory).
    Returns:
        y: Tensor or tuple of Tensors (with the same shape as y0) at time tspan[-1]
    """
    N = len(tspan)
    h = (tspan[N - 1] - tspan[0]) / (N - 1)

    assert 'multi_coefficient' in options, "multi_coefficient must be provided in options"
    coefficient = options['multi_coefficient']

    # Get device from y0 (handle both tensor and tuple cases)
    device = y0[0].device if _is_tuple(y0) else y0.device
    dtype = y0[0].dtype if _is_tuple(y0) else y0.dtype

    # Ensure beta and coefficient are tensors on the correct device
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta, dtype=dtype, device=device)
    else:
        beta = beta.to(device=device, dtype=dtype)

    if not isinstance(coefficient, torch.Tensor):
        coefficient = torch.tensor(coefficient, dtype=dtype, device=device)
    else:
        coefficient = coefficient.to(device=device, dtype=dtype)

    n_terms = len(beta)
    assert len(coefficient) == n_terms, "beta and coefficient must have the same length"

    # Compute GL coefficients c_m^{(α_j)} for each fractional order
    # c[j, m] = c_m^{(α_j)} where c_0 = 1, c_m = (1 - (1+α)/m) * c_{m-1}
    c = torch.zeros(n_terms, N + 1, dtype=dtype, device=device)
    c[:, 0] = 1.0
    for m in range(1, N + 1):
        # Vectorized over all terms: c[j,m] = (1 - (1+β_j)/m) * c[j,m-1]
        c[:, m] = (1 - (1 + beta) / m) * c[:, m - 1]

    # Compute h^{-α_j} for each term
    h_neg_power = torch.pow(h, -beta)  # Shape: (n_terms,)

    # Compute distributed GL weights: c̃_m = Σ w_j * h^{-α_j} * c_m^{(α_j)}
    # Vectorized: (n_terms,) * (n_terms,) * (n_terms, N+1) -> sum over dim 0 -> (N+1,)
    weighted_h = coefficient * h_neg_power  # Shape: (n_terms,)
    c_tilde = torch.sum(weighted_h.unsqueeze(1) * c, dim=0)  # Shape: (N+1,)

    # c̃_0 = Σ w_j * h^{-α_j} (since c_0^{(α_j)} = 1 for all j)
    c_tilde_0 = c_tilde[0]

    y_current = _clone(y0)
    y_history = [y_current]  # y_history[j] stores y_j

    for k in range(N - 1):
        # Current time point t_k
        t_k = tspan[k]

        # Evaluate f(t_k, y_k) - needed for computing y_{k+1}
        f_k = func(t_k, y_current)

        # Initialize accumulator for convolution sum
        convolution_sum = None

        # Determine memory range for short-memory version
        if 'memory' not in options or options['memory'] == -1:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(options['memory'], k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Accumulate: Σ c̃_{k+1-j} * y_j for j from start_idx to k
        # This corresponds to Σ_{m=1}^{min(k+1,M)} c̃_m * y_{k+1-m}
        for j in range(start_idx, k + 1):
            coeff_idx = k + 1 - j  # Index for c̃ coefficient (m = k+1-j)
            if convolution_sum is None:
                convolution_sum = _multiply(c_tilde[coeff_idx], y_history[j])
            else:
                convolution_sum = _add(convolution_sum, _multiply(c_tilde[coeff_idx], y_history[j]))

        # Compute y_{k+1} = (1/c̃_0) * (f(t_k, y_k) - convolution_sum)
        if convolution_sum is None:
            # Edge case: no history terms (shouldn't happen for k >= 0)
            y_current = _multiply(1.0 / c_tilde_0, f_k)
        else:
            y_current = _multiply(1.0 / c_tilde_0, _minus(f_k, convolution_sum))

        # Store y_{k+1} in history
        y_history.append(y_current)

    del y_history
    return y_current