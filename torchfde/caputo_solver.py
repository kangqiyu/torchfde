"""
Deprecation Notice: `predictor` and `l1solver` have been moved to `explicit_solver.py`.

"""

import torch
import math
from .utils_fde import _is_tuple, _clone, _add, _multiply

# def fractional_pow(base, exponent):
#     eps = 1e-4
#     return torch.pow(base, exponent)

def predictor(func, y0, beta, tspan, **options):
    """Use one-step Adams-Bashforth (Euler) method to integrate Caputo equation
        D^beta y(t) = f(t,y)
        From the paper "Detailed error analysis for a fractional adams method,
        Numerical algorithms, vol. 36, no. 1, pp. 31–52, 2004."

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
    # gamma_beta = 1 / math.gamma(beta)
    h_beta_over_beta = torch.pow(h, beta) / (beta * math.gamma(beta))
    # Get device from y0 (handle both tensor and tuple cases)
    device = y0[0].device if _is_tuple(y0) else y0.device
    dtype = y0[0].dtype if _is_tuple(y0) else y0.dtype

    y_current = _clone(y0)
    fhistory = []

    for k in range(N - 1):
        # Current time point t_k
        t_k = tspan[k]

        # Evaluate and store f(t_k, y_k)
        f_k = func(t_k, y_current)
        fhistory.append(f_k)

        # Determine memory range
        if 'memory' not in options or options['memory'] == -1:
            memory_length = k + 1  # Use all available history
        else:
            memory_length = min(options['memory'], k + 1)
            assert memory_length > 0, "memory must be greater than 0"

        start_idx = max(0, k + 1 - memory_length)

        # Compute weights for indices from start_idx to k
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device).unsqueeze(1)
        b_j_k_1 = h_beta_over_beta * (torch.pow(k + 1 - j_vals, beta) - torch.pow(k - j_vals, beta))

        # Initialize accumulator
        # if _is_tuple(y0):
        #     convolution_sum = tuple(torch.zeros_like(y_i) for y_i in y0)
        # else:
        #     convolution_sum = torch.zeros_like(y0)
        convolution_sum = None

        # Accumulate: sum from j=start_idx to k
        for j in range(start_idx, k + 1):
            local_idx = j - start_idx  # Index into b_j_k_1 array
            if convolution_sum is None:
                convolution_sum = _multiply(b_j_k_1[local_idx], fhistory[j])
            else:
                convolution_sum = _add(convolution_sum, _multiply(b_j_k_1[local_idx], fhistory[j]))

        # Compute y_{k+1}
        weight_term = convolution_sum #_multiply(gamma_beta, convolution_sum)
        y_current = _add(y0, weight_term)

    # release memory
    del fhistory
    del b_j_k_1
    return y_current


def l1solver(func, y0, beta, tspan, **options):
    """Use L1 method to integrate Caputo equation
        D^beta y(t) = f(t,y)
        From the paper “A compact finite difference scheme for the fractional sub-diffusion equations,
        Journal of Computational Physics, vol. 230, no. 3, pp. 586–595, 2011."

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

        # Vectorized computation of c_j^(k) weights for indices from start_idx to k
        j_vals = torch.arange(start_idx, k + 1, dtype=dtype, device=device)

        # Compute c_j^(k) for all j values
        # For j >= 1: c_j^(k) = (k-j+2)^{1-α} - 2(k-j+1)^{1-α} + (k-j)^{1-α}
        # For j = 0: c_0^(k) = -[(k+1)^{1-α} - k^{1-α}]

        # Compute the general formula for all j
        kjp2 = torch.pow(k + 2 - j_vals, one_minus_beta)
        kjp1 = torch.pow(k + 1 - j_vals, one_minus_beta)
        kj = torch.pow(k - j_vals, one_minus_beta)
        c_j_k = kjp2 - 2 * kjp1 + kj

        # Special handling for j=0 if it's in the range
        if start_idx == 0:
            c_j_k[0] = -(torch.pow(torch.tensor(k + 1, dtype=dtype, device=device), one_minus_beta) -
                         torch.pow(torch.tensor(k, dtype=dtype, device=device), one_minus_beta))

        # Initialize accumulator for the sum
        # if _is_tuple(y0):
        #     convolution_sum = tuple(torch.zeros_like(y_i) for y_i in y0)
        # else:
        #     convolution_sum = torch.zeros_like(y0)
        convolution_sum = None


        # Accumulate: sum from j=start_idx to k
        for j in range(start_idx, k + 1):
            local_idx = j - start_idx  # Index into c_j_k array
            if convolution_sum is None:
                convolution_sum = _multiply(c_j_k[local_idx], yhistory[j])
            else:
                convolution_sum = _add(convolution_sum, _multiply(c_j_k[local_idx], yhistory[j]))

        # Compute y_{k+1} = h^α * Γ(2-α) * f(t_k, y_k) - sum
        f_term = _multiply(h_alpha_gamma, f_k)
        y_current = _add(f_term, _multiply(-1, convolution_sum))

        # Store y_{k+1} in history
        yhistory.append(y_current)

    # Release memory
    del yhistory
    return y_current



##################################################
# Predictor_Corrector has not been fully checked #
##################################################

def predictor_corrector(func, y0, beta, tspan,**options):
    """Use one-step Adams-Bashforth (Euler) method to integrate Caputo equation
        D^beta y(t) = f(t,y)

        Raises:
          FODEValueError
        See also:
          K. Diethelm et al. (2004) Detailed error analysis for a fractional Adams
             method
          C. Li and F. Zeng (2012) Finite Difference Methods for Fractional
             Differential Equations
        """
    N = len(tspan)
    h = (tspan[-1] - tspan[0]) / (N - 1)
    gamma_beta = 1 / math.gamma(beta)
    a_item = torch.pow(h, beta) / (beta * (beta + 1))
    fhistory = []

    # Get device from y0 (handle both tensor and tuple cases)
    device = y0[0].device if _is_tuple(y0) else y0.device
    dtype = y0[0].dtype if _is_tuple(y0) else y0.dtype

    yn = _clone(y0)

    for k in range(N-1):
        tn = tspan[k]
        f_k = func(tn, yn)
        fhistory.append(f_k)

        # can apply short memory here
        if 'memory' not in options:
            memory = k+1
        else:
            memory = options['memory']
        memory_k = max(0, k+1 - memory)

        j_vals = torch.arange(0, k + 1, dtype=dtype, device=device).unsqueeze(1)
        b_j_k_1 = (torch.pow(h, beta) / beta) * (
                torch.pow(k + 1 - j_vals, beta) - torch.pow(k - j_vals, beta))

        # Initialize accumulator with correct structure (tensor or tuple)
        if _is_tuple(fhistory[memory_k]):
            b_all_k = tuple(torch.zeros_like(f_i) for f_i in fhistory[memory_k])
        else:
            b_all_k = torch.zeros_like(fhistory[memory_k])

        # Loop through the range and accumulate results
        for i in range(memory_k, k + 1):
            b_all_k = _add(b_all_k, _multiply(b_j_k_1[i], fhistory[i]))

        # Final update step
        weight_term = _multiply(gamma_beta, b_all_k)
        yn_ = _add(y0, weight_term)



        a_j_k_1 = a_item * torch.ones((k + 2, 1), dtype=dtype, device=device)
        a_j_k_1[0] = a_item * (torch.pow(k, beta + 1) - (k - beta) * torch.pow(k + 1, beta))
        for j in range(1, k + 1):
            a_j_k_1[j] = a_item * (
                        torch.pow(k + 2 - j, beta + 1) + torch.pow(k - j, beta + 1) - 2 * torch.pow(k + 1 - j,
                                                                                                    beta + 1))
        # corrector
        if 'corrector_step' not in options:
            corrector_step = 1
        else:
            corrector_step = options['corrector_step']

        if _is_tuple(fhistory[memory_k]):
            a_all_k = tuple(torch.zeros_like(f_i) for f_i in fhistory[memory_k])
        else:
            a_all_k = torch.zeros_like(fhistory[memory_k])

        for i in range(memory_k, k + 1):
            a_all_k = _add(a_all_k, _multiply(a_j_k_1[i], fhistory[i]))

        for _ in range(corrector_step):
            a_all_k_final = _add(a_all_k, _multiply(a_j_k_1[k+1], func(tn+h, yn_)))
            # multiple updates for corrector
            weight_term = _multiply(gamma_beta, a_all_k_final)
            yn_ = _add(y0, weight_term)

        yn = yn_

    # release memory
    del fhistory
    del b_j_k_1

    return yn






